import math

import torch

from stable_audio_3.training.utils import bucket_loss_by_sigma

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PADDING_POISON = 1e6  # large value standing in for unconstrained model output on padding


def single_rank_gather(tensor):
    """Emulate Lightning's all_gather with world_size=1: adds a leading world dim."""
    return tensor.unsqueeze(0)


def expected_buckets(world_loss, world_sigmas, world_mask, num_buckets=10):
    """Independently compute per-bucket means over loss-contributing positions.

    Args:
        world_loss: (W, B, C, T) raw per-element loss
        world_sigmas: (W, B) sigma per sample
        world_mask: (W, B, T) boolean loss mask
    """
    bucket_size = 1 / num_buckets
    out = []
    for i in range(num_buckets):
        lo, hi = i * bucket_size, (i + 1) * bucket_size
        vals = []
        for w in range(world_loss.shape[0]):
            for b in range(world_loss.shape[1]):
                if lo <= world_sigmas[w, b] < hi:
                    vals.extend(world_loss[w, b][:, world_mask[w, b]].flatten().tolist())
        out.append(sum(vals) / len(vals) if vals else math.nan)
    return torch.tensor(out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_masked_positions_excluded_from_buckets():
    """Padding positions must not contribute to the bucketed diagnostics."""
    batch, channels, seq_len = 4, 2, 8
    sigmas = torch.tensor([0.05, 0.15, 0.17, 0.95])

    # Signal positions hold a known per-sample constant; padding holds poison
    signal_lengths = [8, 6, 4, 5]
    signal_values = [1.0, 2.0, 4.0, 8.0]
    loss_all = torch.full((batch, channels, seq_len), PADDING_POISON)
    loss_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for b, (length, value) in enumerate(zip(signal_lengths, signal_values)):
        loss_all[b, :, :length] = value
        loss_mask[b, :length] = True

    buckets = bucket_loss_by_sigma(
        loss_all, sigmas[:, None, None], loss_mask, single_rank_gather
    )

    assert buckets.shape == (10,)
    # Bucket 0: sample 0 only (fully valid, so masking is a no-op there)
    assert buckets[0].item() == 1.0
    # Bucket 1: samples 1 and 2 pooled, weighted by their signal element counts
    assert torch.isclose(buckets[1], torch.tensor((2.0 * 12 + 4.0 * 8) / 20))
    # Bucket 9: sample 3 only
    assert buckets[9].item() == 8.0
    # Remaining buckets received no samples
    for i in (2, 3, 4, 5, 6, 7, 8):
        assert torch.isnan(buckets[i])
    # Nothing leaked from the poisoned padding region
    assert buckets[~torch.isnan(buckets)].max() < PADDING_POISON / 2

    expected = expected_buckets(
        loss_all.unsqueeze(0), sigmas.unsqueeze(0), loss_mask.unsqueeze(0)
    )
    torch.testing.assert_close(buckets, expected, equal_nan=True)


def test_multi_rank_gather_pools_signal_positions():
    """Shapes and values must hold through a world_size=2 all_gather."""
    batch, channels, seq_len = 2, 3, 4

    sigmas = [torch.tensor([0.25, 0.55]), torch.tensor([0.28, 0.95])]
    loss = [
        torch.full((batch, channels, seq_len), PADDING_POISON),
        torch.full((batch, channels, seq_len), PADDING_POISON),
    ]
    mask = [torch.zeros(batch, seq_len, dtype=torch.bool) for _ in range(2)]
    # rank 0: sample 0 -> 3 valid steps of value 1.0, sample 1 -> 2 valid steps of 5.0
    loss[0][0, :, :3] = 1.0
    mask[0][0, :3] = True
    loss[0][1, :, :2] = 5.0
    mask[0][1, :2] = True
    # rank 1: sample 0 -> 1 valid step of 3.0 (shares bucket 2 with rank 0's sample 0),
    # sample 1 -> fully valid of 7.0
    loss[1][0, :, :1] = 3.0
    mask[1][0, :1] = True
    loss[1][1] = 7.0
    mask[1][1] = True

    def two_rank_gather(tensor):
        # Each rank NaN-masks its own loss before the collective, so rank 1
        # contributes its own masked tensor; sigmas are gathered unmasked
        if tensor.shape == (batch, 1, 1):
            return torch.stack([tensor, sigmas[1][:, None, None]])
        assert tensor.shape == (batch, channels, seq_len)
        rank1_masked = torch.where(mask[1].unsqueeze(1), loss[1], float("nan"))
        return torch.stack([tensor, rank1_masked])

    buckets = bucket_loss_by_sigma(
        loss[0], sigmas[0][:, None, None], mask[0], two_rank_gather
    )

    assert buckets.shape == (10,)
    expected = expected_buckets(torch.stack(loss), torch.stack(sigmas), torch.stack(mask))
    torch.testing.assert_close(buckets, expected, equal_nan=True)

    # Bucket 2 pools both ranks' sample 0: 9 elements of 1.0 and 3 of 3.0
    assert torch.isclose(buckets[2], torch.tensor((1.0 * 9 + 3.0 * 3) / 12))
    assert torch.isclose(buckets[5], torch.tensor(5.0))
    assert torch.isclose(buckets[9], torch.tensor(7.0))


def test_fully_masked_sample_yields_nan_bucket():
    """A bucket whose only sample has no loss-contributing positions logs as NaN
    (and is then skipped by the isnan filter in training_step)."""
    batch, channels, seq_len = 2, 1, 4
    sigmas = torch.tensor([0.05, 0.95])
    loss_all = torch.full((batch, channels, seq_len), PADDING_POISON)
    loss_all[1] = 2.0
    loss_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    loss_mask[1] = True

    buckets = bucket_loss_by_sigma(
        loss_all, sigmas[:, None, None], loss_mask, single_rank_gather
    )

    assert torch.isnan(buckets[0])
    assert torch.isclose(buckets[9], torch.tensor(2.0))


def test_all_true_mask_matches_unmasked_buckets():
    """With nothing masked, results are identical to the pre-fix computation."""
    torch.manual_seed(0)
    batch, channels, seq_len = 6, 2, 5
    num_buckets = 10
    bucket_size = 1 / num_buckets
    sigmas = torch.rand(batch)
    loss_all = torch.rand(batch, channels, seq_len)
    loss_mask = torch.ones(batch, seq_len, dtype=torch.bool)

    buckets = bucket_loss_by_sigma(
        loss_all, sigmas[:, None, None], loss_mask, single_rank_gather, num_buckets
    )

    # Original (unmasked) bucketing from training_step
    reference = torch.stack(
        [
            loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean()
            for i in torch.arange(0, 1, bucket_size)
        ]
    )
    torch.testing.assert_close(buckets, reference, equal_nan=True)
