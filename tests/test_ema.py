import types

import pytest
import torch
from torch import nn

from stable_audio_3.training.ema import EMA

# ---------------------------------------------------------------------------
# EMA class tests (torch-only, no training extras required)
# ---------------------------------------------------------------------------


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(16, 8)
        self.register_buffer("float_buf", torch.zeros(4))
        self.register_buffer("int_buf", torch.zeros(2, dtype=torch.long))

    def forward(self, x):
        return self.lin2(torch.nn.functional.gelu(self.lin1(x)))


def _perturb(model, offset=1.0):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(offset)
        model.float_buf.add_(offset)
        model.int_buf.add_(1)


def _params_equal(model_a, model_b):
    params_b = dict(model_b.named_parameters())
    return all(
        torch.equal(param, params_b[name].to(param.dtype))
        for name, param in model_a.named_parameters()
    )


def test_decay_schedule_matches_power_law():
    ema = EMA(_ToyModel(), beta=0.9995, power=0.75, inv_gamma=1.0)

    assert ema.get_decay(0) == 0.0
    assert ema.get_decay(1) == pytest.approx(1.0 - 2.0**-0.75)
    assert ema.get_decay(100) == pytest.approx(1.0 - 101.0**-0.75)
    # Far past warmup the decay caps at beta
    assert ema.get_decay(10**9) == 0.9995

    decays = [ema.get_decay(t) for t in range(0, 10_000, 100)]
    assert decays == sorted(decays)


def test_first_update_copies_then_lerps():
    torch.manual_seed(0)
    model = _ToyModel()
    ema = EMA(model)

    # EMA starts as a copy of the online model
    assert _params_equal(ema.ema_model, model)

    # First update is a pure copy (decay(0) == 0)
    _perturb(model)
    ema.update()
    assert ema.step.item() == 1
    assert _params_equal(ema.ema_model, model)
    assert torch.equal(ema.ema_model.float_buf, model.float_buf)
    assert torch.equal(ema.ema_model.int_buf, model.int_buf)

    # Second update lerps with decay(1)
    prev_ema = {n: p.clone() for n, p in ema.ema_model.named_parameters()}
    _perturb(model)
    ema.update()

    decay = ema.get_decay(1)
    assert 0.0 < decay < 1.0
    online = dict(model.named_parameters())
    for name, param in ema.ema_model.named_parameters():
        expected = torch.lerp(prev_ema[name], online[name], 1.0 - decay)
        torch.testing.assert_close(param, expected)
        assert not torch.equal(param, online[name])
    # Buffers are copied, not averaged
    assert torch.equal(ema.ema_model.float_buf, model.float_buf)
    assert torch.equal(ema.ema_model.int_buf, model.int_buf)


def test_update_every_skips_intermediate_steps():
    torch.manual_seed(0)
    model = _ToyModel()
    ema = EMA(model, update_every=2)

    ema.update()  # step 0: applies (copy)
    _perturb(model)
    snapshot = {n: p.clone() for n, p in ema.ema_model.named_parameters()}

    ema.update()  # step 1: skipped
    for name, param in ema.ema_model.named_parameters():
        assert torch.equal(param, snapshot[name])

    ema.update()  # step 2: applies
    assert ema.step.item() == 3
    for name, param in ema.ema_model.named_parameters():
        assert not torch.equal(param, snapshot[name])


def test_state_dict_roundtrip_resumes_identically():
    torch.manual_seed(0)
    model = _ToyModel()
    ema = EMA(model)
    for _ in range(3):
        _perturb(model, 0.5)
        ema.update()

    state = ema.state_dict()
    assert any(key.startswith("ema_model.") for key in state)
    assert state["step"].item() == 3
    # The online model must not be duplicated into the EMA state
    assert not any(key.startswith("_online_model") for key in state)

    torch.manual_seed(1)
    model2 = _ToyModel()
    model2.load_state_dict(model.state_dict())
    ema2 = EMA(model2)
    ema2.load_state_dict(state)

    assert ema2.step.item() == 3
    assert _params_equal(ema2.ema_model, ema.ema_model)

    # Updates after the round-trip stay identical (decay schedule continues)
    _perturb(model, 0.5)
    _perturb(model2, 0.5)
    ema.update()
    ema2.update()
    assert _params_equal(ema2.ema_model, ema.ema_model)


def test_bf16_ema_storage():
    torch.manual_seed(0)
    model = _ToyModel()
    ema = EMA(model, dtype="bfloat16")

    assert all(p.dtype == torch.bfloat16 for p in ema.ema_model.parameters())

    _perturb(model)
    ema.update()
    ema.update()

    online = dict(model.named_parameters())
    for name, param in ema.ema_model.named_parameters():
        assert param.dtype == torch.bfloat16
        torch.testing.assert_close(param.float(), online[name], rtol=1e-2, atol=1e-2)


def test_device_pinning_is_noop_when_unpinned():
    ema = EMA(_ToyModel())
    assert ema.ema_device is None
    ema.pin_ema_device()

    pinned = EMA(_ToyModel(), device="cpu")
    assert pinned.ema_device == torch.device("cpu")
    pinned.update()
    pinned.pin_ema_device()


# ---------------------------------------------------------------------------
# Training wrapper integration tests (require the [training] extras)
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_training_wrapper():
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("wandb")
    from stable_audio_3.models.conditioners import MultiConditioner
    from stable_audio_3.models.diffusion import ConditionedDiffusionModelWrapper
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    def _make(use_ema=True, ema_config=None, seed=0):
        torch.manual_seed(seed)
        cond_wrapper = ConditionedDiffusionModelWrapper(
            _ToyModel(),
            MultiConditioner({}),
            io_channels=8,
            sample_rate=16000,
            min_input_length=64,
            diffusion_objective="rectified_flow",
        )
        return DiffusionCondTrainingWrapper(
            cond_wrapper,
            lr=1e-4,
            use_ema=use_ema,
            ema_config=ema_config,
            sample_rate=16000,
            sample_size=1024,
        )

    return _make


def test_wrapper_builds_ema_and_checkpoint_roundtrips(make_training_wrapper):
    wrapper = make_training_wrapper(use_ema=True, seed=0)
    assert wrapper.diffusion_ema is not None
    assert _params_equal(wrapper.diffusion_ema.ema_model, wrapper.diffusion.model)

    # Let EMA and online weights diverge so the round-trip is meaningful
    _perturb(wrapper.diffusion.model)
    wrapper.diffusion_ema.update()
    _perturb(wrapper.diffusion.model)
    wrapper.diffusion_ema.update()
    assert not _params_equal(wrapper.diffusion_ema.ema_model, wrapper.diffusion.model)

    checkpoint = {"state_dict": wrapper.state_dict()}
    assert any(key.startswith("diffusion_ema.") for key in checkpoint["state_dict"])

    resumed = make_training_wrapper(use_ema=True, seed=1)
    resumed.on_load_checkpoint(checkpoint)
    resumed.load_state_dict(checkpoint["state_dict"], strict=True)

    assert resumed.diffusion_ema.step.item() == 2
    assert _params_equal(
        resumed.diffusion_ema.ema_model, wrapper.diffusion_ema.ema_model
    )


def test_wrapper_seeds_ema_from_checkpoint_without_ema_state(make_training_wrapper):
    wrapper = make_training_wrapper(use_ema=False, seed=0)
    _perturb(wrapper.diffusion.model)
    checkpoint = {"state_dict": wrapper.state_dict()}
    assert not any(key.startswith("diffusion_ema.") for key in checkpoint["state_dict"])

    resumed = make_training_wrapper(use_ema=True, seed=1)
    resumed.on_load_checkpoint(checkpoint)
    resumed.load_state_dict(checkpoint["state_dict"], strict=True)

    # EMA seeded from the checkpoint's online weights, warmup restarted
    assert resumed.diffusion_ema.step.item() == 0
    assert _params_equal(resumed.diffusion_ema.ema_model, wrapper.diffusion.model)
    assert _params_equal(resumed.diffusion.model, wrapper.diffusion.model)


def test_wrapper_drops_ema_state_when_disabled(make_training_wrapper):
    wrapper = make_training_wrapper(use_ema=True, seed=0)
    wrapper.diffusion_ema.update()
    checkpoint = {"state_dict": wrapper.state_dict()}

    resumed = make_training_wrapper(use_ema=False, seed=1)
    resumed.on_load_checkpoint(checkpoint)
    resumed.load_state_dict(checkpoint["state_dict"], strict=True)
    assert resumed.diffusion_ema is None


def test_wrapper_updates_ema_once_per_optimizer_step(make_training_wrapper):
    wrapper = make_training_wrapper(use_ema=True)
    fake_trainer = types.SimpleNamespace(global_step=0, world_size=1)
    wrapper._trainer = fake_trainer
    wrapper.on_fit_start()
    wrapper.on_train_start()

    # No optimizer step yet (e.g. mid gradient accumulation): no EMA update
    wrapper.on_train_batch_end(None, None, 0)
    assert wrapper.diffusion_ema.step.item() == 0

    fake_trainer.global_step = 1
    wrapper.on_train_batch_end(None, None, 1)
    assert wrapper.diffusion_ema.step.item() == 1

    # Same global step again: still no second update
    wrapper.on_train_batch_end(None, None, 2)
    assert wrapper.diffusion_ema.step.item() == 1

    fake_trainer.global_step = 2
    wrapper.on_train_batch_end(None, None, 3)
    assert wrapper.diffusion_ema.step.item() == 2


def test_wrapper_export_model_uses_ema_weights(make_training_wrapper, tmp_path):
    wrapper = make_training_wrapper(use_ema=True, seed=0)
    wrapper.diffusion_ema.update()
    ema_state = {
        name: param.clone()
        for name, param in wrapper.diffusion_ema.ema_model.named_parameters()
    }

    # Diverge the online weights after the EMA update
    _perturb(wrapper.diffusion.model)
    assert not _params_equal(wrapper.diffusion_ema.ema_model, wrapper.diffusion.model)

    export_path = tmp_path / "exported.ckpt"
    wrapper.export_model(str(export_path))
    exported = torch.load(export_path, weights_only=True)["state_dict"]

    for name, expected in ema_state.items():
        torch.testing.assert_close(exported[f"model.{name}"], expected)


def test_lora_mode_disables_ema():
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("wandb")
    from stable_audio_3.models.conditioners import MultiConditioner
    from stable_audio_3.models.diffusion import ConditionedDiffusionModelWrapper
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    torch.manual_seed(0)
    cond_wrapper = ConditionedDiffusionModelWrapper(
        _ToyModel(),
        MultiConditioner({}),
        io_channels=8,
        sample_rate=16000,
        min_input_length=64,
        diffusion_objective="rectified_flow",
    )
    wrapper = DiffusionCondTrainingWrapper(
        cond_wrapper,
        lr=1e-4,
        use_ema=True,
        lora_config={"rank": 2},
        sample_rate=16000,
        sample_size=1024,
    )
    assert wrapper.diffusion_ema is None


def test_ema_deepcopy_excludes_conditioner(make_training_wrapper):
    wrapper = make_training_wrapper(use_ema=True)
    ema_keys = set(wrapper.diffusion_ema.ema_model.state_dict().keys())
    online_keys = set(wrapper.diffusion.model.state_dict().keys())
    assert ema_keys == online_keys


class _ToyDenoiser(nn.Module):
    """Minimal denoiser with the (x, t, **kwargs) interface the DiT exposes."""

    def __init__(self, io_channels=8):
        super().__init__()
        self.net = nn.Conv1d(io_channels, io_channels, 3, padding=1)

    def forward(self, x, t, **kwargs):
        return self.net(x)


def test_trainer_fit_updates_checkpoints_and_resumes_ema(tmp_path):
    """Full pl.Trainer integration: EMA updates once per optimizer step (with
    gradient accumulation), rides along in Lightning checkpoints, and resumes."""
    pl = pytest.importorskip("pytorch_lightning")
    pytest.importorskip("wandb")
    from torch.utils.data import DataLoader, Dataset

    from stable_audio_3.models.conditioners import MultiConditioner
    from stable_audio_3.models.diffusion import ConditionedDiffusionModelWrapper
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    io_channels, seq_len = 8, 64

    class ToyDataset(Dataset):
        def __len__(self):
            return 64

        def __getitem__(self, idx):
            g = torch.Generator().manual_seed(idx)
            return torch.randn(io_channels, seq_len, generator=g), {"prompt": "toy"}

    def collate(batch):
        return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

    def make_wrapper(use_ema=True):
        torch.manual_seed(0)
        cond_wrapper = ConditionedDiffusionModelWrapper(
            _ToyDenoiser(io_channels),
            MultiConditioner({}),
            io_channels=io_channels,
            sample_rate=16000,
            min_input_length=1,
            diffusion_objective="rectified_flow",
        )
        return DiffusionCondTrainingWrapper(
            cond_wrapper,
            lr=1e-3,
            use_ema=use_ema,
            sample_rate=16000,
            sample_size=seq_len,
            ot_coupling=False,
        )

    def make_trainer(max_steps):
        return pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=max_steps,
            accumulate_grad_batches=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

    dl = DataLoader(ToyDataset(), batch_size=4, collate_fn=collate)

    # 8 batches with accumulate_grad_batches=2 -> 4 optimizer steps -> 4 EMA updates
    wrapper = make_wrapper()
    trainer = make_trainer(max_steps=4)
    trainer.fit(wrapper, dl)
    assert wrapper.diffusion_ema.step.item() == 4
    assert not _params_equal(wrapper.diffusion_ema.ema_model, wrapper.diffusion.model)

    ckpt_path = str(tmp_path / "ema.ckpt")
    trainer.save_checkpoint(ckpt_path)
    state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
    assert state_dict["diffusion_ema.step"].item() == 4

    # Resume continues the EMA exactly: 2 more optimizer steps -> step 6, and the
    # accumulation guard must re-anchor (one update per step, none spurious).
    resumed = make_wrapper()
    make_trainer(max_steps=6).fit(resumed, dl, ckpt_path=ckpt_path)
    assert resumed.diffusion_ema.step.item() == 6

    # Resuming a checkpoint saved without EMA state seeds the EMA and trains on
    ckpt = torch.load(ckpt_path, weights_only=False)
    for key in [k for k in ckpt["state_dict"] if k.startswith("diffusion_ema.")]:
        del ckpt["state_dict"][key]
    stripped_path = str(tmp_path / "ema_stripped.ckpt")
    torch.save(ckpt, stripped_path)

    seeded = make_wrapper()
    make_trainer(max_steps=5).fit(seeded, dl, ckpt_path=stripped_path)
    assert seeded.diffusion_ema.step.item() == 1

    # And use_ema=False resumes cleanly from a checkpoint that has EMA state
    no_ema = make_wrapper(use_ema=False)
    make_trainer(max_steps=5).fit(no_ema, dl, ckpt_path=ckpt_path)
    assert no_ema.diffusion_ema is None
