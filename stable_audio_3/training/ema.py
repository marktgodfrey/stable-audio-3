import copy
import typing as tp

import torch
from torch import nn

_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def resolve_dtype(dtype: tp.Union[str, torch.dtype, None]) -> tp.Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unknown dtype '{dtype}', valid: {sorted(_DTYPE_ALIASES)}")
    return _DTYPE_ALIASES[key]


class EMA(nn.Module):
    """Exponential moving average of a model's weights with power-law decay warmup.

    The decay at update step t is:

        decay(t) = min(beta, 1 - (1 + t / inv_gamma) ** -power)

    Defaults (beta=0.9995, power=0.75) follow section 3.5 of the Stable Audio 3
    technical report (arXiv 2605.17991). decay(0) = 0, so the first update copies
    the online weights into the EMA model.

    The averaged copy is exposed as `ema_model` and can be used as a drop-in
    replacement for the online model at inference time. `ema_model` and the update
    step counter are registered on this module, so adding an EMA instance as a
    submodule of a LightningModule checkpoints and resumes EMA state automatically.
    The online model is intentionally NOT registered, so its weights are not
    duplicated in this module's state_dict.

    Updates are plain parameter-space lerps with no gradient flow; under DDP the
    online weights are identical on every rank after each optimizer step, so
    updating on all ranks keeps every rank's EMA copy identical. Not FSDP-aware:
    with sharded parameters the name-based matching below would not see full
    weights.

    Args:
        model: the online model to track.
        beta: maximum (asymptotic) decay rate.
        power: exponent of the power-law warmup.
        inv_gamma: time-scale divisor of the power-law warmup.
        update_every: apply the EMA update only every N calls to update(). The
            decay schedule is not compensated, so values > 1 stretch the effective
            averaging horizon.
        update_after_step: number of update() calls before averaging begins;
            until then the EMA copy mirrors the online weights.
        dtype: storage dtype for the EMA copy (e.g. "bfloat16" to halve memory).
            Defaults to the online model's dtype.
        device: pinned device for the EMA copy (e.g. "cpu" to keep it out of GPU
            memory). Defaults to following the online model. A pinned device adds
            a device transfer to every update, and the copy must be moved back to
            the compute device before it can be sampled from (see
            `pin_ema_device`).
    """

    def __init__(
            self,
            model: nn.Module,
            beta: float = 0.9995,
            power: float = 0.75,
            inv_gamma: float = 1.0,
            update_every: int = 1,
            update_after_step: int = 0,
            dtype: tp.Union[str, torch.dtype, None] = None,
            device: tp.Union[str, torch.device, None] = None,
    ):
        super().__init__()

        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")

        self.beta = beta
        self.power = power
        self.inv_gamma = inv_gamma
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.ema_device = torch.device(device) if device is not None else None

        # Held in a plain list so the online model is not registered as a
        # submodule (keeps it out of this module's state_dict).
        self._online_model = [model]

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        ema_dtype = resolve_dtype(dtype)
        if ema_dtype is not None:
            self.ema_model.to(dtype=ema_dtype)
        if self.ema_device is not None:
            self.ema_model.to(device=self.ema_device)

        # Number of update() calls performed so far. A buffer, so it is saved
        # and restored with the EMA state.
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    @property
    def online_model(self) -> nn.Module:
        return self._online_model[0]

    def get_decay(self, step: tp.Optional[int] = None) -> float:
        """Decay rate used at the given update step (defaults to the current step)."""
        if step is None:
            step = int(self.step.item())
        t = step - self.update_after_step
        if t <= 0:
            return 0.0
        decay = 1.0 - (1.0 + t / self.inv_gamma) ** -self.power
        return min(max(decay, 0.0), self.beta)

    @torch.no_grad()
    def copy_params_from_model_to_ema(self):
        online_params = dict(self.online_model.named_parameters())
        for name, ema_param in self.ema_model.named_parameters():
            ema_param.copy_(online_params[name].to(device=ema_param.device, dtype=ema_param.dtype))
        self._copy_buffers_from_model()

    @torch.no_grad()
    def _copy_buffers_from_model(self):
        online_buffers = dict(self.online_model.named_buffers())
        for name, ema_buffer in self.ema_model.named_buffers():
            ema_buffer.copy_(online_buffers[name].to(device=ema_buffer.device))

    @torch.no_grad()
    def _lerp_params(self, decay: float):
        online_params = dict(self.online_model.named_parameters())
        ema_tensors = []
        online_tensors = []
        fast_path = True
        for name, ema_param in self.ema_model.named_parameters():
            online_param = online_params[name]
            ema_tensors.append(ema_param)
            online_tensors.append(online_param)
            if ema_param.dtype != online_param.dtype or ema_param.device != online_param.device:
                fast_path = False

        if fast_path and hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(ema_tensors, online_tensors, 1.0 - decay)
        else:
            for ema_param, online_param in zip(ema_tensors, online_tensors):
                ema_param.lerp_(online_param.to(device=ema_param.device, dtype=ema_param.dtype), 1.0 - decay)

    @torch.no_grad()
    def update(self):
        """Advance the EMA by one step. Call once per optimizer step."""
        step = int(self.step.item())
        self.step += 1

        if step % self.update_every != 0:
            return

        decay = self.get_decay(step)
        if decay <= 0.0:
            self.copy_params_from_model_to_ema()
        else:
            self._lerp_params(decay)
            # Buffers (e.g. positional caches) are copied, not averaged.
            self._copy_buffers_from_model()

    def pin_ema_device(self):
        """Move the EMA copy back to its pinned device. No-op when not pinned."""
        if self.ema_device is not None:
            self.ema_model.to(self.ema_device)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
