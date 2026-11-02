"""Weight averaging: an EMA of the trajectory, or the mean of the last N epochs.

Why either exists: SGD's last iterate sits wherever the final step left it, and
on a ten-epoch schedule that is a noisier point than the neighbourhood it is in.
Averaging picks a point in the middle of the trajectory instead. It is cheap --
no extra forward passes during training -- and it does not touch the optimiser,
so a run with averaging on is the same run plus a second set of weights.

    from kappa.training.averaging import build_averager

    averager = build_averager(cfg, model)          # None when the config says off
    ...
    averager.after_step(model)                     # EMA updates here
    averager.after_epoch(model)                    # last-N snapshots here
    with averager.applied(model):
        score = validate(model)                    # averaged weights, then restored

**The averaged weights are scored and saved *alongside* the raw ones, never
instead of them.** The point of an arm is the comparison, and a run that reports
one number cannot say whether averaging helped. It also keeps ``select="last"``
meaning what it says: the selection rule is unchanged, and averaging is a second
set of weights derived from the same trajectory rather than a different rule for
picking one.

Two details that are quietly load-bearing:

**Buffers are copied, not averaged.** Running statistics are not points on a loss
surface and averaging them is meaningless. It happens not to matter for the
default backbone -- ``convnext_small`` has **0 BatchNorm layers**, which is why it
was chosen -- but ``tf_efficientnetv2_s`` has 110, and on that backbone averaging
the running means would produce weights that are wrong in a way only a validation
pass would reveal.

**The shadow lives on the model's device and in fp32.** Accumulating an EMA in
bf16 is the classic silent failure: with ``decay=0.999`` the update term is about
1e-3 of the value, which is below bf16's ~3 decimal digits of precision, so the
shadow stops moving and the "EMA" is an expensive copy of the initial weights.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from kappa.config import TrainConfig

__all__ = [
    "WeightAverager",
    "ExponentialMovingAverage",
    "LastNAverage",
    "build_averager",
]


def _float_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """The trainable floating-point parameters, which are the only things worth averaging.

    Membership is decided by ``named_parameters()``, not by matching names like
    ``"running_mean"``: a name pattern has to be kept in sync with every buffer
    a module might carry, while ``named_parameters()`` excludes a buffer by
    construction, custom or not. BatchNorm's ``running_mean``/``running_var``
    are float32 buffers -- exactly the case this used to miss, since they are
    floating-point but not trainable.
    """
    trainable = {name for name, _ in model.named_parameters()}
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name in trainable and tensor.is_floating_point()
    }


class WeightAverager:
    """Shared machinery: hold a state dict, and swap it in reversibly."""

    label = "average"

    def __init__(self, model: nn.Module) -> None:
        self._template = {name: tensor.shape for name, tensor in _float_params(model).items()}

    def after_step(self, model: nn.Module) -> None:
        """Called after every optimiser step. Default: nothing."""

    def after_epoch(self, model: nn.Module) -> None:
        """Called at the end of every epoch. Default: nothing."""

    def state_dict(self) -> dict[str, torch.Tensor]:  # pragma: no cover - overridden
        raise NotImplementedError

    def ready(self) -> bool:
        """Whether there is anything to swap in yet."""
        return True

    @contextlib.contextmanager
    def applied(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily install the averaged weights, then restore the raw ones.

        Restoring matters: training continues from the raw trajectory after a
        mid-run evaluation, and an averaged weight left in place would make every
        later step descend from a point the optimiser's state does not describe.
        """
        averaged = self.state_dict()
        saved = {name: tensor.detach().clone() for name, tensor in _float_params(model).items()}
        try:
            _load(model, averaged)
            yield model
        finally:
            _load(model, saved)


def _load(model: nn.Module, values: Mapping[str, torch.Tensor]) -> None:
    """Copy ``values`` into the model in place, leaving every other entry alone."""
    with torch.no_grad():
        current = model.state_dict()
        for name, tensor in values.items():
            current[name].copy_(tensor.to(current[name].device, current[name].dtype))


class ExponentialMovingAverage(WeightAverager):
    """``shadow = decay * shadow + (1 - decay) * weight``, updated every step.

    Bias correction is deliberately omitted. It matters when the average is read
    within the first ``1 / (1 - decay)`` steps; here the shadow is only ever read
    at the end of training, which on this project is thousands of steps in, and a
    correction that is a no-op by then is a term someone has to verify anyway.
    """

    label = "ema"

    def __init__(self, model: nn.Module, decay: float) -> None:
        super().__init__(model)
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay={decay} must lie in (0, 1); 0 would disable it")
        self.decay = float(decay)
        self._steps = 0
        # fp32 on the model's device: see the module docstring on bf16 underflow.
        self._shadow = {
            name: tensor.detach().to(torch.float32).clone()
            for name, tensor in _float_params(model).items()
        }

    def after_step(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, tensor in _float_params(model).items():
                shadow = self._shadow[name]
                shadow.mul_(self.decay).add_(
                    tensor.detach().to(shadow.dtype), alpha=1.0 - self.decay
                )
        self._steps += 1

    def ready(self) -> bool:
        return self._steps > 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self._shadow.items()}

    @property
    def steps(self) -> int:
        return self._steps

    def __repr__(self) -> str:
        return f"ExponentialMovingAverage(decay={self.decay}, steps={self._steps})"


class LastNAverage(WeightAverager):
    """The unweighted mean of the last ``n`` epoch-end snapshots.

    Snapshots are kept on the CPU. ``n`` copies of a 50M-parameter model is 200 MB
    per snapshot in fp32, and holding them beside the activations on one H200 is
    how a run that trains fine dies at the last epoch.
    """

    label = "swa"

    def __init__(self, model: nn.Module, n: int) -> None:
        super().__init__(model)
        if n < 1:
            raise ValueError(f"n={n} must be at least 1; 0 disables averaging instead")
        self.n = int(n)
        self._snapshots: list[dict[str, torch.Tensor]] = []

    def after_epoch(self, model: nn.Module) -> None:
        self._snapshots.append(
            {
                name: tensor.detach().to("cpu", torch.float32).clone()
                for name, tensor in _float_params(model).items()
            }
        )
        # Bounded, not accumulated: keeping every epoch and slicing at the end would
        # hold ten snapshots to use three.
        if len(self._snapshots) > self.n:
            self._snapshots.pop(0)

    def ready(self) -> bool:
        return bool(self._snapshots)

    @property
    def observed(self) -> int:
        """How many snapshots the average actually covers.

        Not necessarily ``n``: a run stopped early averages what it has. Reported
        rather than assumed, because "the last 5 epochs" describing a mean of 2 is
        a wrong number in a table with nothing to flag it.
        """
        return len(self._snapshots)

    def state_dict(self) -> dict[str, torch.Tensor]:
        if not self._snapshots:
            raise RuntimeError("no snapshots taken; after_epoch was never called")
        total = {name: tensor.clone() for name, tensor in self._snapshots[0].items()}
        for snapshot in self._snapshots[1:]:
            for name, tensor in snapshot.items():
                total[name].add_(tensor)
        for tensor in total.values():
            tensor.div_(len(self._snapshots))
        return total

    def __repr__(self) -> str:
        return f"LastNAverage(n={self.n}, observed={self.observed})"


def build_averager(cfg: TrainConfig, model: nn.Module) -> WeightAverager | None:
    """The averager a config asks for, or ``None`` when it asks for neither.

    ``None`` rather than a no-op object so that an unaveraged run takes the same
    code path it did before this module existed, and so that ``if averager`` in the
    loop reads as "averaging is on" rather than "an object exists".
    """
    if cfg.optim.ema_decay > 0.0:
        return ExponentialMovingAverage(model, decay=cfg.optim.ema_decay)
    if cfg.optim.swa_last_n > 0:
        return LastNAverage(model, n=cfg.optim.swa_last_n)
    return None
