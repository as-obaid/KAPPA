"""Teacher-to-student distillation, and the softmax that would have hidden its failure.

Why this exists, in one sentence: the supervised loss can only learn from the 1.3%
of label cells that are known, and the teacher's soft target exists for **all** of
them. The teacher sees the report-derived features and the student does not; a
frozen teacher checkpoint therefore hands the image-only student a target for every
one of the twelve labels on every study, including the unlabelled majority that
:mod:`kappa.training.losses` masks away. Distilling on those cells is the whole
method, which is why ``distill_unlabelled`` defaults to True and why masking the KD
term to the labelled cells is an *ablation* rather than a safety measure.

The silent failure this module is shaped to prevent is the standard KD recipe.
Every reference implementation of Hinton distillation reads
``F.kl_div(log_softmax(student / T), softmax(teacher / T))``, and a softmax over the
label axis answers "**which one** of the twelve is it". That is not this problem.
The twelve labels are independent binary decisions and they co-occur -- a knee with
both ACL and MCL findings is ordinary -- so a softmax splits one teacher's
confidence between the labels it is confident about, and discards the absolute
probability entirely: a teacher at 0.9 on two labels and a teacher at 0.05 on two
labels produce the identical soft target. Nothing raises. The loss falls, the run
finishes, and the quantity transferred is the wrong one. So the KD here is twelve
independent Bernoulli KLs and there is no softmax anywhere in this file.

The rest is arithmetic that only fails at the extremes, which is where confident
teachers live. ``torch.log(torch.sigmoid(x))`` is ``-inf`` for a logit of -200,
because the probability underflows to exactly 0.0 in fp32 first; every log here goes
through ``F.logsigmoid``, which is exact over the whole range. The ``T * T`` factor
is the KD gradient-scale correction: the derivative of the soft term carries a
``1 / T``, so without it a temperature sweep would be measuring the learning rate.

    from kappa.training.distill import DistillationLoss
    from kappa.training.losses import AsymmetricLoss

    criterion = DistillationLoss(AsymmetricLoss(), temperature=2.0)
    out = criterion(logits, batch["labels"], batch["mask"], teacher_logits)
    out.total.backward()

``teacher_logits=None`` is a supported argument, not an error: it makes the
no-distillation control the same forward pass as the distilled student with the
teacher left out.
Control and treatment then cannot differ by anything except the config.

Validation raises rather than asserts, because ``python -O`` strips ``assert`` and
a diverged teacher must not be optional to detect. The finiteness check covers the
whole teacher tensor unconditionally, for the reason the sibling module documents:
``0.0 * nan`` is ``nan``, so masking a cell out does not neutralise a NaN in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Private, and imported anyway: these are the sibling module's definitions of the
# masking idiom and the final NaN gate. A local copy would be the same three lines
# until the day one of them changed, and then the supervised and distillation terms
# would disagree about what an all-masked batch means.
from kappa.training.losses import _masked_mean, _no_nan

__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_LAMBDA_KD",
    "DEFAULT_LAMBDA_SUP",
    "BinaryKLDistillation",
    "DistillationOutput",
    "DistillationLoss",
]

#: Softening temperature. 2.0 is the usual starting point: high enough that the
#: teacher's ordering among its near-zero probabilities survives, low enough that
#: the target has not been flattened into 0.5 everywhere.
DEFAULT_TEMPERATURE = 2.0
DEFAULT_LAMBDA_KD = 1.0
DEFAULT_LAMBDA_SUP = 1.0


def _check_distill_inputs(
    student_logits: Tensor, teacher_logits: Tensor, mask: Tensor | None
) -> Tensor | None:
    """Validate the pair and return the mask as the loss's own dtype, or None.

    A shape mismatch is checked because ``(B, 12)`` against ``(12,)`` or ``(B, 1)``
    broadcasts rather than failing, and a KD term computed on a broadcast pairing
    trains the student against the wrong teacher cell at full confidence.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"teacher logits {tuple(teacher_logits.shape)} and student logits "
            f"{tuple(student_logits.shape)} differ; mismatched shapes broadcast into a "
            "plausible loss instead of raising, and the student then distills from the "
            "wrong cell"
        )
    if mask is not None and mask.shape != student_logits.shape:
        raise ValueError(
            f"mask {tuple(mask.shape)} and student logits {tuple(student_logits.shape)} differ"
        )
    if not torch.isfinite(teacher_logits).all():
        raise ValueError(
            "teacher logits are not finite. The teacher is a frozen checkpoint, so this "
            "is not a transient spike -- every batch of the student's run takes the same "
            "poison, and a NaN target is precisely the failure kappa.training.losses "
            "exists to refuse. The check is unconditional over the whole tensor because "
            "0.0 * nan is nan: masking the cell out does not remove it."
        )
    if not torch.isfinite(student_logits).all():
        raise ValueError(
            "student logits are not finite; the student has already diverged. Reported "
            "here rather than left to become a NaN gradient, because the KD term and the "
            "supervised term need different fixes and the loss curve shows neither."
        )
    return None if mask is None else mask.to(dtype=student_logits.dtype)


class BinaryKLDistillation(nn.Module):
    """Per-label binary KL between teacher and student logits, at temperature T.

    Twelve independent Bernoulli KL divergences, averaged over the cells kept.
    ``KL(p || q) = p (log p - log q) + (1 - p)(log(1 - p) - log(1 - q))`` with
    ``p = sigmoid(teacher / T)`` and ``q = sigmoid(student / T)``, times ``T * T``.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be positive, got {temperature}. Zero divides the logits "
                "by zero; a negative temperature flips every teacher probability to its "
                "complement, which does not error -- it trains the student to disagree."
            )
        self.temperature = float(temperature)

    def forward(
        self, student_logits: Tensor, teacher_logits: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        weights = _check_distill_inputs(student_logits, teacher_logits, mask)

        # Detached again even though the caller detaches: it is free, and it makes a
        # caller who forgot indistinguishable from one who did not. The alternative is
        # a gradient flowing quietly into a teacher that is supposed to be frozen.
        teacher = teacher_logits.detach().to(dtype=student_logits.dtype)

        soft_teacher = teacher / self.temperature
        soft_student = student_logits / self.temperature

        # All four logs via logsigmoid, none via log(sigmoid(x)). In fp32
        # sigmoid(-200) is exactly 0.0, so torch.log(torch.sigmoid(x)) returns -inf for
        # a confident teacher and puts NaN in the student's gradient; logsigmoid(-200)
        # is -200. p itself is allowed to underflow to 0.0, which only zeroes its own
        # term because the bracket beside it stays finite.
        log_p = F.logsigmoid(soft_teacher)
        log_1mp = F.logsigmoid(-soft_teacher)
        log_q = F.logsigmoid(soft_student)
        log_1mq = F.logsigmoid(-soft_student)
        p = torch.sigmoid(soft_teacher)

        per_cell = p * (log_p - log_q) + (1.0 - p) * (log_1mp - log_1mq)
        # KL is non-negative analytically, and identical logits cancel exactly, so this
        # clamp only removes float rounding where p and q already agree. It cannot hide
        # a sign error, which would be large and would survive it.
        per_cell = per_cell.clamp(min=0.0) * (self.temperature * self.temperature)

        if weights is None:
            # No mask: every cell, which is the default and the point of the method.
            # numel is clamped for the degenerate empty batch, so an exhausted loader
            # gives 0.0 rather than the 0/0 that .mean() would return as NaN.
            return _no_nan(per_cell.sum() / max(per_cell.numel(), 1))
        return _no_nan(_masked_mean(per_cell, weights))

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


@dataclass(frozen=True)
class DistillationOutput:
    """The scalar to optimise, and the terms it was built from.

    Both terms are carried, not just the total, because a run whose teacher stopped
    contributing and a run whose supervised term collapsed produce the same total and
    need opposite fixes. A future feature-KD term is one more field here and one more
    addend in :attr:`total`; nothing else about this contract changes.
    """

    total: Tensor
    supervised: Tensor
    distillation: Tensor

    def as_log_dict(self) -> dict[str, float]:
        """Plain floats for the tracker, detached so logging cannot retain the graph.

        Deliberately unprefixed: the ``fold00/`` namespace convention lives in
        :mod:`kappa.tracking`, so the caller prefixes these exactly as it prefixes
        ``train_loss`` and the two stay comparable in one chart.
        """
        return {
            "loss_total": float(self.total.detach()),
            "loss_supervised": float(self.supervised.detach()),
            "loss_distill": float(self.distillation.detach()),
        }


class DistillationLoss(nn.Module):
    """``L_total = lambda_supervised * L_sup + lambda_distill * L_KD``."""

    def __init__(
        self,
        supervised: nn.Module,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        lambda_supervised: float = DEFAULT_LAMBDA_SUP,
        lambda_distill: float = DEFAULT_LAMBDA_KD,
        distill_unlabelled: bool = True,
    ) -> None:
        super().__init__()
        if lambda_supervised < 0.0 or lambda_distill < 0.0:
            raise ValueError(
                f"loss weights must be non-negative; got lambda_supervised="
                f"{lambda_supervised}, lambda_distill={lambda_distill}. A negative weight "
                "maximises its own term, which does not diverge loudly -- it trains, "
                "downhill, away from the target."
            )
        if lambda_supervised == 0.0 and lambda_distill == 0.0:
            raise ValueError(
                "both loss weights are 0, so the total loss is a constant and every "
                "gradient is zero. The run would complete, log a flat curve and score an "
                "untrained model; drop a term by config, not by weighting it away."
            )
        # Assigned as submodules so .to(device), .state_dict() and .eval() reach the
        # supervised loss's own buffers -- ASL's per-label gamma_neg vector is one.
        self.supervised = supervised
        self.distill = BinaryKLDistillation(temperature=temperature)
        self.lambda_supervised = float(lambda_supervised)
        self.lambda_distill = float(lambda_distill)
        self.distill_unlabelled = bool(distill_unlabelled)

    def forward(
        self,
        student_logits: Tensor,
        targets: Tensor,
        mask: Tensor,
        teacher_logits: Tensor | None,
    ) -> DistillationOutput:
        # Validating (logits, targets, mask) is the supervised loss's job and is not
        # repeated here: it already refuses a NaN target and a -1 "unknown" sentinel.
        supervised = self.supervised(student_logits, targets, mask)

        if teacher_logits is None:
            # The no-distillation control is this same forward with the teacher left
            # out, so it cannot silently differ from the treatment by anything but the
            # config. Multiplying by zero rather than building torch.zeros keeps the
            # scalar attached to the graph, on the student's device and dtype, so
            # backward() and the gradient-accumulation bookkeeping behave identically in
            # both arms -- and the zero gradient it contributes is the correct one.
            # Multiplied before the sum, not after: after, a negative logit total
            # leaves a signed -0.0 in the log, which reads as a bug and is not one.
            distillation = (student_logits * 0.0).sum()
        else:
            # distill_unlabelled: no mask at all. The teacher defines a target for every
            # cell and the label mask keeps 1.3% of them, so this is where the transfer
            # happens. Passing the mask instead is the ablation that prices the rest.
            kd_mask = None if self.distill_unlabelled else mask
            distillation = self.distill(student_logits, teacher_logits, kd_mask)

        total = self.lambda_supervised * supervised + self.lambda_distill * distillation
        return DistillationOutput(total=total, supervised=supervised, distillation=distillation)

    def extra_repr(self) -> str:
        return (
            f"lambda_supervised={self.lambda_supervised}, "
            f"lambda_distill={self.lambda_distill}, "
            f"distill_unlabelled={self.distill_unlabelled}"
        )
