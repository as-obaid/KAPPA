"""Masked multi-label losses, and the assertion that stops NaN reaching them.

Why this exists, in one sentence: ``BCEWithLogitsLoss`` over a raw label array
destroys training **silently**. **98.7% of raw label cells are NaN** in this
project. A NaN target produces a NaN loss, a NaN gradient, and NaN weights within
one step; there is no exception, no warning, and no diverging loss curve to
notice -- just a model that has learned nothing and a fold that looks merely
disappointing. So every loss here refuses NaN input outright, with a ``raise``
rather than an ``assert`` statement, because ``python -O`` strips ``assert`` and
this check must not be optional. The same gate rejects an infinite logit and a
target outside ``[0, 1]``: an infinity becomes NaN the moment it meets a zero
target, and a ``-1`` sentinel target is worse than NaN because it trains quietly.

The same trap has a second mouth. The masking idiom is ``(per_cell_loss *
mask).sum() / mask.sum()``, and ``0.0 * nan`` is ``nan``, not ``0.0``. Multiplying
by a zero mask does **not** neutralise a NaN target, so it is not enough to check
only the cells the mask keeps -- the check is unconditional over the whole tensor.
Upstream honours this: the pinned batch contract fills ``labels`` with 0.0 where
``mask`` is False and never with NaN.

    from kappa.training.losses import AsymmetricLoss

    criterion = AsymmetricLoss()            # gamma_neg=4, gamma_pos=0, clip=0.05
    loss = criterion(logits, batch["labels"], batch["mask"])

Why Asymmetric Loss (arXiv 2009.14119) rather than BCE or focal. Focal loss
applies the same focusing exponent to positives and negatives, so on a multi-label
problem where most cells are negative it accumulates negative gradient and
suppresses the rare positives it was supposed to rescue. ASL splits the exponent
in two -- ``gamma_neg`` down-weights easy negatives while ``gamma_pos=0`` leaves
positives alone -- and hard-thresholds very confident negatives away entirely via
``clip``. The labels here are rare enough for that to matter: ``MCL`` sits at
15.5% prevalence, and every one of the twelve labels contributes an equal share of
the macro AUC no matter how rare it is, so a loss that quietly gives up on the
rare ones is giving up score at full price.

:class:`MaskedBCE` is here so that the ASL-versus-BCE ablation is a config change
and not a rewrite: same signature, same masking, same NaN refusal.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kappa.constants import N_LABELS

__all__ = [
    "AsymmetricLoss",
    "MaskedBCE",
    "LOSSES",
    "build_loss",
]


def _check_masked_inputs(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Validate the three tensors and return the mask as the loss's own dtype.

    Raises, rather than asserts, on NaN. The whole point of this module is that a
    NaN target is undetectable downstream, so the check has to survive ``-O``.
    Both ``targets`` and ``logits`` are checked: a bad target means a broken label
    table, a non-finite logit means the model has already diverged, and the two need
    different fixes but both are worth finding at the step they happen rather than
    at the end of the epoch. Targets are also range-checked, because a sentinel
    ``-1`` for "unknown" is the one poisoned target that does not announce itself --
    it yields a *smaller* loss than a correct cell.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} differ")
    if mask.shape != targets.shape:
        raise ValueError(f"mask {tuple(mask.shape)} and targets {tuple(targets.shape)} differ")
    if torch.isnan(targets).any():
        raise ValueError(
            "targets contain NaN. Multiplying by a zero mask does not remove it -- "
            "0.0 * nan is nan -- so this would poison the whole batch. Emit labels and "
            "label_mask separately and fill masked cells with 0.0."
        )
    if ((targets < 0.0) | (targets > 1.0)).any():
        raise ValueError(
            "targets outside [0, 1]. A target is read as the probability of the positive "
            "class, so a sentinel such as -1 for 'unknown' does not error -- it produces a "
            "small, plausible loss and trains on nothing. Mask the cell instead."
        )
    if not torch.isfinite(logits).all():
        raise ValueError(
            "logits are not finite; the model has already diverged. An infinity is as fatal "
            "as a NaN here: it either leaves an infinite loss or meets a zero target and "
            "becomes NaN through 0 * inf, which then gets blamed on the loss."
        )
    return mask.to(dtype=logits.dtype)


def _masked_mean(per_cell: Tensor, mask: Tensor) -> Tensor:
    """Sum the kept cells and divide by how many were kept.

    **An all-masked batch returns exactly 0.0, not NaN.** It can happen: with
    only 58 of 4,407 studies labelled, a batch drawn from the unlabelled majority
    has every cell masked out. Returning NaN there would put NaN into the gradient
    and end the run, so the denominator is clamped at 1 and the batch contributes
    no loss and a zero gradient. The zero still flows through the graph, which
    keeps ``backward()`` and gradient-accumulation bookkeeping working instead of
    raising on a scalar with no history.
    """
    kept = mask.sum()
    return (per_cell * mask).sum() / kept.clamp(min=1.0)


def _no_nan(loss: Tensor) -> Tensor:
    """Final gate: a NaN scalar must never leave a loss function."""
    if torch.isnan(loss).any():
        raise ValueError("loss is NaN despite clean inputs; this is a bug in the loss itself")
    return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (arXiv 2009.14119), masked per cell and averaged over kept cells.

    ``gamma_neg`` focuses down easy negatives, ``gamma_pos`` leaves positives at
    their full weight by default, and ``clip`` shifts the negative probability so
    that any negative the model already scores below ``clip`` contributes nothing
    at all. Defaults are the paper's.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        gamma_neg_by_label: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if gamma_neg < 0.0 or gamma_pos < 0.0:
            raise ValueError("focusing exponents must be non-negative")
        if not 0.0 <= clip < 1.0:
            raise ValueError("clip is a probability shift and must lie in [0, 1)")
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)

        # Per-label gamma_neg. The scalar above says "positives are rare"; measured
        # on the derived table (n=4,407), that is true of Fracture at 7.4% positive
        # and false of Effusion at 58.8%, where the negatives are the rarer class
        # and a scalar gamma_neg focuses down the cells that column needs most.
        # Every label is 1/12 of a macro score, so it is given up at full price.
        #
        # Registered as a buffer even when None so that .to(device) needs no branch,
        # and so the vector is saved with the checkpoint: a run reloaded with the
        # wrong gammas would train perfectly well and be a different experiment.
        vector: Tensor | None = None
        if gamma_neg_by_label is not None:
            vector = torch.as_tensor(list(gamma_neg_by_label), dtype=torch.float32)
            if vector.ndim != 1 or vector.numel() != N_LABELS:
                raise ValueError(
                    f"gamma_neg_by_label has {tuple(vector.shape)} entries; it must be "
                    f"one per label, in LABELS order ({N_LABELS} of them)"
                )
            if bool((vector < 0.0).any()):
                raise ValueError("focusing exponents must be non-negative")
        self.register_buffer("gamma_neg_by_label", vector)

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        weights = _check_masked_inputs(logits, targets, mask)
        probs = torch.sigmoid(logits)

        # log(p) via logsigmoid rather than probs.log(): for a strongly negative
        # logit the probability underflows to exactly 0 and the log of that is -inf,
        # which the paper's reference implementation papers over with an eps clamp.
        # logsigmoid is exact over the whole range, so the positive term needs no
        # clamp at all.
        log_pos = F.logsigmoid(logits)
        if self.clip > 0.0:
            # Probability shift, capped at 1: a negative the model already scores
            # below `clip` lands at 1.0 here, so its log is 0 and it drops out of the
            # loss entirely. That hard threshold is the second half of the asymmetry.
            prob_neg = (1.0 - probs + self.clip).clamp(max=1.0)
            log_neg = prob_neg.clamp(min=self.eps).log()
        else:
            prob_neg = 1.0 - probs
            log_neg = F.logsigmoid(-logits)

        per_cell = -(targets * log_pos + (1.0 - targets) * log_neg)

        gamma_neg = self._gamma_neg_for(logits)
        if self.gamma_pos > 0.0 or bool(torch.as_tensor(gamma_neg).gt(0.0).any()):
            # Probability assigned to the *true* class, per cell. The exponent is
            # picked per cell from the target, which is the asymmetry: negatives get
            # gamma_neg, positives get gamma_pos.
            p_true = targets * probs + (1.0 - targets) * prob_neg
            gamma = targets * self.gamma_pos + (1.0 - targets) * gamma_neg
            per_cell = per_cell * (1.0 - p_true).clamp(min=0.0).pow(gamma)

        return _no_nan(_masked_mean(per_cell, weights))

    def _gamma_neg_for(self, logits: Tensor) -> Tensor | float:
        """The negative-class exponent: a ``(N_LABELS,)`` row, or the scalar.

        Checked against the logits rather than trusted, because a vector of the
        wrong length would broadcast against a different label count instead of
        failing -- every column would get someone else's exponent and the run
        would converge to a wrong number without complaint.
        """
        vector: Tensor | None = self.gamma_neg_by_label
        if vector is None:
            return self.gamma_neg
        if logits.shape[-1] != vector.numel():
            raise ValueError(
                f"logits have {logits.shape[-1]} labels but gamma_neg_by_label has "
                f"{vector.numel()}; a mismatched vector broadcasts instead of raising"
            )
        return vector.to(device=logits.device, dtype=logits.dtype)

    def extra_repr(self) -> str:
        vector: Tensor | None = self.gamma_neg_by_label
        gamma_neg = f"{self.gamma_neg}" if vector is None else f"per-label{vector.tolist()}"
        return f"gamma_neg={gamma_neg}, gamma_pos={self.gamma_pos}, clip={self.clip}"


class MaskedBCE(nn.Module):
    """Plain masked BCE-with-logits, for the ablation ASL has to beat.

    Identical signature and identical masking, so swapping it in is a config
    change. ``pos_weight`` is available but off by default: it rescales the
    positive term per label, which shifts the loss without changing the ranking the
    metric reads, and this competition scores nothing but the ranking.
    """

    def __init__(self, pos_weight: Tensor | None = None) -> None:
        super().__init__()
        # Registered even when None so that .to(device) moves it without a branch.
        self.register_buffer(
            "pos_weight", None if pos_weight is None else pos_weight.detach().clone()
        )

    def forward(self, logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        weights = _check_masked_inputs(logits, targets, mask)
        pos_weight: Tensor | None = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(dtype=logits.dtype, device=logits.device)
        per_cell = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        return _no_nan(_masked_mean(per_cell, weights))


#: Name to class, so a config can name a loss without importing one.
LOSSES: dict[str, type[nn.Module]] = {
    "asl": AsymmetricLoss,
    "bce": MaskedBCE,
}


def build_loss(name: str = "asl", **kwargs: object) -> nn.Module:
    """Construct a loss by name, failing with the list of names that do exist."""
    try:
        factory = LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; known: {sorted(LOSSES)}") from None
    return factory(**kwargs)  # type: ignore[arg-type]
