"""Macro ROC-AUC over the twelve labels, with the denominator said out loud.

Why this exists: the obvious implementation of "macro AUC" is
``np.nanmean([roc_auc_score(y[:, j], p[:, j]) for j in range(12)])``, and it lies
twice.

**It lies about the denominator.** A label with only one class present in the
evaluation split has no defined AUC. ``nanmean`` drops it and reports the mean of
whatever survived, still called "macro". Eleven labels averaged and labelled
macro-12 is not a rounding error; it is a different quantity, and it drifts fold
to fold as which labels happen to be single-class changes. So :func:`macro_roc_auc`
returns the mean it can actually compute *and* names every label it could not, in
``skipped``, and warns. Read ``skipped`` before quoting ``macro``.

**It lies about coverage.** Only 58 of 4,407 train studies carry the 12 labels and
**98.7% of raw label cells are NaN**, so per-label support is not the row count. A
prior validation table in this project printed a positive rate over all 58 studies
while ten of twelve rows actually covered fewer, overstating coverage on ten rows
out of twelve. Hence ``n_used`` and ``n_pos`` per label, and hence
:meth:`AucResult.table`, which cannot print a rate without printing its
denominator next to it.

    from kappa.evaluation.metrics import macro_roc_auc

    result = macro_roc_auc(y_true, y_score)      # NaN in y_true means "unknown"
    print(result.table())
    if not result.is_complete:
        ...  # result.skipped labels had one class present; macro is over the rest

The point estimate on its own is not evidence. The recorded noise floor for this
task is **0.0020 macro** and the recorded gap between a working solution and a
top-10 place is **~0.006** -- both external figures carried into this project, not
measurements taken in this repository. Two numbers 0.001 apart are the same
number. :func:`fold_summary` therefore reports the across-fold standard deviation
for the macro *and for all twelve labels*, and :func:`is_evidence` exists so that
comparing a delta against the floor is one call rather than a judgement call.

**No calibration, ever.** ROC-AUC depends only on the order of the scores, so
Platt scaling, isotonic regression and temperature all leave every number in this
module bit-identical while consuming a fold. The corollary is the one that gets
broken in practice: averaging two models' probabilities is *not* order-preserving,
because the member with the wider spread wins the average. Combine with
:func:`rank_average`, which is invariant to any monotone rescaling of a member.

**Diagnostic metrics are not the competition metric, and must never be treated as
one.** :func:`diagnostic_metrics` adds average precision, precision, recall and F1
per label -- average precision most of all, because ROC-AUC does not read the
positive rate, and a label like Fracture at 7.5% prevalence can post a
respectable AUC while every usable operating point on it has poor precision.
But every quantity :func:`diagnostic_metrics` returns that depends on a
threshold -- precision, recall and F1 at :data:`FIXED_THRESHOLD`, and the best
achievable F1 with the threshold that finds it -- is fit *to the split it was
measured on*. Picking a threshold, or a model, by looking at ``f1_best`` repeats
the mistake this project already made once: the spent 58-study label anchor is a
regression check, not something to tune against, and a threshold chosen on a
validation fold is the same error with an extra decimal place. Log these for a
human to read. Never let one of them choose a checkpoint, a fold or a model --
the only quantity that does that here is ``macro`` from :func:`macro_roc_auc`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kappa.constants import LABELS

__all__ = [
    "FIXED_THRESHOLD",
    "NOISE_FLOOR_MACRO",
    "AucResult",
    "DiagnosticResult",
    "FoldSummary",
    "average_ranks",
    "diagnostic_metrics",
    "fold_summary",
    "is_evidence",
    "macro_roc_auc",
    "pr_curve_points",
    "rank_average",
    "rank_normalise",
    "roc_curve_points",
]

#: Smallest macro difference that can mean anything. An external figure recorded
#: for this task, not measured in this repository: a competitor measured 0.0020
#: macro of run-to-run spread at three folds. Any delta under it is noise wearing
#: a decimal point.
NOISE_FLOOR_MACRO = 0.0020

#: The one threshold :func:`diagnostic_metrics` binarises a score at for its
#: fixed-threshold precision, recall and F1. Named and exported rather than
#: written inline as ``0.5`` because it is arbitrary and near-useless for a
#: rare label -- Fracture at 7.5% prevalence has no reason to have its useful
#: operating point sit at a bare 0.5 -- so the number must be visible and
#: overridable, not buried in a call site.
FIXED_THRESHOLD = 0.5


@dataclass(frozen=True)
class AucResult:
    """One macro-AUC evaluation, carrying what it could not score.

    ``macro`` is the mean over the labels that *were* scoreable, so it is only the
    competition quantity when ``skipped`` is empty -- which is what
    :attr:`is_complete` is for. ``per_label`` holds ``nan`` for a skipped label
    rather than omitting the key, so a results table always has twelve rows and a
    missing label is visible instead of absent.
    """

    macro: float
    per_label: dict[str, float]
    n_used: dict[str, int]
    n_pos: dict[str, int]
    skipped: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        """True only when all twelve labels were scoreable, i.e. ``macro`` is macro-12."""
        return not self.skipped

    @property
    def n_scored(self) -> int:
        """How many labels the ``macro`` mean is actually over."""
        return len(self.per_label) - len(self.skipped)

    def table(self) -> str:
        """Per-label AUC with its own denominator on the same line.

        Deliberately impossible to print a positive rate without its support: the
        rate and ``n`` are formatted together, because the pair is the fact and the
        rate alone is a claim about a sample size it does not have.
        """
        lines = [f"{'label':<18}{'auc':>8}{'n':>7}{'pos':>6}{'pos%':>7}"]
        for name, auc in self.per_label.items():
            used = self.n_used[name]
            pos = self.n_pos[name]
            rate = f"{100.0 * pos / used:.1f}" if used else "-"
            shown = "skipped" if name in self.skipped else f"{auc:.4f}"
            lines.append(f"{name:<18}{shown:>8}{used:>7}{pos:>6}{rate:>7}")
        lines.append(f"macro {self.macro:.4f} over {self.n_scored} of {len(self.per_label)} labels")
        if self.skipped:
            lines.append(f"skipped (one class present): {', '.join(self.skipped)}")
        return "\n".join(lines)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based ranks of a 1-D array, ties sharing their mid-rank.

    Shared by the AUC and by :func:`rank_normalise` on purpose. Tie handling is
    where a hand-rolled AUC silently diverges from ``sklearn``, and one
    implementation cannot disagree with itself.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    n = values.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=is_new[1:])
    starts = np.flatnonzero(is_new)
    counts = np.diff(np.append(starts, n))
    # A tie group starting at index s with c members spans ranks s+1 .. s+c, whose
    # mean is s + (c + 1) / 2.
    mid = starts + (counts + 1.0) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(mid, counts)
    return ranks


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC for one label, or ``nan`` when only one class is present.

    The Mann-Whitney form, which equals the trapezoidal area ``sklearn`` computes
    including its treatment of tied scores as half-credit.
    """
    positives = y_true == 1.0
    n_pos = int(positives.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(y_score)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    labels: Sequence[str] = LABELS,
) -> AucResult:
    """Macro ROC-AUC over ``labels``, each contributing exactly one equal share.

    ``y_true`` and ``y_score`` are ``(n_studies, n_labels)``. A cell is used only
    when it is not NaN in ``y_true`` and, if ``mask`` is given, ``mask`` is True
    there. NaN in ``y_true`` is read as *unknown*, which is the raw competition
    label array's own encoding and 98.7% of its cells; excluding those cells is
    the whole reason this takes a mask at all.

    NaN in ``y_score`` raises. A NaN prediction is a broken model, not a missing
    observation, and there is no order that a NaN can take part in -- dropping it
    quietly would score a model on the subset of studies it did not break on.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 2:
        raise ValueError(f"y_true must be (n_studies, n_labels), got shape {y_true.shape}")
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true {y_true.shape} and y_score {y_score.shape} disagree")
    if y_true.shape[1] != len(labels):
        raise ValueError(f"y_true has {y_true.shape[1]} columns, expected {len(labels)}")
    if len(set(labels)) != len(labels):
        raise ValueError(
            "labels contains a duplicate name. Per-label results are keyed by name, so a "
            "duplicate silently collapses two columns into one entry and shrinks the "
            "macro's denominator -- the exact failure this function exists to prevent."
        )

    usable = ~np.isnan(y_true)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != y_true.shape:
            raise ValueError(f"mask {mask.shape} does not match y_true {y_true.shape}")
        usable &= mask.astype(bool)

    if np.isnan(y_score[usable]).any():
        raise ValueError(
            "y_score contains NaN in a cell the mask keeps. A NaN prediction cannot be "
            "ranked; fix the model rather than scoring it on the rows it survived."
        )

    per_label: dict[str, float] = {}
    n_used: dict[str, int] = {}
    n_pos: dict[str, int] = {}
    skipped: list[str] = []
    for column, name in enumerate(labels):
        keep = usable[:, column]
        truth = y_true[keep, column]
        if truth.size and not np.isin(truth, (0.0, 1.0)).all():
            raise ValueError(
                f"y_true[{name!r}] holds values outside {{0, 1}}; ROC-AUC needs binary "
                "truth, so this is a probability or a magic float in the label column."
            )
        auc = _binary_auc(truth, y_score[keep, column])
        per_label[name] = auc
        n_used[name] = int(keep.sum())
        n_pos[name] = int((truth == 1.0).sum())
        if not np.isfinite(auc):
            skipped.append(name)

    scored = [auc for auc in per_label.values() if np.isfinite(auc)]
    macro = float(np.mean(scored)) if scored else float("nan")
    if skipped:
        warnings.warn(
            f"macro is the mean of {len(scored)} of {len(labels)} labels; "
            f"{', '.join(skipped)} had one class present and were not scored. "
            "This is not the competition's macro-12.",
            RuntimeWarning,
            stacklevel=2,
        )
    return AucResult(
        macro=macro,
        per_label=per_label,
        n_used=n_used,
        n_pos=n_pos,
        skipped=tuple(skipped),
    )


@dataclass(frozen=True)
class DiagnosticResult:
    """Average precision, precision/recall/F1 and best-F1 over the twelve labels.

    **Diagnostic only -- never a selection signal.** Read the module docstring
    before reading this class: ``precision``, ``recall`` and ``f1`` are
    measured at the single fixed :data:`FIXED_THRESHOLD`, and ``f1_best`` /
    ``threshold_best`` name the best F1 *found on this split* and the
    threshold that found it. Neither may choose a checkpoint, a fold or a
    model; the competition is scored on ``macro`` from :func:`macro_roc_auc`,
    and nothing here.

    Mirrors :class:`AucResult`'s shape on purpose: a label with zero positives
    in the split that was used has none of these quantities defined --
    precision and recall are both a count of *found* positives over a
    positive count of zero -- so it carries ``nan`` in every dict here rather
    than a missing key, and its name in ``skipped``. Unlike :class:`AucResult`,
    a label with zero *negatives* is not skipped: average precision, precision,
    recall and F1 are all well-defined when every used cell is positive (a
    constant-positive column has a trivial but real average precision of 1.0),
    where ROC-AUC has no defined ordering to score without both classes.
    """

    ap: dict[str, float]
    ap_macro: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_best: dict[str, float]
    threshold_best: dict[str, float]
    n_used: dict[str, int]
    n_pos: dict[str, int]
    skipped: tuple[str, ...]
    threshold: float

    @property
    def is_complete(self) -> bool:
        """True only when every label had at least one positive in the split used."""
        return not self.skipped

    @property
    def n_scored(self) -> int:
        """How many labels the macros here are actually averaged over."""
        return len(self.ap) - len(self.skipped)


def _binary_pr_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and threshold at every achievable operating point, one label.

    Ordered by decreasing threshold, so ``recall`` is non-decreasing and ends at
    1.0. A run of tied scores collapses onto a single point -- the same
    ``sklearn.metrics._binary_clf_curve`` convention -- because a heavily tied
    model does not have one operating point per row, and scoring it as if it did
    is exactly the kind of hand-rolled divergence this file exists to avoid (see
    the module docstring on :func:`average_ranks`). Empty arrays when there are
    no positives: there is no curve to speak of, and every caller here treats
    that as its own skip condition rather than dividing zero by zero.
    """
    n_pos = float((y_true == 1.0).sum())
    if n_pos == 0.0 or y_true.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    order = np.argsort(y_score, kind="stable")[::-1]
    truth = y_true[order]
    scores = y_score[order]
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, truth.size - 1]
    tps = np.cumsum(truth)[idx]
    predicted = idx + 1.0
    fps = predicted - tps
    precision = np.divide(tps, predicted, out=np.zeros_like(tps), where=predicted != 0)
    recall = tps / n_pos
    return precision, recall, scores[idx]


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision for one label, or ``nan`` when it has no positives.

    Reproduces ``sklearn.metrics.average_precision_score``: the step-function
    integral of precision over recall, unweighted and uninterpolated, from
    :func:`_binary_pr_curve` with the ``(recall=0, precision=1)`` sentinel
    ``sklearn`` prepends so the curve starts on the axis. ``sklearn`` itself
    returns ``0.0`` with a warning for a zero-positive column; this returns
    ``nan`` instead; see :class:`DiagnosticResult`.
    """
    precision, recall, _ = _binary_pr_curve(y_true, y_score)
    if precision.size == 0:
        return float("nan")
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


def _binary_best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """The largest F1 achievable on this label and the threshold that achieves it.

    **Diagnostic only.** Searches every achievable operating point from
    :func:`_binary_pr_curve`, which already covers every distinct score, so
    this is the true best over that split -- and precisely why it must never
    choose anything: it is fit to the split by construction. Ties in F1 are
    broken by the highest threshold among them (the first one reached, since
    :func:`_binary_pr_curve` is ordered by decreasing threshold), which prefers
    the more conservative operating point when several score identically.
    Returns ``(nan, nan)`` when the label has no positives.
    """
    precision, recall, thresholds = _binary_pr_curve(y_true, y_score)
    if precision.size == 0:
        return float("nan"), float("nan")
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom != 0)
    best = int(np.argmax(f1))
    return float(f1[best]), float(thresholds[best])


def _binary_threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    """Precision, recall and F1 for one label at a single fixed threshold.

    **Diagnostic only; the threshold is a constant, not a fit quantity** -- see
    :data:`FIXED_THRESHOLD` and the module docstring. Reproduces
    ``sklearn.metrics.precision_recall_fscore_support(..., zero_division=0)``
    at a score already binarised by ``>= threshold``. Returns
    ``(nan, nan, nan)`` when the label has no positives, matching
    :func:`_binary_best_f1` rather than ``sklearn``'s ``0.0``-with-a-warning.
    """
    n_pos = float((y_true == 1.0).sum())
    if n_pos == 0.0:
        return float("nan"), float("nan"), float("nan")
    truth = y_true == 1.0
    predicted = y_score >= threshold
    tp = float(np.logical_and(truth, predicted).sum())
    fp = float(np.logical_and(~truth, predicted).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / n_pos
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    return precision, recall, f1


def diagnostic_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    labels: Sequence[str] = LABELS,
    threshold: float = FIXED_THRESHOLD,
) -> DiagnosticResult:
    """Average precision, precision/recall/F1 and best-F1 over ``labels``.

    **Diagnostic only -- never a selection signal.** Read the module docstring
    and :class:`DiagnosticResult` before using this anywhere near a checkpoint
    or fold decision; the competition metric is ``macro`` from
    :func:`macro_roc_auc`, and nothing computed here may substitute for it.

    ``y_true``, ``y_score`` and ``mask`` have exactly :func:`macro_roc_auc`'s
    contract: ``(n_studies, n_labels)``, NaN in ``y_true`` read as *unknown*
    and dropped from every count here, ``mask`` narrowing further, and NaN in
    ``y_score`` raising rather than being silently dropped -- a NaN prediction
    cannot be thresholded or ranked, and dropping it would score a model on the
    subset of studies it did not break on. A label with zero positives in the
    cells actually used has none of these quantities defined and is named in
    ``skipped`` with ``nan`` everywhere in its row, never a missing key or a
    silent 0.0 -- see :class:`DiagnosticResult` for why zero *negatives* is not
    the same failure and is not skipped.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 2:
        raise ValueError(f"y_true must be (n_studies, n_labels), got shape {y_true.shape}")
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true {y_true.shape} and y_score {y_score.shape} disagree")
    if y_true.shape[1] != len(labels):
        raise ValueError(f"y_true has {y_true.shape[1]} columns, expected {len(labels)}")
    if len(set(labels)) != len(labels):
        raise ValueError(
            "labels contains a duplicate name. Per-label results are keyed by name, so a "
            "duplicate silently collapses two columns into one entry and shrinks the "
            "macro's denominator -- the exact failure this function exists to prevent."
        )

    usable = ~np.isnan(y_true)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != y_true.shape:
            raise ValueError(f"mask {mask.shape} does not match y_true {y_true.shape}")
        usable &= mask.astype(bool)

    if np.isnan(y_score[usable]).any():
        raise ValueError(
            "y_score contains NaN in a cell the mask keeps. A NaN prediction cannot be "
            "thresholded or ranked; fix the model rather than scoring it on the rows it "
            "survived."
        )

    ap: dict[str, float] = {}
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    f1_best: dict[str, float] = {}
    threshold_best: dict[str, float] = {}
    n_used: dict[str, int] = {}
    n_pos: dict[str, int] = {}
    skipped: list[str] = []

    for column, name in enumerate(labels):
        keep = usable[:, column]
        truth = y_true[keep, column]
        if truth.size and not np.isin(truth, (0.0, 1.0)).all():
            raise ValueError(
                f"y_true[{name!r}] holds values outside {{0, 1}}; these metrics need "
                "binary truth, so this is a probability or a magic float in the label column."
            )
        scores = y_score[keep, column]
        n_used[name] = int(keep.sum())
        n_pos[name] = int((truth == 1.0).sum())

        ap[name] = _binary_average_precision(truth, scores)
        precision[name], recall[name], f1[name] = _binary_threshold_metrics(
            truth, scores, threshold
        )
        f1_best[name], threshold_best[name] = _binary_best_f1(truth, scores)

        if n_pos[name] == 0:
            skipped.append(name)

    def _macro(values: dict[str, float]) -> float:
        scored = [v for name, v in values.items() if name not in skipped]
        return float(np.mean(scored)) if scored else float("nan")

    ap_macro = _macro(ap)
    precision_macro = _macro(precision)
    recall_macro = _macro(recall)
    f1_macro = _macro(f1)

    if skipped:
        warnings.warn(
            f"{len(skipped)} of {len(labels)} labels had zero positives in the cells "
            f"used ({', '.join(skipped)}); their average precision, precision, recall "
            "and F1 are nan and excluded from every macro here.",
            RuntimeWarning,
            stacklevel=2,
        )

    return DiagnosticResult(
        ap=ap,
        ap_macro=ap_macro,
        precision=precision,
        recall=recall,
        f1=f1,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        f1_best=f1_best,
        threshold_best=threshold_best,
        n_used=n_used,
        n_pos=n_pos,
        skipped=tuple(skipped),
        threshold=float(threshold),
    )


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FPR and TPR, both ascending, at every achievable threshold for one label.

    **Diagnostic and visual only** -- built for a W&B curve panel
    (:func:`kappa.tracking.roc_curve_panel`), not for scoring; the area under
    this curve, if wanted, is :func:`macro_roc_auc`'s per-label AUC, computed
    independently by the Mann-Whitney form and not by integrating this curve.
    Empty arrays when a class is entirely absent -- the same condition
    :func:`macro_roc_auc` calls a skip, since there is no ROC curve without
    both classes present.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = float((y_true == 1.0).sum())
    n_neg = float((y_true == 0.0).sum())
    if n_pos == 0.0 or n_neg == 0.0 or y_true.size == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(y_score, kind="stable")[::-1]
    truth = y_true[order]
    scores = y_score[order]
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, truth.size - 1]
    tps = np.cumsum(truth)[idx]
    fps = (idx + 1.0) - tps
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    return fpr, tpr


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recall and precision, both ascending in recall, at every achievable threshold.

    **Diagnostic and visual only** -- built for a W&B curve panel
    (:func:`kappa.tracking.pr_curve_panel`); the area under this curve is
    :func:`diagnostic_metrics`'s ``ap``, computed independently by
    :func:`_binary_average_precision` and not by integrating this curve.
    Prepends the ``(recall=0, precision=1)`` sentinel ``sklearn`` uses so the
    curve starts on the axis. Empty arrays when there are no positives.
    """
    truth = np.asarray(y_true, dtype=np.float64).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    precision, recall, _ = _binary_pr_curve(truth, scores)
    if precision.size == 0:
        return np.empty(0), np.empty(0)
    return np.r_[0.0, recall], np.r_[1.0, precision]


@dataclass(frozen=True)
class FoldSummary:
    """Across-fold mean and spread, for the macro and for every label.

    The spread is the point. A single fold's macro is a draw from a distribution
    whose width, for this task, is of the same order as the entire gap between a
    working solution and a top-10 place, so a mean without a standard deviation
    invites reading noise as progress. ``std`` is the sample standard deviation
    (``ddof=1``) and is ``nan`` for a single fold, because one observation has no
    spread and reporting 0.0 there would claim certainty.
    """

    n_folds: int
    macro_mean: float
    macro_std: float
    per_label_mean: dict[str, float]
    per_label_std: dict[str, float]
    fold_skips: dict[str, int]

    def table(self) -> str:
        """Per-label mean +/- fold std, with a count of the folds that skipped it."""
        lines = [f"{'label':<18}{'mean':>8}{'std':>8}{'skips':>7}"]
        for name, mean in self.per_label_mean.items():
            std = self.per_label_std[name]
            mean_shown = f"{mean:.4f}" if np.isfinite(mean) else "-"
            std_shown = f"{std:.4f}" if np.isfinite(std) else "-"
            lines.append(f"{name:<18}{mean_shown:>8}{std_shown:>8}{self.fold_skips[name]:>7}")
        std_shown = f"{self.macro_std:.4f}" if np.isfinite(self.macro_std) else "-"
        lines.append(
            f"macro {self.macro_mean:.4f} +/- {std_shown} over {self.n_folds} fold(s); "
            f"noise floor {NOISE_FLOOR_MACRO:.4f}"
        )
        return "\n".join(lines)


def fold_summary(results: Sequence[AucResult]) -> FoldSummary:
    """Aggregate per-fold :class:`AucResult` objects into a mean and a spread.

    Skipped labels stay skipped: a label is averaged over the folds that could
    score it, and ``fold_skips`` records how many could not, so a label whose mean
    rests on one fold cannot be mistaken for one that rests on five.
    """
    if not results:
        raise ValueError("fold_summary needs at least one AucResult")
    names = list(results[0].per_label)
    for result in results[1:]:
        if list(result.per_label) != names:
            raise ValueError("folds disagree on their label set, so they cannot be averaged")

    def _mean_std(values: list[float]) -> tuple[float, float]:
        finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if finite.size == 0:
            return float("nan"), float("nan")
        mean = float(finite.mean())
        std = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        return mean, std

    macro_mean, macro_std = _mean_std([r.macro for r in results])
    per_label_mean: dict[str, float] = {}
    per_label_std: dict[str, float] = {}
    fold_skips: dict[str, int] = {}
    for name in names:
        values = [r.per_label[name] for r in results]
        per_label_mean[name], per_label_std[name] = _mean_std(values)
        fold_skips[name] = sum(1 for r in results if name in r.skipped)
    return FoldSummary(
        n_folds=len(results),
        macro_mean=macro_mean,
        macro_std=macro_std,
        per_label_mean=per_label_mean,
        per_label_std=per_label_std,
        fold_skips=fold_skips,
    )


def is_evidence(delta: float, *, noise_floor: float = NOISE_FLOOR_MACRO) -> bool:
    """Does a macro difference clear the noise floor?

    A one-call answer so that "the new run is better" has to be tested rather than
    eyeballed. A NaN delta is not evidence either.
    """
    return bool(np.isfinite(delta) and abs(delta) > noise_floor)


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Map each column independently onto ``(0, 1)`` by its mid-ranks.

    Independently *per column* is the load-bearing word. The twelve labels are
    scored as twelve separate rankings, so a shared normalisation across labels
    would mix twelve unrelated orderings and change every one of them. Within a
    column the map is monotone, so it leaves that column's AUC bit-identical --
    which is exactly why it is safe to apply before combining models.

    Uses ``(rank - 0.5) / n`` so that a single row maps to 0.5 instead of dividing
    by zero.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"rank_normalise expects (n_studies, n_labels), got {scores.shape}")
    if np.isnan(scores).any():
        raise ValueError("rank_normalise got NaN; a NaN score has no rank")
    out = np.empty_like(scores)
    n = scores.shape[0]
    for column in range(scores.shape[1]):
        out[:, column] = (average_ranks(scores[:, column]) - 0.5) / max(n, 1)
    return out


def rank_average(
    members: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Combine model predictions by per-column rank normalisation, then average.

    The only correct blend for a rank-based metric. Averaging raw probabilities is
    dominated by whichever member happens to have the widest spread, so it silently
    weights members by their confidence calibration -- a quantity this competition
    never scores. Rank-averaging is invariant to any monotone rescaling of any
    member, so a member cannot buy influence by being over-confident.
    """
    if not members:
        raise ValueError("rank_average needs at least one member")
    stacked = [rank_normalise(member) for member in members]
    shape = stacked[0].shape
    for member in stacked[1:]:
        if member.shape != shape:
            raise ValueError("ensemble members disagree on shape")
    if weights is None:
        weights = [1.0] * len(stacked)
    if len(weights) != len(stacked):
        raise ValueError(f"{len(weights)} weights for {len(stacked)} members")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("ensemble weights must sum to something positive")
    blended = np.zeros(shape, dtype=np.float64)
    for weight, member in zip(weights, stacked):
        blended += float(weight) * member
    return blended / total
