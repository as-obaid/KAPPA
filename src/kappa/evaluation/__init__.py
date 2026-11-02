"""Metrics, out-of-fold prediction stores, diagnostics, error analysis.

The macro-AUC surface is re-exported here because every other package reads it --
training reports it per epoch, the fold builder aggregates it, the ensemble step
depends on :func:`rank_average` being the only blend anyone reaches for. One import
path means nobody hand-rolls a second ``nanmean`` over eleven labels and calls it
macro-12.

    from kappa.evaluation import macro_roc_auc, rank_average

Nothing here imports ``torch``: scoring a submission and blending members must work
on a CPU-only machine with the modelling extra uninstalled.
"""

from __future__ import annotations

from kappa.evaluation.metrics import (
    NOISE_FLOOR_MACRO,
    AucResult,
    FoldSummary,
    average_ranks,
    fold_summary,
    is_evidence,
    macro_roc_auc,
    rank_average,
    rank_normalise,
)

__all__ = [
    "NOISE_FLOOR_MACRO",
    "AucResult",
    "FoldSummary",
    "average_ranks",
    "fold_summary",
    "is_evidence",
    "macro_roc_auc",
    "rank_average",
    "rank_normalise",
]
