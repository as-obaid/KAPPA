"""Offline checkpoint loading, scoring, rank blending and the submission writer.

This package is the **image-only** half of the project and its import boundary is a
competition rule, not a preference. Reports are training-only -- test studies carry
none -- so a submission is produced from pixels alone. Nothing reachable from here may read or
interpret report text -- :data:`kappa.inference.predict.FORBIDDEN_MODULES` names the
modules that do, and ``tests/test_loop.py`` imports this package in a clean
interpreter and asserts that none of them loads.

    from kappa.inference import load_checkpoint, predict_studies, write_predictions

Three properties hold across everything here. No path is absolute or resolved from
the environment, because the Kaggle notebook imports this code and runs somewhere
none of the training machine's paths exist. No weight is fetched: checkpoints carry
their own state dict and their own normalisation, so a no-internet notebook cannot
fail halfway through building a model. And nothing is calibrated -- macro ROC-AUC
reads only the order of the scores in a column, so the one supported ensemble is
per-column rank normalisation and probability averaging is never it.

**Why the predict names are resolved lazily.** ``pyproject.toml`` states that only
``kappa.models`` and ``kappa.training`` may import ``torch`` at module scope, and
:mod:`kappa.evaluation` states that scoring and blending work with the ``train``
extra uninstalled. Importing a submodule runs its package ``__init__`` first, so
eagerly importing :mod:`kappa.inference.predict` here made ``import
kappa.inference.submission`` require torch -- and the submission writer, the
header check and the rank blender need nothing but numpy. Both stated rules were
false in practice: verified by blocking ``torch`` on ``sys.meta_path`` and
importing the module, which raised ``ImportError``. The names below still resolve
identically through :pep:`562`; only the cost moved to first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kappa.inference.submission import (
    LINE_TERMINATOR,
    N_FIELDS,
    SubmissionCheck,
    constant_submission,
    expected_header_bytes,
    rank_blend,
    read_submission,
    validate_submission,
    write_submission,
)

if TYPE_CHECKING:  # pragma: no cover - import-time cost is what this avoids
    from kappa.inference.predict import (
        CHECKPOINT_VERSION,
        FORBIDDEN_MODULES,
        NOTEBOOK_LIMIT_SECONDS,
        LoadedCheckpoint,
        PredictionResult,
        load_checkpoint,
        predict_studies,
        resolve_device,
        save_inference_checkpoint,
        transform_scores,
        write_predictions,
    )

#: Names re-exported from :mod:`kappa.inference.predict`, which imports torch at
#: module scope. Resolved on first attribute access so that the torch-free half of
#: this package -- the submission writer and the rank blender -- imports without it.
_PREDICT_EXPORTS = frozenset(
    {
        "CHECKPOINT_VERSION",
        "FORBIDDEN_MODULES",
        "NOTEBOOK_LIMIT_SECONDS",
        "LoadedCheckpoint",
        "PredictionResult",
        "load_checkpoint",
        "predict_studies",
        "resolve_device",
        "save_inference_checkpoint",
        "transform_scores",
        "write_predictions",
    }
)


def __getattr__(name: str) -> object:
    """Resolve the torch-dependent exports on first use (:pep:`562`).

    Raising :class:`AttributeError` for anything else is required, not cosmetic:
    returning ``None`` for an unknown name would make a typo look like a missing
    optional feature, and ``hasattr`` would answer True for names this package
    does not have.
    """
    if name in _PREDICT_EXPORTS:
        from kappa.inference import predict

        return getattr(predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CHECKPOINT_VERSION",
    "FORBIDDEN_MODULES",
    "LINE_TERMINATOR",
    "NOTEBOOK_LIMIT_SECONDS",
    "N_FIELDS",
    "LoadedCheckpoint",
    "PredictionResult",
    "SubmissionCheck",
    "constant_submission",
    "expected_header_bytes",
    "load_checkpoint",
    "predict_studies",
    "rank_blend",
    "read_submission",
    "resolve_device",
    "save_inference_checkpoint",
    "transform_scores",
    "validate_submission",
    "write_predictions",
    "write_submission",
]
