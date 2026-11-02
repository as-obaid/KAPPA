"""Masked losses, the fold training loop, and the noise-floor measurement.

Re-exported here because a training entrypoint should reach for one import path.
``AsymmetricLoss`` is the loss for this task -- plain BCE and symmetric focal are
both wrong on a label set where ``MCL`` sits at 15.5% prevalence and every one of
the twelve labels contributes an equal 1/12 of the macro no matter how rare it is.

    from kappa.training import DataBundle, run_experiment, measure_noise_floor

:func:`measure_noise_floor` is listed beside :func:`run_experiment` on purpose.
The gap between a working solution and a top-10 place on this task is about 0.006
macro, so a single run's number is uninterpretable until the seed-to-seed spread
of the same config has been measured. It is a supported mode, not a manual
exercise.

This package imports ``torch``, which is the optional ``train`` extra. Nothing in
``kappa.labels``, ``kappa.data.manifest`` or ``kappa.evaluation`` may import it,
so the label pipeline still runs on a machine with no CUDA.
"""

from __future__ import annotations

from kappa.training.distill import (
    DEFAULT_LAMBDA_KD,
    DEFAULT_LAMBDA_SUP,
    DEFAULT_TEMPERATURE,
    BinaryKLDistillation,
    DistillationLoss,
    DistillationOutput,
)
from kappa.training.losses import LOSSES, AsymmetricLoss, MaskedBCE, build_loss
from kappa.training.loop import (
    DataBundle,
    EpochRecord,
    ExperimentResult,
    FoldResult,
    NoiseFloorResult,
    build_criterion,
    measure_noise_floor,
    run_experiment,
    seed_everything,
    train_fold,
)

__all__ = [
    "DEFAULT_LAMBDA_KD",
    "DEFAULT_LAMBDA_SUP",
    "DEFAULT_TEMPERATURE",
    "LOSSES",
    "AsymmetricLoss",
    "BinaryKLDistillation",
    "DataBundle",
    "DistillationLoss",
    "DistillationOutput",
    "EpochRecord",
    "ExperimentResult",
    "FoldResult",
    "MaskedBCE",
    "NoiseFloorResult",
    "build_criterion",
    "build_loss",
    "measure_noise_floor",
    "run_experiment",
    "seed_everything",
    "train_fold",
]
