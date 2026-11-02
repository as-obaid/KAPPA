"""The training loop, built around the two mistakes that cost this project real time.

**Mistake one: a run that does not say what it ran.** A fold trained against a
misspelled config key uses the default, prints a plausible loss curve and gets filed
under the wrong hypothesis. So :func:`train_fold` prints the resolved config, its
:func:`kappa.config.config_hash`, the git SHA, the seed, the device and the AMP mode
*before* the first batch, and :class:`kappa.models.StudyClassifier` prints which
backbone it actually loaded and whether the weights were pretrained. A run that does
not print those things is not evidence about that backbone.

**Mistake two: timing the inner step.** Partial timing underestimated a fold's wall
clock by **1.7x, three times running**. Every duration reported here is measured
around the *whole* loop -- dataset construction, dataloading, validation and
checkpointing included -- and :attr:`FoldResult.step_seconds` is reported next to it
so the gap between the two is visible rather than assumed away. Never quote
``step_seconds`` as a fold's cost.

    from kappa.training.loop import DataBundle, run_experiment
    from kappa.config import load_config

    cfg = load_config("configs/baseline_axial_fs1.yaml")
    result = run_experiment(cfg, DataBundle.from_config(cfg))
    print(result.report())

**The noise floor comes before any interpretation.** The gap between a working
solution and a top-10 place on this task is about 0.006 macro, and a competitor
measured 0.0020 of run-to-run spread at three folds. Two numbers 0.001 apart are the
same number. :func:`measure_noise_floor` runs one config under several seeds -- same
config hash, asserted -- and reports the spread, so "better" is a comparison against
a locally measured floor rather than against zero. It is a supported mode of
``scripts/train.py``, not a manual exercise, because a floor nobody runs is a floor
nobody has.

Everything the loop emits is chosen to make a result readable rather than merely
reported: per-label AUC and across-fold standard deviation for all twelve labels (a
mean without a spread invites reading noise as progress), an OOF table keyed by study,
the config hash, the git SHA, the seed, and the measured inference seconds per study
that the efficiency track's ``T / 32400`` term consumes. The validation pass is
:func:`kappa.inference.predict.predict_studies` -- the same function the Kaggle
notebook calls -- so the reported rate is measured by the code that will be timed.

Two things this loop deliberately does not do. It does not calibrate: macro ROC-AUC is
rank-based and threshold-free, so a sigmoid or a temperature leaves every number
bit-identical while consuming a fold. And it selects the **last** epoch, not the best:
picking the best epoch by validation macro fits the very split the number is quoted
from, and with a ten-epoch schedule the difference is noise. ``select="best"`` exists
for the ablation and says so.
"""

from __future__ import annotations

import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from kappa.config import SLOT_NAMES, VIDEO_BACKBONES, TrainConfig, config_hash, describe
from kappa.constants import ID_COLUMN, LABELS, N_LABELS
from kappa.data.augment import build_augment
from kappa.data.dataset import CropSpec, Normalisation, StudyDataset
from kappa.data.folds import TEST_FOLD, check_fold_columns, dev_mask, read_folds
from kappa.evaluation.metrics import (
    NOISE_FLOOR_MACRO,
    AucResult,
    DiagnosticResult,
    FoldSummary,
    diagnostic_metrics,
    fold_summary,
    is_evidence,
    macro_roc_auc,
)
from kappa.inference.predict import (
    NOTEBOOK_LIMIT_SECONDS,
    PredictionResult,
    predict_studies,
    resolve_device,
    save_inference_checkpoint,
)
from kappa.labels.features import REPORT_FEATURE_NAMES, load_features
from kappa.models import StudyClassifier
from kappa.models.teacher import TeacherClassifier
from kappa.tracking import (
    NoOpRun,
    diagnostic_label_metrics,
    epoch_metrics,
    finish_run,
    fold_prefix,
    install_walltime_guard,
    git_sha,
    label_metrics,
    log_standard_artifacts,
    per_fold_table,
    per_label_summary_table,
    per_label_table,
    pr_curve_panel,
    require_run_directory,
    roc_curve_panel,
    start_training_run,
    step_metrics,
    summarise,
)
from kappa.training.averaging import build_averager
from kappa.training.distill import DistillationLoss
from kappa.training.resume import (
    ResumeIdentity,
    ResumeMismatch,
    checkpoint_of,
    clear_resume,
    install_averager,
    load_resume,
    read_sidecar,
    restore_rng,
    save_resume,
)
from kappa.training.report_dataset import (
    build_report_collate,
    build_study_dataset,
    report_collate,
)
from kappa.training.losses import AsymmetricLoss, MaskedBCE

__all__ = [
    "MASK_SUFFIX",
    "NOTEBOOK_LIMIT_SECONDS",
    "OOF_COLUMNS",
    "TRUE_SUFFIX",
    "DataBundle",
    "EpochRecord",
    "ExperimentResult",
    "FoldResult",
    "NoiseFloorResult",
    "build_criterion",
    "build_model",
    "build_optimiser",
    "build_scheduler",
    "measure_noise_floor",
    "oof_columns",
    "run_experiment",
    "seed_everything",
    "slot_indices",
    "train_fold",
]

#: Suffixes in the OOF table. The ground truth and the mask travel *with* the scores
#: rather than being re-joined later: a score column without the mask it was evaluated
#: under can be re-scored against a different mask and quietly produce a different
#: macro, which is the kind of number nobody can reproduce.
TRUE_SUFFIX = "__true"
MASK_SUFFIX = "__mask"


def oof_columns() -> tuple[str, ...]:
    """Out-of-fold table schema: ids, then scores, then truth, then mask."""
    return (
        ID_COLUMN,
        "fold",
        "seed",
        *LABELS,
        *(f"{name}{TRUE_SUFFIX}" for name in LABELS),
        *(f"{name}{MASK_SUFFIX}" for name in LABELS),
    )


#: Materialised once for the assertion in :func:`_oof_frame`.
OOF_COLUMNS: tuple[str, ...] = oof_columns()


def slot_indices(names: Sequence[str]) -> tuple[int, ...]:
    """``("Axial_fs1",)`` to ``(1,)``: config slot names to global slot ids.

    The *global* id, always, even when the run selects one slot. A slot embedding
    trained with ``slots=(1,)`` stays valid when the run widens to all six, so adding
    coronal is a config change rather than a reindexing that invalidates a checkpoint.
    """
    missing = [name for name in names if name not in SLOT_NAMES]
    if missing:
        raise ValueError(f"unknown slot(s) {missing}; legal names are {list(SLOT_NAMES)}")
    return tuple(SLOT_NAMES.index(name) for name in names)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, and make cuDNN pick a fixed algorithm.

    This buys reproducibility of the *data order and the initialisation*, which is what
    a seed-to-seed noise measurement needs. It does **not** claim bitwise determinism:
    ``torch.use_deterministic_algorithms`` is not set, because several reductions have
    no deterministic CUDA kernel and enabling it would either raise mid-epoch or slow
    the run by a factor nobody budgeted. Two seeds differing is the measurement; one
    seed reproducing exactly is not promised.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in this environment
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


#: Corpus median inter-slice gap and through-plane span, in millimetres, measured over
#: the training manifest. Used only to turn a slice count into the coverage it implies in the run
#: log, so that "contiguous, 8 slices" reads as the 24.5 mm of knee it actually is.
_MEDIAN_SLICE_GAP_MM = 3.50
_MEDIAN_THROUGH_PLANE_MM = 105.60


def build_model(cfg: TrainConfig) -> StudyClassifier:
    """Construct the model from config. It prints its own identity; do not suppress it."""
    return StudyClassifier(
        backbone=cfg.model.backbone,
        n_slices=cfg.data.n_slices,
        pretrained=cfg.model.pretrained,
        drop_path_rate=cfg.model.drop_path_rate,
        aux_dim=cfg.model.aux_dim,
        head=cfg.model.head,
        head_num_heads=cfg.model.head_num_heads,
        freeze_backbone_bn=cfg.model.freeze_backbone_bn,
        # `pool` is a default-dropped section, so a config that never mentions it
        # passes False/False here and builds the model this repository has always
        # built -- same modules, same parameter names, same arithmetic.
        per_label_slots=cfg.pool.per_label_slots,
        head_residual=cfg.pool.head_residual,
        # The input matrix has to reach the backbone at CONSTRUCTION, not only the
        # dataloader. A ViT builds its position embedding for one grid: timm's
        # `vit_base_patch14_dinov2` is native 518, and handed a 392 px batch it raises
        # `Input height (392) doesn't match model (518)` on the first forward -- after
        # the job has already queued for and been given a card. Passing it here is
        # what makes `data.size: 392` mean the same thing to the physical crop and to
        # the encoder. A convolutional backbone rejects the kwarg, still builds, and
        # records that it dropped it, so this is a no-op for convnext_small and leaves
        # the measured baseline byte-identical.
        img_size=cfg.data.size,
    )


def build_teacher(cfg: TrainConfig, report_dim: int) -> TeacherClassifier:
    """Construct the teacher. Image branch identical to the student, plus the report."""
    return TeacherClassifier(
        report_dim=report_dim,
        backbone=cfg.model.backbone,
        n_slices=cfg.data.n_slices,
        pretrained=cfg.model.pretrained,
        drop_path_rate=cfg.model.drop_path_rate,
        head=cfg.model.head,
        head_num_heads=cfg.model.head_num_heads,
        freeze_backbone_bn=cfg.model.freeze_backbone_bn,
        report_hidden=cfg.distill.report_hidden,
        report_dropout=cfg.distill.report_dropout,
        # Same reason as `build_model`, and one step worse if omitted: the teacher is
        # written to a checkpoint and rebuilt later from `TeacherClassifier.arch`, so a
        # missing input matrix would survive into the rebuild and assert during
        # distillation rather than during the teacher's own run.
        img_size=cfg.data.size,
    )


def report_frame(cfg: TrainConfig) -> pd.DataFrame | None:
    """Load the report feature table, or ``None`` when the run is image-only.

    The name and order check lives in :func:`kappa.labels.features.load_features`,
    which owns the vocabulary. This only renames the identifier to ``ID_COLUMN`` so
    the frame keys the same way every other table in this project does.
    """
    if cfg.distill.report_features is None:
        return None
    path = Path(cfg.distill.report_features)
    if not path.exists():
        raise FileNotFoundError(
            f"distill.report_features={path} does not exist. Build it with "
            "`python scripts/build_report_features.py`, which reads the extraction "
            "records the label pipeline already wrote."
        )
    uids, values, present = load_features(path)
    frame = pd.DataFrame(values, columns=list(REPORT_FEATURE_NAMES))
    frame.insert(0, ID_COLUMN, uids)
    frame["report_present"] = present
    return frame


def resolve_teacher_ckpt(raw: Path | str, *, fold: int, seed: int) -> Path:
    """Substitute ``{fold}`` and ``{seed}`` in a teacher checkpoint path.

    The folds are the parallelism here -- one array element per fold -- and each
    element needs *its own* fold's teacher, or :func:`load_frozen_teacher` refuses
    it for leaking validation labels. Without a placeholder the alternatives are
    five near-identical configs or a ``--set`` override per element, and both are
    ways to end up distilling fold 3 from fold 0's teacher.

        distill.teacher_ckpt: data/runs/teacher_axial_fs1/fold{fold}_seed{seed}.pt

    A path with no placeholder is returned unchanged, so a single-fold run can
    still name one file directly.
    """
    return Path(str(raw).replace("{fold}", str(fold)).replace("{seed}", str(seed)))


def save_teacher_checkpoint(
    path: Path | str, model: TeacherClassifier, *, meta: dict[str, Any] | None = None
) -> Path:
    """Write a teacher checkpoint, marked so inference cannot mistake it for a student.

    Separate from :func:`save_inference_checkpoint` on purpose. A teacher reads report
    text and can never be submitted, so its file carries ``kappa_teacher: True`` and
    the feature names its report branch was trained on. Loading it into the image-only
    path raises rather than silently failing a shape check.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kappa_teacher": True,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "report_features": list(REPORT_FEATURE_NAMES),
            # The teacher's OWN architecture. Rebuilding it from the student's
            # config instead is a shape mismatch waiting to happen: the two files
            # are different experiments and their report widths need not agree.
            "arch": model.arch(),
            "meta": dict(meta or {}),
        },
        path,
    )
    return path


def load_frozen_teacher(
    path: Path | str,
    *,
    fold: int | None = None,
    device: str | torch.device | None = None,
) -> TeacherClassifier:
    """Rebuild a teacher from its checkpoint, in eval mode with gradients off.

    Frozen is the default because the research question is what the teacher's fixed
    knowledge is worth to the student. A teacher that keeps training during
    distillation makes the student's gain unattributable to the transfer.

    ``pretrained=False``: every weight comes from the file, and asking timm for hub
    weights on a compute node fails after the architecture is already built, which
    makes the traceback blame the backbone.

    **``fold`` is checked, and this is the one guard that protects the result.** A
    teacher trained on fold 0 had fold 1's validation studies in its own training
    set. Distilling fold 1 from that teacher pipes those studies' labels into the
    student through the soft targets, and the fold-1 score comes back inflated with
    nothing raised and no obvious tell. So the folds must match, and a mismatch is
    refused rather than warned about.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload.get("kappa_teacher"):
        raise ValueError(
            f"{path} is not a KAPPA teacher checkpoint. A student checkpoint has no "
            "report branch, so distilling from one would score every study on an "
            "untrained head."
        )
    meta = payload.get("meta") or {}
    trained_fold = meta.get("fold")
    if fold is not None and trained_fold is not None and int(trained_fold) != int(fold):
        raise ValueError(
            f"{path} was trained on fold {trained_fold} but this run trains fold {fold}. "
            f"That teacher had fold {fold}'s validation studies in its own training set, "
            "so its soft targets would leak them into the student and inflate the score. "
            "Use the teacher for this fold."
        )
    names = list(payload.get("report_features") or ())
    if names != list(REPORT_FEATURE_NAMES):
        raise ValueError(
            f"{path} was trained on {len(names)} report features and this build has "
            f"{len(REPORT_FEATURE_NAMES)}. The vocabulary moved, so the teacher would "
            "read a permuted report vector and its soft targets would be noise."
        )
    arch = payload.get("arch")
    if not isinstance(arch, dict):
        raise ValueError(
            f"{path} carries no architecture record, so the teacher cannot be rebuilt "
            "without guessing its report width from the student's config -- which is "
            "how a silent shape mismatch becomes a crash halfway through a fold."
        )
    # pretrained=False unconditionally: every weight is in the file, and a hub
    # download on a compute node fails after the model is already built.
    teacher = TeacherClassifier(**{**arch, "pretrained": False})
    teacher.load_state_dict(payload["state_dict"], strict=True)
    teacher.to(resolve_device(device)).eval()
    teacher.requires_grad_(False)
    return teacher


def build_criterion(cfg: TrainConfig) -> torch.nn.Module:
    """The loss, configured. Imported from :mod:`kappa.training.losses`.

    Not reimplemented here, and not swappable by accident: plain BCE and symmetric
    focal are both wrong for this label distribution, and ``MaskedBCE`` exists in the
    losses module purely so the ablation is a deliberate call rather than a default.
    ``kind: bce`` selects it, and ``kappa.config`` refuses a file that sets an ASL
    focusing exponent next to it rather than accepting a setting that will not run.

    ``gamma_neg_by_label`` is passed through as a vector in ``LABELS`` order. It is
    resolved by the config, so an unset label keeps the scalar and the vector cannot
    silently be short.
    """
    if cfg.loss.kind == "bce":
        return MaskedBCE()
    return AsymmetricLoss(
        gamma_neg=cfg.loss.gamma_neg,
        gamma_pos=cfg.loss.gamma_pos,
        clip=cfg.loss.clip,
        gamma_neg_by_label=(
            cfg.loss.gamma_neg_vector() if cfg.loss.gamma_neg_by_label else None
        ),
    )


def positive_rates(bundle: DataBundle, study_uids: Sequence[str]) -> dict[str, float]:
    """Per-label P(positive | known) over ``study_uids``, in ``LABELS`` order.

    Printed at startup because the loss's defaults encode an assumption about it.
    ``gamma_neg=4.0`` means "positives are rare, focus down the easy negatives",
    which is a claim about this table -- and on the derived table it is true of
    Fracture at 7.4% and false of Effusion at 58.8%, where it focuses down the
    rarer class instead. A run that does not state the rates it trained against
    cannot be read as evidence about that choice, and the rates move whenever the
    derivation or the `unmentioned` policy changes.
    """
    values, mask = _truth_for(bundle, study_uids)
    rates: dict[str, float] = {}
    for column, name in enumerate(LABELS):
        known = mask[:, column]
        n_known = int(known.sum())
        rates[name] = float(values[known, column].mean()) if n_known else float("nan")
    return rates


#: Parameter-name prefix that marks the PRETRAINED trunk, and nothing else.
#:
#: ``encoder.`` -- the obvious prefix, and the one this started as -- is WRONG, and
#: wrong in the direction that matters. Both encoders hang their conditioning tables
#: off the encoder alongside the trunk: ``SliceEncoder`` builds ``plane_film`` and
#: ``contrast_film`` as ``nn.Embedding`` and initialises them with ``nn.init.normal_``
#: (``encoder.py:450-453``), and ``VolumeEncoder`` builds them as ``nn.Parameter(
#: torch.randn(...) * _FILM_INIT_STD)`` (``volume_encoder.py:258-259``). They are as
#: freshly initialised as the head is, and under ``encoder.`` they would have trained
#: at the SLOWED trunk rate -- a fresh table given a pretrained table's learning rate,
#: which is the exact error this split exists to correct, applied backwards.
#:
#: Checked by enumerating ``named_parameters()`` on a built
#: ``StudyClassifier`` of each kind: ``test_convnext`` (``SliceEncoder``, 147 tensors)
#: and ``r3d_18`` (``VolumeEncoder``, 108 tensors). Both name their trunk
#: ``self.backbone``, so one prefix covers both; the set of names matching
#: ``encoder.backbone.`` equals the set of ``model.encoder.backbone.parameters()``
#: exactly, in both directions, on both; and the ONLY parameters under ``encoder.``
#: but outside ``encoder.backbone.`` are the two FiLM tables, on both.
#:
#: ``TeacherClassifier`` nests a whole ``StudyClassifier`` under ``self.image``
#: (``teacher.py:154``), so its trunk is named ``image.encoder.backbone.`` and this
#: prefix does not reach it. That is why :func:`build_optimiser` refuses a differential
#: teacher run outright instead of quietly filing the entire teacher into the head
#: groups: the run would look like a differential-LR experiment, be a uniform one, and
#: report a null effect for a technique it never applied. *The teacher arm has never
#: run, so that path is reasoned from the parameter names, not observed.*
_BACKBONE_PREFIX = "encoder.backbone."

#: The four param groups :func:`build_optimiser` builds, in the order it builds them.
#: **The order is a contract, not a detail** -- see that function's docstring for why
#: group 0 is head-decay -- so it lives in one named table that a test can pin.
_PARAM_GROUP_NAMES = ("head-decay", "head-no-decay", "backbone-decay", "backbone-no-decay")


def build_optimiser(cfg: TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW in four groups: {head, backbone} x {decay, no-decay}.

    Decaying a bias or a LayerNorm gain shrinks it toward zero for no regularisation
    benefit -- there is no capacity in a per-channel scalar to over-fit with -- and on
    a small labelled set it measurably fights the pretrained statistics. The split is
    by tensor rank, which catches every norm weight without naming modules.

    The second split is by PROVENANCE, and it is what ``optim.backbone_lr_mult``
    controls: the trunk trains at ``mult x`` the head's rate. "Trunk" means the
    transferred weights and only those -- ``_BACKBONE_PREFIX`` is
    ``encoder.backbone.``, not ``encoder.``, because the encoder also owns two randomly
    initialised FiLM tables which belong with the head. At the default ``1.0``
    all four groups carry ``cfg.optim.lr`` and this function is arithmetically the
    two-group version it replaced -- the split exists, and costs nothing, until a
    config asks for it.

    **Group order is part of the interface.** Group 0 is head-decay because the
    training loop logs ``optimiser.param_groups[0]["lr"]`` as *the* learning rate of
    the run (``loop.py`` step metrics and ``EpochRecord.lr_last``). Putting the head
    first keeps that series meaning what it has always meant -- the unscaled
    ``cfg.optim.lr`` trajectory -- so a uniform run from before this change and a
    differential run after it can be read on one axis, with the trunk's rate
    recoverable as ``mult x`` the logged curve and the multiplier itself printed in the
    run-log header. All four groups are constructed even when one of them is empty, so
    the index means the same thing on every backbone and head combination; AdamW
    tolerates an empty group and a scheduler still assigns it a rate.
    """
    buckets: dict[str, list[torch.nn.Parameter]] = {name: [] for name in _PARAM_GROUP_NAMES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        part = "backbone" if name.startswith(_BACKBONE_PREFIX) else "head"
        decay = "no-decay" if parameter.ndim <= 1 or name.endswith(".bias") else "decay"
        buckets[f"{part}-{decay}"].append(parameter)

    head_lr = float(cfg.optim.lr)
    backbone_lr = head_lr * float(cfg.optim.backbone_lr_mult)
    # A multiplier that matched nothing is the failure this whole path exists to avoid:
    # the run trains uniformly, logs a config that says otherwise, and its null result
    # gets read as evidence about differential learning rates. Refuse it at build time,
    # naming the prefix, rather than at analysis time. Deliberately conditional on the
    # multiplier: at 1.0 there is nothing to erase, and a model with no trainable trunk
    # (a fully frozen encoder) is then still a legal thing to build.
    if cfg.optim.backbone_lr_mult != 1.0 and not (
        buckets["backbone-decay"] or buckets["backbone-no-decay"]
    ):
        raise ValueError(
            f"optim.backbone_lr_mult={cfg.optim.backbone_lr_mult} asks the backbone to "
            f"train slower than the head, but no trainable parameter of "
            f"{type(model).__name__} is named {_BACKBONE_PREFIX!r}*, so the multiplier "
            "would apply to nothing and the run would be a uniform-LR run wearing a "
            "differential-LR config. TeacherClassifier nests its trunk at "
            "'image.encoder.backbone.'; widen _BACKBONE_PREFIX before running that arm."
        )
    return torch.optim.AdamW(
        [
            {
                "name": "head-decay",
                "params": buckets["head-decay"],
                "lr": head_lr,
                "weight_decay": cfg.optim.weight_decay,
            },
            {
                "name": "head-no-decay",
                "params": buckets["head-no-decay"],
                "lr": head_lr,
                "weight_decay": 0.0,
            },
            {
                "name": "backbone-decay",
                "params": buckets["backbone-decay"],
                "lr": backbone_lr,
                "weight_decay": cfg.optim.weight_decay,
            },
            {
                "name": "backbone-no-decay",
                "params": buckets["backbone-no-decay"],
                "lr": backbone_lr,
                "weight_decay": 0.0,
            },
        ],
        lr=head_lr,
        betas=(0.9, 0.999),
    )


def build_scheduler(
    cfg: TrainConfig, optimiser: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """OneCycle or cosine-with-warmup, stepped **per optimiser step**, not per epoch.

    Per step because the schedule is short: ten epochs over a few hundred batches is a
    few thousand steps, and a per-epoch cosine would spend its entire warmup on epoch
    one. ``total_steps`` counts optimiser steps after gradient accumulation, so
    changing ``grad_accum`` does not silently change the shape of the schedule.

    Both branches preserve the PER-GROUP rates :func:`build_optimiser` set, so a
    backbone running at ``optim.backbone_lr_mult x`` the head's rate keeps that ratio
    for the whole schedule. Neither branch preserves them for free, they preserve them
    for different reasons, and getting either one wrong is silent -- so both reasons
    are written out at the branch that depends on them.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")

    # Both branches read each param group's CURRENT `lr` as that group's intended peak
    # (`max_lr` below; `base_lrs` inside LambdaLR), which is only the same thing while
    # nothing has stepped. Every scheduler stamps `initial_lr` into the groups it
    # takes over, so its presence is the cheap, exact signal that this optimiser has
    # already been scheduled -- by an earlier `build_scheduler`, or by a resume path
    # that restored a mid-run optimiser. Either way the rates found here would be
    # annealed values, and the new schedule would peak at wherever the old one had
    # got to: a silently under-trained run, and on a differential run a silently
    # different backbone/head ratio too. The single call site today builds the
    # optimiser on the line above, so this cannot fire; it exists so that the day a
    # resume path is added, it fails here instead of in the results table.
    scheduled = [
        group.get("name", str(index))
        for index, group in enumerate(optimiser.param_groups)
        if "initial_lr" in group
    ]
    if scheduled:
        raise ValueError(
            f"param group(s) {scheduled} already carry 'initial_lr', so this optimiser "
            "has been driven by a scheduler before. build_scheduler reads each group's "
            "current lr as its peak, which is only valid on a freshly built optimiser; "
            "on this one it would read annealed rates and build a schedule that peaks "
            "below the configured learning rate. Build the optimiser and the scheduler "
            "together, or set each group's lr back to its peak first."
        )

    # OneCycleLR divides by the width of its warmup phase, so it needs at least two
    # steps inside it and at least two after it: on a 4-step schedule -- which is what
    # a two-epoch smoke run over a handful of studies produces -- it raises
    # ZeroDivisionError from inside its own get_lr. Rather than let a wiring run die on
    # an arithmetic edge in the scheduler, that case falls through to the cosine branch
    # and says so. `warmup_pct=0.0` is also legal in the config and means "no warmup",
    # which in a stepped schedule is the smallest phase OneCycleLR will accept.
    if cfg.optim.schedule == "onecycle":
        minimum_pct = 2.0 / total_steps
        if total_steps >= 4 and minimum_pct <= 0.5:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimiser,
                # A LIST, one peak per param group, and NEVER a scalar. `max_lr` goes
                # through OneCycleLR's `_format_param`, which broadcasts a single
                # number across every group: handed a scalar it writes the same peak
                # -- and the same `initial_lr = max_lr/div_factor` -- into all of them,
                # silently erasing the backbone/head split `build_optimiser` just set
                # up. Checked on torch 2.14: two groups constructed at 1e-3 and 1e-4
                # both read 4e-5 immediately after the scalar-form constructor and both
                # read 1e-3 after one `.step()`, while the list form holds the 0.1 ratio
                # at every step. A differential run built on the scalar form would be
                # numerically identical to a uniform one, and its null result would be
                # read as evidence against differential learning rates rather than as a
                # wiring bug. `tests/test_loop.py` asserts the ratio AFTER a
                # `scheduler.step()` for exactly this reason -- asserting it on the
                # freshly built optimiser does not catch this.
                #
                # Read off the optimiser rather than recomputed from cfg: each group's
                # `lr` IS its intended peak here (nothing has stepped yet -- this is
                # called immediately after `build_optimiser`), so this stays correct
                # whatever order or number of groups that function chooses.
                max_lr=[group["lr"] for group in optimiser.param_groups],
                total_steps=total_steps,
                pct_start=min(max(cfg.optim.warmup_pct, minimum_pct), 0.5),
                anneal_strategy="cos",
                cycle_momentum=False,
            )
        warnings.warn(
            f"schedule=onecycle over {total_steps} optimiser step(s) has no room for a "
            "warmup phase, so this run uses cosine-with-warmup instead. A schedule this "
            "short is a wiring run, not an experiment -- do not compare its number to a "
            "full one.",
            stacklevel=2,
        )

    warmup = int(round(cfg.optim.warmup_pct * total_steps))

    # LambdaLR is safe where the scalar `max_lr` above was not, and the difference is
    # worth stating so nobody "fixes" this branch into a form that breaks it. One
    # lambda is applied to every param group, but as a MULTIPLIER on that group's own
    # `initial_lr`, which `LRScheduler.__init__` captures from `group["lr"]`. The
    # backbone/head ratio is therefore preserved through warmup and the whole cosine
    # tail, and a differential run keeps its split. Checked on torch 2.14: groups at
    # 1e-3 and 1e-4 still read a 0.1 ratio after `.step()`. Do not rewrite this to set
    # `group["lr"]` from a single number.
    def factor(step: int) -> float:
        if warmup and step < warmup:
            # step+1 so the first step is not exactly zero: a zero learning rate on
            # step 0 is a wasted batch and, with gradient accumulation, a wasted set.
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)


def _amp_setup(
    cfg: TrainConfig, device: torch.device
) -> tuple[torch.dtype | None, torch.amp.GradScaler | None, str]:
    """Resolve ``(autocast dtype, grad scaler, label)`` for this device.

    AMP is a CUDA path here. On CPU it is switched off and *said out loud* rather than
    quietly enabled, because CPU autocast is not the arithmetic a GPU run performs: a
    CPU timing taken under it describes neither machine, and a CPU smoke test that
    reported ``amp=bf16`` would be claiming to have exercised a path it did not.

    A scaler exists only for fp16. bf16 has fp32's exponent range, so there is no
    underflow to rescue and a scaler would add a synchronisation per step for nothing.
    """
    if not cfg.optim.amp or device.type != "cuda":
        why = "disabled in config" if not cfg.optim.amp else f"{device.type} is not a cuda device"
        return None, None, f"off ({why})"
    dtype = torch.bfloat16 if cfg.optim.amp_dtype == "bf16" else torch.float16

    # bf16 needs Ampere (sm_80) or newer, and a shared GPU partition is often
    # heterogeneous: T4 (sm_75), V100 (sm_70), A100 (sm_80) and H200 (sm_90) can all
    # live in one. Asked for bf16 on a T4, cuDNN does not report an unsupported dtype
    # -- it fails deep inside ConvNeXt's depthwise convolution with "GET was unable to
    # find an engine to execute this computation", which reads like a broken install
    # rather than the wrong card. slurm/train.sbatch pins --gres=gpu:h200:1 so this
    # should not arise; the check is here because the failure it replaces costs an
    # hour to understand.
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        raise RuntimeError(
            f"config asks for amp_dtype=bf16 but {name} (sm_{major}{minor}) does not "
            "support it -- bf16 needs sm_80 (Ampere) or newer. Pin the card with "
            "--gres=gpu:h200:1, or set optim.amp_dtype=fp16. Left alone this does not "
            "fail here; it fails inside cuDNN with 'unable to find an engine'."
        )

    scaler = torch.amp.GradScaler("cuda") if dtype is torch.float16 else None
    return dtype, scaler, f"cuda/{cfg.optim.amp_dtype}"


@dataclass(frozen=True)
class DataBundle:
    """Manifest, labels, folds and the cache directory, resolved once.

    Held together rather than passed as four arguments so that a fold split cannot be
    taken from one table and the pixels from another. ``folds`` carries one of the
    pinned :data:`kappa.data.folds.FOLD_SCHEMAS`, including both ``fold`` (grouped by
    report content) and ``fold_scanner`` (grouped by report content *and* scanner).
    Comparing a score across the two columns is how site memorisation becomes
    measurable -- worth about 0.05 of a random-fold score -- so which column a run
    used belongs in its log.

    A v2 table additionally carries ``split``, and its ``test`` rows belong to
    **neither** side of any fold; see :meth:`split`.
    """

    manifest: pd.DataFrame
    labels: pd.DataFrame
    folds: pd.DataFrame
    cache_dir: Path

    def __post_init__(self) -> None:
        check_fold_columns(self.folds.columns)
        if ID_COLUMN not in self.labels.columns:
            raise KeyError(f"label table is missing its identifier column {ID_COLUMN!r}")

    @classmethod
    def from_config(cls, cfg: TrainConfig, *, root: Path | str | None = None) -> DataBundle:
        """Read the three tables named by the config, from paths relative to ``root``.

        Every path is a config value, never an environment lookup, because the same
        config has to resolve on a laptop, on a cluster and inside a notebook.
        """
        base = Path(root) if root is not None else Path.cwd()

        def resolve(value: Path | None, what: str) -> Path:
            if value is None:
                raise ValueError(
                    f"config does not name {what}; set data.{what} (or folds.file) so the "
                    "run states where its inputs came from"
                )
            path = Path(value)
            return path if path.is_absolute() else base / path

        manifest_path = resolve(cfg.data.manifest, "manifest")
        labels_path = resolve(cfg.data.labels, "labels")
        folds_path = resolve(cfg.folds.file, "file")
        cache = resolve(cfg.data.cache_dir, "cache_dir")
        return cls(
            manifest=pd.read_parquet(manifest_path),
            labels=pd.read_parquet(labels_path),
            folds=read_folds(folds_path),
            cache_dir=cache,
        )

    def split(self, fold: int, *, column: str = "fold") -> tuple[tuple[str, ...], tuple[str, ...]]:
        """``(train_uids, val_uids)`` for one fold, asserted disjoint.

        The assertion is not defensive padding. A fold table built with the wrong
        grouping puts one patient's two studies on both sides, which inflates the
        validation macro by an amount that looks exactly like progress.

        **A row that does not participate is on neither side.** ``folds_v2`` marks
        its frozen held-out test set with ``split == "test"`` and ``fold == -1``, and
        the obvious ``train = uids[assignment != fold]`` puts all 661 of those
        studies into *every* fold's training set. Nothing raises, no shape changes,
        and the only symptom is a cross-validation score that is quietly too high --
        contradicted in late October by the private leaderboard, when nothing can be
        changed. Participation is therefore decided by
        :func:`kappa.data.folds.dev_mask`, which tests for it positively instead of
        inferring it from an inequality against one fold index.

        A negative ``fold`` is refused outright rather than returning the test set as
        a validation fold: ``-1`` is the sentinel for "in no fold", not a fold.
        """
        if column not in self.folds.columns:
            raise KeyError(
                f"fold table has no {column!r} column; it has {tuple(self.folds.columns)}"
            )
        if int(fold) < 0:
            raise ValueError(
                f"fold {fold} is negative, and a negative fold index marks a row that is "
                f"in no fold at all -- folds_v2 writes {TEST_FOLD} for the frozen "
                f"held-out test set. There is nothing to validate on and nothing to "
                f"train on; the test set is read once, at the end."
            )
        assignment = self.folds[column].astype(int).to_numpy()
        eligible = dev_mask(self.folds, column=column)
        present = sorted({int(value) for value in assignment[eligible]})
        if int(fold) not in present:
            raise ValueError(f"fold {fold} does not appear in {column!r}: {present}")
        uids = self.folds[ID_COLUMN].astype(str).to_numpy()
        val = tuple(uids[eligible & (assignment == fold)])
        train = tuple(uids[eligible & (assignment != fold)])
        overlap = set(train) & set(val)
        if overlap:
            raise ValueError(
                f"{len(overlap)} study/studies are in both the train and validation side of "
                f"fold {fold}, e.g. {sorted(overlap)[:3]}. A leaked study inflates the macro "
                "by an amount indistinguishable from a real improvement."
            )
        return train, val

    def usable_cells(self, study_uids: Sequence[str]) -> int:
        """How many label cells are usable across these studies, i.e. mask is True.

        Reported per fold because it is the denominator of everything downstream. 98.7%
        of *raw* label cells are unknown; a run whose derived table is barely denser
        than that is training on almost nothing, and the loss curve does not say so.
        """
        wanted = set(str(uid) for uid in study_uids)
        rows = self.labels[self.labels[ID_COLUMN].astype(str).isin(wanted)]
        columns = [f"{name}{MASK_SUFFIX}" for name in LABELS if f"{name}{MASK_SUFFIX}" in rows]
        if not columns:
            return 0
        return int(rows[columns].to_numpy(dtype=bool).sum())


@dataclass(frozen=True)
class EpochRecord:
    """One epoch: what the loss did, what the macro did, and what it cost.

    ``skipped`` is carried per epoch on purpose. A label with one class present in the
    validation fold has no AUC, so the macro is a mean over fewer than twelve labels --
    a different quantity that drifts as which labels are single-class changes. Printing
    it every epoch means a macro can never be read without its denominator.
    """

    epoch: int
    train_loss: float
    n_batches: int
    n_optimiser_steps: int
    lr_last: float
    macro: float
    n_scored: int
    skipped: tuple[str, ...]
    train_seconds: float
    val_seconds: float
    step_seconds: float

    def line(self) -> str:
        note = f"  skipped {len(self.skipped)}" if self.skipped else ""
        return (
            f"epoch {self.epoch:>3}  loss {self.train_loss:.4f}  "
            f"macro {self.macro:.4f} over {self.n_scored}/{N_LABELS}  "
            f"lr {self.lr_last:.2e}  "
            f"train {self.train_seconds:.1f}s val {self.val_seconds:.1f}s{note}"
        )


@dataclass(frozen=True)
class FoldResult:
    """Everything one fold is evidence for, including what it cost to produce.

    :attr:`wall_seconds` is the whole fold. :attr:`step_seconds` is the forward,
    backward and optimiser step alone. Both are here so that
    :attr:`partial_timing_ratio` -- what quoting the inner step would have
    underestimated by -- is a number in the log rather than a lesson someone relearns.
    """

    fold: int
    seed: int
    name: str
    config_hash: str
    git_sha: str
    backbone: str
    device: str
    amp: str
    n_train: int
    n_val: int
    usable_train_cells: int
    usable_val_cells: int
    auc: AucResult
    #: Average precision / precision / recall / F1 for the SELECTED epoch. Diagnostic
    #: only: :attr:`macro` is what this fold is evidence for, and nothing here may
    #: choose a checkpoint, a fold or a model.
    diagnostics: DiagnosticResult
    oof: pd.DataFrame
    epochs: tuple[EpochRecord, ...]
    selected_epoch: int
    wall_seconds: float
    step_seconds: float
    inference: PredictionResult
    checkpoint: Path | None
    #: The averaged-weights arm, when ``optim.ema_decay`` or ``optim.swa_last_n``
    #: asked for one. ``None`` when averaging is off, which is the default. It sits
    #: BESIDE :attr:`auc` rather than replacing it: the arm's question is whether
    #: averaging helped, and one number cannot answer that.
    averaged: AucResult | None = None
    averaged_checkpoint: Path | None = None
    averaging: str | None = None

    @property
    def macro(self) -> float:
        return self.auc.macro

    @property
    def partial_timing_ratio(self) -> float:
        """Whole fold divided by inner steps. Quoting the inner step lies by this much."""
        return self.wall_seconds / max(self.step_seconds, 1e-9)

    @property
    def seconds_per_study(self) -> float:
        """Measured inference cost, from the same function the notebook calls."""
        return self.inference.seconds_per_study

    def report(self) -> str:
        lines = [
            f"fold {self.fold} seed {self.seed}  config {self.config_hash}  sha {self.git_sha}",
            f"  backbone {self.backbone} on {self.device}, amp {self.amp}",
            f"  studies  {self.n_train} train / {self.n_val} val; usable label cells "
            f"{self.usable_train_cells} train / {self.usable_val_cells} val",
            f"  selected epoch {self.selected_epoch} of {len(self.epochs)}",
            self.auc.table(),
            # On one line and labelled, because this is the text that lands in the
            # Slurm log and gets read first. AP is here and ROC-AUC is above because
            # AP reads the positive rate and ROC-AUC does not -- Fracture at 7.5% can
            # hold a respectable AUC and a near-zero AP at the same time.
            f"  DIAGNOSTIC ONLY, never a selection signal: AP macro "
            f"{self.diagnostics.ap_macro:.4f}; at threshold "
            f"{self.diagnostics.threshold:g} precision {self.diagnostics.precision_macro:.4f}, "
            f"recall {self.diagnostics.recall_macro:.4f}, F1 {self.diagnostics.f1_macro:.4f}",
            f"  wall clock {self.wall_seconds:.1f} s TOTAL "
            f"(inner steps {self.step_seconds:.1f} s = "
            f"{100.0 / max(self.partial_timing_ratio, 1e-9):.0f}%); timing only the step "
            f"underestimated a fold by 1.7x three times on this project, so the total is "
            f"the number to quote",
            "  " + self.inference.timing_report().replace("\n", "\n  "),
        ]
        if self.checkpoint is not None:
            lines.append(f"  checkpoint {self.checkpoint}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ExperimentResult:
    """All folds of one config at one seed, aggregated with its spread.

    ``summary`` carries the across-fold standard deviation for the macro **and for all
    twelve labels**, because a per-label mean without a spread is unreadable on this
    task: the whole gap between working and top-10 is about 0.006, which is the same
    order as a single fold's draw.
    """

    name: str
    seed: int
    config_hash: str
    git_sha: str
    folds: tuple[FoldResult, ...]
    summary: FoldSummary
    oof: pd.DataFrame
    oof_path: Path | None
    wall_seconds: float

    @property
    def macro(self) -> float:
        return self.summary.macro_mean

    @property
    def seconds_per_study(self) -> float:
        """Mean measured inference rate across folds."""
        return float(np.mean([fold.seconds_per_study for fold in self.folds]))

    def report(self) -> str:
        total_studies = int(sum(fold.n_val for fold in self.folds))
        projected = self.seconds_per_study * total_studies
        lines = [
            f"experiment : {self.name}  config {self.config_hash}  sha {self.git_sha}  "
            f"seed {self.seed}",
            self.summary.table(),
            f"wall clock  : {self.wall_seconds:.1f} s TOTAL for "
            f"{len(self.folds)} fold(s), including data loading, validation and "
            f"checkpointing",
            f"inference   : {self.seconds_per_study:.4f} s/study MEASURED; "
            f"{projected:.1f} s projected for {total_studies} studies = runtime term "
            f"{projected / NOTEBOOK_LIMIT_SECONDS:.4f} against the "
            f"{NOTEBOOK_LIMIT_SECONDS} s notebook limit",
        ]
        if self.oof_path is not None:
            lines.append(f"oof         : {self.oof_path}")
        if not math.isfinite(self.summary.macro_std):
            lines.append(
                "spread      : one fold, so there is no spread. Read nothing into the mean "
                f"until it clears a measured floor -- run --noise-floor "
                f"(external reference floor {NOISE_FLOOR_MACRO:.4f})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class NoiseFloorResult:
    """The same config under several seeds, and the spread that makes a delta readable.

    This is the first measurement any experiment needs and the one most often skipped.
    Until it exists locally, no comparison of two runs can be interpreted: a competitor
    measured 0.0020 macro of run-to-run spread at three folds, and the entire distance
    between a working solution and a top-10 place is about 0.006. Every seed here shares
    one :func:`kappa.config.config_hash`, which is asserted -- two different configs
    would measure config sensitivity and call it noise.
    """

    name: str
    config_hash: str
    git_sha: str
    seeds: tuple[int, ...]
    experiments: tuple[ExperimentResult, ...]

    @property
    def macros(self) -> tuple[float, ...]:
        return tuple(experiment.macro for experiment in self.experiments)

    @property
    def spread(self) -> float:
        """Max minus min. The blunt version, and the one to quote for two seeds."""
        finite = [value for value in self.macros if math.isfinite(value)]
        return float(max(finite) - min(finite)) if len(finite) > 1 else float("nan")

    @property
    def std(self) -> float:
        """Sample standard deviation, ``nan`` for a single seed rather than 0.0."""
        finite = np.asarray(
            [value for value in self.macros if math.isfinite(value)], dtype=np.float64
        )
        return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    def is_readable(self, delta: float) -> bool:
        """Would a macro difference of ``delta`` clear *this measured* floor?

        Falls back to the external reference floor when only one seed was run, and the
        caller should say which one it used -- a locally measured floor and a figure
        carried in from someone else's competition are not the same claim.
        """
        floor = self.spread if math.isfinite(self.spread) else NOISE_FLOOR_MACRO
        return is_evidence(delta, noise_floor=floor)

    def report(self) -> str:
        lines = [
            f"noise floor : {self.name}  config {self.config_hash}  sha {self.git_sha}",
            f"seeds       : {list(self.seeds)}",
        ]
        for seed, macro in zip(self.seeds, self.macros):
            lines.append(f"  seed {seed:<4} macro {macro:.4f}")
        lines.append(
            f"spread      : {self.spread:.4f} (max-min), std {self.std:.4f} over "
            f"{len(self.seeds)} seed(s)"
        )
        lines.append(
            f"reference   : {NOISE_FLOOR_MACRO:.4f} macro, an external figure for this task "
            "and not a measurement from this repository"
        )
        lines.append(
            "read as     : a macro difference below the spread above is the same number. "
            "The gap between working and top-10 on this task is about 0.006."
        )
        return "\n".join(lines)


def _oof_frame(
    prediction: PredictionResult,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    """Assemble the OOF rows: scores, ground truth and the mask they were scored under."""
    data: dict[str, Any] = {
        ID_COLUMN: list(prediction.study_uids),
        "fold": np.full(prediction.n_studies, int(fold), dtype=np.int64),
        "seed": np.full(prediction.n_studies, int(seed), dtype=np.int64),
    }
    for position, name in enumerate(LABELS):
        data[name] = prediction.scores[:, position]
        data[f"{name}{TRUE_SUFFIX}"] = truth[:, position]
        data[f"{name}{MASK_SUFFIX}"] = mask[:, position]
    frame = pd.DataFrame(data, columns=list(OOF_COLUMNS))
    if tuple(frame.columns) != OOF_COLUMNS:
        raise AssertionError(f"OOF columns {tuple(frame.columns)} != {OOF_COLUMNS}")
    return frame


def _truth_for(bundle: DataBundle, study_uids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(values, mask)`` aligned to ``study_uids``, with unlabelled studies masked out.

    An absent study is all-unknown, never all-negative: twelve zeros with a True mask
    would be twelve fabricated negatives and would move every AUC.
    """
    index = {str(uid): position for position, uid in enumerate(bundle.labels[ID_COLUMN])}
    values = np.zeros((len(study_uids), N_LABELS), dtype=np.float64)
    mask = np.zeros((len(study_uids), N_LABELS), dtype=bool)
    label_values = bundle.labels[list(LABELS)].to_numpy(dtype=np.float64, na_value=np.nan)
    mask_columns = [f"{name}{MASK_SUFFIX}" for name in LABELS]
    if all(column in bundle.labels.columns for column in mask_columns):
        label_mask = bundle.labels[mask_columns].to_numpy(dtype=bool)
    else:
        # A table with no mask columns encodes unknown as NaN. Honour it rather than
        # reading NaN as a value, which is how a NaN reaches the metric.
        label_mask = ~np.isnan(label_values)
    for position, uid in enumerate(study_uids):
        row = index.get(str(uid))
        if row is None:
            continue
        values[position] = np.nan_to_num(label_values[row], nan=0.0)
        mask[position] = label_mask[row]
    return values, mask


def _masked_truth(truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Truth with unusable cells set to NaN, the shape every scorer here expects.

    One helper rather than the same ``np.where`` spelled at each call site, because
    the AUC, the diagnostics and the curve panels must agree on exactly which cells
    exist. A scorer reading one cell the others drop is not a different number, it is
    a different denominator, and the macro moves without anything saying so.
    """
    return np.where(mask, truth, np.nan)


def _score(truth: np.ndarray, mask: np.ndarray, scores: np.ndarray) -> AucResult:
    """Macro AUC over the masked cells, with the skip warning turned into a return value.

    The warning is suppressed and the information kept: ``AucResult.skipped`` already
    names every label with one class present, and it is printed on every epoch line, so
    a per-epoch ``RuntimeWarning`` would bury the signal it carries under repetition.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return macro_roc_auc(_masked_truth(truth, mask), scores)


def _diagnostics(truth: np.ndarray, mask: np.ndarray, scores: np.ndarray) -> DiagnosticResult:
    """Average precision, precision/recall/F1 over the same cells :func:`_score` uses.

    **Diagnostic only -- never a selection signal.** Checkpoint selection reads
    ``auc.macro`` and nothing computed here; ``f1_best`` in particular is fitted on
    the very fold it is quoted from, which is the same reason ``select="last"`` is
    the default. These exist so that Fracture at a 7.5% positive rate is visible
    *during* a run rather than after the leaderboard, because ROC-AUC does not read
    the positive rate and average precision does.

    Warnings are suppressed for the same reason as in :func:`_score`: a label with no
    positives in this split is already named in ``DiagnosticResult.skipped``, and
    re-raising that once per epoch buries it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return diagnostic_metrics(_masked_truth(truth, mask), _probabilities(scores))


def _probabilities(scores: np.ndarray) -> np.ndarray:
    """Logits to probabilities, for the threshold-based diagnostics only.

    Why this exists: ``PredictionResult.scores`` are **raw logits** -- the project
    applies no sigmoid anywhere, because macro ROC-AUC is rank-based and a monotone
    transform cannot move it. But precision, recall and F1 are *not* rank-based:
    they need an operating point, and ``FIXED_THRESHOLD = 0.5`` compared against a
    raw logit is the implicit probability cutoff ``sigmoid(0.5) = 0.62``, not 0.5.
    That reports every label at a conservative point nobody chose -- high precision,
    low recall -- and the numbers look plausible enough not to question.

    So the diagnostics are computed on probabilities, which also puts
    ``threshold_best`` in the space a reader can act on. The macro AUC is
    deliberately **not** routed through here: it is computed on the logits and is
    identical either way.

    Clipped before ``exp`` so a saturated logit cannot raise an overflow warning
    on the way to a value that was going to round to 0.0 or 1.0 regardless.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(scores, dtype=np.float64), -60.0, 60.0)))


def predict_teacher_studies(
    model: TeacherClassifier,
    dataset: StudyDataset,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    amp: bool | str | None = None,
) -> PredictionResult:
    """Score studies with the teacher, which :func:`predict_studies` cannot do.

    ``predict_studies`` takes ``(pixels, present, slot_ids)`` because that is the
    entire inference contract, and it stays that way -- widening it to accept report
    text would put the privileged input on the path that ships. So the teacher gets
    its own pass here, in the training package, where report text is legal.

    No TTA. The teacher exists to produce soft targets, not to be submitted, and a
    multi-view teacher score is not comparable to the student's single-view one.
    """
    resolved = resolve_device(device)
    pin = pin_memory if pin_memory is not None else resolved.type == "cuda"
    if int(getattr(dataset, "report_dim", 0)) == 0:
        raise ValueError(
            "the teacher was handed a dataset with no report features; it would read "
            "an all-zero report for every study and score them all alike"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=report_collate,
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(amp)) if amp else None
    model.eval()
    uids: list[str] = []
    chunks: list[np.ndarray] = []
    started = time.perf_counter()
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            args = (
                batch["pixels"].to(resolved, non_blocking=pin),
                batch["present"].to(resolved, non_blocking=pin),
                batch["slot_ids"].to(resolved, non_blocking=pin),
                batch["report"].to(resolved, non_blocking=pin),
                batch["report_present"].to(resolved, non_blocking=pin),
            )
            if dtype is None:
                logits = model(*args).logits
            else:
                with torch.autocast(device_type=resolved.type, dtype=dtype):
                    logits = model(*args).logits
            chunks.append(logits.float().cpu().numpy())
            uids.extend(batch["study_uid"])
            n_batches += 1
    return PredictionResult(
        study_uids=tuple(uids),
        scores=np.concatenate(chunks, axis=0) if chunks else np.zeros((0, N_LABELS), np.float32),
        seconds=time.perf_counter() - started,
        n_batches=n_batches,
        device=str(resolved),
        amp=str(amp) if amp else "off",
    )


def train_fold(
    cfg: TrainConfig,
    bundle: DataBundle,
    fold: int,
    *,
    seed: int | None = None,
    device: str | torch.device | None = None,
    out_dir: Path | str | None = None,
    fold_column: str | None = None,
    select: Literal["last", "best"] = "last",
    save_checkpoint: bool = True,
    normalisation: Normalisation | None = None,
    run: Any = None,
    verbose: bool = True,
    grad_checkpoint: bool = False,
    resume: bool = False,
) -> FoldResult:
    """Train one fold and return everything needed to interpret it.

    The clock starts on the first line and stops on the last, so ``wall_seconds``
    includes dataset construction, every dataloader stall, validation on every epoch
    and the checkpoint write. ``step_seconds`` is reported beside it; the ratio between
    them is in :attr:`FoldResult.partial_timing_ratio`.

    ``normalisation`` defaults to the backbone's own resolved data config, read from the
    model rather than written down anywhere. Passing
    ``Normalisation.identity()`` is correct for a wiring run and wrong for a reported
    score, which is why it has to be passed explicitly to happen.

    ``grad_checkpoint`` trades backbone compute for activation memory. It is an
    argument and not a config field, deliberately: ``_HASH_EXCLUDED`` is only
    name/out_dir/seed/seeds, so a ``TrainConfig`` field would move every hash and
    rename the experiment it is meant to rescue. The same reasoning already governs
    ``--cpus-per-task`` and ``--select``. Unlike ``--select`` this one is a
    RESULT-NEUTRAL knob -- ``torch.utils.checkpoint`` recomputes the same activations
    and defaults to ``preserve_rng_state=True``, so dropout and drop-path draw the same
    numbers -- which is why hashing identically across it is correct here and is a
    hazard there. ``tests/test_loop.py`` pins both halves of that claim.

    ``resume`` is **off by default and that default is load-bearing**, not a taste.
    A job queued before the code changed under it must still train exactly the fold
    it was submitted for, and a resume adds file reads, file writes and a restored
    RNG stream, any of which changes a run that did not ask for one. So every line
    of the resume path sits behind this flag -- with it false, no
    :class:`~kappa.training.resume.ResumeIdentity` is constructed, no sidecar is read,
    no checkpoint is written and no RNG stream is touched, so the fold is
    bit-identical to the one this function trained before the flag existed.
    ``tests/test_resume.py`` asserts the no-file half of that directly.

    Like ``--select`` and ``grad_checkpoint`` this is an argument and not a config
    field: ``_HASH_EXCLUDED`` is only name/out_dir/seed/seeds, so a ``TrainConfig``
    field would move every existing ``config_hash`` -- including the one the measured
    0.7280 baseline is filed under -- and rename every experiment it is meant to
    rescue. See :mod:`kappa.training.resume`, rule 1.
    """
    started = time.perf_counter()
    resolved_seed = cfg.seed if seed is None else int(seed)
    # The fold column defaults from the config, not from a literal here: which
    # grouping a run used is part of what the number means, and a function default
    # that disagreed with the config would make the log wrong rather than absent.
    column = cfg.folds.column if fold_column is None else str(fold_column)
    device_t = resolve_device(device)
    digest = config_hash(cfg)
    sha = git_sha()
    out_root = Path(out_dir) if out_dir is not None else Path(cfg.out_dir) / cfg.name

    if verbose:
        print(f"\n=== fold {fold}  seed {resolved_seed}  ({column}) ===")
        print(describe(cfg))
        print(f"git sha    : {sha}")
        print(f"seed       : {resolved_seed}")
        print(f"device     : {device_t}")

    seed_everything(resolved_seed)
    dtype, scaler, amp_label = _amp_setup(cfg, device_t)
    if verbose:
        print(f"amp        : {amp_label}")
        print(f"out_dir    : {out_root}")

    train_uids, val_uids = bundle.split(fold, column=column)
    if not train_uids or not val_uids:
        # The usual cause is a fold table where every study landed in one fold, which
        # otherwise surfaces as "num_samples=0" out of the DataLoader and looks like a
        # torch problem rather than a fold-table problem.
        raise ValueError(
            f"fold {fold} of {column!r} has {len(train_uids)} train and {len(val_uids)} "
            "validation studies; both sides need at least one. Check the fold table."
        )
    slots = slot_indices(cfg.data.slots)
    crop = CropSpec(fov_mm=cfg.data.physical_crop_mm, out_px=cfg.data.size)

    # The three runs of the research question, chosen here and nowhere else.
    is_teacher = cfg.distill.role == "teacher"
    reports = report_frame(cfg)
    report_dim = 0 if reports is None else len(REPORT_FEATURE_NAMES)

    model = build_teacher(cfg, report_dim) if is_teacher else build_model(cfg)
    if grad_checkpoint:
        # A 24-slice config runs out of memory on an H200 (139.81 GiB) in every fold,
        # dying in the ConvNeXt shortcut add. The batch is (B, 6, n_slices, 3, H, W),
        # so one step pushes B*6*n_slices = 8*6*24 = 1152 images at 384 px through
        # the backbone, and activations rather than weights exhaust the card. With one
        # GPU per job there is no second card to spill onto.
        model.set_grad_checkpointing(True)
        if verbose:
            print("grad ckpt  : ON -- backbone recomputes activations in backward")
    norm = (
        normalisation
        if normalisation is not None
        else Normalisation.from_data_config(model.data_config)
    )
    model.to(device_t)

    # Frozen, eval mode, gradients off. Its logits are soft targets, not a second
    # thing being trained; see :func:`load_frozen_teacher`.
    teacher_path = (
        resolve_teacher_ckpt(cfg.distill.teacher_ckpt, fold=fold, seed=resolved_seed)
        if cfg.distill.distilling
        else None
    )
    teacher = (
        load_frozen_teacher(teacher_path, fold=fold, device=device_t)
        if teacher_path is not None
        else None
    )
    distilling = teacher is not None
    # Read once, here, so the ablation cannot be true while `distilling` is false --
    # that combination is an arm that hashes as the ablation and trains as the
    # control. `DistillConfig` refuses it too; this is the second lock.
    ablate_teacher_report = distilling and cfg.distill.teacher_report_ablate
    if verbose and distilling:
        print(f"teacher    : {teacher_path} (frozen, eval)")
        if ablate_teacher_report:
            print(
                "residue    : ABLATED -- the teacher's report branch is gated off for "
                "every study, so its soft targets carry image evidence only"
            )

    common = {
        "normalisation": norm,
        "labels": bundle.labels,
        "slots": slots,
        "n_slices": cfg.data.n_slices,
        "crop": crop,
        "report_features": reports,
        # Train and validation share ONE mode, from the config, because the two must
        # sample the same anatomy: a model trained on centred blocks and scored on
        # full-extent spreads reports a number about a distribution it never saw, and
        # nothing raises. This is why the mode lives in `common` and not in either
        # call site.
        "slice_mode": cfg.volume.mode,
    }
    train_set = build_study_dataset(
        bundle.manifest,
        bundle.cache_dir,
        study_uids=train_uids,
        training=True,
        series_dropout=cfg.data.series_dropout,
        seed=resolved_seed,
        augment=build_augment(cfg),
        **common,
    )
    # No augment and no series dropout, for the same reason: the validation number
    # has to describe the pixels the model is judged on. `build_report_collate(training=
    # False)` returns bare `collate_studies`, so CutMix cannot reach this side
    # either -- it would mix the very labels the score is computed against.
    val_set = build_study_dataset(
        bundle.manifest,
        bundle.cache_dir,
        study_uids=val_uids,
        training=False,
        series_dropout=0.0,
        seed=resolved_seed,
        **common,
    )

    pin = cfg.data.pin_memory and device_t.type == "cuda"
    # NAMED rather than built inline in the DataLoader call, which is what it was
    # until the resume path existed. *Symptom of leaving it inline:* a resumed fold
    # draws the same shuffle permutation it drew for epoch 0, so the epochs after a
    # resume see the data in an order the uninterrupted run never used -- and nothing
    # raises, because a permutation is a permutation. *Cause:* this generator is the
    # ONLY stateful RNG in the loop (`RandomSampler` advances it once per epoch), and
    # a throwaway local cannot have `.get_state()` read at the epoch boundary. *Fix:*
    # one named object, captured by `save_resume` and put back by `set_state` below.
    # Constructing it here rather than inline consumes nothing from any other stream
    # -- `torch.Generator()` and `Generator.manual_seed` do not draw from the global
    # RNG -- so this hoist alone leaves an unresumed run bit-identical.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(resolved_seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=pin,
        collate_fn=build_report_collate(cfg, seed=resolved_seed, training=True),
        drop_last=False,
        generator=loader_generator,
    )

    steps_per_epoch = max(math.ceil(len(train_loader) / cfg.optim.grad_accum), 1)
    total_steps = steps_per_epoch * cfg.optim.epochs
    criterion = build_criterion(cfg)
    if distilling:
        # Wraps, never replaces, the supervised loss: the labelled cells are scored
        # by exactly the same ASL the baseline used, so a student's gain cannot come
        # from having quietly changed the supervised objective too.
        criterion = DistillationLoss(
            criterion,
            temperature=cfg.distill.temperature,
            lambda_supervised=cfg.distill.lambda_supervised,
            lambda_distill=cfg.distill.lambda_distill,
            distill_unlabelled=cfg.distill.distill_unlabelled,
        )
    optimiser = build_optimiser(cfg, model)
    scheduler = build_scheduler(cfg, optimiser, total_steps)
    # None unless the config asks, so an unaveraged run takes the code path it took
    # before this existed. The averaged weights are scored and saved ALONGSIDE the
    # raw ones -- the arm's whole question is the comparison.
    averager = build_averager(cfg, model)

    # -----------------------------------------------------------------------
    # RESUME. Everything from here to `truth, mask = ...` is dead code unless
    # `resume=True`, and the guard is the first line so it reads that way.
    # -----------------------------------------------------------------------
    # Without this, a fold killed at the walltime limit leaves no artifact, so a
    # scheduler that derives outstanding work from artifacts on disk submits it
    # again and it dies the same way. Every piece of training state used to live in
    # locals of this function, and the only file written was the inference
    # checkpoint, which deliberately holds no optimiser or scheduler state.
    # `kappa.training.resume` is read here and written at each epoch boundary below.
    start_epoch = 0
    restored: Any = None
    identity: ResumeIdentity | None = None
    if resume:
        identity = ResumeIdentity(
            config_hash=digest,
            git_sha=sha,
            # `select` is in the identity BECAUSE it is not in `config_hash` -- it is
            # a CLI argument that never reaches TrainConfig, so two runs differing
            # only in selection policy hash identically and nothing downstream can
            # tell them apart. Without it here a resume would continue a `last`-policy
            # run under `best` and produce a policy-mixed fold that no artifact can
            # tell apart from a clean one.
            select=str(select),
            epochs=int(cfg.optim.epochs),
            seed=resolved_seed,
            fold=int(fold),
            name=cfg.name,
        )
        # The CHEAP read first. `read_sidecar` is a few hundred bytes of JSON and
        # returns None when there is nothing there, which keeps the ~700 MB
        # `torch.load` off the path a fresh run takes -- and keeps a missing file
        # from arriving as a FileNotFoundError that reads like a bug.
        if read_sidecar(out_root, fold, resolved_seed) is None:
            if verbose:
                print(f"resume     : ON, no checkpoint at {out_root}; starting at epoch 0")
        else:
            try:
                restored = load_resume(out_root, expect=identity)
            except (ResumeMismatch, OSError) as exc:
                # Non-fatal, and loud. Raising would abort a job that a scheduler
                # then resubmits, which loops; a bare `except` would let the fold
                # silently continue another experiment's schedule. A mismatch is
                # normal -- a commit landed, `--select` changed, the epoch budget
                # moved -- and a sidecar can also outlive its `.pt` after a scratch
                # purge, which is the OSError. So say what disagreed, then train the
                # fold from epoch 0, which is what would have happened without the flag. Only
                # these two are caught: the atomic `.tmp` + `os.replace` publish means
                # a torn `.pt` cannot exist, so a RuntimeError out of `torch.load`
                # here is a real torch failure and must not be dressed up as "started
                # fresh".
                print(f"resume     : REFUSED -- {exc}")
                print("resume     : training this fold from epoch 0 instead")
                restored = None
    if restored is not None:
        # The order below is mandated by `load_resume`'s docstring and is the whole
        # trap. `build_optimiser` then `build_scheduler` (both already run, above)
        # then `optimiser.load_state_dict` then `scheduler.load_state_dict`.
        # *Symptom of swapping the middle two:* `build_scheduler` raises, naming the
        # param groups that already carry `initial_lr` -- which is the GOOD outcome it
        # was written for. *Symptom of the bad ordering a future refactor could still
        # reach:* a scheduler built on annealed rates peaks wherever the old schedule
        # had got to, so the run is silently under-trained with a completely
        # correct-looking config, and on a differential-LR run the backbone/head ratio
        # is silently different too.
        model.load_state_dict(restored.model)
        optimiser.load_state_dict(restored.optimiser)
        scheduler.load_state_dict(restored.scheduler)
        if scaler is not None and restored.scaler is not None:
            scaler.load_state_dict(restored.scaler)
        elif (scaler is None) != (restored.scaler is None):
            # Not an error, and not silent either. `optim.amp_dtype` is inside
            # config_hash so the two halves cannot disagree about fp16-vs-bf16; what
            # CAN differ is the DEVICE, which is not hashed -- `_amp_setup` returns
            # scaler=None on CPU whatever the config says. So this is a cuda/fp16 run
            # being continued on a CPU or the reverse, which is a real thing to do for
            # a wiring check and a strange thing to do for a result.
            print(
                "resume     : NOTE the GradScaler is present on one side only "
                f"(this run {'has' if scaler is not None else 'has no'} scaler, the "
                f"checkpoint {'has' if restored.scaler is not None else 'has no'} "
                "state). amp_dtype is hashed, so this is a device change, not a "
                "config change. Loss scaling restarts at its default."
            )
        loader_generator.set_state(restored.loader_generator)
        # LAST of the restores, because everything above it -- model construction,
        # dataset and collate setup -- draws from these streams on the way here. Put
        # back at this point they hold exactly what the uninterrupted run held when it
        # finished epoch `next_epoch - 1`, which is the only point at which they are
        # comparable.
        restore_rng(restored.rng)
        # An EMA restarted empty is an expensive copy of the mid-run weights, which is
        # not the quantity an EMA run is meant to measure. `install_averager` refuses a
        # payload of the wrong kind rather than installing half of it.
        install_averager(averager, restored.averager)
        start_epoch = int(restored.next_epoch)
        if verbose:
            print(
                f"resume     : RESUMED from {checkpoint_of(out_root, fold, resolved_seed)} "
                f"at epoch {start_epoch} of {cfg.optim.epochs} "
                f"({len(restored.epochs)} epoch(s) already trained, "
                f"{restored.global_step} optimiser steps)"
            )
            if restored.wandb_run_id:
                # Recorded, not reattached. Reattaching means passing an id into
                # `wandb.init`, which happens in `run_experiment` through
                # `kappa.tracking.start_training_run` -- and that function has no id
                # parameter today. So a resumed fold currently appears as a SECOND
                # W&B run, and this line is what lets a reader join the two by hand
                # instead of wondering which is the fold.
                print(
                    f"resume     : the interrupted half logged to W&B run "
                    f"{restored.wandb_run_id}; this half starts a new run"
                )

    truth, mask = _truth_for(bundle, val_uids)
    if verbose:
        print(
            f"data       : {len(train_set)} train / {len(val_set)} val studies, "
            f"{len(train_loader)} batches x {cfg.optim.grad_accum} accum = "
            f"{steps_per_epoch} steps/epoch, {total_steps} total"
        )
        print(f"usable     : {int(mask.sum())} of {mask.size} validation label cells")
        # The sampler is printed even when it is the default, because the whole point
        # of the volumetric arm is that two runs can differ ONLY here -- and
        # `volume.mode` is a section that is dropped from config_hash when it is at
        # its default, so the hash alone cannot always tell a reader which one ran.
        span_mm = (cfg.data.n_slices - 1) * _MEDIAN_SLICE_GAP_MM
        coverage = (
            f"~{span_mm:.0f} mm of the ~{_MEDIAN_THROUGH_PLANE_MM:.0f} mm joint at the "
            f"corpus median {_MEDIAN_SLICE_GAP_MM} mm gap"
            if cfg.volume.mode == "contiguous"
            else "full geometric extent"
        )
        print(
            f"sampling   : {cfg.volume.mode} over {cfg.data.n_slices} slices -> {coverage}"
        )
        # The rates the loss's asymmetry is an assumption about. Named, not summarised:
        # a macro mean would hide exactly the columns where the assumption inverts.
        rates = positive_rates(bundle, train_uids)
        print(
            "train pos% : "
            + "  ".join(f"{name} {100 * rate:.1f}" for name, rate in rates.items())
        )
        if isinstance(criterion, AsymmetricLoss):
            inverted = [name for name, rate in rates.items() if rate > 0.5]
            if inverted and criterion.gamma_neg_by_label is None:
                print(
                    f"NOTE       : {inverted} are majority-positive, so a scalar "
                    f"gamma_neg={criterion.gamma_neg} focuses down their RARER class. "
                    "loss.gamma_neg_by_label sets these per label."
                )

    def _forward_logits(px: Tensor, pr: Tensor, sid: Tensor, rep, rp) -> Tensor:
        """Student takes pixels only; teacher additionally takes the report."""
        if is_teacher:
            return model(px, pr, sid, rep, rp).logits
        return model(px, pr, sid).logits

    def _soft_targets(px: Tensor, pr: Tensor, sid: Tensor, rep, rp) -> Tensor | None:
        if teacher is None:
            return None
        if ablate_teacher_report and rp is not None:
            # The residue ablation. Gate the presence FLAG, not the feature tensor:
            # TeacherClassifier multiplies its report EMBEDDING by this flag, so a
            # zeroed flag removes the branch's contribution exactly as it already
            # does for a study whose extraction failed. Zeroing `rep` instead would
            # still push a genuine all-zero vector through the report MLP and pick
            # up its biases, which is a different and smaller ablation wearing this
            # one's name.
            rp = torch.zeros_like(rp)
        with torch.no_grad():
            return teacher(px, pr, sid, rep, rp).logits.detach()

    def _loss_of(logits: Tensor, y: Tensor, m: Tensor, soft) -> tuple[Tensor, dict[str, float]]:
        if distilling:
            parts = criterion(logits, y, m, soft)
            return parts.total, parts.as_log_dict()
        return criterion(logits, y, m), {}

    def _save_weights(path: Path, meta: dict[str, Any]) -> Path:
        """Write this run's weights. A teacher is a different kind of file.

        A teacher reads report text and can never be submitted, so it must not be
        loadable by the image-only path. :func:`save_teacher_checkpoint` marks it
        and records the feature vocabulary its report branch expects.
        """
        if is_teacher:
            return save_teacher_checkpoint(path, model, meta=meta)
        return save_inference_checkpoint(
            path,
            model,
            normalisation=norm,
            crop=crop,
            slots=slots,
            n_slices=cfg.data.n_slices,
            meta=meta,
        )

    def _predict_val() -> PredictionResult:
        """Score the validation set with whichever model this run trains.

        A teacher cannot go through :func:`predict_studies` -- that function's
        signature *is* the inference contract, and widening it would put report
        text on the path that ships.
        """
        if is_teacher:
            return predict_teacher_studies(
                model,
                val_set,
                device=device_t,
                batch_size=cfg.optim.batch_size,
                num_workers=cfg.data.num_workers,
                pin_memory=pin,
                amp=cfg.optim.amp_dtype if dtype is not None else False,
            )
        return predict_studies(
            model,
            val_set,
            device=device_t,
            batch_size=cfg.optim.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=pin,
            amp=cfg.optim.amp_dtype if dtype is not None else False,
            # Validation runs the SAME augmentation the notebook will. Scoring
            # without it would quote a CV for a pipeline nobody ships.
            tta=cfg.eval.tta,
        )

    epochs: list[EpochRecord] = []
    step_total = 0.0
    best: tuple[float, int, PredictionResult] | None = None
    last: tuple[int, PredictionResult] | None = None
    # CUMULATIVE over the whole fold, and deliberately NOT the loop's `n_steps`,
    # which is re-zeroed at the top of every epoch below. *Symptom of merging them:*
    # the schedule guard three lines into the step block -- `if n_steps <
    # total_steps: scheduler.step()`, where `total_steps = steps_per_epoch * epochs`
    # -- is dead code today for every budget >= 2 epochs, and a cumulative `n_steps`
    # would ACTIVATE it, so the scheduler would stop stepping near the end of
    # training and the run would anneal to the wrong place while looking entirely
    # healthy. *Fix:* leave `n_steps` alone and carry a second counter. See
    # `save_resume`'s docstring, which says the same thing from the other side.
    global_step = 0

    if restored is not None:
        # The two-line reconstruction `ResumeState.epochs` documents: EpochRecord
        # lives here and `resume.py` must not import this module, so the rows come
        # back as plain dicts. `skipped` is re-tupled because the safe torch loader
        # hands sequences back as it finds them and EpochRecord is frozen with a
        # tuple annotation -- a list there would compare unequal to a fresh run's
        # record for no reason a reader could see.
        epochs = [
            EpochRecord(**{**row, "skipped": tuple(row["skipped"])}) for row in restored.epochs
        ]
        step_total = float(restored.step_total)
        best = restored.best
        last = restored.last
        global_step = int(restored.global_step)

    for epoch in range(start_epoch, cfg.optim.epochs):
        train_set.set_epoch(epoch)
        model.train()
        epoch_started = time.perf_counter()
        loss_sum, n_batches, n_steps, step_seconds = 0.0, 0, 0, 0.0
        distill_sums: dict[str, float] = {}
        optimiser.zero_grad(set_to_none=True)

        for index, batch in enumerate(train_loader):
            step_started = time.perf_counter()
            pixels = batch["pixels"].to(device_t, non_blocking=pin)
            present = batch["present"].to(device_t, non_blocking=pin)
            slot_ids = batch["slot_ids"].to(device_t, non_blocking=pin)
            labels = batch["labels"].to(device_t, non_blocking=pin)
            cell_mask = batch["mask"].to(device_t, non_blocking=pin)

            report = batch.get("report")
            report_present = batch.get("report_present")
            if report is not None:
                report = report.to(device_t, non_blocking=pin)
                report_present = report_present.to(device_t, non_blocking=pin)
                if is_teacher and cfg.distill.report_drop_rate > 0.0:
                    # Withhold the whole channel for some studies. Reuses the
                    # teacher's own presence gating, so a withheld report is
                    # indistinguishable from a study that never had one.
                    keep = torch.rand(report_present.shape, device=report_present.device)
                    report_present = report_present & (keep >= cfg.distill.report_drop_rate)

            if dtype is None:
                logits = _forward_logits(pixels, present, slot_ids, report, report_present)
                soft = _soft_targets(pixels, present, slot_ids, report, report_present)
                loss, loss_parts = _loss_of(logits.float(), labels, cell_mask, soft)
            else:
                with torch.autocast(device_type=device_t.type, dtype=dtype):
                    logits = _forward_logits(pixels, present, slot_ids, report, report_present)
                    soft = _soft_targets(pixels, present, slot_ids, report, report_present)
                # The loss runs in fp32 whatever the forward pass ran in. ASL takes a
                # log of a clipped probability, and in fp16 a confident negative
                # underflows that clip and the term vanishes -- silently changing the
                # loss the config says is being used. The KD term takes the same
                # log of a saturating probability and has the same exposure.
                loss, loss_parts = _loss_of(
                    logits.float(), labels, cell_mask, None if soft is None else soft.float()
                )
            for _key, _value in loss_parts.items():
                distill_sums[_key] = distill_sums.get(_key, 0.0) + _value

            scaled = loss / cfg.optim.grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            is_last = index + 1 == len(train_loader)
            if (index + 1) % cfg.optim.grad_accum == 0 or is_last:
                if scaler is not None:
                    scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                if averager is not None:
                    averager.after_step(model)
                # Guard the schedule rather than trusting the arithmetic: OneCycleLR
                # raises once total_steps is exceeded, and a final partial accumulation
                # group is exactly how the count drifts by one.
                if n_steps < total_steps:
                    scheduler.step()
                n_steps += 1
                global_step += 1
                if run is not None:
                    # Logged per optimiser step, not per epoch: the epoch budget is
                    # ten, and ten points cannot show a diverging run or a bad
                    # warmup. Read the LR before the scheduler moves again so the
                    # point matches the step that used it.
                    run.log(
                        step_metrics(
                            fold=fold,
                            step=n_steps,
                            loss=float(loss.detach()),
                            lr=float(optimiser.param_groups[0]["lr"]),
                            parts=loss_parts,
                        )
                    )
            loss_sum += float(loss.detach())
            n_batches += 1
            step_seconds += time.perf_counter() - step_started

        train_seconds = time.perf_counter() - epoch_started
        step_total += step_seconds

        val_started = time.perf_counter()
        prediction = _predict_val()
        aligned = _align(prediction, val_uids)
        auc = _score(truth[aligned], mask[aligned], prediction.scores)
        val_seconds = time.perf_counter() - val_started

        record = EpochRecord(
            epoch=epoch,
            train_loss=loss_sum / max(n_batches, 1),
            n_batches=n_batches,
            n_optimiser_steps=n_steps,
            lr_last=float(optimiser.param_groups[0]["lr"]),
            macro=auc.macro,
            n_scored=auc.n_scored,
            skipped=auc.skipped,
            train_seconds=train_seconds,
            val_seconds=val_seconds,
            step_seconds=step_seconds,
        )
        epochs.append(record)
        if verbose:
            print(record.line())
        if run is not None:
            # The key convention lives in kappa.tracking, not here. A call site that
            # spells its own metric names is how two folds end up under two
            # namespaces and stop being comparable in one chart.
            # Computed here rather than beside `auc` so that it stays outside the
            # `val_seconds` measurement above -- a timing number that silently
            # includes the logger is how a fold gets misattributed.
            run.log(
                {
                    **epoch_metrics(
                        auc,
                        fold=fold,
                        epoch=epoch,
                        train_loss=record.train_loss,
                        lr=record.lr_last,
                        train_seconds=record.train_seconds,
                        val_seconds=record.val_seconds,
                        step_seconds=record.step_seconds,
                    ),
                    **diagnostic_label_metrics(
                        _diagnostics(truth[aligned], mask[aligned], prediction.scores),
                        prefix=fold_prefix(fold),
                    ),
                    # The KD split, averaged over the epoch's batches. Empty for an
                    # image-only run. Logged because a distillation run whose two
                    # terms are not separable is unreadable: a total that falls while
                    # the supervised term rises means the student is fitting the
                    # teacher and losing the labels, and that is invisible in the sum.
                    **{
                        f"{fold_prefix(fold)}/{key}": value / max(n_batches, 1)
                        for key, value in distill_sums.items()
                    },
                }
            )

        if averager is not None:
            averager.after_epoch(model)

        last = (epoch, prediction)
        if best is None or (math.isfinite(auc.macro) and auc.macro > best[0]):
            best = (auc.macro, epoch, prediction)

        if identity is not None:
            # At the epoch boundary and nowhere else, after `best`/`last` have moved,
            # so what is on disk is a whole number of epochs and never a half one.
            # Saved mid-epoch, the optimiser moments would belong to step 231 while
            # the dataloader permutation belongs to the top of the epoch, and the
            # resumed fold would train a schedule no uninterrupted run would have
            # followed; only the boundary is a state the two halves can agree on.
            # One ~700 MB write per epoch costs a few seconds against a measured
            # 47.4 min per fold for the cheapest config, far less than retraining.
            #
            # `next_epoch = epoch + 1` INCLUDING after the final epoch, deliberately:
            # a fold killed during the post-loop work -- the selection pass, the
            # averaged validation pass, the 238 MB checkpoint write -- then resumes
            # with zero epochs left to train and finishes from the serialised
            # `best`/`last` instead of retraining the whole budget.
            save_resume(
                out_root,
                identity=identity,
                next_epoch=epoch + 1,
                global_step=global_step,
                step_total=step_total,
                model=model,
                optimiser=optimiser,
                scheduler=scheduler,
                scaler=scaler,
                loader_generator=loader_generator,
                epochs=epochs,
                best=best,
                last=last,
                averager=averager,
                # A NoOpRun carries a plausible `id` ("untracked") and an honest
                # `offline = True`, and neither distinguishes it from a real offline
                # run -- the same trap `require_run_directory` documents. Writing
                # that string into the sidecar would hand a scheduler a run id that
                # does not exist, so it is normalised to null here.
                wandb_run_id=(
                    None
                    if run is None or isinstance(run, NoOpRun)
                    else getattr(run, "id", None)
                ),
                device=device_t,
            )

    if last is None:  # pragma: no cover - epochs >= 1 is enforced by the config
        raise RuntimeError("no epoch ran; cfg.optim.epochs must be at least 1")
    if select == "last":
        selected_epoch, selected = last
    else:
        selected_epoch, selected = best[1], best[2]  # type: ignore[index]
    aligned = _align(selected, val_uids)
    auc = _score(truth[aligned], mask[aligned], selected.scores)
    diagnostics = _diagnostics(truth[aligned], mask[aligned], selected.scores)
    oof = _oof_frame(selected, truth[aligned], mask[aligned], fold=fold, seed=resolved_seed)
    if run is not None:
        # Curves for the selected epoch only. One panel per fold rather than per
        # epoch: the panel is an image, and ten of them per fold turns the run page
        # into a flipbook nobody reads. Both helpers return None without wandb
        # installed, and logging a None is the one way a missing logger breaks a run.
        y_true_selected = _masked_truth(truth[aligned], mask[aligned])
        panels = {
            f"{fold_prefix(fold)}/roc_curves": roc_curve_panel(
                y_true_selected, selected.scores
            ),
            f"{fold_prefix(fold)}/pr_curves": pr_curve_panel(
                y_true_selected, selected.scores
            ),
        }
        drawn = {key: value for key, value in panels.items() if value is not None}
        if drawn:
            run.log(drawn)

    checkpoint: Path | None = None
    if save_checkpoint:
        checkpoint = _save_weights(
            out_root / f"fold{fold}_seed{resolved_seed}.pt",
            {
                "backbone": cfg.model.backbone,
                # Recorded so inference rebuilds the SAMPLER and the depth, not just the
                # weights. `--select` taught this lesson once already: a policy that is
                # not in the artifact is a policy nothing downstream can tell apart.
                "slice_mode": cfg.volume.mode,
                "volumetric": cfg.model.backbone.split(".", 1)[0] in VIDEO_BACKBONES,
                "n_slices": cfg.data.n_slices,
                "drop_path_rate": cfg.model.drop_path_rate,
                "aux_dim": cfg.model.aux_dim,
                # The READOUT, recorded for the same reason the sampler is: inference
                # rebuilds the architecture from this dict and nothing else, so a
                # shape that is not written down here is a shape the notebook cannot
                # reconstruct. `head` was missing until `pool` was added, which is why
                # `load_checkpoint` defaults every one of these four to the value the
                # architecture had before the key existed.
                "head": cfg.model.head,
                "head_num_heads": cfg.model.head_num_heads,
                "per_label_slots": cfg.pool.per_label_slots,
                "head_residual": cfg.pool.head_residual,
                "role": cfg.distill.role,
                # Which KD arm produced these weights. `config_hash` already separates
                # the arms, but a hash is a lookup and this is a reading: the ablation
                # and the full-KD student are the same architecture trained against
                # different soft targets, so without these two keys the only way to
                # tell one checkpoint from the other is to go back to the config that
                # made it. `--select` is the standing lesson -- a policy that is not in
                # the artifact is a policy nothing downstream can tell apart.
                "distilling": cfg.distill.distilling,
                "teacher_report_ablate": bool(ablate_teacher_report),
                "config_hash": digest,
                "git_sha": sha,
                "seed": resolved_seed,
                "fold": fold,
                "fold_column": column,
                "epoch": selected_epoch,
                "name": cfg.name,
                "macro": auc.macro,
            },
        )

    averaged_auc: AucResult | None = None
    averaged_checkpoint: Path | None = None
    if averager is not None and averager.ready():
        # A second validation pass, not a re-score of the raw one: averaging changes
        # the weights, so the predictions have to be recomputed. This is the only
        # cost of the arm and it is paid once per fold, not per epoch.
        with averager.applied(model):
            averaged_prediction = _predict_val()
            averaged_aligned = _align(averaged_prediction, val_uids)
            averaged_auc = _score(
                truth[averaged_aligned], mask[averaged_aligned], averaged_prediction.scores
            )
            if save_checkpoint:
                # A separate file, so the raw weights stay recoverable. Overwriting
                # them would make the arm unfalsifiable after the fact.
                averaged_checkpoint = _save_weights(
                    out_root / f"fold{fold}_seed{resolved_seed}_{averager.label}.pt",
                    {
                        "backbone": cfg.model.backbone,
                        # Recorded so inference rebuilds the SAMPLER and the depth, not just the
                        # weights. `--select` taught this lesson once already: a policy that is
                        # not in the artifact is a policy nothing downstream can tell apart.
                        "slice_mode": cfg.volume.mode,
                        "volumetric": cfg.model.backbone.split(".", 1)[0] in VIDEO_BACKBONES,
                        "n_slices": cfg.data.n_slices,
                        "drop_path_rate": cfg.model.drop_path_rate,
                        "aux_dim": cfg.model.aux_dim,
                        # Same four as the raw checkpoint above: the averaged weights
                        # are the same architecture and must rebuild the same way.
                        "head": cfg.model.head,
                        "head_num_heads": cfg.model.head_num_heads,
                        "per_label_slots": cfg.pool.per_label_slots,
                        "head_residual": cfg.pool.head_residual,
                        "role": cfg.distill.role,
                        "config_hash": digest,
                        "git_sha": sha,
                        "seed": resolved_seed,
                        "fold": fold,
                        "fold_column": column,
                        "epoch": selected_epoch,
                        "name": cfg.name,
                        "macro": averaged_auc.macro,
                        "weights": averager.label,
                    },
                )
        if verbose:
            print(
                f"{averager.label:<11}: macro {averaged_auc.macro:.4f} "
                f"vs {auc.macro:.4f} raw  ({averager!r})"
            )
        if run is not None:
            run.log({f"{fold_prefix(fold)}/{averager.label}_macro": averaged_auc.macro})

    result = FoldResult(
        fold=fold,
        seed=resolved_seed,
        name=cfg.name,
        config_hash=digest,
        git_sha=sha,
        backbone=cfg.model.backbone,
        device=str(device_t),
        amp=amp_label,
        n_train=len(train_set),
        n_val=len(val_set),
        usable_train_cells=bundle.usable_cells(train_uids),
        usable_val_cells=int(mask.sum()),
        auc=auc,
        diagnostics=diagnostics,
        oof=oof,
        epochs=tuple(epochs),
        selected_epoch=selected_epoch,
        wall_seconds=time.perf_counter() - started,
        step_seconds=step_total,
        inference=selected,
        checkpoint=checkpoint,
        averaged=averaged_auc,
        averaged_checkpoint=averaged_checkpoint,
        averaging=None if averager is None else averager.label,
    )
    if identity is not None:
        # On success only, and after every artifact this fold owes has been written.
        # Left behind, a sidecar beside a finished fold makes a completed run look
        # like an interrupted one to a scheduler, and at roughly 700 MB a file (227
        # MiB of model plus ~454 MiB of AdamW moments) a sweep's worth would fill a
        # scratch filesystem. The resume file is scaffolding; the inference
        # checkpoint and the OOF parquet are the artifacts. So it is deleted here,
        # where "the fold finished" is known.
        #
        # Deliberately NOT gated on `save_checkpoint` or `write_oof`. A --no-oof or
        # --no-checkpoint run still finished this fold, and a resume file left behind
        # by one of those runs would be read by the next run that does write
        # artifacts -- resuming a fold it never started. The noise-floor mode is
        # exactly that shape: both flags off, several seeds through this function.
        clear_resume(out_root, fold, resolved_seed)
    if verbose:
        print(result.report())
        # AFTER the epochs, because the counter only fills as items are read. A
        # non-empty count means some studies contributed fewer DISTINCT planes than
        # data.n_slices asked for: the series is shorter than the block, so indices
        # repeat. That is the documented handling -- a duplicated real slice beats a
        # black one, the rule `neighbour_indices` already follows -- but it silently
        # weakens the deepest configs of a depth sweep, so it is stated as a number
        # rather than left to be inferred from the corpus statistics.
        short = {**train_set.short_series, **val_set.short_series}
        if short:
            shortest = min(short.values())
            worst = 100.0 * shortest / cfg.data.n_slices
            print(
                f"short      : {len(short)} series carry fewer than "
                f"{cfg.data.n_slices} cached slices (min {shortest}, i.e. {worst:.0f}% "
                "distinct planes); indices repeat to fill the depth. This arm sees "
                "fewer distinct planes than it asks for on those studies."
            )
    return result


def _align(prediction: PredictionResult, val_uids: Sequence[str]) -> np.ndarray:
    """Row indices into the truth array matching the order predictions came back in.

    The dataloader iterates the dataset's own study order and ``shuffle=False``, so this
    is almost always the identity -- but "almost always" is not a guarantee, and a
    silent misalignment here scores every study against another study's labels while
    producing a perfectly plausible macro around 0.5.
    """
    position = {str(uid): index for index, uid in enumerate(val_uids)}
    missing = [uid for uid in prediction.study_uids if uid not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} predicted study/studies are not in the validation fold, e.g. "
            f"{missing[:3]}; the truth array cannot be aligned to them"
        )
    return np.asarray([position[uid] for uid in prediction.study_uids], dtype=np.int64)


def run_experiment(
    cfg: TrainConfig,
    bundle: DataBundle,
    *,
    seed: int | None = None,
    folds: Sequence[int] | None = None,
    device: str | torch.device | None = None,
    out_dir: Path | str | None = None,
    fold_column: str | None = None,
    select: Literal["last", "best"] = "last",
    save_checkpoint: bool = True,
    write_oof: bool = True,
    normalisation: Normalisation | None = None,
    track: bool = False,
    job_type: str = "train",
    verbose: bool = True,
    grad_checkpoint: bool = False,
    resume: bool = False,
) -> ExperimentResult:
    """Train every selected fold and aggregate them into a mean and a spread.

    ``folds`` defaults to ``cfg.folds.selected()``, so a Slurm array element that
    passes one fold and a laptop run that passes none share this function. Folds are
    independent by construction -- nothing is carried between them -- which is what
    makes them safe to run as parallel one-GPU jobs. Where a scheduler caps GPUs per
    job, a job over the cap can stay pending forever rather than run slowly.

    ``resume`` is forwarded to every fold and is OFF by default; see
    :func:`train_fold`. It is **intra-fold only** -- it restarts an interrupted fold
    at its last epoch boundary and does not skip a fold that already has artifacts on
    disk. That omission is deliberate and is not a gap to fill here. Filling it would
    produce an experiment reporting a mean over five folds when it retrained one, or a
    crash in the aggregation below: the artifacts a finished fold leaves --
    ``fold<k>_seed<s>.pt`` and the OOF parquet -- carry no ``EpochRecord`` history,
    no ``step_seconds``, no ``usable_*_cells`` and no diagnostics, so a skipped fold
    has no :class:`FoldResult` to put into ``results`` and the OOF concat, the
    ``fold_summary`` and the W&B tables all read one fold short. Artifact existence
    is also not doneness: a run directory records no ``config_hash``, no selection
    policy and no epoch budget, so keying a skip on a filename is how a policy-mixed
    fold gets read as a good one. Deciding which folds still need to run belongs to
    whatever external scheduler submits the jobs.
    """
    started = time.perf_counter()
    resolved_seed = cfg.seed if seed is None else int(seed)
    chosen = tuple(int(f) for f in (folds if folds is not None else cfg.folds.selected()))
    if not chosen:
        raise ValueError("no folds selected; cfg.folds.selected() and folds= are both empty")
    digest = config_hash(cfg)
    sha = git_sha()
    out_root = Path(out_dir) if out_dir is not None else Path(cfg.out_dir) / cfg.name

    column = cfg.folds.column if fold_column is None else str(fold_column)
    run = None
    if track:
        # One run per *job*, started through kappa.tracking so the name, the group and
        # the flattened config entities all follow the one convention. Folds are
        # parallel Slurm jobs, so a job training one fold names it and a job training
        # several passes ``fold=None`` -- rendered as ``all`` -- while the group is
        # keyed on the config hash, which excludes the seed. That grouping is the only
        # thing that puts a seed sweep back together, and a noise floor is unreadable
        # without it.
        # Guarded, because this is the logger's code path. A fold table that cannot
        # be split is train_fold's error to raise, with train_fold's message naming
        # the fold and the column; it must not surface from here as a stack trace
        # inside the tracker. An experiment never fails because of its logger.
        try:
            first_train, first_val = bundle.split(chosen[0], column=column)
            n_train, n_val = len(first_train), len(first_val)
        except Exception:
            n_train, n_val = None, None
        run = start_training_run(
            cfg,
            fold=chosen[0] if len(chosen) == 1 else None,
            seed=resolved_seed,
            job_type=job_type,
            n_train=n_train,
            n_val=n_val,
            extra={
                "folds_selected": ",".join(str(fold) for fold in chosen),
                "fold_column": column,
            },
        )
        # ``--track`` used to be able to fail silently into a NoOpRun --
        # WANDB_API_KEY unset, wandb not installed, or wandb.init raising -- and the
        # job would train for hours, report a macro, and leave nothing on disk to
        # sync, because a compute node without internet cannot sync itself
        # afterwards. require_run_directory turns that into an immediate, named
        # failure BEFORE any GPU time is spent in the fold loop below, instead of
        # sync_wandb.sh reporting success over an empty set hours later.
        #
        # KAPPA_NO_TRACKING=1 is excluded on purpose: kappa.tracking's own docstring
        # names it "what tests should do" to force a NoOpRun regardless of --track,
        # and that deliberate override is a different intent than --track silently
        # not working and must stay usable for a wandb-free smoke test.
        if os.environ.get("KAPPA_NO_TRACKING") != "1":
            require_run_directory(run, tracked=track)
        # Commit this run to disk if Slurm signals the walltime. Without it an
        # 8-hour job that runs out of time writes a 7-byte file and loses every
        # metric it logged; see install_walltime_guard for the measurement.
        install_walltime_guard(run)

    try:
        results = [
            train_fold(
                cfg,
                bundle,
                fold,
                seed=resolved_seed,
                device=device,
                out_dir=out_root,
                fold_column=fold_column,
                select=select,
                save_checkpoint=save_checkpoint,
                normalisation=normalisation,
                run=run,
                verbose=verbose,
                grad_checkpoint=grad_checkpoint,
                resume=resume,
            )
            for fold in chosen
        ]
    except BaseException:
        # An exception here (CUDA OOM, a data bug, Ctrl-C on the login node) is the
        # SAME failure as a walltime kill from the run's point of view: wandb's
        # offline transaction log is block-buffered at 32 KiB with no time-based
        # flush, so anything logged so far sits in a buffer that only run.finish()
        # commits. install_walltime_guard only fires on SIGUSR1/SIGTERM; it does not
        # fire here, so without this the process unwinds, the interpreter exits, and
        # the run is a 7-byte stub that still matches sync_wandb.sh's glob and still
        # looks syncable.
        if run is not None:
            summarise(run, {"crashed": True, "finished_cleanly": False})
            finish_run(run)
        raise

    oof = pd.concat([result.oof for result in results], ignore_index=True)
    duplicated = oof[ID_COLUMN].duplicated()
    if bool(duplicated.any()):
        raise ValueError(
            f"{int(duplicated.sum())} study/studies appear in more than one fold's OOF rows, "
            "so the out-of-fold table is not out of fold. Check the fold table's grouping."
        )

    oof_path: Path | None = None
    if write_oof:
        # The folds this invocation actually trained belong in the FILENAME whenever it
        # trained only some of them. Without that, a five-task job array over one config
        # -- which is how the fusion x backbone sweep runs -- has all five tasks writing
        # this one path, each holding only its own fold's rows. They do not merge: the
        # last writer wins and the other four folds' predictions are gone, and because
        # the tasks overlap in time it is also a torn-write race. Nothing raises, the
        # per-fold AUCs still print, and the loss only shows up later as a rank
        # ensemble built from one fold of out-of-fold predictions.
        #
        # A run that trained every fold keeps the original name, so existing artifacts
        # and the whole-experiment path are untouched.
        trained = tuple(sorted(int(fold) for fold in chosen))
        complete = trained == tuple(range(int(cfg.folds.n_folds)))
        suffix = "" if complete else "_fold" + "-".join(str(fold) for fold in trained)
        oof_path = out_root / f"oof_seed{resolved_seed}{suffix}.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(oof_path, index=False)

    summary = fold_summary([result.auc for result in results])
    experiment = ExperimentResult(
        name=cfg.name,
        seed=resolved_seed,
        config_hash=digest,
        git_sha=sha,
        folds=tuple(results),
        summary=summary,
        oof=oof,
        oof_path=oof_path,
        wall_seconds=time.perf_counter() - started,
    )
    if verbose:
        print()
        print(experiment.report())
    if run is not None:
        # Into the summary, not only into the step history: the runs table sorts and
        # filters on the summary, so a macro that exists only as the last point of a
        # series is invisible in the one view used to compare a hundred runs.
        # `wall_seconds` and `step_seconds` go side by side because quoting the inner
        # step underestimated a fold by 1.7x three times on this project.
        summarise(
            run,
            {
                **label_metrics(results[0].auc, prefix="cv", summary=summary),
                # Same source as the line above -- results[0] -- because folds run as
                # separate Slurm jobs and each job summarises the fold it trained.
                **diagnostic_label_metrics(results[0].diagnostics, prefix="cv"),
                "cv/n_folds": len(results),
                "cv/wall_seconds": experiment.wall_seconds,
                "cv/step_seconds": sum(result.step_seconds for result in results),
                "cv/seconds_per_study": experiment.seconds_per_study,
            },
        )
        tables = {
            "results/per_label": per_label_table(results[0].auc, summary=summary),
            # Across-fold means with their spread AND their skip count. Without it, a
            # mean resting on one of five folds renders identically to one resting
            # on all five -- the misreading the table's own docstring exists to
            # prevent.
            "results/per_label_summary": per_label_summary_table(summary),
            "results/per_fold": per_fold_table(
                [
                    {
                        "fold": result.fold,
                        "seed": result.seed,
                        "macro": result.macro,
                        "n_scored": result.auc.n_scored,
                        "n_skipped": len(result.auc.skipped),
                        "skipped": ",".join(result.auc.skipped),
                        "n_train": result.n_train,
                        "n_val": result.n_val,
                        "wall_seconds": result.wall_seconds,
                        "step_seconds": result.step_seconds,
                        "seconds_per_study": result.seconds_per_study,
                        "checkpoint": str(result.checkpoint) if result.checkpoint else "",
                    }
                    for result in results
                ]
            ),
        }
        # per_*_table returns None when wandb is absent, and logging a None would be
        # the one way a missing logger breaks a run.
        live = {key: value for key, value in tables.items() if value is not None}
        if live:
            run.log(live)
        log_standard_artifacts(
            run,
            cfg=cfg,
            oof=oof_path,
            fold_file=cfg.folds.file,
            weights=[r.checkpoint for r in results if r.checkpoint is not None],
        )
        finish_run(run)
    return experiment


def measure_noise_floor(
    cfg: TrainConfig,
    bundle: DataBundle,
    *,
    seeds: Sequence[int] | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> NoiseFloorResult:
    """Run one config under several seeds and report the macro spread.

    A first-class mode, not a manual exercise. Every seed shares one config hash --
    asserted, not assumed -- so the spread measures run-to-run noise rather than the
    difference between two configs. Two seeds give a max-minus-min; three or more give
    a standard deviation worth quoting.

    Checkpoints and OOF files are off by default here: the output of this mode is one
    number, and writing several gigabytes of weights to produce it wastes the disk that
    the real runs need.
    """
    resolved = tuple(int(s) for s in (seeds if seeds is not None else cfg.seed_list()))
    if len(resolved) < 2:
        raise ValueError(
            f"a noise floor needs at least two seeds, got {list(resolved)}. One run has no "
            "spread, and reporting 0.0 there would claim a certainty that does not exist."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"seeds repeat: {list(resolved)}; the same seed twice measures nothing")

    # job_type separates a floor measurement from a real run in the W&B runs table,
    # and checkpoints and OOF files are off because the output of this mode is one
    # number -- writing a set of weights per seed to get it spends the disk the real
    # runs need.
    options: dict[str, Any] = {
        "save_checkpoint": False,
        "write_oof": False,
        "job_type": "noise-floor",
        "verbose": verbose,
    }
    options.update(kwargs)
    digest = config_hash(cfg)
    experiments: list[ExperimentResult] = []
    for seed in resolved:
        experiment = run_experiment(cfg, bundle, seed=seed, **options)
        if experiment.config_hash != digest:
            raise AssertionError(
                f"seed {seed} ran config {experiment.config_hash}, not {digest}. A noise floor "
                "over two different configs measures config sensitivity and calls it noise."
            )
        experiments.append(experiment)

    result = NoiseFloorResult(
        name=cfg.name,
        config_hash=digest,
        git_sha=git_sha(),
        seeds=resolved,
        experiments=tuple(experiments),
    )
    if verbose:
        print()
        print(result.report())
    return result
