"""Typed, validated experiment configuration.

Why this exists: **a config that silently accepts an unknown key tests something
other than what it claims.** A YAML file with ``droppath_rate: 0.3`` in it looks
like it configured stochastic depth; with a plain ``dict`` it configures nothing
at all, the run uses the default 0.2, and the result is filed under the wrong
hypothesis. Nothing in the training log would ever say so. Every model here sets
``extra="forbid"``, so that file fails at load with the offending key named.

    from kappa.config import load_config, config_hash

    cfg = load_config("configs/baseline_axial_fs1.yaml")
    print(cfg.model.backbone, config_hash(cfg))

The second job of this module is the hash. ``config_hash`` is a stable digest of
the resolved configuration with ``name``, ``out_dir``, ``seed`` and ``seeds``
excluded, so it identifies the *experiment* rather than the *run*: two seeds of
one config share a hash, which is what makes a seed-to-seed spread readable as a
noise floor rather than as two different experiments. Log it next to the git SHA
and the seed, and a number in a table can be traced back to the file that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kappa.constants import LABELS

__all__ = [
    "ConfigError",
    "DataConfig",
    "AugmentConfig",
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "FoldsConfig",
    "EvalConfig",
    "DistillConfig",
    "VolumeConfig",
    "PoolConfig",
    "TrainConfig",
    "SLOT_NAMES",
    "TTA_TRANSFORMS",
    "VIDEO_BACKBONES",
    "load_config",
    "loads_config",
    "apply_overrides",
    "config_hash",
    "describe",
]

#: The six plane x contrast slots, named ``<plane>_fs<bit>``, in the order of
#: ``SLOT_ORDER`` from the manifest contract -- a slot's index in this tuple is
#: its ``slot_ids`` value in the batch. Written out rather than derived so a
#: config can be validated without the manifest module on disk; ``tests/
#: test_config.py`` cross-checks it against ``SLOT_ORDER`` when that lands.
SLOT_NAMES = (
    "Axial_fs0",
    "Axial_fs1",
    "Coronal_fs0",
    "Coronal_fs1",
    "Sagittal_fs0",
    "Sagittal_fs1",
)

#: Test-time augmentations that are legal on a knee MRI, and the complete list.
#: Each one must map a study to a *differently rendered version of the same study*
#: without changing which label is true. Rotation and zoom qualify; a horizontal
#: flip does not, and is absent from this tuple deliberately -- see
#: :class:`AugmentConfig` for the anatomy.
TTA_TRANSFORMS = (
    "identity",
    "rotate_cw",
    "rotate_ccw",
    "zoom_in",
    "zoom_out",
)

#: The 3D (video) backbones the volumetric arm may use, and the complete list.
#: Naming one of these in ``model.backbone`` is what switches
#: :class:`kappa.models.StudyClassifier` from the 2.5D :class:`SliceEncoder` onto
#: the volumetric :class:`kappa.models.volume_encoder.VolumeEncoder`. The switch is
#: the backbone name and nothing else, deliberately: a separate ``volume.enabled``
#: flag could disagree with the backbone, and a config whose two halves disagree is
#: the exact failure this module exists to prevent.
#:
#: These are ``torchvision.models.video`` names, NOT timm names -- timm has no 3D
#: backbone. Their pretrained weights come from ``download.pytorch.org`` into
#: ``$TORCH_HOME``, which is a second weight store beside the HF cache and has to be
#: staged on the login node (``slurm/stage_torch_weights.sh``); compute nodes have no
#: outbound internet, so an unstaged weight is a hang, not an error.
VIDEO_BACKBONES = (
    "r3d_18",
    "mc3_18",
    "r2plus1d_18",
)

#: The Kaggle notebook wall-clock limit, in seconds. The efficiency track's score
#: carries a ``T / 32400`` runtime term, so a full-length run contributes 1.0
#: against accuracy gaps two orders of magnitude smaller. Inference timing is
#: reported against this number, never guessed at.
NOTEBOOK_LIMIT_SECONDS = 32_400


class ConfigError(ValueError):
    """A config file that cannot be trusted to mean what it says."""


class _Section(BaseModel):
    """Base for every config section: unknown keys are an error, values are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Section):
    """Which pixels reach the model, and at what physical scale."""

    slots: tuple[str, ...] = ("Axial_fs1",)
    n_slices: int = Field(16, ge=1, le=64)
    stack: int = 3
    size: int = Field(384, ge=64, le=1024)
    physical_crop_mm: float = Field(150.0, gt=0.0, le=400.0)
    cache_dir: Path | None = None
    manifest: Path | None = None
    labels: Path | None = None
    #: Include the 58 anchor studies with their ground truth overriding the
    #: derivation. **Consumed by the step that writes ``labels``, not by the training
    #: loop** -- the loop reads whatever ``derived.parquet`` says and has no access to
    #: the ground-truth table, which lives in read-only quarantine. It is a config
    #: field rather than a flag on the derive script so that the run's hash records
    #: which label table it trained against. ``scripts/derive_labels.py`` currently
    #: only *scores* against the anchors, so this field is declared and not yet
    #: honoured anywhere; see the report accompanying this change.
    anchors_override: bool = True
    series_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    num_workers: int = Field(4, ge=0, le=64)
    pin_memory: bool = True

    @field_validator("slots")
    @classmethod
    def _slots_are_known_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("at least one slot is required")
        unknown = [name for name in value if name not in SLOT_NAMES]
        if unknown:
            raise ValueError(f"unknown slot(s) {unknown}; legal names are {list(SLOT_NAMES)}")
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate slot in {list(value)}")
        return value

    @field_validator("stack")
    @classmethod
    def _stack_is_the_pinned_three(cls, value: int) -> int:
        # The cross-agent batch is (B, N_SLOTS, n_slices, 3, H, W) with the 3
        # pinned. A config asking for 5 would be accepted here and then ignored
        # by the loader, which is precisely the silent mismatch this module is
        # for. Changing it is a contract change, not a config change.
        if value != 3:
            raise ValueError(
                f"stack={value}: the pinned batch fixes the 2.5D neighbour stack at 3. "
                "Changing it requires changing the batch contract in every consumer."
            )
        return value

    @field_validator("size")
    @classmethod
    def _size_is_on_a_grid_some_backbone_can_use(cls, value: int) -> int:
        # Every convnext / maxvit stage downsamples by 2 at least five times, so a
        # size off the 32-grid gets silently rounded inside the backbone and the
        # effective mm/px stops matching what the config says. A patch14 ViT has
        # the same exposure against a different modulus: it tokenises in 14 px
        # patches, so 14 is its grid and 32 is not (392 = 14 x 28 is legal for it
        # and illegal for everything else).
        #
        # WHICH of the two moduli applies is a property of ``model.backbone``, and a
        # field validator on ``data`` cannot see it. So this half only refuses a
        # size that no supported backbone could use at all, and the pairing itself
        # is checked by :meth:`TrainConfig._backbone_and_size_agree`. Loosening this
        # check without that one would let ``convnext_small`` at 392 through.
        if value % 32 and value % 14:
            raise ValueError(
                f"size={value} is neither a multiple of 32 (conv and window-attention "
                "backbones) nor of 14 (patch14 ViTs), so no supported backbone can "
                "take it without rounding the matrix internally"
            )
        return value

    @property
    def mm_per_px(self) -> float:
        """Physical resolution implied by the crop and the matrix size."""
        return self.physical_crop_mm / self.size


class AugmentConfig(_Section):
    """Training-time augmentation. **Every field defaults to off.**

    Off by default so that the baseline this project already measured stays the
    baseline: an augmentation that arrives switched on turns the next run into two
    changes at once, and the gap between "works" and top-10 here is ~0.006 against a
    0.0020 noise floor quoted in published write-ups. Enable these in a separate ablation config, one
    arm at a time.

    ``rotate_deg``, ``brightness`` and ``contrast`` are per-sample and run inside
    ``StudyDataset``; ``cutmix_*`` is per-batch and runs inside ``collate_studies``,
    because CutMix mixes two studies and must mix ``labels`` and ``mask`` with them.
    """

    #: Maximum absolute rotation, in degrees, sampled uniformly in +/- this value.
    #: Worth +0.025 CV in a controlled off-dataset ablation -- the largest single
    #: augmentation effect in the published write-ups.
    rotate_deg: float = Field(0.0, ge=0.0, le=45.0)
    #: Maximum additive intensity shift, in units of the backbone's normalised std.
    brightness: float = Field(0.0, ge=0.0, le=1.0)
    #: Maximum multiplicative contrast deviation from 1.0, sampled in +/- this value.
    contrast: float = Field(0.0, ge=0.0, le=1.0)
    #: Probability that a batch is CutMixed. The measured recipe uses p~1.0, where
    #: CutMix beat flip, rotate, brightness, contrast, blur *and* mixup.
    cutmix_prob: float = Field(0.0, ge=0.0, le=1.0)
    #: Beta parameter for the CutMix area fraction.
    cutmix_alpha: float = Field(1.0, gt=0.0, le=10.0)
    #: **Tombstone. This must stay False, and the validator explains why.** Declared
    #: rather than omitted so that a config asking for it fails with the anatomy
    #: instead of with pydantic's generic "extra inputs are not permitted", which
    #: reads like a typo and invites someone to add the field for real.
    hflip: bool = False

    @field_validator("hflip")
    @classmethod
    def _hflip_is_not_expressible_on_a_knee(cls, value: bool) -> bool:
        # A horizontal flip turns a left knee into a right knee, which swaps medial
        # and lateral. Two label pairs could be swapped back (Medial/Lateral
        # Meniscus, Medial/Lateral OA), but **MCL has no LCL column** and **PF OA is
        # not lateralised**, so there is no relabelling that makes a flipped study
        # correct for all twelve. The result is not a crash -- it is a plausible
        # wrong number on the compartment-specific columns.
        if value:
            raise ValueError(
                "hflip=True is refused. Flipping a knee horizontally swaps medial and "
                "lateral. Medial/Lateral Meniscus and Medial/Lateral OA could be swapped "
                "back, but MCL has no LCL column and PF OA is not lateralised, so no "
                "relabelling makes a flipped study correct for all twelve labels. Use "
                "rotation instead; it is the larger measured effect anyway."
            )
        return value

    @property
    def per_sample_enabled(self) -> bool:
        """Whether anything runs in the dataset's per-study ``augment`` hook."""
        return self.rotate_deg > 0.0 or self.brightness > 0.0 or self.contrast > 0.0

    @property
    def batch_enabled(self) -> bool:
        """Whether anything runs in the collate's ``batch_augment`` hook."""
        return self.cutmix_prob > 0.0

    @property
    def enabled(self) -> bool:
        """Whether this config asks for any augmentation at all."""
        return self.per_sample_enabled or self.batch_enabled


class ModelConfig(_Section):
    """Backbone identity, head shape, and regularisation."""

    backbone: str = "convnext_small"
    pretrained: bool = True
    drop_path_rate: float = Field(0.2, ge=0.0, lt=1.0)
    aux_dim: int = Field(0, ge=0)
    #: How the pooled study vector becomes twelve logits. ``linear`` is a single
    #: ``nn.Linear(dim, N_LABELS)``: the twelve "heads" share every parameter up to
    #: the final weight row, so no label has any capacity of its own. ``query`` gives
    #: each label a learned query that cross-attends the pooled tokens, which is the
    #: only one of the two where "per-label" means anything. Worth ~0.006 by the
    #: published ablation -- small, and roughly 20x less than where the weights come
    #: from, so it is an ablation arm rather than a default.
    head: Literal["linear", "query"] = "linear"
    #: Attention heads used by ``head="query"``, and meaningless to ``linear``.
    head_num_heads: int = Field(8, ge=1, le=32)
    #: Hold the backbone's ``BatchNorm`` layers in eval mode for the whole of
    #: training, so a training step and a validation step read the identical running
    #: statistics rather than the current minibatch's own mean/var. Off by default:
    #: every existing config keeps training exactly as it does today. Wired only for
    #: the volumetric arm -- ``kappa.models.volume_encoder.VolumeEncoder`` overrides
    #: ``train()`` to re-freeze its backbone's ``BatchNorm3d`` layers every time it is
    #: called, which is required because ``nn.Module.train()`` recurses into every
    #: submodule once per epoch and would otherwise silently thaw them again. See
    #: ``TrainConfig._frozen_bn_needs_a_volumetric_backbone`` for the refusal on any
    #: other family. The motivating case: ``r3d_18`` carries 20 ``BatchNorm3d``
    #: layers, trains at micro-batch 4, and the 3D arm overfits from epoch 3 -- a
    #: batch of 4 is exactly the regime where BatchNorm's own minibatch statistics are
    #: noisiest, and this flag is how to find out whether that noise is the cause.
    freeze_backbone_bn: bool = False

    @model_validator(mode="after")
    def _head_options_match_the_head(self) -> ModelConfig:
        # Setting a query-head option while asking for the linear head is the exact
        # failure this module exists to prevent: the file claims a setting, the run
        # ignores it, and the result is filed under the wrong hypothesis.
        if "head_num_heads" in self.model_fields_set and self.head != "query":
            raise ValueError(
                f"head_num_heads is set but head={self.head!r} ignores it; "
                'set head: "query" or remove head_num_heads'
            )
        return self


class LossConfig(_Section):
    """Asymmetric loss parameters. Plain BCE and symmetric focal are both wrong here."""

    #: Which loss. ASL-versus-BCE is *asserted* in the write-ups rather than ablated,
    #: and ``MaskedBCE`` exists so that checking it is a config change, not a rewrite.
    #: Spelled ``kind`` rather than ``name`` because ``name`` is the run's identity
    #: everywhere else in this file, and one word meaning two things in one config is
    #: how a reader ends up excluding the wrong field from the hash.
    kind: Literal["asl", "bce"] = "asl"
    gamma_neg: float = Field(4.0, ge=0.0, le=10.0)
    gamma_pos: float = Field(0.0, ge=0.0, le=10.0)
    clip: float = Field(0.05, ge=0.0, lt=1.0)
    #: Per-label override of ``gamma_neg``, keyed by the exact label name.
    #:
    #: ``gamma_neg=4.0`` encodes "positives are rare, so focus down the easy
    #: negatives". Measured on the derived table the loss actually sees (n=4,407),
    #: that holds for Fracture at 7.4% and Synovitis at 10.6% and fails for
    #: **Effusion at 58.8% positive** and **Medial Meniscus at 50.8%**, where the
    #: negatives are the rarer class and the default focuses down the very cells
    #: those columns need. Every label is 1/12 of the score,
    #: so a column the loss quietly gives up on is given up at full price. Unset
    #: labels keep the scalar ``gamma_neg``.
    gamma_neg_by_label: dict[str, float] | None = None

    @field_validator("gamma_neg_by_label")
    @classmethod
    def _labels_are_the_twelve(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return None
        unknown = sorted(set(value) - set(LABELS))
        if unknown:
            # Names are taken from kappa.constants, never retyped: "Baker's" has an
            # apostrophe and five labels contain a space.
            raise ValueError(f"unknown label(s) {unknown}; legal names are {list(LABELS)}")
        bad = {k: v for k, v in value.items() if not 0.0 <= v <= 10.0}
        if bad:
            raise ValueError(f"gamma_neg outside [0, 10] for {bad}")
        return dict(value)

    @model_validator(mode="after")
    def _asl_knobs_need_the_asl_loss(self) -> LossConfig:
        # Same rule as ModelConfig.head_num_heads: a file that sets a focusing
        # exponent and then asks for BCE has documented an experiment that will not
        # happen. gamma/clip are ASL's asymmetry and MaskedBCE has no use for them.
        if self.kind != "asl":
            claimed = sorted(
                self.model_fields_set & {"gamma_neg", "gamma_pos", "clip", "gamma_neg_by_label"}
            )
            if claimed:
                raise ValueError(
                    f"{claimed} set but kind={self.kind!r} ignores them; "
                    'they are ASL\'s asymmetry. Use kind: "asl" or remove them.'
                )
        return self

    def gamma_neg_vector(self) -> tuple[float, ...]:
        """``gamma_neg`` for each of the twelve labels, in ``LABELS`` order.

        One vector whether or not the override is set, so the loss has a single code
        path and the per-label arm cannot diverge from the scalar one.
        """
        overrides = self.gamma_neg_by_label or {}
        return tuple(float(overrides.get(name, self.gamma_neg)) for name in LABELS)


class OptimConfig(_Section):
    """Optimiser and schedule. Defaults sit inside the range that converged across
    published RSNA write-ups; the bounds refuse the settings that did not."""

    lr: float = Field(5e-5, gt=0.0, lt=1e-1)
    #: The BACKBONE's learning rate as a multiple of ``lr``, which is the HEAD's. A
    #: pretrained trunk and a randomly-initialised ``nn.Linear(dim, 12)`` do not want
    #: the same rate: the trunk arrives with Kinetics-400 or ImageNet features worth
    #: keeping, the head arrives with noise, and at one shared rate either the head
    #: crawls or the head's early gradients drag the trunk off its pretraining. The
    #: 3D arm's over-fit from epoch 3 is a candidate instance of the second.
    #: Bounded at ``1.0`` because the direction under test is a SLOWER trunk; a trunk
    #: faster than a fresh head is not a hypothesis anyone here holds, and the bound
    #: refuses a mistyped ``10`` before it spends eight hours on an H200 rather than
    #: after. ``1.0`` -- the default -- IS the single-rate optimiser this project ran
    #: before the field existed: every group gets ``lr`` and the arithmetic is
    #: unchanged, so a config that omits this field trains exactly as it did.
    #:
    #: It is deliberately NOT exempt from ``config_hash``. ``optim`` is not a
    #: default-droppable section -- no real config leaves it at its default -- so
    #: adding a field here moved every hash in the repository. That was accepted
    #: rather than worked around: the alternative is a fifth config section for one
    #: float. The earlier hashes are recorded next to the new ones in
    #: ``tests/test_volume.py``.
    backbone_lr_mult: float = Field(1.0, gt=0.0, le=1.0)
    weight_decay: float = Field(1e-2, ge=0.0, le=1.0)
    schedule: Literal["onecycle", "cosine"] = "onecycle"
    warmup_pct: float = Field(0.1, ge=0.0, le=0.5)
    epochs: int = Field(10, ge=1, le=50)
    batch_size: int = Field(8, ge=1, le=256)
    grad_accum: int = Field(1, ge=1, le=64)
    max_grad_norm: float = Field(1.0, gt=0.0)
    amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    #: Exponential moving average of the weights; ``0.0`` disables it. The EMA is
    #: evaluated and checkpointed alongside the raw weights, never instead of them,
    #: so the baseline number stays readable next to it.
    ema_decay: float = Field(0.0, ge=0.0, lt=1.0)
    #: Average the weights of the last N epochs; ``0`` disables it. An alternative to
    #: ``ema_decay``, not a companion -- running both produces a third set of weights
    #: nobody asked for.
    swa_last_n: int = Field(0, ge=0, le=50)

    @model_validator(mode="after")
    def _one_averaging_scheme_at_a_time(self) -> OptimConfig:
        if self.ema_decay > 0.0 and self.swa_last_n > 0:
            raise ValueError(
                f"ema_decay={self.ema_decay} and swa_last_n={self.swa_last_n} are both "
                "set; they are two answers to the same question. Pick one."
            )
        if self.swa_last_n > self.epochs:
            raise ValueError(
                f"swa_last_n={self.swa_last_n} exceeds epochs={self.epochs}; it would "
                "average fewer checkpoints than it claims to"
            )
        return self


class FoldsConfig(_Section):
    """Cross-validation layout. The fold table is CSV because ``.gitignore`` globs
    ``*.parquet`` and would silently drop a parquet fold file from the repo."""

    n_folds: int = Field(5, ge=2, le=20)
    file: Path | None = None
    only: tuple[int, ...] = ()
    #: Which assignment column of the fold table to train against. ``fold`` groups by
    #: patient; ``fold_scanner`` groups by patient *and* scanner fingerprint. This is
    #: an experiment-defining choice and therefore part of ``config_hash``: 408
    #: distinct per-study fingerprints exist and site memorisation is worth about 0.05 of a
    #: random-fold score, so the same weights scored under the two columns are two
    #: different numbers and a run that does not record which one it used cannot be
    #: compared to anything.
    column: Literal["fold", "fold_scanner"] = "fold"

    @field_validator("file")
    @classmethod
    def _fold_file_is_csv(cls, value: Path | None) -> Path | None:
        if value is not None and value.suffix.lower() != ".csv":
            raise ValueError(
                f"fold file {value} is not .csv -- .gitignore globs *.parquet, so a "
                "parquet fold table would be silently excluded from the repository"
            )
        return value

    @model_validator(mode="after")
    def _only_is_in_range(self) -> FoldsConfig:
        bad = [fold for fold in self.only if not 0 <= fold < self.n_folds]
        if bad:
            raise ValueError(f"fold(s) {bad} outside 0..{self.n_folds - 1}")
        return self

    def selected(self) -> tuple[int, ...]:
        """Folds this run should train, honouring ``only`` when it is set."""
        return self.only or tuple(range(self.n_folds))


class EvalConfig(_Section):
    """How predictions are produced at validation and at inference.

    Applies to both, and deliberately so: ``predict_studies`` is the same function
    the training loop's validation pass and the Kaggle notebook call, so a TTA
    setting that changed only one of them would make the reported CV describe a
    pipeline that never runs.
    """

    #: Test-time augmentations to average over, by rank. Empty is a single plain
    #: pass. Every name must be label-safe on a knee; ``TTA_TRANSFORMS`` is the
    #: complete legal set and contains no flip.
    tta: tuple[str, ...] = ()

    @field_validator("tta")
    @classmethod
    def _tta_is_known_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [name for name in value if name not in TTA_TRANSFORMS]
        if unknown:
            raise ValueError(f"unknown tta {unknown}; legal names are {list(TTA_TRANSFORMS)}")
        if len(set(value)) != len(value):
            raise ValueError(f"duplicate tta transform in {list(value)}")
        return value


class DistillConfig(_Section):
    """Teacher/student wiring: who sees the report text, and how it transfers.

    Off by default, and that default is load-bearing -- an existing image-only
    config keeps its exact meaning and its exact ``config_hash`` when this
    section is absent (see :func:`canonical_dump`).

    The research question is whether report text available only at training time
    can improve an image-only model. That makes three distinct runs, and the
    difference between them lives entirely here:

    * ``role="student"``, no ``teacher_ckpt`` -- the image-only reference.
    * ``role="teacher"`` -- image + report features. Cannot be submitted; the
      competition's test studies carry no report. Exists only to be distilled from.
    * ``role="student"`` with a ``teacher_ckpt`` -- image-only weights, trained
      against the teacher's soft targets. This is the one that can be submitted.
    """

    #: Which model the loop builds. ``teacher`` reads ``report_features`` and is
    #: never an inference artifact; ``student`` is image-only at forward time
    #: whether or not it is distilling.
    role: Literal["student", "teacher"] = "student"

    #: Parquet written by ``kappa.labels.features.write_features``: one row per
    #: study, 78 anatomy clusters x 2 polarity channels. Required by a teacher,
    #: and required to *distil* (the frozen teacher still needs its text input).
    report_features: Path | None = None

    #: A frozen teacher checkpoint. Set it and the student distils; leave it and
    #: the student trains on labels alone. Both are the same code path.
    teacher_ckpt: Path | None = None

    #: Softening applied to both sides of the KD term. Scaled by ``T**2`` in the
    #: loss so changing it does not silently rescale the learning rate.
    temperature: float = Field(2.0, gt=0.0, le=20.0)

    #: Weight on the masked supervised loss over the real labels.
    lambda_supervised: float = Field(1.0, ge=0.0)

    #: Weight on the KD term. Ignored when ``teacher_ckpt`` is unset.
    lambda_distill: float = Field(1.0, ge=0.0)

    #: Distil on every cell, not just the labelled ones. **True is the point of
    #: the method**: 98.7% of raw label cells are unknown, so the supervised term
    #: sees almost nothing and the transfer happens on the unlabelled majority.
    #: False is the ablation that measures what those cells were worth.
    distill_unlabelled: bool = True

    #: Probability of withholding the **whole** report channel for a study during
    #: teacher training, by clearing its presence flag. Distinct from
    #: ``report_dropout``, which drops units inside the branch.
    #:
    #: This is the leak check. 4,349 of the 4,407 label rows were themselves made
    #: from the reports, so a teacher can minimise its loss by relearning the label
    #: pipeline and never reading the pixels -- and that failure looks exactly like
    #: success. A teacher whose score collapses when this is raised never learned
    #: the images. Read only when ``role="teacher"``.
    report_drop_rate: float = Field(0.0, ge=0.0, lt=1.0)

    #: Width of the teacher's report branch, and its dropout. Read only when
    #: ``role="teacher"``.
    report_hidden: int = Field(256, ge=1, le=4096)
    report_dropout: float = Field(0.1, ge=0.0, lt=1.0)

    #: Clear the frozen teacher's presence flag for **every** study while
    #: distilling, so its report branch contributes nothing to the soft targets.
    #:
    #: This is the residue ablation, and it is the control that tells two
    #: explanations apart. A student that gains from KD may be gaining from the
    #: report residue the teacher read, or merely from having *any* smooth second
    #: opinion over the 98.7% of cells the labels leave unknown -- and those two
    #: are indistinguishable in a macro mean. Setting this true keeps the same
    #: fold-matched teacher weights, the same architecture and the same soft-target
    #: machinery, and removes only the residue's contribution: the teacher's fusion
    #: layer sees a zeroed report embedding, exactly as it already does for a study
    #: whose extraction failed. The gap between this arm and the full KD arm is
    #: what the residue was worth.
    #:
    #: Read only while distilling. Setting it on a teacher run is refused below,
    #: because a teacher trained on a permanently-gated report is the image-only
    #: baseline wearing a teacher's name -- the same failure ``role="teacher"``
    #: with no ``report_features`` is refused for.
    teacher_report_ablate: bool = False

    @model_validator(mode="after")
    def _roles_and_inputs_agree(self) -> DistillConfig:
        if self.role == "teacher":
            if self.report_features is None:
                raise ValueError(
                    "role='teacher' with no distill.report_features. A teacher that "
                    "cannot read the report is just the image-only baseline under "
                    "another name, and would be filed as a teacher result."
                )
            if self.teacher_ckpt is not None:
                raise ValueError(
                    "role='teacher' with distill.teacher_ckpt set. A teacher does not "
                    "distil from a teacher; set role='student' to train the student."
                )
        if self.teacher_ckpt is not None and self.report_features is None:
            raise ValueError(
                "distill.teacher_ckpt without distill.report_features. The frozen "
                "teacher still reads report text at student-training time -- that is "
                "legal (training only) and required, or the teacher scores every "
                "study on an all-zero report."
            )
        if self.teacher_ckpt is not None and self.lambda_distill == 0.0:
            raise ValueError(
                "distill.teacher_ckpt set with lambda_distill=0.0. This loads and runs "
                "a teacher, pays for it, and discards its output. For the no-KD control "
                "leave teacher_ckpt unset, so the control differs from the treatment by "
                "one field."
            )
        if self.lambda_supervised == 0.0 and self.lambda_distill == 0.0:
            raise ValueError("both loss weights are 0.0; this run optimises nothing")
        if self.teacher_report_ablate and self.teacher_ckpt is None:
            # Silently doing nothing is the failure mode here: the ablation arm and
            # the control arm would hash differently, run identically, and be filed
            # as two measurements of the same thing.
            raise ValueError(
                "distill.teacher_report_ablate=true with no distill.teacher_ckpt. "
                "This flag gates the FROZEN TEACHER's report branch while distilling, "
                "so without a teacher it changes nothing and the arm would be the "
                "no-KD control under a name that claims otherwise."
            )
        if self.teacher_report_ablate and self.role == "teacher":
            raise ValueError(
                "distill.teacher_report_ablate=true with role='teacher'. This flag is "
                "read while DISTILLING, not while training a teacher; a teacher whose "
                "report is gated off for every study is the image-only baseline under "
                "another name. Set it on the student arm that distils from this teacher."
            )
        return self

    @property
    def distilling(self) -> bool:
        """True when a teacher checkpoint will be loaded and used."""
        return self.teacher_ckpt is not None and self.lambda_distill > 0.0


class VolumeConfig(_Section):
    """How the ``n_slices`` indices are chosen along the series axis.

    **This section exists to hold exactly one decision, and it is a section rather
    than a field on** :class:`DataConfig` **for a hashing reason.**
    :func:`canonical_dump` hashes the full ``model_dump()``, so a new field on an
    existing section moves **every** hash in the repository -- including the one the
    measured baseline (macro 0.7280) is filed under. A whole section can be dropped
    before hashing when it is at its default, which is the trick ``distill`` already
    uses, and it is the only way to add this knob without moving existing hashes.
    Keep it that way: putting a second field here is free, putting one on ``data`` is
    not.

    **``spanning`` is the default because it is what the loader already did.** A
    config that does not mention this section hashes exactly as it did before this
    section existed, and trains exactly as it did.

    **The two modes measure different things**, and reading one as the other is the
    mistake this docstring exists to prevent. Measured over the 4,719 ``Axial_fs1``
    series of the real manifest: median 32 slices, and the corpus median inter-slice
    spacing is 3.50 mm over a 105.60 mm through-plane span.

    * ``spanning`` -- ``n_slices`` indices at equal intervals across the **full**
      geometric extent, i.e. ``sample_slice_indices``, unchanged. Coverage is 100% of
      the joint at every depth, so a depth sweep under this mode varies **sampling
      density** and holds anatomical coverage fixed.
    * ``contiguous`` -- ``n_slices`` **adjacent** indices centred on the series
      middle. Coverage is ``(n_slices - 1) * 3.50 mm`` of a 105.60 mm joint: 24.5 mm
      (23%) at depth 8, 52.5 mm (50%) at 16, 80.5 mm (76%) at 24, 108.5 mm (100%+) at
      32. Depth and coverage therefore move **together**, so a monotone result under
      this mode is not evidence that volumetric context helps -- it is equally
      consistent with "more of the knee was in frame". The ``spanning`` runs at the
      same depths are the control that separates them.

    A series shorter than the requested depth repeats indices rather than being
    zero-padded, because a black slice tells the encoder the anatomy ended, which is
    false -- the volume ended. Measured on the same 4,719 series, the fraction shorter
    than the depth is **0.00% at 8, 1.36% at 16, 15.47% at 24 and 48.46% at 32**. The
    depth-32 figure is nearly half the corpus and follows from the median being 32, so
    that run is partly measuring interpolation rather than depth. The dataset counts
    how often it fired and the run log prints it rather than hiding it.
    """

    mode: Literal["spanning", "contiguous"] = "spanning"


class PoolConfig(_Section):
    """How the six ``plane x contrast`` slots are weighted, and who reads the result.

    **A section, not two fields on** :class:`ModelConfig`, **for the reason
    ``volume`` is one.** :func:`canonical_dump` hashes the full ``model_dump()``, so a
    new field on an existing section moves *every* hash in the repository --
    ``optim.backbone_lr_mult`` and ``model.freeze_backbone_bn`` each did exactly that,
    and artifacts trained before them are filed under the digests they moved off. A
    whole section can be dropped before hashing
    while it is at its default. **Both fields therefore default to ``False``, and
    ``False`` is today's arithmetic exactly** -- a config that does not mention this
    section hashes and trains as it did before the section existed.

    ``per_label_slots`` -- give each of the twelve labels its own softmax over the
    present slots, added to the shared one rather than replacing it.
    :class:`kappa.models.aggregator.StudyAggregator` pools the slots with **one**
    :class:`~kappa.models.aggregator.GatedAttentionPool`, so all twelve labels read a
    single set of plane weights. Knee MRI's best plane is not one plane: a meniscal
    tear is read sagittally and a collateral ligament coronally. This repository's own
    one-slot to three-plane step is the evidence that the aggregate hides the
    disagreement -- it
    bought its gain on **Medial Meniscus +0.082** and **Lateral Meniscus +0.054**
    while **MCL moved -0.006**. One shared
    softmax cannot serve both ends of that spread.

    ``head_residual`` -- add the pooled study vector back into what
    :class:`kappa.models.aggregator.LabelQueryHead` reads, and so **requires**
    ``model.head: query``. With ``head="query"`` and ``aux_dim=0`` the two pooling
    modules currently receive **no gradient at all**: the query head cross-attends the
    encoder's slice tokens directly and nothing downstream consumes the pooled vector,
    which ``kappa.models.study_classifier.StudyClassifier`` states outright in its own
    docstring. The pools are built, run, and saved at their initial values.

    The published ablation this project cites separates the two shapes and they are
    not close: class-agnostic pooling alone **82.1**, class-specific alone **84.2**,
    and **both together 85.3**. Replacing the pooled vector with a per-label one is
    the losing arrangement; adding to it is the winning one. Both fields here are
    therefore *residual* by construction -- see
    :meth:`kappa.models.study_classifier.StudyClassifier.forward` for the arithmetic
    -- and neither can be configured to replace.

    Neither is a default and neither has been measured **in this repository**: the
    82.1/84.2/85.3 figures are carried in, exactly like ``NOISE_FLOOR_MACRO``, and
    quoting them as this project's own result would be the mistake
    ``kappa.evaluation.metrics`` is annotated to prevent.
    """

    #: Per-label slot weighting, added to the shared pool. ``n_labels x n_slots``
    #: extra weights -- 72 numbers for twelve labels over six slots, which is why
    #: this arm costs nothing next to the encoder.
    per_label_slots: bool = False
    #: Feed the pooled study vector back into the query head. Refused unless
    #: ``model.head == "query"``, because the linear head already reads it.
    head_residual: bool = False


class TrainConfig(_Section):
    """One experiment, resolved. Hash it, log it, and the run becomes evidence."""

    name: str
    seed: int = 0
    seeds: tuple[int, ...] = ()
    out_dir: Path = Path("data/runs")
    data: DataConfig = DataConfig()
    augment: AugmentConfig = AugmentConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optim: OptimConfig = OptimConfig()
    folds: FoldsConfig = FoldsConfig()
    eval: EvalConfig = EvalConfig()
    distill: DistillConfig = DistillConfig()
    volume: VolumeConfig = VolumeConfig()
    pool: PoolConfig = PoolConfig()

    @model_validator(mode="after")
    def _absent_slots_need_series_dropout(self) -> TrainConfig:
        # The presence mask is load-bearing: zero-filling an absent plane x
        # contrast slot without also dropping present series at random teaches
        # the model that "absent" and "black" are the same input. With a single
        # 100%-coverage slot nothing is ever absent, so dropout 0.0 is correct;
        # the moment a second slot appears it is not.
        if len(self.data.slots) > 1 and self.data.series_dropout == 0.0:
            raise ValueError(
                f"{len(self.data.slots)} slots configured with series_dropout=0.0. "
                "Fusing slots that are not at 100% coverage means zero-filling absent "
                "ones, and without random series dropout the model cannot tell an "
                "absent slot from a black one. Set data.series_dropout (~0.1)."
            )
        return self

    @model_validator(mode="after")
    def _backbone_and_size_agree(self) -> TrainConfig:
        """Refuse an input matrix the configured backbone's pretrained weights cannot take.

        This lives on ``TrainConfig`` rather than on ``DataConfig`` or ``ModelConfig``
        because it is the only level that sees **both** ``model.backbone`` and
        ``data.size``; a field validator on either section can only see its own half
        of the pair, and the pair is the thing that is wrong.

        Left to timm, the same mistake has two outcomes and the bad one is the quiet
        one. A ``maxvit_*_tf_384`` weight set carries window partitioning and
        relative-position tables built for a 384 px input; a patch14 ViT interpolates
        its pretrained position embedding onto a ``size/14`` token grid. Off-spec,
        timm either **raises deep inside position-embedding interpolation** -- an hour
        into a queued job, with the GPU already spent -- or **silently interpolates
        onto a grid the pretrained weights were never trained on**, in which case the
        run finishes, prints a loss curve and reports a macro AUC that is evidence
        about nothing. Rejecting the pair at config time turns both into one
        immediate, named error, before the job is submitted.
        """
        backbone, size = self.model.backbone, self.data.size
        # timm names are `family.pretrained_tag`; the constraint is the family's.
        family = backbone.split(".", 1)[0]
        if family in VIDEO_BACKBONES:
            # torchvision's VideoResNet stem is stride (1,2,2) and layers 2-4 are
            # stride (2,2,2), so the spatial axis is divided by 16 and the temporal
            # axis by 8. The 32-grid below is stricter than the 16 this actually
            # needs, and is kept anyway so the volumetric arm can be read against the
            # 384 px 2.5D arms at the identical matrix. Only the temporal floor is
            # checked here, because it is the one the 2D branch cannot express.
            if size % 32:
                raise ValueError(
                    f"model.backbone={backbone!r} with data.size={size}: not a multiple "
                    "of 32. The volumetric arm exists to be read against the 384 px 2.5D "
                    "arms, which requires the same matrix."
                )
            if self.data.n_slices < 8:
                raise ValueError(
                    f"model.backbone={backbone!r} with data.n_slices={self.data.n_slices}. "
                    "data.n_slices IS the volumetric depth D on this path, and the three "
                    "temporal stride-2 stages divide it by 8 -- below 8 the depth axis is "
                    "collapsed to a single element before layer4 and the run measures a 2D "
                    "model with extra cost. Use n_slices >= 8."
                )
            return self
        if family.startswith("maxvit_") and family.endswith("_tf_384") and size != 384:
            raise ValueError(
                f"model.backbone={backbone!r} with data.size={size}. A maxvit *_tf_384 "
                "weight set is trained at exactly 384 px: its window partitioning needs "
                "the matrix on the 32-grid and its relative-position tables are sized "
                f"for 384, so {size} is either a raise inside timm or a silent resize of "
                "those tables onto a grid the weights never saw. Set data.size: 384, or "
                "choose a backbone whose weights match the size you want."
            )
        if "patch14" in family:
            if size % 14:
                raise ValueError(
                    f"model.backbone={backbone!r} with data.size={size}. A patch14 "
                    f"backbone tokenises in 14 px patches and {size}/14 = {size / 14:.2f} "
                    "is not an integer, so timm would interpolate the pretrained position "
                    "embedding onto a fractional token grid it never trained on -- which "
                    "does not crash and does not stop the run reporting a number. Use a "
                    "multiple of 14; 392 = 14 x 28 holds the 150 mm crop at 0.383 mm/px, "
                    "2% off the 384 px arms and identical in field of view."
                )
        elif size % 32:
            raise ValueError(
                f"model.backbone={backbone!r} with data.size={size}: not a multiple of 32. "
                "The 14-grid is legal only for patch14 ViTs; a conv or window-attention "
                f"backbone downsamples by 2 at least five times, so {size} is rounded "
                "inside the backbone and the effective mm/px stops matching this config."
            )
        return self

    @model_validator(mode="after")
    def _a_video_backbone_ignores_the_2d_knobs(self) -> TrainConfig:
        """Refuse a volumetric config that sets a knob only the 2.5D path reads.

        Same rule as ``ModelConfig.head_num_heads`` and ``LossConfig``'s ASL knobs, one
        level up: a file that states a setting the run then ignores has documented an
        experiment that will not happen, and nothing in the log would say so.

        ``drop_path_rate`` is the live case. It is timm's stochastic-depth argument and
        ``torchvision.models.video`` has no equivalent, so a volumetric config left at
        the baseline's 0.2 would claim a regulariser the 3D arm does not have. It has
        to be written as 0.0, which also states plainly in the file that the 3D arm is
        less regularised than the 2.5D arms it will be compared against.
        """
        if self.model.backbone.split(".", 1)[0] not in VIDEO_BACKBONES:
            return self
        if self.model.drop_path_rate != 0.0:
            raise ValueError(
                f"model.backbone={self.model.backbone!r} with "
                f"drop_path_rate={self.model.drop_path_rate}. That is timm's "
                "stochastic-depth argument and torchvision.models.video has no "
                "equivalent, so the value would be silently dropped and the 3D arm would "
                "be compared against 2.5D arms that really do carry it. Set "
                "model.drop_path_rate: 0.0 and read the comparison knowing the "
                "difference."
            )
        if self.model.head == "query":
            raise ValueError(
                f"model.backbone={self.model.backbone!r} with head='query'. The "
                "volumetric encoder emits ONE token per slot, not one per slice, so a "
                "per-label query would cross-attend a sequence of length len(slots) and "
                'the "per-label capacity over slice tokens" the query head is for does '
                "not exist on this path. Use head: linear."
            )
        return self

    @model_validator(mode="after")
    def _frozen_bn_needs_a_volumetric_backbone(self) -> TrainConfig:
        """Refuse ``freeze_backbone_bn`` on a backbone that cannot honour it.

        Inverse of ``_a_video_backbone_ignores_the_2d_knobs``, same reason: only
        :class:`kappa.models.volume_encoder.VolumeEncoder` overrides ``train()`` to
        re-freeze its backbone's ``BatchNorm3d`` layers, so a 2.5D config setting this
        flag would have it silently ignored -- whether or not the chosen timm family
        carries BatchNorm of its own (``tf_efficientnetv2_s`` carries 110). Refused
        here, before the job runs, rather than left for the run log to contradict.
        """
        if not self.model.freeze_backbone_bn:
            return self
        if self.model.backbone.split(".", 1)[0] not in VIDEO_BACKBONES:
            raise ValueError(
                f"model.freeze_backbone_bn=True with model.backbone={self.model.backbone!r}. "
                "Only the volumetric arm (kappa.models.volume_encoder.VolumeEncoder) is "
                "wired to hold its BatchNorm layers in eval mode during training; a 2.5D "
                f"backbone would silently ignore this flag. Use one of {list(VIDEO_BACKBONES)}, "
                "or remove freeze_backbone_bn."
            )
        return self

    @model_validator(mode="after")
    def _contiguous_mode_needs_a_volumetric_encoder(self) -> TrainConfig:
        """``volume.mode`` is read by the loader alone, so pair it with the 3D arm.

        ``contiguous`` is legal input to the 2.5D loader -- it would hand
        :class:`kappa.models.SliceEncoder` a centred block instead of a spread -- and
        that is a real experiment, but it is NOT one this repository has planned, and a
        2.5D config that quietly acquired it would be filed against the 2.5D slot
        sweep while measuring something else. Refused here so that choosing it is deliberate;
        lift this validator, do not work around it, if a 2.5D contiguous arm is ever
        wanted.
        """
        if self.volume.mode == "contiguous" and (
            self.model.backbone.split(".", 1)[0] not in VIDEO_BACKBONES
        ):
            raise ValueError(
                f"volume.mode='contiguous' with model.backbone={self.model.backbone!r}, "
                "which is a 2D timm backbone. Contiguous sampling collapses coverage "
                "from the full joint to (n_slices-1) x 3.50 mm, so this is a different "
                "experiment from the 2.5D slot sweep and must not be filed against it. "
                f"Use one of {list(VIDEO_BACKBONES)}, or volume.mode: spanning."
            )
        return self

    @model_validator(mode="after")
    def _the_head_residual_needs_the_query_head(self) -> TrainConfig:
        """Refuse ``pool.head_residual`` on a head that already reads the study vector.

        Same refusal as ``ModelConfig._head_options_match_the_head``, one level up
        because it is the only level that sees both sections. ``head="linear"`` *is*
        ``nn.Linear(dim, N_LABELS)`` over the pooled study vector, so "add the pooled
        study vector back in" is not a modification of it -- there is nothing to add
        it back to, and :class:`kappa.models.study_classifier.StudyClassifier` would
        ignore the flag outright. A config that claims a setting the run ignores is
        the exact failure this module exists to prevent: the file reads as an
        experiment, the run is the baseline, and the result is filed under the wrong
        hypothesis with a hash that says it was configured.
        """
        if self.pool.head_residual and self.model.head != "query":
            raise ValueError(
                f"pool.head_residual=True with model.head={self.model.head!r}. The "
                "residual adds the pooled study vector into what LabelQueryHead reads, "
                "which only exists under head: query -- the linear head already reads "
                "that vector and nothing else. Set model.head: query, or remove "
                "pool.head_residual."
            )
        return self

    @model_validator(mode="after")
    def _report_text_cannot_be_cutmixed(self) -> TrainConfig:
        # CutMix mixes two studies' pixels, labels and masks together. There is no
        # sane way to mix two report vectors: the result would claim findings that
        # neither blended study has. A teacher trained this way reads study A's
        # report over a pixel blend of A and B and still prints a loss curve, so
        # this is refused rather than documented.
        reads_report = self.distill.role == "teacher" or self.distill.distilling
        if reads_report and self.augment.cutmix_prob > 0.0:
            raise ValueError(
                f"augment.cutmix_prob={self.augment.cutmix_prob} with a model that reads "
                "report features. CutMix blends two studies' pixels but cannot blend "
                "their reports, so the report would describe a study that does not "
                "exist. Set cutmix_prob: 0.0 for teacher and distillation runs."
            )
        return self

    def seed_list(self) -> tuple[int, ...]:
        """Seeds for a **noise-floor** measurement: ``seeds`` when set, else ``seed``.

        Not what a normal run uses. ``run_experiment`` trains ``seed`` alone;
        ``measure_noise_floor`` defaults to this list, which is why populating
        ``seeds`` in a config file does not silently turn one experiment into two.
        """
        return self.seeds or (self.seed,)


def _as_mapping(raw: object, source: str) -> Mapping[str, Any]:
    """A YAML document as a mapping, or ``ConfigError`` naming the source.

    An empty file parses to ``None``, which is a valid "all defaults" config; a
    list or a bare scalar is not, and ``TrainConfig(**raw)`` on either raises a
    ``TypeError`` that names no file.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{source}: top level must be a mapping, got {type(raw).__name__}")
    return raw


def _validate(raw: Mapping[str, Any], source: str) -> TrainConfig:
    """Build a :class:`TrainConfig`, re-raising as ``ConfigError`` naming the source.

    Both entry points funnel through here. They used to carry a copy each, and had
    already drifted -- one treated an empty document as ``{}`` explicitly, the other
    via ``or {}``, which also swallows a document that is legitimately falsey.
    """
    try:
        return TrainConfig(**raw)
    except Exception as exc:  # pydantic ValidationError, or a bad nested type
        raise ConfigError(f"{source}: {exc}") from exc


def loads_config(text: str, *, source: str = "<string>") -> TrainConfig:
    """Parse and validate YAML text. Raises ``ConfigError`` naming the bad key."""
    import yaml

    return _validate(_as_mapping(yaml.safe_load(text), source), source)


def load_config(path: str | Path, *, overrides: Sequence[str] = ()) -> TrainConfig:
    """Load a YAML config, apply ``key.path=value`` overrides, and validate.

    Overrides are applied to the raw mapping *before* validation, so a misspelled
    override key is rejected by the same ``extra="forbid"`` that guards the file.
    """
    import yaml

    path = Path(path)
    source = str(path)
    raw = _as_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), source)
    return _validate(apply_overrides(raw, overrides), source)


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``raw`` with dotted ``a.b=value`` assignments applied.

    Values are parsed as YAML, so ``optim.epochs=2`` is an int and
    ``model.pretrained=false`` is a bool rather than the string ``"false"`` --
    which would be truthy and would quietly download weights in an offline job.
    """
    import yaml

    merged: dict[str, Any] = json.loads(json.dumps(_to_plain(raw)))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override {item!r} is not of the form key.path=value")
        target = merged
        parts = key.strip().split(".")
        for part in parts[:-1]:
            nxt = target.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                target[part] = nxt
            target = nxt
        target[parts[-1]] = yaml.safe_load(value)
    return merged


def _to_plain(value: Any) -> Any:
    """Recursively convert mappings and sequences to plain JSON-able containers."""
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


#: Fields excluded from ``config_hash``: they name or locate a run rather than
#: define the experiment it performs.
_HASH_EXCLUDED = ("name", "out_dir", "seed", "seeds")

#: Sections dropped from ``config_hash`` when they are at their default, so that
#: adding one does not move every hash that already exists. **A section may only be
#: listed here if its default is a no-op**: the entry is a promise that a config
#: which omits the section behaves exactly as it did before the section was
#: introduced. ``distill`` defaults to image-only; ``volume`` defaults to
#: ``mode="spanning"``, which is the slice sampling the loader has always done;
#: ``pool`` defaults to both flags off, which is the single shared slot pool and the
#: query head reading slice tokens alone -- the arithmetic every existing checkpoint
#: was trained under.
_DEFAULTED_SECTIONS: tuple[tuple[str, type[_Section]], ...] = (
    ("distill", DistillConfig),
    ("volume", VolumeConfig),
    ("pool", PoolConfig),
)


def canonical_dump(cfg: TrainConfig) -> str:
    """The exact string that gets hashed. Sorted keys, no whitespace slack.

    A ``distill`` or ``volume`` section left at its default is dropped before
    hashing. Adding a config section would otherwise move **every** existing hash,
    including the one the first measured baseline (macro 0.7280) was filed under, and a baseline that cannot be hash-matched to its own config file
    stops being a reference point. An image-only 2.5D run therefore hashes
    identically whether or not these sections exist; a teacher, a distilling student
    or a contiguous-mode volumetric run hashes differently, which is what the hash is
    for.

    ``_DEFAULTED_SECTIONS`` is a table rather than two hand-written ``if``s so that
    the next section added to :class:`TrainConfig` cannot be added there and forgotten
    here -- the omission would be silent and would move every 2.5D hash at once.
    ``tests/test_config.py`` pins the measured configs' hashes against this.
    """
    payload = cfg.model_dump(mode="json")
    for field in _HASH_EXCLUDED:
        payload.pop(field, None)
    for section, factory in _DEFAULTED_SECTIONS:
        if payload.get(section) == factory().model_dump(mode="json"):
            payload.pop(section)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def config_hash(cfg: TrainConfig, *, length: int = 12) -> str:
    """Stable short digest of the experiment, ignoring name, out_dir and seed."""
    return hashlib.sha256(canonical_dump(cfg).encode("utf-8")).hexdigest()[:length]


def _gamma_neg_summary(loss: LossConfig) -> str:
    """``gamma_neg``, naming the overridden labels rather than hiding them in a hash."""
    if not loss.gamma_neg_by_label:
        return f"{loss.gamma_neg}"
    overrides = ", ".join(f"{k}={v}" for k, v in sorted(loss.gamma_neg_by_label.items()))
    return f"{loss.gamma_neg} [{overrides}]"


def _loss_summary(loss: LossConfig) -> str:
    """The loss and only the parameters it actually reads."""
    if loss.kind != "asl":
        return f"{loss.kind.upper()} (masked)"
    return (
        f"ASL gamma_neg={_gamma_neg_summary(loss)} "
        f"gamma_pos={loss.gamma_pos} clip={loss.clip}"
    )


def _lr_summary(optim: OptimConfig) -> str:
    """The learning rate -- both of them, when the backbone runs slower than the head.

    Two numbers rather than one and a multiplier because this line is read by eye in a
    Slurm log: a differential run and a uniform run at the same ``lr`` would otherwise
    print the identical string, and the run that would be misread is precisely the one
    whose claim is that the trunk moved slower. The multiplier is printed as well so
    the line can be matched against the config that produced it.
    """
    if optim.backbone_lr_mult == 1.0:
        return f"lr={optim.lr:g} (one rate: backbone and head)"
    return (
        f"lr={optim.lr:g} head / {optim.lr * optim.backbone_lr_mult:g} backbone "
        f"(backbone x{optim.backbone_lr_mult:g})"
    )


def _augment_summary(augment: AugmentConfig) -> str:
    """What augmentation this run applies, or ``off``. Never silently empty.

    A run whose log does not say it was CutMixed is not evidence about CutMix, and
    this line is the only place a reader of the Slurm log would find out.
    """
    parts = []
    if augment.rotate_deg > 0.0:
        parts.append(f"rotate=+/-{augment.rotate_deg:g}deg")
    if augment.brightness > 0.0:
        parts.append(f"brightness=+/-{augment.brightness:g}")
    if augment.contrast > 0.0:
        parts.append(f"contrast=+/-{augment.contrast:g}")
    if augment.cutmix_prob > 0.0:
        parts.append(f"cutmix p={augment.cutmix_prob:g} alpha={augment.cutmix_alpha:g}")
    return "  ".join(parts) if parts else "off"


def _averaging_summary(optim: OptimConfig) -> str:
    """EMA, last-N weight averaging, or neither. The three are different runs."""
    if optim.ema_decay > 0.0:
        return f"EMA decay={optim.ema_decay:g}"
    if optim.swa_last_n > 0:
        return f"average of the last {optim.swa_last_n} epochs"
    return "off (raw final weights)"


def _pool_summary(cfg: TrainConfig) -> str:
    """Which slot pooling ran, and who reads it. Never silently empty.

    A run whose log does not say the twelve labels shared one set of plane weights is
    not evidence about per-label pooling, and this is the only line that says so.
    """
    parts = [
        "per-label slot softmax (12 x 6), added to the shared pool"
        if cfg.pool.per_label_slots
        else "one shared pool -- all 12 labels read the same plane weights"
    ]
    if cfg.pool.head_residual:
        parts.append("study vector fed back into the query head (pools get gradient)")
    return "  |  ".join(parts)


def _distill_summary(distill: DistillConfig) -> str:
    """Which of the three runs this is, spelled out rather than left to the hash."""
    if distill.role == "teacher":
        return (
            f"TEACHER -- image + report ({distill.report_features}), "
            f"hidden={distill.report_hidden} dropout={distill.report_dropout:g}. "
            "Training only; never an inference artifact"
        )
    if not distill.distilling:
        return "off -- image-only student, labels only (no teacher)"
    cells = "all cells" if distill.distill_unlabelled else "labelled cells only"
    residue = (
        "  |  RESIDUE ABLATED (teacher report gated off; soft targets are image-only)"
        if distill.teacher_report_ablate
        else ""
    )
    return (
        f"STUDENT <- {distill.teacher_ckpt}  T={distill.temperature:g} "
        f"lambda_sup={distill.lambda_supervised:g} lambda_kd={distill.lambda_distill:g} "
        f"on {cells}{residue}"
    )


def describe(cfg: TrainConfig) -> str:
    """Multi-line rendering of the resolved config, for the top of a run log.

    Printed at startup on purpose: a run that does not state which backbone and
    which config it loaded is not evidence about that backbone.
    """
    lines = [
        f"config     : {cfg.name}  (hash {config_hash(cfg)})",
        f"backbone   : {cfg.model.backbone}  pretrained={cfg.model.pretrained} "
        f"drop_path={cfg.model.drop_path_rate} freeze_bn={cfg.model.freeze_backbone_bn}",
        f"input      : slots={list(cfg.data.slots)} n_slices={cfg.data.n_slices} "
        f"stack={cfg.data.stack} size={cfg.data.size}",
        # The sampler is printed ALWAYS, including at its default. `volume` is dropped
        # from config_hash when it is at default, so the hash alone cannot tell a
        # reader which sampler ran -- and on a video backbone `n_slices` stops meaning
        # "slices pooled by a transformer" and starts meaning "depth convolved over by
        # 3D kernels", which is not a difference any other line here would show.
        f"sampling   : {cfg.volume.mode}"
        + (
            f"  -- VOLUMETRIC: n_slices={cfg.data.n_slices} IS the depth D, "
            f"{(cfg.data.n_slices - 1) * 3.50:.1f} mm of the ~105.6 mm joint"
            if cfg.model.backbone.split(".", 1)[0] in VIDEO_BACKBONES
            and cfg.volume.mode == "contiguous"
            else "  -- VOLUMETRIC: n_slices IS the depth D, full geometric extent"
            if cfg.model.backbone.split(".", 1)[0] in VIDEO_BACKBONES
            else ""
        ),
        f"scale      : {cfg.data.physical_crop_mm:.0f} mm crop / {cfg.data.size} px "
        f"= {cfg.data.mm_per_px:.3f} mm/px",
        f"loss       : {_loss_summary(cfg.loss)}",
        f"optim      : AdamW {_lr_summary(cfg.optim)} wd={cfg.optim.weight_decay:g} "
        f"{cfg.optim.schedule} warmup={cfg.optim.warmup_pct:g} epochs={cfg.optim.epochs} "
        f"batch={cfg.optim.batch_size}x{cfg.optim.grad_accum}",
        f"amp        : {cfg.optim.amp} ({cfg.optim.amp_dtype})",
        # Printed ALWAYS, default included, for the reason `sampling` is: `pool` is
        # dropped from config_hash at its default, so the hash alone cannot tell a
        # reader which pooling ran, and "one set of plane weights for all twelve
        # labels" is precisely the assumption the one-slot to three-plane per-label
        # spread questions.
        f"slot pool  : {_pool_summary(cfg)}",
        f"head       : {cfg.model.head}"
        + (f" ({cfg.model.head_num_heads} attention heads)" if cfg.model.head == "query" else "")
        + f"  aux_dim={cfg.model.aux_dim}",
        f"augment    : {_augment_summary(cfg.augment)}",
        f"averaging  : {_averaging_summary(cfg.optim)}",
        f"tta        : {list(cfg.eval.tta) if cfg.eval.tta else 'off (single pass)'}",
        f"folds      : {cfg.folds.n_folds} -> training {list(cfg.folds.selected())} "
        f"column={cfg.folds.column} table={cfg.folds.file}",
        f"distill    : {_distill_summary(cfg.distill)}",
        f"seeds      : {list(cfg.seed_list())}",
        f"out_dir    : {cfg.out_dir}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    for _path in sys.argv[1:]:
        print(describe(load_config(_path)))

