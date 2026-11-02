"""Score studies from a checkpoint, offline, with the runtime measured rather than guessed.

Why this exists, and why it is the *inference* package rather than a function inside
the training loop: **submissions are image-only.** Reports are training-only -- the
competition's test studies carry none -- so the notebook that produces
``submission.csv`` has no report text available, and any accidental dependency on report-derived state is not a bug
that degrades a score, it is a submission that cannot run at all. This module
therefore names its own boundary in :data:`FORBIDDEN_MODULES` and
``tests/test_loop.py`` asserts that importing it loads none of them. The boundary is
a competition-legality constraint, not a style preference.

Two further constraints shape every signature here.

**No absolute local paths.** The Kaggle notebook *imports* ``src/kappa/``, and it runs
on a machine where none of the training machine's paths exist. Every
path is an argument. Nothing is resolved from the environment, and
:func:`load_checkpoint` never reaches the network: ``pretrained=False`` is passed
unconditionally, because the weights are in the checkpoint and a hub download inside
a no-internet notebook fails after the model has already been built.

**Normalisation travels in the checkpoint.** The one mistake that silently invalidates
a whole run is feeding a backbone statistics it was not trained with -- it still runs,
still emits scores, and reports a number that means nothing. Re-resolving ``mean`` and
``std`` from timm at inference would need the hub and could disagree with what training
used, so :func:`save_inference_checkpoint` writes them into the file and
:func:`load_checkpoint` reads them back. Pixels and weights then agree by construction.

    from kappa.inference.predict import load_checkpoint, predict_studies

    loaded = load_checkpoint("fold0.pt", device="cpu")
    result = predict_studies(loaded.model, loaded.dataset(manifest, cache_dir), device="cpu")
    print(result.timing_report(n_studies=3_000))

**The runtime is part of the score.** The efficiency track carries a ``T / 32400``
term, where 32,400 s is the nine-hour notebook limit, so a full-length run contributes
1.0 against accuracy gaps of 0.01-0.05. :class:`PredictionResult` therefore always
carries measured seconds and seconds-per-study, and :meth:`PredictionResult.projection`
turns them into the term the leaderboard actually adds. Measuring it now is cheap;
retrofitting it after a submission is not.

**No calibration and no probability averaging.** Macro ROC-AUC reads only the order of
the scores within a column, so a sigmoid, Platt scaling and temperature all leave every
number bit-identical, and averaging two members' probabilities hands the blend to
whichever member is more confident. The default ``transform="rank"`` is per-column rank
normalisation from :mod:`kappa.evaluation.metrics` -- order-preserving by construction,
therefore AUC-identical to the raw logits, and in ``[0, 1]`` so no grader can reject it.
Ensembling *across members* is :func:`kappa.inference.submission.rank_blend` and nothing
else.

**Test-time augmentation lives here, and it is off unless asked for.** ``tta=()`` is a
single plain pass down exactly the code this module ran before TTA existed. When names
are given, each batch is scored once per view and the views are pooled **in logit
space**, which is the one place this project pools by mean rather than by rank; the
reasoning, and why the ensembling ruling does not reach inside a single model, is set
out in full on :func:`predict_studies`. Two properties of the legal set matter enough to
restate at the top of the file:

*No flip, ever, and not as an option.* This is a knee. A horizontal flip swaps medial
and lateral, and no relabelling repairs it for all twelve columns: Medial/Lateral
Meniscus and Medial/Lateral OA could be swapped back, but MCL has no LCL column to swap
with and PF OA is not lateralised at all. So a flipped study is simply mislabelled on at
least two of the twelve, and ``TTA_TRANSFORMS`` contains no flip by construction.

*TTA multiplies the runtime term.* The efficiency track divides by 32,400 s, so three
views is three times the forward cost against an accuracy gap of 0.01-0.05.
:class:`PredictionResult` therefore records which views ran, the timer still wraps the
whole pass, and :meth:`PredictionResult.timing_report` names the multiplier rather than
quoting a per-study rate that looks like a single pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kappa.constants import ID_COLUMN, LABELS, N_LABELS
from kappa.data.dataset import CropSpec, Normalisation, StudyDataset, collate_studies
from kappa.evaluation.metrics import rank_normalise
from kappa.inference.submission import rank_blend, write_submission
from kappa.models import StudyClassifier

__all__ = [
    "CHECKPOINT_VERSION",
    "FORBIDDEN_MODULES",
    "NOTEBOOK_LIMIT_SECONDS",
    "TTA_ROTATE_DEGREES",
    "TTA_TRANSFORMS",
    "TTA_ZOOM_FACTOR",
    "LoadedCheckpoint",
    "PredictionResult",
    "load_checkpoint",
    "predict_studies",
    "resolve_device",
    "save_inference_checkpoint",
    "transform_scores",
    "write_predictions",
]

#: Bumped whenever the checkpoint payload changes shape. A checkpoint written by an
#: older version is refused by name rather than silently missing a key, because the
#: key most likely to go missing is ``normalisation`` and its absence is invisible.
CHECKPOINT_VERSION = 1

#: The nine-hour Kaggle notebook limit, in seconds. Mirrors
#: ``kappa.config.NOTEBOOK_LIMIT_SECONDS``; restated here so that the inference path
#: has no reason to import the training-side config module inside a notebook.
NOTEBOOK_LIMIT_SECONDS = 32_400

#: Modules that read or interpret report text. **None of these may be reachable from
#: this module**, at import time or at call time. Submissions are image-only
#: (test studies carry no report), so a report-derived import is not a slow path -- it is a notebook
#: that cannot run. ``tests/test_loop.py`` imports this module in a clean interpreter
#: and asserts that not one of these names appears in ``sys.modules``.
#:
#: **This list is a floor, not the guard.** It was one once, and it failed: every
#: `kappa.labels.*` module was named here except `normalise`, which was the only one
#: inference actually pulled in (via `kappa.data.manifest`), so the check passed while
#: the boundary was broken. The test now matches on the `kappa.labels` prefix, which
#: cannot drift as modules are added. These names are kept so the test has something
#: concrete to assert against.
FORBIDDEN_MODULES: tuple[str, ...] = (
    "kappa.labels.anatomy_map",
    "kappa.labels.discover",
    "kappa.labels.engine",
    "kappa.labels.label_map",
    "kappa.labels.masking",
    "kappa.labels.polarity",
    "kappa.labels.sample",
    "kappa.labels.schema",
    "kappa.labels.normalise",
    "kappa.labels.shard",
)

#: The prefix every report-reading module shares. Matching on this is what actually
#: enforces the image-only boundary; see the note on FORBIDDEN_MODULES above.
FORBIDDEN_PREFIX = "kappa.labels"

#: Every test-time augmentation this module will run, and the complete legal set.
#: Mirrors ``kappa.config.TTA_TRANSFORMS`` and is restated for the same reason
#: :data:`NOTEBOOK_LIMIT_SECONDS` is: the notebook imports this module and must not be
#: made to import the training-side config module to score a study.
#: ``tests/test_tta.py`` asserts the two tuples are identical, so they cannot drift --
#: and they must not, because a config-legal ``eval.tta`` that this module refused
#: would fail at the end of a nine-hour run rather than at parse time.
#:
#: **There is no flip and there will not be.** A horizontal flip swaps medial and
#: lateral on a knee. Medial/Lateral Meniscus and Medial/Lateral OA could be swapped
#: back, but MCL has no LCL column and PF OA is not lateralised, so no relabelling
#: makes a flipped study correct for all twelve labels -- it is a wrong-label
#: augmentation that raises nothing and quietly costs AUC on the columns it corrupts.
TTA_TRANSFORMS: tuple[str, ...] = (
    "identity",
    "rotate_cw",
    "rotate_ccw",
    "zoom_in",
    "zoom_out",
)

#: Half-angle of the ``rotate_cw`` / ``rotate_ccw`` pair, in degrees.
#:
#: 10 deg because that is the train-time ``augment.rotate_deg`` in
#: ``configs/ablate_rotate.yaml``: a TTA view drawn from the distribution the model was
#: trained over is a view it has seen, and a larger one tests an invariance nobody
#: trained for. The pair is symmetric about 0 on purpose -- two opposite rotations
#: average to the unrotated geometry, whereas a single signed rotation would bias every
#: study the same way and just move the model off-centre.
TTA_ROTATE_DEGREES = 10.0

#: Linear magnification of ``zoom_in``; ``zoom_out`` uses its reciprocal, so the two are
#: symmetric in log-scale about 1.0 for the same reason the rotations are symmetric
#: about 0.
#:
#: 1.1 is deliberately small. Unlike rotation there is **no** train-time scale jitter in
#: ``AugmentConfig``, so a zoom view is outside the training distribution by
#: construction; this is an arm to measure, not a default, and its magnitude is kept
#: near the identity so that being wrong about it is cheap.
TTA_ZOOM_FACTOR = 1.1


def _zoom(pixels: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale ``(..., H, W)`` about its centre by ``factor``, onto the *same* pixel grid.

    Written to the same contract as :func:`kappa.data.augment.rotate`, and for the same
    reason: the crop is a physical measurement, ``physical_crop_mm / size`` mm/px, and a
    transform that returned a different ``H x W`` would silently change the scale the
    run reports while every downstream shape check still passed. ``affine_grid`` maps
    the output grid onto the input, so the output shape is the input shape by
    construction and a resize is not expressible here.

    ``factor > 1`` magnifies -- the frame samples a sub-region of the input. ``factor <
    1`` shrinks the anatomy and the vacated border is filled with 0.0, which downstream
    of the backbone's normalisation is the mean pixel rather than black; black is a real
    intensity a scanner can produce and would read as signal.

    Note what this does *not* change: ``CropSpec.mm_per_pixel`` still describes the grid
    that comes back, because the grid is untouched. What moves is the apparent scale of
    the anatomy within that fixed grid, which is the entire point of a scale TTA.
    """
    if factor <= 0.0:
        raise ValueError(f"zoom factor must be positive, got {factor}")
    if factor == 1.0:
        return pixels
    lead, height, width = pixels.shape[:-2], pixels.shape[-2], pixels.shape[-1]
    flat = pixels.reshape(-1, 1, height, width)
    # affine_grid maps output coords to input coords, so the matrix is the *inverse*
    # of the magnification: a factor of 1.1 samples the input over +/- 1/1.1 of the
    # normalised frame, i.e. a smaller region shown larger.
    inverse = 1.0 / float(factor)
    matrix = torch.tensor(
        [[inverse, 0.0, 0.0], [0.0, inverse, 0.0]], dtype=pixels.dtype, device=pixels.device
    ).expand(flat.shape[0], 2, 3)
    grid = F.affine_grid(matrix, list(flat.shape), align_corners=False)
    scaled = F.grid_sample(
        flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return scaled.reshape(*lead, height, width)


def _tta_views(names: Sequence[str]) -> tuple[Callable[[torch.Tensor], torch.Tensor], ...]:
    """Resolve TTA names to deterministic, shape-preserving pixel operations.

    Deterministic is load-bearing. Training-time augmentation draws from an RNG; a TTA
    view that did the same would give one checkpoint a different macro AUC on every
    evaluation, and the gap between working and top-10 on this task is about 0.006 --
    smaller than the noise a random TTA would inject into the number used to compare
    runs. Every view here is a fixed angle or a fixed factor.

    ``rotate`` is imported from :mod:`kappa.data.augment` rather than reimplemented, so
    the rotation the model meets at test time is the identical operator it met at train
    time, down to the interpolation and the zero fill. The import is inside the function
    on purpose: ``kappa.data.augment`` reaches ``kappa.config``, and this module's
    header commits to keeping the training-side config module out of the notebook. With
    ``tta=()`` -- the default -- nothing here runs and the import graph of a scoring
    notebook is exactly what it was before TTA existed.
    """
    resolved = tuple(str(name) for name in names)
    unknown = [name for name in resolved if name not in TTA_TRANSFORMS]
    if unknown:
        raise ValueError(
            f"unknown tta {unknown}; legal names are {list(TTA_TRANSFORMS)}. There is no "
            "flip in that set and adding one would mislabel every lateralised column."
        )
    if len(set(resolved)) != len(resolved):
        # A repeat is not a harmless no-op: it double-weights one view in the mean and
        # pays a full extra forward pass for it. kappa.config refuses the same thing, so
        # refusing here keeps a hand-called pass and a config-driven one in agreement.
        raise ValueError(f"duplicate tta transform in {list(resolved)}")
    if not resolved:
        return ()

    from kappa.data.augment import rotate  # see the docstring: lazy, and why

    # Positive degrees turns the image anticlockwise on screen under `rotate`'s
    # output-to-input convention (verified against a corner marker), so `rotate_cw`
    # takes the negative angle.
    table: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "identity": lambda pixels: pixels,
        "rotate_cw": lambda pixels: rotate(pixels, -TTA_ROTATE_DEGREES),
        "rotate_ccw": lambda pixels: rotate(pixels, TTA_ROTATE_DEGREES),
        "zoom_in": lambda pixels: _zoom(pixels, TTA_ZOOM_FACTOR),
        "zoom_out": lambda pixels: _zoom(pixels, 1.0 / TTA_ZOOM_FACTOR),
    }
    return tuple(table[name] for name in resolved)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Pick a device, defaulting to CUDA only when it is actually usable.

    ``torch.cuda.is_available()`` rather than a build check: on the cluster's ``gpu``
    partition a cu130 wheel imports fine and reports ``False`` here, which is exactly
    the case that must fall back to CPU instead of raising halfway through a forward
    pass. See ``slurm/env_build.sh`` for the measured driver situation.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_dtype(device: torch.device, amp: bool | str | None) -> torch.dtype | None:
    """Resolve the autocast dtype, or ``None`` for a full-precision pass.

    ``None`` means auto: bfloat16 on CUDA, off elsewhere. CPU autocast is deliberately
    not the default -- it is not the arithmetic a GPU run performs, so a CPU timing
    taken under it would not describe either machine.
    """
    if amp is False or amp == "off":
        return None
    if amp is None or amp is True:
        return torch.bfloat16 if device.type == "cuda" else None
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[str(amp)]


@dataclass(frozen=True)
class PredictionResult:
    """Scores for a set of studies, with the wall clock that produced them.

    ``scores`` are the model's raw logits in ``LABELS`` order -- no sigmoid, because
    the metric ranks within a column and a sigmoid is monotone, so it would change
    nothing except to make the numbers look calibrated. Apply
    :func:`transform_scores` at the moment of writing, not here.

    ``seconds`` is measured around the whole pass including the dataloader, not around
    the forward call. That is the number the notebook limit applies to.

    ``tta`` names the views that were averaged into ``scores``, in the order they ran,
    and is empty for a single plain pass. It is carried rather than dropped because a
    seconds-per-study figure is not comparable between a one-view and a three-view run,
    and a result that did not say which it was would invite exactly that comparison.
    """

    study_uids: tuple[str, ...]
    scores: np.ndarray
    seconds: float
    n_batches: int
    device: str
    amp: str
    tta: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scores.shape != (len(self.study_uids), N_LABELS):
            raise ValueError(
                f"scores {self.scores.shape} do not match {len(self.study_uids)} studies "
                f"x {N_LABELS} labels"
            )
        if not np.isfinite(self.scores).all():
            raise ValueError(
                "prediction contains a non-finite score. A NaN cannot be ranked and a "
                "submission carrying one is rejected outright, so this fails here rather "
                "than at upload."
            )

    @property
    def n_studies(self) -> int:
        return len(self.study_uids)

    @property
    def n_passes(self) -> int:
        """Forward passes per study. ``1`` with no TTA, ``len(tta)`` with it."""
        return max(len(self.tta), 1)

    @property
    def n_forward_batches(self) -> int:
        """Batches the *model* saw, i.e. loader batches times TTA views.

        ``n_batches`` counts what the loader produced; this counts what was paid for.
        They differ by the TTA multiplier and quoting the wrong one understates the
        cost by exactly that factor.
        """
        return self.n_batches * self.n_passes

    @property
    def seconds_per_study(self) -> float:
        """Measured, not estimated. The one number the efficiency term needs.

        Already inclusive of every TTA view: the timer in :func:`predict_studies` wraps
        the whole pass, so a three-view run reports roughly three times the rate of a
        one-view run rather than the rate of one of its views. Dividing this by
        ``n_passes`` to get a "per view" figure would produce a number that no
        submission can achieve.
        """
        return self.seconds / max(self.n_studies, 1)

    def projection(self, n_studies: int) -> tuple[float, float]:
        """``(projected seconds, runtime term)`` for a full test set of ``n_studies``.

        The runtime term is ``T / 32400``. Linear in the study count on purpose: unlike
        the vLLM extraction path, whose throughput is measured to improve 7x from n=5 to
        n=300, a fixed-shape image batch has no batch-composition effect to extrapolate
        wrongly. It is still a projection, not a measurement, and the caller should say
        so when quoting it.
        """
        seconds = self.seconds_per_study * float(n_studies)
        return seconds, seconds / NOTEBOOK_LIMIT_SECONDS

    def timing_report(self, n_studies: int | None = None) -> str:
        """One block stating what was measured and what was only projected from it.

        With no TTA the block is byte-for-byte what it was before TTA existed. The extra
        line only appears when extra passes were actually paid for, so a plain run's log
        cannot be misread as a TTA run's.
        """
        lines = [
            f"inference  : {self.seconds:.2f} s for {self.n_studies} studies "
            f"({self.n_batches} batches) on {self.device}, amp={self.amp}",
            f"           : {self.seconds_per_study:.4f} s/study MEASURED",
        ]
        if self.tta:
            lines.append(
                f"tta        : {self.n_passes} view(s) {list(self.tta)}, averaged in "
                "logit space (never by rank -- see predict_studies)"
            )
            lines.append(
                f"           : {self.n_forward_batches} forward batches; the rate above "
                "already includes all of them"
            )
        if n_studies:
            seconds, term = self.projection(n_studies)
            lines.append(
                f"projection : {seconds:.1f} s for {n_studies} studies = "
                f"{100.0 * seconds / NOTEBOOK_LIMIT_SECONDS:.1f}% of the "
                f"{NOTEBOOK_LIMIT_SECONDS} s notebook limit, runtime term {term:.4f}"
            )
            lines.append(
                "           : ^ projected from the measured rate, not measured at that size"
            )
        return "\n".join(lines)

    def frame(self) -> pd.DataFrame:
        """``StudyInstanceUID`` plus the twelve score columns, in submission order."""
        data: dict[str, Any] = {ID_COLUMN: list(self.study_uids)}
        for position, name in enumerate(LABELS):
            data[name] = self.scores[:, position]
        return pd.DataFrame(data)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A model restored from disk, together with the preprocessing it was trained with.

    The preprocessing is the point. Building a dataset from anything other than
    :attr:`normalisation` and :attr:`crop` gives the backbone pixels it was not trained
    on, which trains nothing and reports a number -- so :meth:`dataset` is the supported
    way to build a loader and it takes no normalisation argument at all.
    """

    model: StudyClassifier
    normalisation: Normalisation
    crop: CropSpec
    slots: tuple[int, ...]
    n_slices: int
    meta: dict[str, Any]
    path: Path

    @property
    def backbone(self) -> str:
        return str(self.meta.get("backbone", "unknown"))

    @property
    def slice_mode(self) -> str:
        """Which slice sampler this checkpoint was TRAINED with.

        Read from ``meta`` rather than from a top-level key, so no checkpoint version
        had to move: every file written before the volumetric arm existed is a
        ``spanning`` run, which is exactly what the default says. Scoring a
        contiguous-trained model with the spanning sampler would feed it a
        full-extent spread it never saw and still print an AUC, so this is not
        cosmetic -- it is the same class of failure as a wrong ``normalisation``, and
        it is why :meth:`dataset` takes no sampler argument.
        """
        return str(self.meta.get("slice_mode", "spanning"))

    def describe(self) -> str:
        """What this checkpoint is evidence about. Print it before scoring anything."""
        return "\n".join(
            [
                f"checkpoint : {self.path}",
                f"backbone   : {self.backbone}  (weights from the checkpoint, "
                f"pretrained=False so nothing is fetched)",
                f"trained as : config {self.meta.get('config_hash')} seed "
                f"{self.meta.get('seed')} fold {self.meta.get('fold')} "
                f"epoch {self.meta.get('epoch')} sha {self.meta.get('git_sha')}",
                f"sampling   : {self.slice_mode} over {self.n_slices} slices"
                + (
                    "  (VOLUMETRIC: n_slices is the depth D)"
                    if self.meta.get("volumetric")
                    else ""
                ),
                f"input      : slots={list(self.slots)} n_slices={self.n_slices} "
                f"{self.crop.fov_mm:.0f} mm / {self.crop.out_px} px = "
                f"{self.crop.mm_per_pixel:.4f} mm/px",
                f"normalise  : mean={self.normalisation.mean} std={self.normalisation.std} "
                f"(from the checkpoint, never re-resolved)",
            ]
        )

    def dataset(
        self,
        manifest: pd.DataFrame,
        cache_dir: Path | str,
        *,
        study_uids: Sequence[str] | None = None,
        verify_cache: bool = True,
    ) -> StudyDataset:
        """Build an inference dataset with this checkpoint's own geometry.

        ``training=False`` and ``series_dropout=0.0`` are not arguments: dropping a
        series at inference would throw away signal for no benefit, and augmentation
        at inference is a different model from the one that was validated.
        """
        return StudyDataset(
            manifest,
            cache_dir,
            normalisation=self.normalisation,
            labels=None,
            study_uids=study_uids,
            slots=self.slots,
            n_slices=self.n_slices,
            crop=self.crop,
            training=False,
            series_dropout=0.0,
            verify_cache=verify_cache,
            slice_mode=self.slice_mode,
        )


def save_inference_checkpoint(
    path: Path | str,
    model: StudyClassifier,
    *,
    normalisation: Normalisation,
    crop: CropSpec,
    slots: Sequence[int],
    n_slices: int,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint that carries everything inference needs and nothing it does not.

    Deliberately no optimiser or scheduler state. This file is read by an offline
    notebook that only scores, so shipping optimiser moments would double its size for
    a tensor nobody loads. Training resumption is a separate concern and there is no
    resumable long run here -- the schedule is ten epochs, not thirty.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kappa_checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "normalisation": {
            "mean": list(normalisation.mean),
            "std": list(normalisation.std),
        },
        "crop": {"fov_mm": float(crop.fov_mm), "out_px": int(crop.out_px)},
        "slots": [int(s) for s in slots],
        "n_slices": int(n_slices),
        "meta": dict(meta or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> LoadedCheckpoint:
    """Rebuild the model and its preprocessing from a checkpoint, touching no network.

    ``pretrained=False`` unconditionally: every weight comes from the file. Asking timm
    for published weights here would try to reach the hub, which fails on a compute node
    and inside a Kaggle notebook -- and it would fail *after* the architecture was
    built, so the traceback would blame the backbone rather than the network.
    """
    path = Path(path)
    resolved = resolve_device(device)
    # weights_only=True refuses to unpickle arbitrary objects. This payload is tensors
    # and plain containers by construction, so the safe loader is sufficient, and using
    # it means a checkpoint from an untrusted source cannot execute code.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a KAPPA checkpoint mapping")
    # A teacher reads report text, which no test study has. Its
    # state dict carries a report branch and a fusion layer this architecture has
    # no slot for, so a strict load would fail here anyway -- but it would fail
    # with a wall of unexpected-key names that reads like a version mismatch. Say
    # the real thing instead: the wrong kind of checkpoint was pointed at
    # inference, and the fix is to submit the student.
    if payload.get("kappa_teacher"):
        raise ValueError(
            f"{path} is a teacher checkpoint and cannot be used for inference. The "
            "teacher reads report features, which test studies do not have. Submit "
            "the distilled student instead."
        )

    version = payload.get("kappa_checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} has checkpoint version {version!r}, this build reads "
            f"{CHECKPOINT_VERSION}. Refusing rather than guessing: the key most likely "
            "to be missing is 'normalisation', and a wrong mean/std still produces "
            "scores."
        )
    for key in ("state_dict", "normalisation", "crop", "slots", "n_slices"):
        if key not in payload:
            raise ValueError(f"{path} is missing the {key!r} key")

    meta = dict(payload.get("meta") or {})
    model = StudyClassifier(
        backbone=str(meta.get("backbone", "convnext_small")),
        n_slices=int(payload["n_slices"]),
        pretrained=False,
        drop_path_rate=float(meta.get("drop_path_rate", 0.0)),
        aux_dim=int(meta.get("aux_dim", 0)),
        # Every one of these four defaults to what the architecture was BEFORE the key
        # existed, which is what keeps the checkpoints already on the cluster loadable:
        # none of them carries `head`, `per_label_slots` or `head_residual`, and every
        # one of them was trained with the shared linear readout these defaults rebuild.
        # A checkpoint written after `pool` landed carries all four, so a per-label run
        # rebuilds as a per-label model rather than loading into the wrong shape and
        # failing on `study_pool.label_slot_logits` -- or, with `strict=False`,
        # succeeding and scoring with an untrained table.
        head=str(meta.get("head", "linear")),
        head_num_heads=int(meta.get("head_num_heads", 8)),
        per_label_slots=bool(meta.get("per_label_slots", False)),
        head_residual=bool(meta.get("head_residual", False)),
        # Taken from `crop.out_px`, which IS the input matrix -- the dataset resizes
        # every crop to exactly that -- so there is nothing to add to `meta` and no
        # checkpoint version to bump. It matters because a ViT is built for one token
        # grid: rebuilding a 392 px DINOv2 checkpoint at its native 518 loads every
        # weight successfully, reports nothing wrong, and then asserts on the first
        # batch -- inside the Kaggle notebook, with the 9 h budget already spent.
        img_size=int(payload["crop"]["out_px"]),
    )
    model.load_state_dict(payload["state_dict"], strict=strict)
    model.eval().to(resolved)

    return LoadedCheckpoint(
        model=model,
        normalisation=Normalisation(
            tuple(float(v) for v in payload["normalisation"]["mean"]),
            tuple(float(v) for v in payload["normalisation"]["std"]),
        ),
        crop=CropSpec(
            fov_mm=float(payload["crop"]["fov_mm"]), out_px=int(payload["crop"]["out_px"])
        ),
        slots=tuple(int(s) for s in payload["slots"]),
        n_slices=int(payload["n_slices"]),
        meta=meta,
        path=path,
    )


def _forward_logits(
    model: StudyClassifier,
    pixels: torch.Tensor,
    present: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    device_type: str,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """One forward pass, returned as float32 logits still on the device.

    float32 before anything else happens to them: a bf16 logit carries about three
    decimal digits, ROC-AUC reads ties, and keeping reduced precision would manufacture
    ties between studies the model actually separated. Under TTA it would also make the
    mean of the views depend on the order they were summed in.
    """
    if dtype is None:
        out = model(pixels, present, slot_ids)
    else:
        with torch.autocast(device_type=device_type, dtype=dtype):
            out = model(pixels, present, slot_ids)
    return out.logits.detach().float()


@torch.no_grad()
def predict_studies(
    model: StudyClassifier,
    dataset: StudyDataset,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    amp: bool | str | None = None,
    tta: Sequence[str] = (),
) -> PredictionResult:
    """Score every study in ``dataset`` and measure how long the whole pass took.

    The same function runs the training loop's validation pass and the Kaggle notebook's
    scoring pass, so the seconds-per-study reported by a training run is measured by the
    code that will actually be timed. Two separate implementations would drift, and the
    one that drifted would be the one nobody times. It is also why ``tta`` is a
    parameter *here* rather than a block of cells in the notebook: a TTA that existed
    only at submission time would make the reported CV describe a pipeline that never
    runs.

    The timer starts before the ``DataLoader`` is iterated and stops after the last
    batch, so decoding, cropping and collation are inside it. Timing only the forward
    call underestimated a fold's wall clock by 1.7x three times running on this project;
    the same mistake at inference underestimates the runtime term. With ``tta`` set the
    timer still wraps everything, so ``seconds_per_study`` is the real multi-view rate;
    three views cost roughly three times one view against a 32,400 s limit, and that is
    a trade to make deliberately.

    ``tta`` names views from :data:`TTA_TRANSFORMS`, in any order, without repeats.
    ``()`` -- the default -- takes the plain single-pass branch below, which performs
    the same operations in the same order as the code this function ran before TTA
    existed, and returns bit-identical scores; ``tests/test_tta.py`` asserts that
    equality rather than a tolerance. The loader is iterated **once** whichever branch
    runs: decoding and cropping dominate a CPU pass, and re-reading the cache per view
    would multiply the part of the cost that TTA has no reason to pay.

    **The views are pooled in logit space, not by rank, and that is a considered
    exception rather than an oversight.**
    This project's standing ruling is "per-column rank normalisation, never average
    probabilities, never calibrate". It is the right rule for *ensembling members* and
    it does not reach inside a single model's TTA, for four reasons.

    1. **The premise is absent.** The ruling exists because two members are two
       independent training runs whose logit scales are incomparable, so a mean is
       silently dominated by whichever run happens to be more spread out -- influence
       bought with confidence, which the competition never scores. TTA views share one
       set of weights and one output head. There is no second scale to reconcile; the
       thing rank-averaging protects against cannot happen here.
    2. **Rank normalisation is transductive and this function is not.** A study's rank
       depends on which *other* studies were in the same call. Pooling views by rank
       would make one study's score a function of its batch-mates: score a single study
       on its own and every column collapses to 0.5, in silence, with no exception and
       a perfectly well-formed submission. Logit averaging is per-study and local, so
       chunking the test set cannot change an answer.
    3. **``scores`` is contractually raw logits, and the caller concatenates.**
       ``kappa.training.loop`` stacks each fold's ``scores`` into one OOF table. If this
       function returned fold-internal ranks, the concatenation would hold five
       incomparable rank scales and a global OOF AUC over it would be a plausible
       meaningless number -- precisely the failure mode this file exists to refuse.
    4. **The rank step still happens, once, where it belongs.** ``transform_scores`` and
       ``rank_blend`` rank-normalise at write time over the full scored set. Ranking a
       second time inside TTA adds nothing the ruling asks for.

    None of that makes the two equivalent -- a mean of logits and a mean of ranks are
    different combinations and macro AUC will differ between them, so this is a real
    choice. A mean of logits is a log-odds pool over correlated views of one subject,
    which is the right granularity for TTA, and its failure modes are local rather than
    set-dependent.
    """
    views = _tta_views(tta)
    resolved = resolve_device(device)
    dtype = _autocast_dtype(resolved, amp)
    amp_label = "off" if dtype is None else f"{resolved.type}/{str(dtype).split('.')[-1]}"
    was_training = model.training
    model.eval().to(resolved)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(pin_memory) if pin_memory is not None else resolved.type == "cuda",
        collate_fn=collate_studies,
        drop_last=False,
    )

    uids: list[str] = []
    chunks: list[np.ndarray] = []
    n_batches = 0
    started = time.perf_counter()
    for batch in loader:
        pixels = batch["pixels"].to(resolved, non_blocking=True)
        present = batch["present"].to(resolved, non_blocking=True)
        slot_ids = batch["slot_ids"].to(resolved, non_blocking=True)
        if not views:
            # Kept as its own branch rather than folded into the loop below with an
            # identity view. The default has to be the pre-TTA arithmetic exactly --
            # one extra tensor op here is a change to every number this project has
            # already measured, and it would arrive with no config to attribute it to.
            logits = _forward_logits(
                model, pixels, present, slot_ids, device_type=resolved.type, dtype=dtype
            )
        else:
            # The view runs outside autocast, on the float32 pixels the loader
            # produced. Resampling at bf16's ~3 decimal digits would blur the geometry
            # the view exists to perturb, and the cost is a rounding error next to the
            # forward pass it feeds.
            total = None
            for view in views:
                seen = _forward_logits(
                    model,
                    view(pixels),
                    present,
                    slot_ids,
                    device_type=resolved.type,
                    dtype=dtype,
                )
                total = seen if total is None else total + seen
            # Mean of logits, not of ranks -- see this function's docstring for why the
            # ensembling ruling stops at the model boundary. Dividing by 1.0 is exact,
            # so a single-view tta gives bit-identical scores to no tta at all.
            logits = total / float(len(views))
        chunks.append(logits.cpu().numpy())
        uids.extend(batch["study_uid"])
        n_batches += 1
    seconds = time.perf_counter() - started

    if was_training:
        model.train()
    scores = (
        np.concatenate(chunks, axis=0) if chunks else np.zeros((0, N_LABELS), dtype=np.float32)
    )
    return PredictionResult(
        study_uids=tuple(uids),
        scores=np.ascontiguousarray(scores, dtype=np.float64),
        seconds=seconds,
        n_batches=n_batches,
        device=str(resolved),
        amp=amp_label,
        tta=tuple(str(name) for name in tta),
    )


def transform_scores(
    scores: np.ndarray, transform: Literal["rank", "none"] = "rank"
) -> np.ndarray:
    """Prepare logits for writing. ``"rank"`` is per-column rank normalisation.

    Both options give a **bit-identical** macro ROC-AUC, because both are monotone
    within a column and the metric reads nothing else. ``"rank"`` is the default only
    because it lands in ``[0, 1]``, so no downstream reader can object to the range.
    Neither is calibration: there is no free parameter fitted to anything here, and
    calibration cannot move a rank-based metric in the first place.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if transform == "none":
        return scores
    if transform != "rank":
        raise ValueError(f"unknown transform {transform!r}; use 'rank' or 'none'")
    if scores.shape[0] == 0:
        return scores
    return rank_normalise(scores)


def write_predictions(
    path: Path | str,
    results: Sequence[PredictionResult],
    *,
    transform: Literal["rank", "none"] = "rank",
    expected_uids: Sequence[str] | None = None,
) -> Path:
    """Blend one or more members by rank and write ``submission.csv``.

    Several members are combined with :func:`kappa.inference.submission.rank_blend` --
    per-column rank normalisation, then a mean of the ranks. Never a mean of the
    probabilities: that blend is dominated by whichever member happens to have the
    widest spread, which is a property this competition does not score.

    ``expected_uids`` reorders the rows to the sample submission's order. The
    leaderboard joins on ``StudyInstanceUID``, but a row-order mismatch is the kind of
    thing that is free to fix here and expensive to discover later, so it is checked.
    """
    if not results:
        raise ValueError("write_predictions needs at least one PredictionResult")
    uid_lists = [result.study_uids for result in results]
    if len({tuple(u) for u in uid_lists}) != 1:
        raise ValueError(
            "members disagree on their study order; rank_blend combines by position, so "
            "a differing order would blend one study's rank with another's"
        )

    uids = list(uid_lists[0])
    if len(results) == 1:
        blended = transform_scores(results[0].scores, transform)
    else:
        # rank_blend already rank-normalises each member, so the blend is in [0, 1]
        # whatever `transform` says; applying it again would be a monotone no-op.
        blended = rank_blend([result.scores for result in results])

    if expected_uids is not None:
        wanted = [str(u) for u in expected_uids]
        position = {uid: index for index, uid in enumerate(uids)}
        missing = [uid for uid in wanted if uid not in position]
        if missing:
            raise ValueError(
                f"{len(missing)} expected study UID(s) were not scored, e.g. {missing[:3]}. "
                "Every test study needs a row: a short submission scores as an error."
            )
        order = [position[uid] for uid in wanted]
        blended = blended[order]
        uids = wanted

    return write_submission(path, uids, blended)
