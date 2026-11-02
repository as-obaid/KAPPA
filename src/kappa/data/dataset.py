"""Study-level batches whose presence mask and label mask are both honest.

Why this exists: a knee study is not one image and its label row is not twelve
numbers. It is up to six plane x contrast slots, of which only ``Axial_fs1`` is
present in every study (``Axial_fs0`` is present in 19.4%), and its twelve
labels are 98.7% unknown at the cell level. Both facts are silent killers if the
tensor handed to the model hides them:

* **Zero-fill without a flag.** An absent slot still has to occupy its place in
  a fixed-shape tensor, so it is filled with zeros. If the batch then claims the
  slot is present, the model learns from a black volume it believes is anatomy.
  This module always emits ``present=False`` for such a slot, and the three
  features -- zero-fill, an honest ``present``, and ~10% random series dropout
  during training -- are **one design, not three options**. Zero-fill plus an
  honest flag with no dropout means the encoder never sees an empty slot while
  training and has no reason to learn what one looks like, so the flag stops
  carrying information at inference. Constructing this dataset with
  ``training=True``, more than one slot and ``series_dropout=0.0`` warns for
  exactly that reason.
* **NaN in the labels.** ``BCEWithLogitsLoss`` (and every other loss) propagates
  NaN silently: the run trains, the loss prints ``nan``, and nothing raises.
  ``labels`` is therefore guaranteed finite *everywhere*, including cells where
  ``mask`` is False, where it is filled with 0.0 and is meaningless by contract.
  Both ``__getitem__`` and :func:`collate_studies` raise on a non-finite label,
  so the guarantee cannot be lost by editing one of them.

Two geometry decisions are also load-bearing:

* **A physical-millimetre crop, never a fixed pixel crop.** In-plane spacing in
  this corpus spans 0.14-0.89 mm/px (median 0.31) over 13 distinct aspect
  ratios, so a fixed 384px centre crop makes the knee fill a different fraction
  of the frame at every site. :class:`CropSpec` fixes the *field of view* in
  millimetres and lets the pixel crop float, which is the only way the encoder
  sees one scale. The cache is letterboxed to square before being resized, so
  the cached pixel size is ``spacing * max(rows, cols) / cache_size`` and not
  ``spacing`` -- see :func:`cache_mm_per_pixel`.
* **Normalisation is a required argument.** There is no ImageNet default here on
  purpose. A backbone expecting greyscale statistics fed ImageNet constants
  still trains and still reports a number; it just reports the wrong one. Pass
  ``Normalisation.from_data_config(timm.data.resolve_data_config(...))``, or
  ``Normalisation.identity()`` when you are testing wiring rather than accuracy.

    from kappa.data.dataset import CropSpec, Normalisation, StudyDataset, collate_studies

    train = StudyDataset.from_paths(
        manifest_path, cache_dir, labels_path,
        normalisation=Normalisation.from_data_config(cfg),
        slots=(1,),               # Axial_fs1 only: 100% coverage, no fusion
        training=True, series_dropout=0.1,
    )
    loader = DataLoader(train, batch_size=8, collate_fn=collate_studies)

Budget the memory before choosing a batch size: ``pixels`` costs
``len(slots) * n_slices * 3 * out_px**2 * 4`` bytes per study, which is 162 MiB
at the six-slot defaults and 27 MiB for a single slot -- both confirmed by
measurement, along with 18.5 ms and 94.6 ms per study respectively to build one
from a 64px synthetic cache on CPU. A batch of 8 single-slot studies is 216 MiB
of pixels, so `num_workers` costs real memory and the six-slot config wants
either a smaller `out_px` or gradient accumulation rather than a bigger batch.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from kappa.constants import ID_COLUMN, LABELS, N_LABELS

__all__ = [
    "SLOT_ORDER",
    "N_SLOTS",
    "Normalisation",
    "CropSpec",
    "StudyDataset",
    "collate_studies",
    "cache_mm_per_pixel",
    "sample_slice_indices",
    "contiguous_slice_indices",
    "SLICE_SAMPLERS",
    "neighbour_indices",
]

# kappa.data.manifest is the canonical home for the slot taxonomy. This was once a
# try/except with a duplicated SLOT_ORDER in the fallback, for the window before
# manifest.py landed. That window closed, and the branch went on to do harm: when
# manifest.py imported kappa.labels.normalise -- a report-side import on the
# image-only inference path -- the except arm swallowed the ImportError and inference carried on
# against the second copy, so the boundary broke without a single test going red. A
# hard import is the point: if the slot taxonomy cannot be loaded, that must stop the
# run rather than quietly substitute a copy that no longer has to agree.
from kappa.data.manifest import N_SLOTS, SLOT_ORDER

# The manifest keys studies as `study_uid` while the label table and the
# submission key them as `StudyInstanceUID` (kappa.constants.ID_COLUMN). Two
# spellings of one identifier is a real trap, so both are named explicitly here
# rather than one being assumed.
ID_COLUMN_MANIFEST = "study_uid"


@dataclass(frozen=True)
class Normalisation:
    """Per-channel mean and standard deviation, taken from the backbone.

    A single-element ``mean``/``std`` is broadcast across the three 2.5D
    channels, which is what greyscale-pretrained medical backbones report.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or not self.std:
            raise ValueError("normalisation mean and std must be non-empty")
        if any(s == 0.0 for s in self.std):
            raise ValueError(f"normalisation std contains zero: {self.std}")

    @classmethod
    def identity(cls) -> Normalisation:
        """Leave pixels in [0, 1]. For wiring tests, never for a reported score."""
        return cls((0.0,), (1.0,))

    @classmethod
    def from_data_config(cls, config: Mapping[str, Any]) -> Normalisation:
        """Build from a timm resolved data config, i.e. its ``mean``/``std`` keys."""
        try:
            mean, std = config["mean"], config["std"]
        except KeyError as exc:  # a config without these keys is not a data config
            raise KeyError(f"data config has no {exc} key; got keys {sorted(config)}") from exc
        return cls(tuple(float(v) for v in mean), tuple(float(v) for v in std))

    def tensors(self, channels: int = 3) -> tuple[Tensor, Tensor]:
        """Return ``(mean, std)`` shaped ``(channels, 1, 1)`` for broadcasting."""
        mean = _broadcast(self.mean, channels, "mean")
        std = _broadcast(self.std, channels, "std")
        return mean.view(channels, 1, 1), std.view(channels, 1, 1)


def _broadcast(values: Sequence[float], channels: int, name: str) -> Tensor:
    if len(values) == channels:
        return torch.tensor(values, dtype=torch.float32)
    if len(values) == 1:
        return torch.full((channels,), float(values[0]), dtype=torch.float32)
    raise ValueError(f"normalisation {name} has {len(values)} entries, need 1 or {channels}")


@dataclass(frozen=True)
class CropSpec:
    """A centre crop of ``fov_mm`` millimetres resampled to ``out_px`` pixels.

    The defaults give 150 mm at 384 px, i.e. 0.390625 mm/px, chosen so a knee
    fills the frame at every site's spacing rather than at the median one.
    """

    fov_mm: float = 150.0
    out_px: int = 384

    def __post_init__(self) -> None:
        if self.fov_mm <= 0.0 or self.out_px < 1:
            raise ValueError(f"CropSpec needs fov_mm > 0 and out_px >= 1, got {self}")

    @property
    def mm_per_pixel(self) -> float:
        """Physical size of one output pixel. Identical for every series, by design."""
        return self.fov_mm / self.out_px


def cache_mm_per_pixel(spacing: float, rows: int, cols: int, cache_size: int) -> float:
    """Millimetres per pixel of a cached slice.

    The cache letterboxes each slice to a square of ``max(rows, cols)`` before
    resizing to ``cache_size``, so the cached pixel is *not* ``spacing`` unless
    the source happened to be square at exactly ``cache_size``. Getting this
    wrong scales every crop by the aspect ratio and is invisible in the images.
    """
    if spacing <= 0.0 or rows < 1 or cols < 1 or cache_size < 1:
        raise ValueError(f"bad geometry: spacing={spacing} rows={rows} cols={cols}")
    return float(spacing) * float(max(int(rows), int(cols))) / float(cache_size)


def sample_slice_indices(n_available: int, n_slices: int) -> np.ndarray:
    """``n_slices`` indices at equal intervals across the full geometric extent.

    The cache is already ordered by the projection of ``ImagePositionPatient``
    onto the slice normal, so position in the array *is* geometric order and no
    re-sorting happens here. Slice counts run 11-320 with a median of 30, so a
    series shorter than ``n_slices`` repeats indices rather than padding: a
    duplicated real slice is closer to the truth than a black one.
    """
    if n_available < 1:
        raise ValueError(f"n_available must be >= 1, got {n_available}")
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    if n_available == 1:
        return np.zeros(n_slices, dtype=np.int64)
    return np.rint(np.linspace(0.0, n_available - 1, n_slices)).astype(np.int64)


def contiguous_slice_indices(n_available: int, n_slices: int) -> np.ndarray:
    """``n_slices`` **adjacent** indices, centred on the middle of the series.

    The volumetric arm's sampler, and the opposite trade to
    :func:`sample_slice_indices`. Position in the cached array is geometric order --
    the cache is written sorted by the projection of ``ImagePositionPatient`` onto the
    slice normal -- so adjacent indices are adjacent anatomy and the returned block is
    non-decreasing by construction.

    **THIS TRADES COVERAGE FOR CONTIGUITY, AND THE TRADE IS THE EXPERIMENT.** Corpus
    median inter-slice spacing is 3.50 mm over a 105.60 mm through-plane span, so a block of ``n_slices`` covers
    ``(n_slices - 1) * 3.50 mm``: 24.5 mm at 8, 52.5 mm at 16, 80.5 mm at 24. The
    spanning sampler covers all 105.60 mm at every depth. A depth sweep built on this
    function therefore varies depth **and** coverage together, and a monotone result
    is not by itself evidence about volumetric context -- see
    :class:`kappa.config.VolumeConfig`, which is where the control arm is described.

    **A series shorter than ``n_slices`` falls back to the spanning sampler**, and that
    is not a shortcut. When the whole series is shorter than the requested block, the
    contiguous block *is* the whole series, so the only question left is how to repeat
    indices to reach length ``n_slices`` -- and edge replication answers it badly. On
    the shortest real series (12 slices) at depth 32, clipping an ``arange`` yields ten
    copies of slice 0 and eleven of slice 11 with each interior plane appearing once:
    two thirds of the volume would be two duplicated end planes, and the 3D kernels
    would see a constant slab at each end that no real acquisition ever produces.
    Delegating to :func:`sample_slice_indices` covers the identical extent -- every
    slice, and no others -- while spreading the repeats evenly, which is also exactly
    what this loader has always done for a short series. Same extent, same convention,
    no fabricated slab.

    Replication of any kind means the study contributes fewer *distinct* planes than
    the config asked for. Measured over the 4,719 ``Axial_fs1`` series of the real
    manifest: min 12, median 32, and the fraction shorter than the
    requested depth is **0.00% at 8, 1.36% at 16, 15.47% at 24 and 48.46% at 32**. That
    last figure is not a corner case -- it is nearly half the corpus, and it follows
    directly from the median being 32. :class:`StudyDataset` counts it and the run log
    prints the count, because a depth sweep whose deepest run is quietly a
    replication sweep is not a depth sweep.

    **The block is contiguous in CACHE index, which is acquisition order only up to
    the cache's own cap.** ``kappa.data.dicom.select_slice_indices`` decimates a series
    longer than ``MAX_SLICES`` (64) at equal intervals before it is written, so for
    such a series a "contiguous" block is a uniformly stretched one. Measured on the
    same 4,719 series: 6.72% exceed 64 slices, so this affects roughly one
    ``Axial_fs1`` series in fifteen. It is a property of the cache, not of this
    function, and re-caching at a higher cap is the only fix.
    """
    if n_available < 1:
        raise ValueError(f"n_available must be >= 1, got {n_available}")
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    if n_available <= n_slices:
        # The whole series is the block. See the docstring: this is the same extent an
        # arange-and-clip would give, without the duplicated end slabs.
        return sample_slice_indices(n_available, n_slices)
    # Floor division, so an odd leftover slice goes to the high-index end rather than
    # being split. Deterministic and stated: a tie-break that varied would make two
    # runs of one config sample different anatomy while reporting the same config.
    start = (n_available - n_slices) // 2
    return np.arange(start, start + n_slices, dtype=np.int64)


#: The slice samplers, by ``kappa.config.VolumeConfig.mode``. A mapping rather than an
#: ``if`` so the legal modes have exactly one spelling, shared by the config and the
#: loader: a mode the config accepts and the loader does not know is precisely the
#: silent mismatch this project keeps paying for.
SLICE_SAMPLERS = {
    "spanning": sample_slice_indices,
    "contiguous": contiguous_slice_indices,
}


def neighbour_indices(centres: Iterable[int], n_available: int) -> np.ndarray:
    """``(len(centres), 3)`` indices of ``(previous, centre, next)`` per centre.

    Edges replicate: the first slice has no predecessor, so its stack is
    ``(0, 0, 1)``, and the last is ``(S-2, S-1, S-1)``. Replication rather than
    zero padding, because a black channel at the edge tells the encoder the
    anatomy ended, which is false -- the volume ended.
    """
    if n_available < 1:
        raise ValueError(f"n_available must be >= 1, got {n_available}")
    centre = np.asarray(list(centres), dtype=np.int64)
    if centre.size and (centre.min() < 0 or centre.max() >= n_available):
        raise ValueError(f"centre index outside [0, {n_available - 1}]")
    stacked = np.stack([centre - 1, centre, centre + 1], axis=1)
    return np.clip(stacked, 0, n_available - 1)


def _crop_to_fov(volume: Tensor, mm_per_px: float, crop: CropSpec) -> Tensor:
    """Centre-crop ``volume`` (N, C, H, W) to ``crop.fov_mm`` then resize to ``out_px``.

    When the requested field of view is wider than the image -- which happens at
    the fine end of the 0.14 mm/px spacing range -- the crop is zero padded
    rather than shrunk. Shrinking would silently restore the fixed-pixel-crop
    bug this function exists to remove.
    """
    size = int(volume.shape[-1])
    side = max(1, int(round(crop.fov_mm / mm_per_px)))
    start = (size - side) // 2
    if start < 0:
        pad = -start
        volume = F.pad(volume, (pad, pad, pad, pad))
        start = (int(volume.shape[-1]) - side) // 2
    volume = volume[..., start : start + side, start : start + side]
    if side != crop.out_px:
        volume = F.interpolate(
            volume,
            size=(crop.out_px, crop.out_px),
            mode="bilinear",
            align_corners=False,
            antialias=side > crop.out_px,
        )
    return volume


@dataclass(frozen=True)
class _SeriesRef:
    """Everything needed to load one slot, resolved once at construction."""

    series_uid: str
    path: Path
    spacing: float
    rows: int
    cols: int


class StudyDataset(Dataset):
    """One item per study: up to six slot volumes, twelve labels, two masks.

    ``__getitem__`` returns the per-study half of the pinned batch contract::

        pixels    FloatTensor (n_slots, n_slices, 3, out_px, out_px)
        present   BoolTensor  (n_slots,)   False = the study has no such series
        slot_ids  LongTensor  (n_slots,)   global slot ids, 0..5
        labels    FloatTensor (12,)        finite everywhere; 0.0 where mask is False
        mask      BoolTensor  (12,)        False = excluded from the loss
        study_uid str

    ``slot_ids`` carries the *global* slot index even when ``slots`` selects a
    subset, so a slot embedding trained on one slot (``slots=(1,)``) stays valid
    when all six are used. Widening the run is a config change, not a reindexing.

    Duplicate series in one slot are resolved to the one with the most cached
    slices, tie-broken by ``series_uid``, so the choice does not drift between
    a training run and its inference run.

    ``slice_mode`` picks which ``n_slices`` indices are taken from the cached series:
    ``"spanning"`` (the default, and what this loader has always done) spreads them
    across the full geometric extent, while ``"contiguous"`` takes an adjacent block
    centred on the series middle for the volumetric arm. **The batch shape is
    identical either way** -- only the indices move -- so nothing downstream has to
    know which was used, and the 2.5D path is untouched by the option existing. See
    :func:`contiguous_slice_indices` for the coverage that the contiguous mode trades
    away, and :class:`kappa.config.VolumeConfig` for why that trade needs a control.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: Path | str,
        *,
        normalisation: Normalisation,
        labels: pd.DataFrame | None = None,
        study_uids: Sequence[str] | None = None,
        slots: Sequence[int] | None = None,
        n_slices: int = 16,
        crop: CropSpec | None = None,
        training: bool = False,
        series_dropout: float = 0.0,
        augment: Callable[[Tensor, np.random.Generator], Tensor] | None = None,
        seed: int = 0,
        verify_cache: bool = True,
        slice_mode: str = "spanning",
    ) -> None:
        for column in (ID_COLUMN_MANIFEST, "series_uid", "slot", "spacing", "rows", "cols"):
            if column not in manifest.columns:
                raise KeyError(f"manifest is missing the {column!r} column")
        if not 0.0 <= series_dropout <= 1.0:
            raise ValueError(f"series_dropout must be in [0, 1], got {series_dropout}")
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        if slice_mode not in SLICE_SAMPLERS:
            raise ValueError(
                f"unknown slice_mode {slice_mode!r}; known: {sorted(SLICE_SAMPLERS)}"
            )

        self.cache_dir = Path(cache_dir)
        self.crop = crop if crop is not None else CropSpec()
        self.n_slices = int(n_slices)
        # Defaults to the sampler the loader has always used, so every existing caller
        # -- inference, the report dataset, the synthetic corpus, every 2.5D config --
        # is unchanged by the volumetric arm existing.
        self.slice_mode = str(slice_mode)
        self._sample_indices = SLICE_SAMPLERS[self.slice_mode]
        # Series too short for `n_slices` under `contiguous`, counted rather than
        # warned per item: a per-study warning over ~3,000 studies is noise nobody
        # reads, and a silent replication is exactly what this project refuses. Read
        # it after an epoch via `short_series`; the training loop prints it.
        self._short_series: Counter[str] = Counter()
        self.normalisation = normalisation
        self.training = bool(training)
        self.series_dropout = float(series_dropout)
        self.seed = int(seed)
        self._augment = augment
        self._epoch = 0
        self._mean, self._std = normalisation.tensors(3)

        self._slots = _validate_slots(slots)
        self._slot_ids = torch.tensor(self._slots, dtype=torch.long)
        self._series = self._index_series(manifest, verify_cache=verify_cache)

        if study_uids is None:
            uids = sorted(str(u) for u in manifest[ID_COLUMN_MANIFEST].unique())
        else:
            uids = [str(u) for u in study_uids]
        self._study_uids = tuple(uids)

        self._label_index, self._label_values, self._label_mask = _index_labels(labels)

        # A study with no cached series in the selected slots is emitted honestly
        # -- zeros with present=False -- but it carries no signal at all, and the
        # usual cause is a half-built cache rather than a genuinely empty study.
        # Silence here means a run trains on black volumes and reports a number.
        self._empty_study_uids = tuple(uid for uid in self._study_uids if not self._series.get(uid))
        if self._empty_study_uids:
            warnings.warn(
                f"{len(self._empty_study_uids)} of {len(self._study_uids)} studies have no "
                f"cached series in slots {self._slots} (e.g. {self._empty_study_uids[0]}); "
                "they are emitted as zeros with present=False everywhere, so they contribute "
                "no signal. Check the cache is complete before reading the score.",
                stacklevel=2,
            )

        if self.training and len(self._slots) > 1 and self.series_dropout == 0.0:
            warnings.warn(
                "training over multiple slots with series_dropout=0.0: absent slots are "
                "zero-filled and flagged, but the encoder never meets an empty slot during "
                "training, so it has no reason to read the presence flag at inference. "
                "Zero-fill, an honest presence flag and ~10% series dropout are one design; "
                "pass series_dropout=0.1.",
                stacklevel=2,
            )

    # Slice selection is deterministic; only dropout and the augment hook consume
    # randomness, and both draw from a generator keyed on (seed, epoch, index) so
    # a run is reproducible while still differing between epochs.
    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation and dropout stream. Call once per epoch."""
        self._epoch = int(epoch)

    @classmethod
    def from_paths(
        cls,
        manifest_path: Path | str,
        cache_dir: Path | str,
        labels_path: Path | str | None = None,
        **kwargs: Any,
    ) -> StudyDataset:
        """Read the pinned manifest and label parquets, then build the dataset."""
        manifest = pd.read_parquet(manifest_path)
        labels = pd.read_parquet(labels_path) if labels_path is not None else None
        return cls(manifest, cache_dir, labels=labels, **kwargs)

    @property
    def slots(self) -> tuple[int, ...]:
        """Global slot ids emitted by this dataset, in batch order."""
        return self._slots

    @property
    def study_uids(self) -> tuple[str, ...]:
        return self._study_uids

    @property
    def empty_study_uids(self) -> tuple[str, ...]:
        """Studies with no cached series in the selected slots, i.e. all-absent items."""
        return self._empty_study_uids

    @property
    def short_series(self) -> dict[str, int]:
        """``series_uid -> cached slice count`` for every series shorter than ``n_slices``.

        Empty until items have actually been read, because it is populated in
        ``_load_slot``. Non-empty means some studies contributed fewer distinct planes
        than the config asked for -- under ``contiguous`` the end slices replicated,
        under ``spanning`` interior indices repeated. Measured over the 4,719
        ``Axial_fs1`` series, the fraction affected is 0.00% at depth 8,
        1.36% at 16, 15.47% at 24 and **48.46% at 32**. Print it; do not let a depth
        sweep quietly become a replication sweep.
        """
        return dict(self._short_series)

    def __len__(self) -> int:
        return len(self._study_uids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        uid = self._study_uids[index]
        rng = np.random.default_rng((self.seed, self._epoch, index))
        n_slots = len(self._slots)

        pixels = torch.zeros(
            (n_slots, self.n_slices, 3, self.crop.out_px, self.crop.out_px),
            dtype=torch.float32,
        )
        present = torch.zeros(n_slots, dtype=torch.bool)
        available = self._series.get(uid, {})
        for position, slot in enumerate(self._slots):
            ref = available.get(slot)
            if ref is None:
                continue  # left as exact zeros, and present stays False
            pixels[position] = self._load_slot(ref, rng)
            present[position] = True

        if self.training and self.series_dropout > 0.0:
            self._drop_series(present, pixels, rng)

        labels, mask = self._labels_for(uid)
        if not bool(torch.isfinite(labels).all()):
            raise ValueError(
                f"study {uid} produced non-finite labels {labels.tolist()}; a NaN here "
                "reaches the loss and destroys the run without raising"
            )

        return {
            "pixels": pixels,
            "present": present,
            "slot_ids": self._slot_ids.clone(),
            "labels": labels,
            "mask": mask,
            "study_uid": uid,
        }

    def _index_series(
        self, manifest: pd.DataFrame, *, verify_cache: bool
    ) -> dict[str, dict[int, _SeriesRef]]:
        """Resolve study -> slot -> the one series to load, dropping uncached ones."""
        wanted = set(self._slots)
        index: dict[str, dict[int, _SeriesRef]] = {}
        for row in manifest.itertuples(index=False):
            slot = int(getattr(row, "slot"))
            if slot not in wanted:
                continue
            uid = str(getattr(row, ID_COLUMN_MANIFEST))
            series_uid = str(getattr(row, "series_uid"))
            path = self.cache_dir / uid / f"{series_uid}.npy"
            if verify_cache and not path.is_file():
                continue  # no cache file is genuinely absent, not silently black
            ref = _SeriesRef(
                series_uid=series_uid,
                path=path,
                spacing=float(getattr(row, "spacing")),
                rows=int(getattr(row, "rows")),
                cols=int(getattr(row, "cols")),
            )
            slots = index.setdefault(uid, {})
            incumbent = slots.get(slot)
            if incumbent is None or _prefer(ref, incumbent, manifest_rows=row):
                slots[slot] = ref
        return index

    def _load_slot(self, ref: _SeriesRef, rng: np.random.Generator) -> Tensor:
        volume = np.load(ref.path, mmap_mode="r")
        if volume.ndim != 3 or volume.shape[1] != volume.shape[2]:
            raise ValueError(
                f"{ref.path} has shape {volume.shape}; the cache contract is (S, N, N) uint8"
            )
        n_available = int(volume.shape[0])
        if n_available < self.n_slices:
            # Counted, not warned and not silently absorbed. Under `spanning` the
            # linspace repeats interior indices; under `contiguous` the end slices
            # replicate. Both are deliberate (a duplicated real slice beats a black
            # one) and both mean this study contributes fewer distinct planes than
            # the config asks for, which the run log has to be able to state.
            self._short_series[ref.series_uid] = n_available
        centres = self._sample_indices(n_available, self.n_slices)
        stack = np.asarray(volume[neighbour_indices(centres, n_available)])

        pixels = torch.from_numpy(np.ascontiguousarray(stack)).to(torch.float32).div_(255.0)
        # Crop in millimetres before normalising: the crop is geometry and the
        # normalisation is the backbone's, and swapping the order would apply the
        # backbone statistics to padding that is not part of the image.
        cache_size = int(volume.shape[-1])
        pixels = _crop_to_fov(
            pixels, cache_mm_per_pixel(ref.spacing, ref.rows, ref.cols, cache_size), self.crop
        )
        pixels = (pixels - self._mean) / self._std
        if self._augment is not None:
            # Measured ordering from an external controlled ablation, not from this
            # dataset: CutMix at p~1.0 beat flip, rotate, brightness, contrast, blur
            # and mixup, and rotation was worth +0.025 CV. CutMix mixes two studies,
            # so it belongs in collate_studies; only per-study geometry and intensity
            # transforms belong here. This hook deliberately has no default policy.
            pixels = self._augment(pixels, rng)
            if tuple(pixels.shape[-2:]) != (self.crop.out_px, self.crop.out_px):
                raise ValueError(
                    f"augment hook changed the crop to {tuple(pixels.shape)}; it must "
                    f"preserve ({self.crop.out_px}, {self.crop.out_px})"
                )
        return pixels

    def _drop_series(self, present: Tensor, pixels: Tensor, rng: np.random.Generator) -> None:
        """Zero and un-flag some present slots, in place. Never all of them."""
        live = [int(i) for i in torch.nonzero(present).flatten().tolist()]
        if len(live) <= 1:
            return  # dropping the only slot leaves the study with no signal at all
        doomed = [i for i in live if rng.random() < self.series_dropout]
        if len(doomed) == len(live):
            doomed.remove(int(rng.choice(np.asarray(doomed))))
        for position in doomed:
            present[position] = False
            pixels[position].zero_()

    def _labels_for(self, uid: str) -> tuple[Tensor, Tensor]:
        row = self._label_index.get(uid)
        if row is None:
            # An unlabelled study is entirely unknown, not entirely negative. Twelve
            # zeros with a True mask would be twelve fabricated negatives.
            return torch.zeros(N_LABELS, dtype=torch.float32), torch.zeros(
                N_LABELS, dtype=torch.bool
            )
        return (
            torch.from_numpy(self._label_values[row].copy()),
            torch.from_numpy(self._label_mask[row].copy()),
        )


def _validate_slots(slots: Sequence[int] | None) -> tuple[int, ...]:
    if slots is None:
        return tuple(range(N_SLOTS))
    chosen = tuple(int(s) for s in slots)
    if not chosen:
        raise ValueError("slots must select at least one slot")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"slots repeats an entry: {chosen}")
    bad = [s for s in chosen if not 0 <= s < N_SLOTS]
    if bad:
        raise ValueError(f"slot ids {bad} outside [0, {N_SLOTS - 1}]")
    return chosen


def _prefer(candidate: _SeriesRef, incumbent: _SeriesRef, *, manifest_rows: Any) -> bool:
    """True when ``candidate`` should replace ``incumbent`` for the same slot.

    More cached slices wins, then the lexicographically smaller ``series_uid``,
    so the same series is chosen on every machine and in every run.
    """
    del manifest_rows  # the tie-break deliberately uses only stable series identity
    candidate_slices = _cached_slice_count(candidate.path)
    incumbent_slices = _cached_slice_count(incumbent.path)
    if candidate_slices != incumbent_slices:
        return candidate_slices > incumbent_slices
    return candidate.series_uid < incumbent.series_uid


def _cached_slice_count(path: Path) -> int:
    """Slice count read from the .npy header, or 0 when the file is unreadable."""
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except (OSError, ValueError):
        return 0


def _index_labels(
    labels: pd.DataFrame | None,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Flatten the derived label table into finite values plus a usable-cell mask.

    Sanitising here rather than per item means every consumer of this dataset
    gets the guarantee, and a NaN in the parquet turns into ``mask=False`` rather
    than into a NaN gradient.
    """
    if labels is None:
        return {}, np.zeros((0, N_LABELS), dtype=np.float32), np.zeros((0, N_LABELS), dtype=bool)
    missing = [c for c in (ID_COLUMN, *LABELS, *(f"{n}__mask" for n in LABELS))
               if c not in labels.columns]
    if missing:
        raise KeyError(f"label table is missing columns {missing}")

    values = labels[list(LABELS)].to_numpy(dtype=np.float32, copy=True)
    # Fill the mask column before casting it, because ``bool(nan)`` is True. A
    # NaN mask cell would otherwise promote an unknown label to usable and pair
    # it with the 0.0 written below, which is a fabricated negative -- the exact
    # failure this whole module exists to prevent, arriving through the mask
    # rather than through the value. ``& isfinite(values)`` does not catch it,
    # since the value cell can be perfectly finite.
    raw_mask = labels[[f"{name}__mask" for name in LABELS]].to_numpy(copy=True)
    filled_mask = np.where(pd.isna(raw_mask), False, raw_mask)
    mask = np.asarray(filled_mask, dtype=bool) & np.isfinite(values)
    values = np.where(mask, np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    values = values.astype(np.float32, copy=False)

    uids = [str(uid) for uid in labels[ID_COLUMN]]
    index = {uid: position for position, uid in enumerate(uids)}
    if len(index) != len(uids):
        # The derived table is one row per study by contract. Two rows for one
        # study means two derivations disagreeing, and taking the last silently
        # makes the training labels depend on row order in a parquet file.
        repeated = sorted(uid for uid, count in Counter(uids).items() if count > 1)
        raise ValueError(
            f"label table has {len(uids) - len(index)} duplicate study rows, e.g. "
            f"{repeated[:3]}; the derived table is one row per study, and keeping the "
            "last row would make the labels depend on parquet row order"
        )
    return index, np.ascontiguousarray(values), np.ascontiguousarray(mask)


def collate_studies(
    samples: Sequence[Mapping[str, Any]],
    *,
    batch_augment: Callable[[dict[str, Any], np.random.Generator], dict[str, Any]] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Stack per-study samples into the pinned batch, re-checking the NaN guarantee.

    ``batch_augment`` is where CutMix belongs, because it mixes two studies and
    must mix ``labels`` and ``mask`` together with ``pixels``. The finiteness
    check runs *after* the hook, so a mixing bug that introduces NaN raises here
    instead of printing ``nan`` for eleven hours. Bind the keyword arguments with
    ``functools.partial`` to pass this as a DataLoader ``collate_fn``.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch: dict[str, Any] = {
        "pixels": torch.stack([s["pixels"] for s in samples]),
        "present": torch.stack([s["present"] for s in samples]),
        "slot_ids": torch.stack([s["slot_ids"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]).to(torch.float32),
        "mask": torch.stack([s["mask"] for s in samples]),
        "study_uid": [str(s["study_uid"]) for s in samples],
    }
    if batch_augment is not None:
        batch = batch_augment(batch, np.random.default_rng(seed))
    if not bool(torch.isfinite(batch["labels"]).all()):
        raise ValueError(
            "batch labels contain a non-finite value; 98.7% of raw label cells are NaN "
            "and every loss propagates them silently, so this must never leave the collate"
        )
    return batch
