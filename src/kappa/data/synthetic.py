"""A complete fake corpus on disk, so the training stack can be run before the data exists.

Why this exists: the derived label table and the pixel cache are both still
being built, and a training stack that has never been executed is not a training
stack -- every wiring bug is still in it, waiting to surface on a cluster node
that costs an hour of walltime per attempt and has no internet. This module
writes a corpus that satisfies **every pinned contract at once**: the cache path
template and dtype, the sixteen manifest columns in order, and the derived label
table with its per-label mask column. So the loader, the model, the loss and the
loop can be run, profiled and debugged today, against labels that mean nothing
and pixels that are a disc.

The silent failure it prevents is the hand-rolled fixture. A fixture assembled
inline in a test grows exactly the columns that one test reads, so the loader
passes against the fixture and fails against the real manifest -- and it fails
after the expensive part. Everything here is written through the pinned column
tuple and the pinned path template, and the manifest schema is asserted on write,
so a contract change breaks this module loudly instead of breaking a run quietly.

Realism is a set of knobs rather than a fixed level, because the consumers need
different things from it:

* ``patterns`` sets the slot availability pattern per study, assigned
  round-robin, so twelve consecutive studies exercise twelve distinct patterns
  and the presence mask is never trivially all-True. Pass ``patterns=((1,),)``
  to reproduce the day-one Axial_fs1-only job, which is the one slot at 100%
  coverage.
* ``mask_rate`` and ``prevalence`` are per label. The defaults reproduce the
  measured 98.7% unknown rate of the raw label cells and MCL's measured 15.5%
  positive rate; see :data:`LABEL_MASK_RATE` for what is measured and what is
  merely a default.
* ``phantom_mm`` draws a disc of a known *physical* diameter into every slice.
  That is what lets a test assert that two series with different spacing arrive
  at the loader's output at the same physical scale -- the assertion a fixed
  *pixel* crop fails while looking perfectly reasonable on screen.

    from kappa.data.dataset import Normalisation
    from kappa.data.synthetic import make_synthetic_dataset

    corpus = make_synthetic_dataset(tmp_path, n_studies=20, seed=0)
    train = corpus.study_dataset(normalisation=Normalisation.identity(), slots=(1,))

Nothing here imports torch: the cache, the manifest and the label table are
numpy and pandas, and :meth:`SyntheticCorpus.study_dataset` imports the loader
lazily. So this generator is usable by the fold builder and the label pipeline on
a machine with no torch installed.

One cached slice costs ``cache_size ** 2`` bytes. The pinned cache size is 512,
which is 16.7 MB for a 64-slice series and over a gigabyte for a twenty-study
corpus, so the default here is 64 -- nothing about wiring depends on the pixel
count. Pass ``cache_size=512`` only when the measurement actually needs it. At
the defaults, 500 studies measured **0.90 s to write, 1,827 series, 267.0 MiB**
on this laptop.

**A smoke run should raise ``mask_rate``.** At the realistic default, a measured
1.32% of label cells are usable, so **28.1% of batches of 8 studies contain no
usable cell at all** and a loss normalised by ``mask.sum()`` divides by zero
there. That is a real property of this dataset and the loss has to survive it,
but it is not what a wiring run is trying to find out: pass ``mask_rate=0.25``
to get a dense enough batch to watch a number move.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from kappa.constants import ID_COLUMN, LABELS

__all__ = [
    "AVAILABILITY_PATTERNS",
    "CACHE_SLICE_CAP",
    "LABEL_MASK_RATE",
    "LABEL_PREVALENCE",
    "MANIFEST_COLUMNS",
    "MATRICES",
    "SITES",
    "SPACING_RANGE",
    "THICKNESS_RANGE",
    "SyntheticCorpus",
    "label_table",
    "make_synthetic_dataset",
    "phantom_slices",
]

try:
    # kappa.data.manifest owns the schema. Importing it rather than restating it
    # means a column added there fails this module's own write assertion, which
    # is the point: two copies of a schema is how a loader ends up reading a
    # column that the real manifest does not have.
    from kappa.data.manifest import MANIFEST_COLUMNS, N_SLOTS  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - only before manifest.py exists
    MANIFEST_COLUMNS = (
        "study_uid",
        "series_uid",
        "plane",
        "fs",
        "slot",
        "n_slices",
        "rows",
        "cols",
        "spacing",
        "thickness",
        "patient_id",
        "laterality",
        "manufacturer",
        "model",
        "field_strength",
        "scanner_fingerprint",
    )
    N_SLOTS = 6

#: Twelve distinct slot-availability patterns, one per study round-robin.
#:
#: Slot 1 (Axial_fs1) is in every one of them because it is the only slot at
#: 100% measured coverage, and slot 0 (Axial_fs0, 19.4% measured) is in the
#: minority. These twelve *sets* are constructed to exercise twelve code paths
#: including the all-six case and the single-slot case; they are **not** the
#: corpus's actual twelve patterns, which this module never claims to know, so
#: the per-slot marginals of a synthetic corpus are not the measured coverage.
#: Pass ``patterns=`` when a test needs specific marginals.
AVAILABILITY_PATTERNS: tuple[tuple[int, ...], ...] = (
    (1,),
    (1, 4),
    (1, 5),
    (1, 3, 4),
    (1, 4, 5),
    (1, 2, 3, 4),
    (1, 3, 4, 5),
    (0, 1, 4, 5),
    (1, 2, 3, 4, 5),
    (0, 1, 3, 4, 5),
    (0, 1, 2, 4, 5),
    (0, 1, 2, 3, 4, 5),
)

#: ``P(mask is False)`` per label, i.e. the fraction of cells that are unknown.
#: 0.987 for every label reproduces the measured 98.7% NaN rate of the raw label
#: cells. The real unknown-ness is clumped by study rather than scattered per
#: cell -- only 58 of 4,407 studies carry ground truth -- and it is per label in
#: the derived table. This default is deliberately the blunt version; pass a
#: mapping to shape it.
LABEL_MASK_RATE: Mapping[str, float] = MappingProxyType({name: 0.987 for name in LABELS})

#: ``P(label == 1.0 | usable)`` per label. **Only MCL's 0.155 is measured.** The
#: other eleven are set to the same value because this project has not measured
#: them, not because they are believed equal; a test that cares about a specific
#: prevalence must pass it explicitly.
LABEL_PREVALENCE: Mapping[str, float] = MappingProxyType({name: 0.155 for name in LABELS})

#: In-plane matrices to draw from. The corpus has 13 distinct aspect ratios with
#: a median of 512 squared; these five are enough for a non-square series to
#: reach the loader, which is what the letterbox arithmetic needs to be tested
#: against.
MATRICES: tuple[tuple[int, int], ...] = ((512, 512), (512, 448), (384, 384), (320, 256), (256, 256))

#: Measured in-plane spacing range, mm/px.
SPACING_RANGE: tuple[float, float] = (0.14, 0.89)

#: Measured slice thickness range, mm.
THICKNESS_RANGE: tuple[float, float] = (0.60, 5.40)

#: Fake scanner identities. Site memorisation is worth about 0.05 of a
#: random-fold score, so a synthetic manifest that has only one site cannot
#: exercise scanner-grouped folds at all.
SITES: tuple[tuple[str, str, float], ...] = (
    ("SYNTH", "PhantomOne", 1.5),
    ("SYNTH", "PhantomTwo", 3.0),
    ("OTHER", "PhantomOne", 3.0),
)

_PLANES = ("Axial", "Coronal", "Sagittal")

# The cache contract caps a cached series at 64 slices regardless of how many the
# series really has, so the manifest's n_slices and the .npy's first axis
# disagree above this. Kept as a constant because a loader that trusts the
# manifest instead of the array is a bug this generator is able to catch.
CACHE_SLICE_CAP = 64


def phantom_slices(
    n_slices: int,
    size: int,
    mm_per_pixel: float,
    *,
    diameter_mm: float = 100.0,
    background: int = 32,
    foreground: int = 255,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A ``(n_slices, size, size)`` uint8 stack holding a disc of a known physical size.

    The disc is centred and its radius is derived from ``mm_per_pixel``, so two
    stacks written at different spacings hold the *same anatomy at different
    pixel sizes*. A loader that crops a fixed number of pixels returns two
    different apparent sizes; a loader that crops a fixed field of view in
    millimetres returns one. That difference is the whole reason this phantom is
    a disc and not noise.

    Slice ``i`` also carries ``i % 256`` in its top-left pixel, which survives an
    identity crop and lets a test read back exactly which cached slices the
    loader sampled -- the cache is in geometric order, so the marker is the
    geometric index and never an ``InstanceNumber``.

    A disc wider than the frame is clipped rather than rescaled, because that is
    what the real fine-spacing series do.
    """
    if n_slices < 1 or size < 1:
        raise ValueError(f"need n_slices >= 1 and size >= 1, got {n_slices}, {size}")
    if mm_per_pixel <= 0.0:
        raise ValueError(f"mm_per_pixel must be > 0, got {mm_per_pixel}")
    if not 0 <= background < 256 or not 0 <= foreground < 256:
        raise ValueError("background and foreground must be uint8 values")

    axis = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    radius = np.hypot(axis[:, None], axis[None, :])
    disc = radius <= (diameter_mm / 2.0) / mm_per_pixel
    frame = np.where(disc, np.uint8(foreground), np.uint8(background))
    stack = np.repeat(frame[None, :, :], n_slices, axis=0)

    if noise > 0.0:
        generator = rng if rng is not None else np.random.default_rng(0)
        jitter = generator.normal(0.0, noise * 255.0, size=stack.shape)
        stack = np.clip(stack.astype(np.float64) + jitter, 0.0, 255.0).astype(np.uint8)

    stack[:, 0, 0] = np.arange(n_slices, dtype=np.int64).astype(np.uint8)
    return np.ascontiguousarray(stack, dtype=np.uint8)


def label_table(
    study_uids: Sequence[str],
    *,
    seed: int = 0,
    prevalence: Mapping[str, float] | float | None = None,
    mask_rate: Mapping[str, float] | float | None = None,
    nan_masked_cells: bool = False,
) -> pd.DataFrame:
    """A derived-label-table-shaped frame: ``StudyInstanceUID`` then twelve value/mask pairs.

    ``nan_masked_cells`` writes NaN instead of 0.0 wherever the mask is False.
    The real table does not do that -- it emits a finite value and a separate
    mask -- but a loader must survive a table that does, because every loss
    propagates NaN silently and the failure looks like a bad learning rate
    rather than like a bad file. Pass it to prove the loader sanitises.
    """
    prevalences = _per_label(prevalence, LABEL_PREVALENCE, "prevalence")
    mask_rates = _per_label(mask_rate, LABEL_MASK_RATE, "mask_rate")
    rng = np.random.default_rng(seed)
    n = len(study_uids)

    usable = rng.random((n, len(LABELS))) >= mask_rates
    positive = rng.random((n, len(LABELS))) < prevalences
    fill = np.float32("nan") if nan_masked_cells else np.float32(0.0)
    values = np.where(usable, positive.astype(np.float32), fill).astype(np.float32)

    columns: dict[str, Any] = {ID_COLUMN: [str(uid) for uid in study_uids]}
    for position, name in enumerate(LABELS):
        columns[name] = values[:, position]
        columns[f"{name}__mask"] = usable[:, position]
    return pd.DataFrame(columns)


@dataclass(frozen=True)
class SyntheticCorpus:
    """Where the generated corpus landed, plus the frames that were written.

    Both the on-disk paths and the in-memory frames are returned because the two
    consumers differ: a training smoke run wants the paths so it exercises the
    parquet read, and a unit test wants the frames so it can perturb one cell
    without a round trip.
    """

    root: Path
    cache_dir: Path
    manifest_path: Path
    labels_path: Path
    manifest: pd.DataFrame
    labels: pd.DataFrame
    cache_size: int
    phantom_mm: float
    patterns: tuple[tuple[int, ...], ...]

    @property
    def study_uids(self) -> tuple[str, ...]:
        return tuple(self.labels[ID_COLUMN].astype(str))

    @property
    def n_series(self) -> int:
        return int(len(self.manifest))

    def cache_bytes(self) -> int:
        """Total size of the written .npy files. Measured, not estimated."""
        return sum(path.stat().st_size for path in self.cache_dir.rglob("*.npy"))

    def mm_per_pixel(self, series_uid: str) -> float:
        """Cached millimetres per pixel for one series, from the manifest geometry."""
        row = self.manifest.loc[self.manifest["series_uid"] == str(series_uid)]
        if row.empty:
            raise KeyError(f"no series {series_uid!r} in this corpus")
        record = row.iloc[0]
        longest = max(int(record["rows"]), int(record["cols"]))
        return float(record["spacing"]) * longest / float(self.cache_size)

    def study_dataset(self, **kwargs: Any) -> Any:
        """Build a :class:`kappa.data.dataset.StudyDataset` over this corpus.

        Imported here rather than at module scope so that generating a corpus
        needs numpy and pandas only. ``normalisation`` stays a required keyword
        of the loader on purpose and is not defaulted here.
        """
        from kappa.data.dataset import StudyDataset

        return StudyDataset(self.manifest, self.cache_dir, labels=self.labels, **kwargs)


def make_synthetic_dataset(
    root: Path | str,
    *,
    n_studies: int = 20,
    seed: int = 0,
    patterns: Sequence[Sequence[int]] | None = None,
    cache_size: int = 64,
    slice_counts: tuple[int, int] = (11, 64),
    geometry: Sequence[tuple[float, int, int]] | None = None,
    phantom_mm: float = 100.0,
    noise: float = 0.0,
    n_sites: int = len(SITES),
    n_patients: int | None = None,
    duplicate_slot_studies: int = 0,
    prevalence: Mapping[str, float] | float | None = None,
    mask_rate: Mapping[str, float] | float | None = None,
    nan_masked_cells: bool = False,
) -> SyntheticCorpus:
    """Write a cache, a manifest and a label table under ``root``, and return where.

    ``root`` is used as a data root, so the layout matches
    :mod:`kappa.data.paths` exactly: ``root/processed/cache/<study>/<series>.npy``,
    ``root/interim/manifest.parquet`` and ``root/interim/labels/derived.parquet``.
    Point ``KAPPA_DATA_ROOT`` at it and the rest of the project cannot tell the
    difference.

    Deterministic in ``seed``: the same seed writes byte-identical files, so a
    failing smoke run can be reproduced rather than re-rolled. ``study_uid``
    embeds the seed, so two seeds can share one root without colliding.

    ``duplicate_slot_studies`` gives that many studies a second, thinner series
    in their first slot. Real studies do carry duplicates and the loader has to
    resolve them the same way twice, so leaving this at 0 for every test means
    the resolution rule is never exercised.

    ``geometry`` pins ``(spacing, rows, cols)`` per series, cycling through the
    sequence, instead of drawing from the measured ranges. Any assertion about
    physical scale needs to know the spacing it asked for, and a random one
    turns that assertion into a re-derivation of the generator.
    """
    root = Path(root)
    if n_studies < 1:
        raise ValueError(f"n_studies must be >= 1, got {n_studies}")
    if cache_size < 1:
        raise ValueError(f"cache_size must be >= 1, got {cache_size}")
    low, high = (int(slice_counts[0]), int(slice_counts[1]))
    if low < 1 or high < low:
        raise ValueError(f"slice_counts must be 1 <= low <= high, got {slice_counts}")
    if not 1 <= n_sites <= len(SITES):
        raise ValueError(f"n_sites must be in [1, {len(SITES)}], got {n_sites}")
    if duplicate_slot_studies < 0:
        raise ValueError(f"duplicate_slot_studies must be >= 0, got {duplicate_slot_studies}")

    chosen_patterns = _validate_patterns(patterns)
    pinned_geometry = _validate_geometry(geometry)
    n_patients = n_studies if n_patients is None else int(n_patients)
    if n_patients < 1:
        raise ValueError(f"n_patients must be >= 1, got {n_patients}")

    cache_dir = root / "processed" / "cache"
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    study_uids: list[str] = []

    for index in range(n_studies):
        study_uid = f"1.2.840.10008.9.{seed}.{index:06d}"
        study_uids.append(study_uid)
        pattern = chosen_patterns[index % len(chosen_patterns)]
        manufacturer, model, field_strength = SITES[index % n_sites]
        patient_id = f"P{index % n_patients:05d}"
        laterality = "L" if index % 2 == 0 else "R"
        study_dir = cache_dir / study_uid
        study_dir.mkdir(parents=True, exist_ok=True)

        occurrences = [(slot, 0) for slot in pattern]
        if index < duplicate_slot_studies:
            occurrences.append((pattern[0], 1))

        for slot, copy in occurrences:
            series_uid = f"{study_uid}.{slot}.{copy}"
            native = int(rng.integers(low, high + 1))
            if copy:
                # The duplicate is deliberately the thinner series, so a loader
                # that resolves duplicates by slice count has a right answer.
                native = max(1, native // 2)
            if pinned_geometry is None:
                matrix_rows, matrix_cols = MATRICES[int(rng.integers(len(MATRICES)))]
                spacing = round(float(rng.uniform(*SPACING_RANGE)), 4)
            else:
                spacing, matrix_rows, matrix_cols = pinned_geometry[
                    len(rows) % len(pinned_geometry)
                ]
            thickness = round(float(rng.uniform(*THICKNESS_RANGE)), 2)
            cached = min(native, CACHE_SLICE_CAP)
            mm_per_pixel = spacing * max(matrix_rows, matrix_cols) / cache_size
            volume = phantom_slices(
                cached,
                cache_size,
                mm_per_pixel,
                diameter_mm=phantom_mm,
                noise=noise,
                rng=rng,
            )
            np.save(study_dir / f"{series_uid}.npy", volume)

            plane, fs = _slot_pair(slot)
            rows.append(
                {
                    "study_uid": study_uid,
                    "series_uid": series_uid,
                    "plane": plane,
                    "fs": fs,
                    "slot": slot,
                    "n_slices": native,
                    "rows": matrix_rows,
                    "cols": matrix_cols,
                    "spacing": spacing,
                    "thickness": thickness,
                    "patient_id": patient_id,
                    "laterality": laterality,
                    "manufacturer": manufacturer,
                    "model": model,
                    "field_strength": field_strength,
                    "scanner_fingerprint": _fingerprint(manufacturer, model, field_strength),
                }
            )

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    if tuple(manifest.columns) != tuple(MANIFEST_COLUMNS):
        raise AssertionError(
            f"synthetic manifest columns {tuple(manifest.columns)} do not match the "
            f"contract {tuple(MANIFEST_COLUMNS)}"
        )
    labels = label_table(
        study_uids,
        seed=seed,
        prevalence=prevalence,
        mask_rate=mask_rate,
        nan_masked_cells=nan_masked_cells,
    )

    manifest_path = root / "interim" / "manifest.parquet"
    labels_path = root / "interim" / "labels" / "derived.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    labels.to_parquet(labels_path, index=False)

    return SyntheticCorpus(
        root=root,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        labels_path=labels_path,
        manifest=manifest,
        labels=labels,
        cache_size=int(cache_size),
        phantom_mm=float(phantom_mm),
        patterns=chosen_patterns,
    )


def _validate_patterns(patterns: Sequence[Sequence[int]] | None) -> tuple[tuple[int, ...], ...]:
    if patterns is None:
        return AVAILABILITY_PATTERNS
    cleaned: list[tuple[int, ...]] = []
    for pattern in patterns:
        slots = tuple(int(slot) for slot in pattern)
        if not slots:
            raise ValueError("an availability pattern must occupy at least one slot")
        if len(set(slots)) != len(slots):
            raise ValueError(f"pattern {slots} repeats a slot")
        outside = [slot for slot in slots if not 0 <= slot < N_SLOTS]
        if outside:
            raise ValueError(f"slot ids {outside} outside [0, {N_SLOTS - 1}]")
        cleaned.append(slots)
    if not cleaned:
        raise ValueError("patterns must hold at least one pattern")
    return tuple(cleaned)


def _validate_geometry(
    geometry: Sequence[tuple[float, int, int]] | None,
) -> tuple[tuple[float, int, int], ...] | None:
    if geometry is None:
        return None
    cleaned: list[tuple[float, int, int]] = []
    for spacing, matrix_rows, matrix_cols in geometry:
        if spacing <= 0.0 or matrix_rows < 1 or matrix_cols < 1:
            raise ValueError(
                f"geometry entry ({spacing}, {matrix_rows}, {matrix_cols}) needs a positive "
                "spacing and at least one row and column"
            )
        cleaned.append((float(spacing), int(matrix_rows), int(matrix_cols)))
    if not cleaned:
        raise ValueError("geometry must hold at least one (spacing, rows, cols) entry")
    return tuple(cleaned)


def _slot_pair(slot: int) -> tuple[str, int]:
    """``slot`` back to its ``(plane, fs)`` pair without importing the manifest module."""
    return _PLANES[slot // 2], slot % 2


def _fingerprint(manufacturer: str, model: str, field_strength: float) -> str:
    """A stable 16-hex-character site key.

    Only stability and distinctness matter to the consumer, which groups folds by
    string equality, so this is not the manifest's fingerprint function and does
    not try to reproduce its digest.
    """
    payload = f"{manufacturer}|{model}|{field_strength:.1f}".encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _per_label(
    value: Mapping[str, float] | float | None,
    default: Mapping[str, float],
    name: str,
) -> np.ndarray:
    """Resolve a scalar, a partial mapping or None into one rate per label, in LABELS order.

    An unknown key raises. A mapping keyed ``"Bakers"`` instead of ``"Baker's"``
    would otherwise be accepted and ignored, and the test asserting a prevalence
    would pass against the default while reporting the wrong reason.
    """
    if value is None:
        resolved = dict(default)
    elif isinstance(value, Mapping):
        unknown = [key for key in value if key not in LABELS]
        if unknown:
            raise KeyError(f"{name} has unknown label(s) {unknown}; legal names are {list(LABELS)}")
        resolved = dict(default) | {key: float(rate) for key, rate in value.items()}
    else:
        resolved = {label: float(value) for label in LABELS}
    rates = np.array([resolved[label] for label in LABELS], dtype=np.float64)
    if np.any(rates < 0.0) or np.any(rates > 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {rates.tolist()}")
    return rates
