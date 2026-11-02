"""One row per MRI series: what plane, what contrast, what scanner, whose knee.

Why this exists: the dataset ships **24,371 series across 4,407 studies** with no
single table that says what each series *is* and where it came from. Without one,
three separate silent failures are waiting.

**1. A study is not a fixed-size tensor.** Coverage of the six plane x contrast
slots is ragged -- measured here from ``train_series.csv``: Axial_fs1 100%,
Sagittal_fs0 96.8%, Coronal_fs1 96.4%, Sagittal_fs1 94.2%, Coronal_fs0 77.3%,
**Axial_fs0 19.4%**, in **12 distinct availability patterns**, with only 566 of
4,407 studies carrying all six. Code that assumes six slots and zero-fills the
gaps teaches the model that "absent" looks like "black". The manifest is what
lets the loader build an honest presence mask instead.

**2. Site memorisation.** ``scanner_fingerprint`` is a stable hash of
manufacturer, model, field strength and a spacing signature. It is load-bearing,
not decoration. A **published competition-forum probe, not a KAPPA run**, fitted
a gradient-boosting model to DICOM headers alone -- no pixels at all -- on this
data and reached macro AUC **0.6516 with random folds against 0.5981 with
scanner-grouped folds**. About **0.05 of a random-fold score is the model
recognising the scanner**, which is the size of the whole top-ten leaderboard
spread. You cannot subtract what you never recorded.

Note the granularity, because it decides what a fingerprint count means. The
fingerprint is per *series*, and the spacing signature varies by sequence, so one
study carries several: measured over the full manifest (n=4,407),
**1,361 distinct per-series fingerprints but only 408 per-study keys, over 58
distinct (manufacturer, model, field strength) triples**, the largest single
scanner covering 256 studies (5.8%). Group folds on the per-study key from
:func:`study_scanner_fingerprint`, never on the raw column.

**3. Patient recurrence.** ``patient_id`` is harvested from a DICOM header
per series, for every study, because if patients recur across studies and the
folds are not grouped by patient then every cross-validation number is inflated
and the project finds out from the private leaderboard.

The state of that question, measured on the 558 studies staged locally -- 58
anchors plus a uniform random 500, so a genuinely random sample and not a sorted
prefix: **558 of 558 studies carry a ``PatientID``, all 558 are distinct, and no
study has series that disagree about it.** The IDs are 12-character truncated
UUIDs, which is what a de-identifier emits, and it is unknowable from the format
whether it minted one per patient or one per study. Read
:class:`PatientLinkage` before drawing a conclusion from that: the
duplicate-report cross-check that would distinguish the two has **zero** of its
five discriminating study pairs in the local subset, so the question is **not
settled here**. It is settled on the full 4,407-study pass -- see
``slurm/manifest.sbatch`` and :data:`kappa.data.folds.DEFAULT_GROUP_KEYS`.

    from kappa.data.manifest import build_manifest, write_manifest

    manifest = build_manifest(raw_dir() / "train_series.csv", raw_dir() / "train_series")
    write_manifest(manifest, interim_dir() / "manifest.parquet")

``plane`` and ``fs`` are **taken from ``train_series.csv``, not re-derived from
``SeriesDescription``.** The competition already labels every one of the 24,371
rows with ``Anatomical_Plane``, ``Fluid_Sensitive`` and ``Fat_Suppression``, with
no nulls. Guessing the plane from a free-text sequence name in nine languages
when the answer is given would be inventing a failure mode.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "SLOT_ORDER",
    "N_SLOTS",
    "MANIFEST_COLUMNS",
    "PLANES",
    "slot_index",
    "scanner_fingerprint",
    "normalise_laterality",
    "read_series_index",
    "build_manifest",
    "write_manifest",
    "read_manifest",
    "series_dicom_files",
    "study_scanner_fingerprint",
    "study_table",
    "slot_coverage",
    "availability_patterns",
    "PatientLinkage",
    "patient_linkage",
]

#: The six plane x contrast slots, in the order that fixes the ``slot`` index.
#: Pinned across agents -- the loader, the manifest and the model all index the
#: slot axis with these positions, so reordering this tuple silently permutes
#: every cached tensor.
SLOT_ORDER: tuple[tuple[str, int], ...] = (
    ("Axial", 0),
    ("Axial", 1),
    ("Coronal", 0),
    ("Coronal", 1),
    ("Sagittal", 0),
    ("Sagittal", 1),
)

N_SLOTS = len(SLOT_ORDER)

PLANES = ("Axial", "Coronal", "Sagittal")

#: Manifest schema, exactly. Consumers assert against this, so a column added
#: in the middle is a breaking change and must be appended instead.
MANIFEST_COLUMNS: tuple[str, ...] = (
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

_SLOT_LOOKUP = {pair: index for index, pair in enumerate(SLOT_ORDER)}

# Column names as the competition writes them. `Fat_Suppression` is read only to
# be checked against `Fluid_Sensitive`; see `read_series_index`.
_PLANE_COLUMN = "Anatomical_Plane"
_FLUID_COLUMN = "Fluid_Sensitive"
_FAT_COLUMN = "Fat_Suppression"

# macOS writes an AppleDouble sidecar named `._<file>` for every file on an exFAT
# volume, and the local 558-study subset lives on one. Those sidecars match
# `*.dcm`, so an unfiltered glob doubles `n_slices` and hands pydicom a 4 KiB
# resource fork. Filtering here rather than at each call site because every
# caller would otherwise have to remember.
_APPLEDOUBLE_PREFIX = "._"

# DICOM defines `Laterality` as `L` or `R`, but this corpus also writes `RIGHT` --
# measured on the local 558-study subset, 10 of 3,095 series. Left alone that is a
# third value, so anything counting sides reports three of them and a per-side
# split silently drops the ten. Unrecognised spellings are uppercased and kept
# rather than mapped to empty, so a new one surfaces in a value count instead of
# disappearing into the missing bucket.
_LATERALITY_MAP = {"L": "L", "LEFT": "L", "R": "R", "RIGHT": "R"}

# How many hex characters of the fingerprint digest to keep. Matches
# `labels.normalise.content_hash` so the two kinds of key look alike in a table.
# 16 hex chars is 64 bits; against the few hundred distinct scanners in this
# corpus the collision probability is negligible.
_FINGERPRINT_CHARS = 16

# Alias table for reading a `SeriesHeader` produced by `kappa.data.dicom`. The
# manifest and the header are built by different agents in the same session, so
# the glue tolerates either spelling of a field rather than failing on a name.
# It is glue, not a second implementation: nothing here reads a DICOM.
_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": ("patient_id", "PatientID", "patient"),
    "laterality": ("laterality", "Laterality", "side"),
    "manufacturer": ("manufacturer", "Manufacturer"),
    "model": ("model", "model_name", "manufacturer_model", "ManufacturerModelName"),
    "field_strength": ("field_strength", "magnetic_field_strength", "MagneticFieldStrength"),
    "rows": ("rows", "Rows", "n_rows"),
    "cols": ("cols", "columns", "Columns", "n_cols"),
    "spacing": ("spacing", "pixel_spacing", "in_plane_spacing", "PixelSpacing"),
    "thickness": ("thickness", "slice_thickness", "SliceThickness"),
    "n_slices": ("n_slices", "n_files", "slice_count"),
}


def slot_index(plane: str, fs: int) -> int:
    """Position of a (plane, contrast) pair in :data:`SLOT_ORDER`.

    Raises rather than returning a sentinel. A series whose plane is not one of
    the three known values is a data problem to look at, and a ``-1`` slot would
    quietly index the last element of the slot axis instead.
    """
    try:
        return _SLOT_LOOKUP[(plane, int(fs))]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"no slot for plane={plane!r} fs={fs!r}; expected one of {SLOT_ORDER}"
        ) from None


def _spacing_signature(spacing: float | None, thickness: float | None) -> str:
    """Coarse geometry key for the scanner fingerprint.

    Rounded on purpose. In-plane spacing is a float derived from field of view
    over matrix size, so two acquisitions from the same protocol on the same
    magnet differ in the fourth decimal place and would fingerprint as two
    scanners. Two decimals on spacing and one on thickness keep protocol-level
    distinctions -- 0.31 mm against 0.22 mm, 3.0 mm slices against 2.5 mm --
    while collapsing that float noise.
    """
    parts = []
    for value, places in ((spacing, 2), (thickness, 1)):
        parts.append("na" if value is None or pd.isna(value) else f"{float(value):.{places}f}")
    return "/".join(parts)


def scanner_fingerprint(
    manufacturer: str | None,
    model: str | None,
    field_strength: float | None,
    spacing: float | None = None,
    thickness: float | None = None,
) -> str:
    """Stable site proxy: hash of make, model, field strength and geometry.

    **Never built with ``hash()``.** Python salts string hashing per process, so
    a fingerprint from ``hash()`` would differ between the manifest build and the
    fold build, and the resulting scanner-grouped folds would leak while looking
    correct. SHA-256 of a normalised joined string is reproducible across runs,
    machines and Python versions.

    The fingerprint identifies an *acquisition setup*, which is the best
    available stand-in for a site: the corpus carries no institution tag. It is
    computed per series, so a study that mixes protocols carries more than one --
    use :func:`study_scanner_fingerprint` to reduce a study to one key.
    """
    fields = (
        (manufacturer or "").strip().casefold(),
        (model or "").strip().casefold(),
        "na"
        if field_strength is None or pd.isna(field_strength)
        else f"{float(field_strength):.1f}",
        _spacing_signature(spacing, thickness),
    )
    joined = "|".join(fields)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


def read_series_index(series_csv: Path | str) -> pd.DataFrame:
    """Load ``train_series.csv``/``test_series.csv`` and derive plane, fs and slot.

    Asserts that ``Fluid_Sensitive`` and ``Fat_Suppression`` agree on every row.
    Measured on ``train_series.csv``: they are byte-identical across all 24,371
    rows, so this is **one contrast bit, not two**. The assertion is here rather
    than in a comment because treating them as two independent bits would create
    four contrast states, half of which never occur, and a model given a
    four-way one-hot would spend capacity on two dead columns. If a future data
    drop separates them, this fails loudly instead of silently changing meaning.
    """
    frame = pd.read_csv(series_csv)
    missing = {"StudyInstanceUID", "SeriesInstanceUID", _PLANE_COLUMN, _FLUID_COLUMN} - set(
        frame.columns
    )
    if missing:
        raise ValueError(f"{series_csv} is missing required columns: {sorted(missing)}")

    if _FAT_COLUMN in frame.columns:
        disagree = int((frame[_FLUID_COLUMN] != frame[_FAT_COLUMN]).sum())
        if disagree:
            raise ValueError(
                f"{_FLUID_COLUMN} and {_FAT_COLUMN} disagree on {disagree} of "
                f"{len(frame)} rows. They were byte-identical when the slot axis was "
                f"designed as one contrast bit; two bits now need four slots per plane."
            )

    bad_plane = sorted(set(frame[_PLANE_COLUMN]) - set(PLANES))
    if bad_plane:
        raise ValueError(f"{series_csv} has unknown planes {bad_plane}; expected {PLANES}")

    out = pd.DataFrame(
        {
            "study_uid": frame["StudyInstanceUID"].astype(str),
            "series_uid": frame["SeriesInstanceUID"].astype(str),
            "plane": frame[_PLANE_COLUMN].astype(str),
            "fs": frame[_FLUID_COLUMN].astype(int),
        }
    )
    out["slot"] = [slot_index(p, f) for p, f in zip(out["plane"], out["fs"])]
    return out


def series_dicom_files(series_dir: Path) -> list[Path]:
    """DICOM files in a series directory, AppleDouble sidecars excluded.

    Sorted by filename only, which is enough for counting and for picking one
    header. **It is not slice order** -- slices must be ordered by the projection
    of ``ImagePositionPatient`` onto the slice normal, which is the pixel
    reader's job, not this module's.
    """
    return sorted(
        path
        for path in series_dir.glob("*.dcm")
        if not path.name.startswith(_APPLEDOUBLE_PREFIX)
    )


def _header_field(header: Any, name: str) -> Any:
    """Read one manifest field from a SeriesHeader, tolerating its spelling."""
    for alias in _HEADER_ALIASES[name]:
        if isinstance(header, dict):
            if alias in header:
                return header[alias]
        elif hasattr(header, alias):
            return getattr(header, alias)
    raise AttributeError(
        f"the series header carries no field for {name!r} (tried "
        f"{_HEADER_ALIASES[name]}). The manifest contract needs "
        f"{sorted(_HEADER_ALIASES)}; see kappa.data.dicom.SeriesHeader."
    )


def _default_header_reader() -> Callable[[Path], Any]:
    """Resolve ``kappa.data.dicom.read_series_header`` on first use.

    Imported lazily and by name so that this module -- and the slot, fingerprint
    and fold logic that the rest of the pipeline tests -- imports on a machine
    with no pydicom and no DICOMs staged. The header reader is owned elsewhere
    and is deliberately not reimplemented here.
    """
    try:
        from kappa.data.dicom import read_series_header
    except ImportError as exc:  # pragma: no cover - depends on sibling module
        raise ImportError(
            "building a manifest needs kappa.data.dicom.read_series_header "
            "(header-only DICOM read). Pass header_reader= to build_manifest to "
            "supply your own."
        ) from exc
    return read_series_header


def build_manifest(
    series_csv: Path | str,
    series_root: Path | str,
    *,
    header_reader: Callable[[Path], Any] | None = None,
    studies: Iterable[str] | None = None,
    require_all: bool = False,
    progress_every: int = 200,
) -> pd.DataFrame:
    """One manifest row per series, joining the given index to DICOM headers.

    ``series_root`` holds ``<study_uid>/<series_uid>/<sop_uid>.dcm``. Series with
    no directory on disk are skipped unless ``require_all``, because the local
    subset is 558 of 4,407 studies and a partial manifest is the normal case off
    the cluster; the returned frame's ``study_uid`` count is the honest record of
    what was actually read.

    ``studies`` restricts the build to a set of study UIDs, which is how the
    Slurm array shards the full pass.
    """
    index = read_series_index(series_csv)
    if studies is not None:
        wanted = set(studies)
        index = index[index["study_uid"].isin(wanted)].reset_index(drop=True)

    root = Path(series_root)
    read_header = header_reader or _default_header_reader()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for position, record in enumerate(index.itertuples(index=False)):
        series_dir = root / record.study_uid / record.series_uid
        if not series_dir.is_dir():
            missing.append(record.series_uid)
            continue

        header = read_header(series_dir)
        n_slices = _slice_count(header, series_dir)
        spacing = _as_float(_header_field(header, "spacing"))
        thickness = _as_float(_header_field(header, "thickness"))
        manufacturer = _as_str(_header_field(header, "manufacturer"))
        model = _as_str(_header_field(header, "model"))
        field_strength = _as_float(_header_field(header, "field_strength"))

        rows.append(
            {
                "study_uid": record.study_uid,
                "series_uid": record.series_uid,
                "plane": record.plane,
                "fs": int(record.fs),
                "slot": int(record.slot),
                "n_slices": n_slices,
                "rows": int(_as_float(_header_field(header, "rows")) or 0),
                "cols": int(_as_float(_header_field(header, "cols")) or 0),
                "spacing": spacing,
                "thickness": thickness,
                "patient_id": _as_str(_header_field(header, "patient_id")),
                "laterality": normalise_laterality(_header_field(header, "laterality")),
                "manufacturer": manufacturer,
                "model": model,
                "field_strength": field_strength,
                "scanner_fingerprint": scanner_fingerprint(
                    manufacturer, model, field_strength, spacing, thickness
                ),
            }
        )
        if progress_every and position and position % progress_every == 0:
            print(f"  {position}/{len(index)} series", file=sys.stderr, flush=True)

    if missing and require_all:
        raise FileNotFoundError(
            f"{len(missing)} of {len(index)} series have no directory under {root}; "
            f"first missing: {missing[0]}"
        )
    if missing:
        print(
            f"note: skipped {len(missing)} of {len(index)} series with no directory "
            f"under {root}",
            file=sys.stderr,
        )

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    return manifest.sort_values(["study_uid", "slot", "series_uid"]).reset_index(drop=True)


def _slice_count(header: Any, series_dir: Path) -> int:
    """Slices in a series, from the header if it says, else by counting files."""
    try:
        value = _header_field(header, "n_slices")
    except AttributeError:
        value = None
    if value is None:
        return len(series_dicom_files(series_dir))
    return int(value)


def _as_float(value: Any) -> float | None:
    """Coerce a DICOM value to float, mapping empty and unparseable to None.

    DICOM writes an absent numeric as an empty string as often as it omits the
    tag, and ``float("")`` raises. Returning None keeps the column nullable so a
    missing field stays visibly missing instead of becoming a 0.0 that reads as
    a measurement.
    """
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[0] if len(value) else None
        if value is None:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str:
    """Coerce a DICOM value to a stripped string; absent becomes empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = value[0] if len(value) else ""
    return str(value).strip()


def normalise_laterality(value: Any) -> str:
    """Fold ``Laterality`` onto ``L``, ``R`` or empty. See :data:`_LATERALITY_MAP`."""
    text = _as_str(value).upper()
    return _LATERALITY_MAP.get(text, text)


def write_manifest(manifest: pd.DataFrame, path: Path | str) -> Path:
    """Write the manifest, asserting the column list first.

    The schema check is here, on the write, because that is the last point where
    a wrong column order is cheap to fix. Downstream the manifest is read by two
    other components that index it positionally in places.
    """
    if tuple(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            f"manifest columns are {tuple(manifest.columns)}, contract requires "
            f"{MANIFEST_COLUMNS}"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out, index=False)
    return out


def read_manifest(path: Path | str) -> pd.DataFrame:
    """Read a manifest back, asserting the column list."""
    manifest = pd.read_parquet(path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            f"{path} has columns {tuple(manifest.columns)}, contract requires "
            f"{MANIFEST_COLUMNS}"
        )
    return manifest


def study_scanner_fingerprint(manifest: pd.DataFrame, canonical_slot: int = 1) -> pd.Series:
    """Reduce each study to one scanner key, indexed by ``study_uid``.

    Folds group by study, so the per-series fingerprint has to collapse. It is
    taken from a fixed slot rather than by majority vote because the spacing
    signature inside the fingerprint depends on the sequence: a majority over a
    study whose sagittal and axial protocols differ would flip with the number
    of series acquired in each plane, which has nothing to do with the scanner.

    ``canonical_slot`` defaults to 1 -- Axial_fs1 -- because that slot is present
    in **4,407 of 4,407** studies as measured from ``train_series.csv``, so the
    choice never falls back. Studies missing it use their lowest present slot.
    """
    keys: dict[str, str] = {}
    for study, group in manifest.groupby("study_uid", sort=True):
        preferred = group[group["slot"] == canonical_slot]
        chosen = preferred if len(preferred) else group[group["slot"] == group["slot"].min()]
        keys[str(study)] = str(chosen.sort_values("series_uid")["scanner_fingerprint"].iloc[0])
    return pd.Series(keys, name="scanner_fingerprint").rename_axis("study_uid")


def study_table(manifest: pd.DataFrame, canonical_slot: int = 1) -> pd.DataFrame:
    """Collapse the manifest to one row per study: exactly the keys folds group on.

    Returns ``study_uid, patient_id, scanner_fingerprint, laterality, n_series``.
    The fold builder needs one grouping value per study, and computing it in two
    places would let the manifest and the fold table disagree about which
    scanner a study belongs to -- which is the one disagreement that makes a
    scanner-grouped fold leak while still passing an overlap check.

    ``laterality`` is the set of sides seen across the study's series, as ``"L"``,
    ``"R"``, ``"LR"`` or ``""``. It is not reduced to a single side because the
    corpus leaves the tag blank on roughly half the series (measured: 1,539 of
    3,095 in the local subset), so a majority vote would invent a side.
    """
    for column in ("study_uid", "patient_id", "scanner_fingerprint", "laterality", "slot"):
        if column not in manifest.columns:
            raise KeyError(f"manifest is missing the {column!r} column")

    fingerprints = study_scanner_fingerprint(manifest, canonical_slot=canonical_slot)
    records = []
    for study, group in manifest.groupby("study_uid", sort=True):
        patients = sorted({p for p in group["patient_id"] if p})
        sides = sorted({s for s in group["laterality"] if s})
        records.append(
            {
                "study_uid": str(study),
                "patient_id": patients[0] if patients else "",
                "scanner_fingerprint": str(fingerprints[str(study)]),
                "laterality": "".join(sides),
                "n_series": len(group),
            }
        )
    return pd.DataFrame(records)


def slot_coverage(index: pd.DataFrame) -> pd.DataFrame:
    """Fraction of studies carrying at least one series in each of the six slots.

    Takes either a manifest or the frame from :func:`read_series_index`, so
    coverage can be measured from the given CSV without reading a single DICOM.
    """
    per_study = index.groupby("study_uid")["slot"].apply(set)
    n_studies = len(per_study)
    records = []
    for position, (plane, fs) in enumerate(SLOT_ORDER):
        present = sum(1 for slots in per_study if position in slots)
        records.append(
            {
                "slot": position,
                "plane": plane,
                "fs": fs,
                "studies": present,
                "fraction": present / n_studies if n_studies else 0.0,
            }
        )
    return pd.DataFrame(records)


def availability_patterns(index: pd.DataFrame) -> pd.Series:
    """Count of studies per distinct set of occupied slots, commonest first.

    Measured on the full ``train_series.csv``: **12 distinct patterns**, with
    566 studies carrying all six slots. That distribution is the reason the
    presence mask exists, so it is worth being able to recompute in one call.
    """
    per_study = index.groupby("study_uid")["slot"].apply(lambda s: tuple(sorted(set(s))))
    return per_study.value_counts()


@dataclass(frozen=True)
class PatientLinkage:
    """Evidence about whether patients recur across studies -- and its limits.

    ``n_studies``/``n_patients`` is the headline. The rest is what stops it being
    over-read, because **a de-identifier that re-keys ``PatientID`` per study
    erases linkage rather than measuring its absence**, and both outcomes look
    like "every PatientID is distinct".

    ``duplicate_text_groups`` counts groups of studies whose normalised report
    text is identical; ``duplicate_text_concordant`` counts how many of those
    groups share one ``PatientID``.

    **A discordant group is only evidence when the shared text could not have
    been written twice for two different patients, and which groups those are is
    measured, not assumed.** Of the 52 duplicate-text groups in ``train.csv``, 31
    have report text under 250 characters and the three largest -- 37, 14 and 12
    studies -- are template negatives ("Sin anomalias", "Diz eklemi ici sivi
    miktari normal"). Hundreds of different patients share a template negative
    report, so a discordant boilerplate group is the *expected* outcome for
    genuinely distinct patients and says nothing about re-keying.

    Length alone does not separate the two, and this was the trap: the longest
    discordant group in the local subset is 538 characters of fully structured
    all-normal English findings, recurring six times corpus-wide. That is a
    department template that happens to be long. A group is therefore counted as
    specific only when it is **both** at least ``min_specific_chars`` long
    **and** no larger than ``max_specific_members`` studies -- a report that
    recurs six times is a template whatever its length, while a long report
    appearing exactly twice is plausibly one patient filed twice. On
    ``train.csv`` that criterion selects 5 of the 52 groups, and none of the
    three template negatives.

    ``split_studies`` is the sanity check that must stay zero: a study whose
    series disagree about ``PatientID`` means the harvest picked the wrong file.
    """

    n_studies: int
    n_patients: int
    n_studies_with_id: int
    max_studies_per_patient: int
    repeated_patients: int
    split_studies: tuple[str, ...]
    duplicate_text_groups: int
    duplicate_text_concordant: int
    duplicate_text_discordant: int
    duplicate_text_specific: int
    duplicate_text_discordant_specific: int
    duplicate_text_specific_out_of_scope: int
    min_specific_chars: int
    max_specific_members: int

    @property
    def patients_recur(self) -> bool:
        """True only if the same PatientID is attached to more than one study."""
        return self.repeated_patients > 0

    @property
    def rekeying_proven(self) -> bool:
        """True only if a *clinically specific* report spans more than one PatientID.

        This is the case that invalidates ``patients_recur is False``: a long,
        rare report filed twice under two IDs means the ID is per-study, so
        absence of recurrence becomes unmeasurable from the ID alone. Boilerplate
        duplicates are excluded on purpose -- see the class docstring; counting
        them here is exactly how a weak signal gets promoted to a conclusion.
        """
        return self.duplicate_text_discordant_specific > 0

    @property
    def verdict(self) -> str:
        """One of ``recur``, ``rekeyed``, ``distinct`` or ``undecided``."""
        if self.patients_recur:
            return "recur"
        if self.rekeying_proven:
            return "rekeyed"
        if self.duplicate_text_specific and not self.duplicate_text_discordant_specific:
            return "distinct"
        return "undecided"

    def summary(self) -> str:
        """One-paragraph verdict, hedged exactly as far as the evidence reaches."""
        lines = [
            f"studies {self.n_studies}, with a PatientID {self.n_studies_with_id}, "
            f"distinct PatientIDs {self.n_patients}",
            f"patients attached to more than one study: {self.repeated_patients} "
            f"(largest {self.max_studies_per_patient} studies)",
            f"identical-report groups in scope: {self.duplicate_text_groups} "
            f"({self.duplicate_text_concordant} share one PatientID, "
            f"{self.duplicate_text_discordant} span several)",
            f"of those, discriminating (>= {self.min_specific_chars} chars and "
            f"<= {self.max_specific_members} studies): {self.duplicate_text_specific} "
            f"({self.duplicate_text_discordant_specific} span several PatientIDs)",
            f"discriminating groups the manifest cannot test, because fewer than two "
            f"of their studies are in it: {self.duplicate_text_specific_out_of_scope}",
        ]
        if self.split_studies:
            lines.append(
                f"WARNING {len(self.split_studies)} studies have series with "
                f"disagreeing PatientIDs, e.g. {self.split_studies[0]}"
            )
        verdict = self.verdict
        if verdict == "recur":
            lines.append("VERDICT patients recur across studies: fold grouping is required.")
        elif verdict == "rekeyed":
            lines.append(
                "VERDICT no PatientID repeats, but a long, rare report spans "
                "several IDs, so the IDs are re-keyed per study. Recurrence is NOT "
                "ruled out and folds must group on something other than PatientID."
            )
        elif verdict == "distinct":
            lines.append(
                "VERDICT no PatientID repeats and every discriminating "
                "identical-report group is internally consistent. Distinct patients is "
                "the better-supported reading, on the strength of the discriminating "
                "groups actually in scope."
            )
        else:
            lines.append(
                "VERDICT no PatientID repeats, and no discriminating duplicate "
                "report is in scope to test the IDs against. Consistent with both "
                "distinct patients and per-study re-keying. UNDECIDED -- keep grouping "
                "the folds by PatientID, because that is the safe reading either way."
            )
        return "\n".join(lines)


def patient_linkage(
    manifest: pd.DataFrame,
    reports: Mapping[str, str] | None = None,
    *,
    min_specific_chars: int = 400,
    max_specific_members: int = 2,
) -> PatientLinkage:
    """Answer "do patients recur?" and record how far the answer reaches.

    ``reports`` maps ``study_uid`` to raw report text. It is hashed here, through
    ``normalise_report`` then ``content_hash``, rather than taking a pre-computed
    key: hashing raw text misses duplicates that differ only in whitespace or
    Unicode composition, and measured on ``train.csv`` that is the difference
    between 46 groups over 177 studies and 52 groups over 197. Doing it in one
    place removes the chance of the caller hashing the wrong string.

    The duplicate-text signal supplies the counter-evidence a clean "all IDs
    distinct" result needs, because de-identification that re-keys ``PatientID``
    per study **erases** linkage rather than measuring its absence. The signal is
    weaker than it looks, and ``min_specific_chars``/``max_specific_members`` are
    where that weakness is handled -- see :class:`PatientLinkage`. The defaults
    select 5 of the 52 duplicate groups in ``train.csv``: the pairs whose text is
    long and appears exactly twice. Those five pairs are the discriminating test
    for patient recurrence, and **none of the ten studies involved is in the local 558-study
    subset**, so the question cannot be settled off the cluster.
    """
    # Imported here, not at module scope. `kappa.data.manifest` is on the inference
    # path (`dataset.py` reads SLOT_ORDER from it), and submissions are image-only
    # because test studies carry no report -- a module-level `kappa.labels` import made every Kaggle
    # notebook load a report reader. `patient_linkage` is a training-time analysis
    # and is never called from inference, so the dependency belongs in its body.
    from kappa.labels.normalise import content_hash, normalise_report

    per_study = manifest.groupby("study_uid")["patient_id"].apply(
        lambda ids: sorted({i for i in ids if i})
    )
    split = tuple(str(study) for study, ids in per_study.items() if len(ids) > 1)
    ids_by_study = {str(study): (ids[0] if ids else "") for study, ids in per_study.items()}

    studies_per_patient: dict[str, list[str]] = {}
    for study, patient in ids_by_study.items():
        if patient:
            studies_per_patient.setdefault(patient, []).append(study)

    groups = concordant = discordant = specific = discordant_specific = 0
    specific_out_of_scope = 0
    if reports is not None:
        # Grouped over every study in `reports`, not only the ones in the
        # manifest, because `max_specific_members` has to see the corpus-wide
        # group size. A six-study template with two members in a partial
        # manifest would otherwise look like a two-study pair and be promoted to
        # evidence -- which is the exact mistake this criterion exists to stop.
        by_key: dict[str, list[str]] = {}
        length: dict[str, int] = {}
        for study, text in reports.items():
            normalised = normalise_report(text)
            key = content_hash(normalised)
            by_key.setdefault(key, []).append(str(study))
            length[key] = len(normalised)
        for key, members in by_key.items():
            if len(members) < 2:
                continue
            is_specific = (
                length[key] >= min_specific_chars and len(members) <= max_specific_members
            )
            in_scope = [s for s in members if s in ids_by_study]
            if len(in_scope) < 2:
                specific_out_of_scope += int(is_specific)
                continue
            groups += 1
            specific += int(is_specific)
            if len({ids_by_study[s] for s in in_scope if ids_by_study[s]}) <= 1:
                concordant += 1
            else:
                discordant += 1
                discordant_specific += int(is_specific)

    return PatientLinkage(
        n_studies=len(ids_by_study),
        n_patients=len(studies_per_patient),
        n_studies_with_id=sum(1 for v in ids_by_study.values() if v),
        max_studies_per_patient=max((len(v) for v in studies_per_patient.values()), default=0),
        repeated_patients=sum(1 for v in studies_per_patient.values() if len(v) > 1),
        split_studies=split,
        duplicate_text_groups=groups,
        duplicate_text_concordant=concordant,
        duplicate_text_discordant=discordant,
        duplicate_text_specific=specific,
        duplicate_text_discordant_specific=discordant_specific,
        duplicate_text_specific_out_of_scope=specific_out_of_scope,
        min_specific_chars=min_specific_chars,
        max_specific_members=max_specific_members,
    )
