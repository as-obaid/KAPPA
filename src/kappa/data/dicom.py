"""Decode one DICOM series into an ordered, windowed, square ``uint8`` volume.

Why this exists: **nothing else in this repository reads a DICOM**, and the
obvious implementation is wrong in four ways that all produce a volume which
trains without complaint and is quietly worthless.

**1. Slice order comes from geometry, never from ``InstanceNumber``.** Sorting by
``InstanceNumber`` is the reflex, and on this corpus it is wrong more often than
right: of 30 series probed in the local subset, **16 disagree** with physical
order, and every disagreement is a full reversal (``InstanceNumber`` counts down
as the patient axis counts up). A reversed volume is anatomically a mirror in z.
Nothing downstream can detect it -- the 2.5D neighbour stack still has three
plausible slices, the loss still falls -- so the error survives to the
leaderboard. The order used here is the projection of ``ImagePositionPatient``
onto the slice normal (the cross product of the two direction cosines in
``ImageOrientationPatient``), ascending, and :attr:`SeriesVolume.slice_positions`
is returned so a consumer can assert it rather than trust it.

**2. Windowing is per series, not per slice and not global.** Intensity in MR is
uninterpretable in absolute terms; it is a property of the scanner and protocol,
and this corpus carries 408 distinct per-study scanner fingerprints. Windowing
per *slice* destroys the contrast between slices, which is the signal for
anything with extent -- an effusion is bright relative to its neighbours.
Windowing *globally* over the corpus lets one bright site set the scale for
every other. The 1st and 99th percentiles of the whole series clip the coil
hot-spots and the background without touching anatomy.

**3. Letterbox first, then resize.** In-plane matrices run 192x192 to 1280 on the
long side, over **87 distinct matrices and 45 distinct aspect ratios** (measured
over all 3,095 series of the local subset). Only 26.4% of series are
512 on the long side: 37.9% are larger and get downsampled, 35.7% smaller and get
upsampled, so 512 is the median rather than a majority. Resizing a 384x512 slice straight to 512x512
stretches it 1.33x in one axis only, so the same meniscus is a different shape
depending on the site that scanned it. Padding to square first makes the
resample isotropic and the distortion zero.

**4. No isotropic resampling, ever.** Anisotropy on this data is **11.7x**
(in-plane spacing median 0.31 mm against slice thickness median 3.00 mm) and
**90.5% of series have an inter-slice gap**. Resampling to an isotropic grid
would therefore synthesise roughly ten interpolated voxels for every measured
one, and a 3D convolution over that grid spends most of its capacity on
invented data. That single measurement is the reason this project is **2.5D**
(three neighbouring slices as channels into a 2D encoder) rather than 3D, and it
is recorded here because this module is where someone would be tempted to
"fix" it.

    from kappa.data.dicom import read_series, read_series_header

    volume = read_series(series_dir)              # (S, 512, 512) uint8
    assert (np.diff(volume.slice_positions) >= 0).all()

    header = read_series_header(series_dir)       # one file, no pixel decode
    print(header.manufacturer, header.rows, header.cols)

:func:`read_series_header` exists as a separate entry point because the manifest
build needs geometry and scanner fields for every series of all 4,407 studies
and has no use for pixels; decoding them would be the entire cost of that pass
for none of its value.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "SeriesHeader",
    "SeriesVolume",
    "CacheReport",
    "read_series",
    "read_series_header",
    "series_files",
    "study_dirs",
    "slice_normal",
    "slice_projections",
    "select_slice_indices",
    "letterbox",
    "resize_plane",
    "window_to_uint8",
    "cache_series",
    "build_cache",
    "CACHE_SIZE",
    "MAX_SLICES",
    "WINDOW_PERCENTILES",
]

# The cache is built at 512, not 224 or 384, and that decision cannot be undone
# without a second multi-hour pass over 819,635 DICOMs. Three reasons: the median
# native in-plane matrix *is* 512, so 512 is near-lossless where 224 throws away
# three quarters of the measured pixels; a dataloader can downsample 512 to
# whatever an experiment wants but cannot invent detail back; and the space costs
# nothing -- measured 7.8 MB per cached series over 114 real series, so ~190 GB
# for the 24,371 train series, against 428 T of scratch.
CACHE_SIZE = 512

# Slices per series run 11-320 with a median of 30. The cap bounds the worst case
# rather than the typical one: without it a 320-slice series is 10x the median
# cost for z-resolution no consumer asks for, since the batch contract samples
# 16 slices. Series at or under the cap keep their native count.
MAX_SLICES = 64

WINDOW_PERCENTILES = (1.0, 99.0)

# macOS writes an AppleDouble sidecar named `._<file>` beside every file on an
# exFAT volume, and the local 558-study subset lives on one. Measured: a series
# directory holding 34 DICOMs returns 68 entries from `glob("*.dcm")`, half of
# them 4 KiB resource forks. Handing one to pydicom raises, and counting one
# doubles the slice count -- so the filter belongs in the enumerator, not at each
# call site.
_APPLEDOUBLE_PREFIX = "._"

# MONOCHROME1 means low stored value renders *white*. It is rare in MR but not
# impossible, and an un-inverted MONOCHROME1 series is a photographic negative:
# fluid reads dark, fat reads bright, and every learned filter is inverted
# relative to the rest of the corpus. Cheap to handle, invisible if ignored.
_INVERTED_PHOTOMETRIC = "MONOCHROME1"


@dataclass(frozen=True)
class SeriesHeader:
    """Geometry and scanner fields for one series, read from a single file.

    The field names are the manifest contract's column names verbatim
    (``kappa.data.manifest.MANIFEST_COLUMNS`` minus ``plane``, ``fs``, ``slot``
    and the derived ``scanner_fingerprint``) so that the manifest build is a
    rename-free copy. Numeric fields are ``None`` rather than ``0.0`` when the
    tag is absent or empty, because a missing spacing must stay visibly missing:
    a ``0.0`` in that column would flow into ``scanner_fingerprint`` and merge
    two sites.
    """

    study_uid: str
    series_uid: str
    patient_id: str
    laterality: str
    manufacturer: str
    model: str
    field_strength: float | None
    rows: int
    cols: int
    spacing: float | None
    thickness: float | None
    n_slices: int


@dataclass(frozen=True)
class SeriesVolume:
    """One decoded series: pixels plus everything needed to audit them.

    ``slice_positions`` and ``sop_uids`` are not decoration. The positions are
    the evidence that ordering happened geometrically -- a consumer can assert
    they are non-decreasing, which is impossible to fake from pixels alone -- and
    they also expose duplicate z, which is how a multi-echo series interleaved
    into one directory announces itself. ``sop_uids`` maps every plane of the
    array back to the file it came from, so a suspicious slice is traceable
    without re-deriving the sort.
    """

    pixels: np.ndarray
    slice_positions: np.ndarray
    sop_uids: tuple[str, ...]
    header: SeriesHeader

    @property
    def n_slices(self) -> int:
        """Slices actually decoded, after the :data:`MAX_SLICES` cap."""
        return int(self.pixels.shape[0])


def series_files(series_dir: Path | str) -> list[Path]:
    """DICOM files in a series directory, AppleDouble sidecars excluded.

    Sorted by filename, which is a stable enumeration order and **not slice
    order**. Falls back to every regular file when nothing matches ``*.dcm``,
    because a DICOM is identified by its preamble and not by its extension, and
    at least one public mirror of this competition ships the files bare.
    """
    directory = Path(series_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a series directory")
    files = sorted(
        path
        for path in directory.glob("*.dcm")
        if path.is_file() and not path.name.startswith(_APPLEDOUBLE_PREFIX)
    )
    if files:
        return files
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(_APPLEDOUBLE_PREFIX)
    )


def slice_normal(orientation: Sequence[float]) -> np.ndarray:
    """Unit normal of the image plane from the six ``ImageOrientationPatient`` cosines.

    The tag holds two orthogonal direction cosines, row then column; their cross
    product is the through-plane direction. Normalised so that the projection it
    is used for is a distance in millimetres and not an arbitrary scale.
    """
    cosines = np.asarray(orientation, dtype=np.float64).reshape(-1)
    if cosines.size != 6:
        raise ValueError(
            f"ImageOrientationPatient must hold 6 direction cosines, got {cosines.size}"
        )
    normal = np.cross(cosines[:3], cosines[3:])
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        raise ValueError(
            "ImageOrientationPatient gives a zero-length slice normal; its two "
            "direction cosines are parallel, so the plane is degenerate"
        )
    return normal / length


def slice_projections(
    positions: Sequence[Sequence[float]], orientation: Sequence[float]
) -> np.ndarray:
    """Signed through-plane coordinate of each ``ImagePositionPatient``, in millimetres.

    This is the one true slice order. It is a dot product with the plane normal,
    so it is invariant to how the scanner numbered the instances and to which
    corner of the patient the origin sits in.
    """
    normal = slice_normal(orientation)
    origins = np.asarray(positions, dtype=np.float64).reshape(len(positions), 3)
    return origins @ normal


def select_slice_indices(n_available: int, max_slices: int = MAX_SLICES) -> np.ndarray:
    """Indices to keep from an ordered series, capped at ``max_slices``.

    Decimates at equal intervals across the **full** extent rather than taking a
    centre block. A 320-slice series covers the whole joint; keeping its middle
    64 would silently crop the anatomy away, whereas keeping every fifth slice
    costs z-resolution the consumer does not use -- the batch contract samples 16
    slices from whatever is cached. Matches
    ``kappa.data.dataset.sample_slice_indices``, which decimates the same way, so
    the two stages never fight over which end to drop.
    """
    if n_available < 1:
        raise ValueError(f"n_available must be >= 1, got {n_available}")
    if max_slices < 1:
        raise ValueError(f"max_slices must be >= 1, got {max_slices}")
    if n_available <= max_slices:
        return np.arange(n_available, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0.0, n_available - 1, max_slices)).astype(np.int64))


def window_to_uint8(
    stack: np.ndarray,
    percentiles: tuple[float, float] = WINDOW_PERCENTILES,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Clip a whole series to its own percentile window and scale to 0-255 float.

    Returns float, not ``uint8``: the caller still has to pad and resize, and
    doing that in float and rounding once at the end avoids stacking two
    quantisation errors. The window is computed over **every pixel of every
    retained slice at once** -- that is what makes it per-series.

    A series whose window has zero width (a uniformly flat acquisition, or one
    where the 1st and 99th percentiles coincide because 98% of the frame is
    background) falls back to full range, and if that is flat too the result is
    all zeros. Returning zeros is honest for a constant input; dividing by zero
    would return NaN, and NaN pixels reach the loss.
    """
    low_pct, high_pct = float(percentiles[0]), float(percentiles[1])
    if not 0.0 <= low_pct < high_pct <= 100.0:
        raise ValueError(f"percentiles must satisfy 0 <= low < high <= 100, got {percentiles}")
    values = np.asarray(stack, dtype=np.float32)
    low, high = (float(v) for v in np.percentile(values, (low_pct, high_pct)))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values)
    # Clipped again after scaling, not only before: (high - low) is a float32
    # reciprocal, so the top of the range lands at 255.00002 and would round to
    # 256, which wraps to 0 on the cast to uint8 -- the brightest pixels of the
    # series would come back black.
    scaled = np.clip((np.clip(values, low, high) - low) * (255.0 / (high - low)), 0.0, 255.0)
    return 255.0 - scaled if invert else scaled


def letterbox(plane: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Pad a 2-D plane to square, centred, so a later resize is isotropic.

    Padding before resizing is the whole point: 45 distinct in-plane aspect
    ratios exist in this corpus, so a direct resize to a square scales the two
    axes by different factors and the same anatomy comes out a different shape
    per site. ``fill`` is 0 because windowing has already run, which puts
    background at 0 -- so the pad is indistinguishable from air rather than
    being a grey band the encoder can key on.
    """
    if plane.ndim != 2:
        raise ValueError(f"letterbox expects a 2-D plane, got shape {plane.shape}")
    height, width = plane.shape
    side = max(height, width)
    if height == width:
        return plane
    top = (side - height) // 2
    left = (side - width) // 2
    out = np.full((side, side), fill, dtype=plane.dtype)
    out[top : top + height, left : left + width] = plane
    return out


def _resample_weights(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangle-filter taps for a 1-D resize, widened when downsampling.

    Widening the filter to the downsample factor is what makes this anti-aliased.
    Plain bilinear at 1024 -> 512 reads two of every four input pixels and
    ignores the rest, which turns fine trabecular texture into noise that changes
    with sub-pixel alignment. A triangle of radius ``n_in / n_out`` averages the
    pixels the output actually covers. At ``n_in == n_out`` the taps collapse to
    the identity, so a native-512 series passes through untouched.

    Border indices are clamped, i.e. the edge pixel is replicated. The weights
    are normalised before clamping, and duplicate indices simply accumulate, so
    every output pixel still has total weight 1 and no edge darkening appears.
    """
    if n_in < 1 or n_out < 1:
        raise ValueError(f"resize needs positive sizes, got {n_in} -> {n_out}")
    scale = n_in / n_out
    support = max(1.0, scale)
    centres = (np.arange(n_out, dtype=np.float64) + 0.5) * scale
    n_taps = int(np.ceil(2.0 * support)) + 1
    left = np.floor(centres - support + 0.5).astype(np.int64)
    index = left[:, None] + np.arange(n_taps, dtype=np.int64)[None, :]
    distance = np.abs((index + 0.5) - centres[:, None]) / support
    weight = np.clip(1.0 - distance, 0.0, None)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.clip(index, 0, n_in - 1), weight.astype(np.float32)


def _resample_axis(values: np.ndarray, axis: int, n_out: int) -> np.ndarray:
    """Resize one axis of an array with the anti-aliased triangle filter.

    Gathers only the few taps each output pixel needs instead of multiplying by
    a dense ``(n_out, n_in)`` matrix. At 768 -> 512 that is 4 taps rather than
    768, which is the difference between a cache build measured in hours and one
    measured in days.
    """
    n_in = values.shape[axis]
    if n_in == n_out:
        return values
    index, weight = _resample_weights(n_in, n_out)
    # np.take inserts the index array's two dimensions at `axis`; move them to
    # the front so one einsum handles either axis.
    gathered = np.moveaxis(np.take(values, index, axis=axis), (axis, axis + 1), (0, 1))
    reduced = np.einsum("ot,ot...->o...", weight, gathered, optimize=True)
    return np.moveaxis(reduced, 0, axis).astype(np.float32)


def resize_plane(plane: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2-D plane to ``size`` x ``size`` with an anti-aliased separable filter.

    Pure NumPy on purpose. ``cv2``, ``PIL`` and ``scipy`` are none of them
    declared dependencies of this package, and ``torch`` is confined to
    ``kappa.models`` and ``kappa.training`` by the packaging contract -- so a
    resize that reached for any of them would make the cache build, which runs on
    a CPU-only node, depend on the modelling extra.
    """
    if plane.ndim != 2:
        raise ValueError(f"resize_plane expects a 2-D plane, got shape {plane.shape}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    out = np.asarray(plane, dtype=np.float32)
    out = _resample_axis(out, 0, size)
    return _resample_axis(out, 1, size)


def _first(dataset: object, *names: str, default: object = None) -> object:
    """First present, non-empty attribute among ``names``, else ``default``."""
    for name in names:
        value = getattr(dataset, name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        return value
    return default


def _as_float(value: object) -> float | None:
    """Coerce a DICOM value to float; absent, empty and unparseable become None.

    DICOM writes an absent numeric as an empty string as often as it omits the
    tag, and ``float("")`` raises. None keeps a missing field visibly missing
    instead of turning it into a 0.0 that reads like a measurement.
    """
    if value is None:
        return None
    if isinstance(value, (bytes, str)):
        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(value, Sequence):
        return _as_float(value[0]) if len(value) else None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_str(value: object) -> str:
    """Coerce a DICOM value to a stripped string; absent becomes empty."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    if not isinstance(value, str) and isinstance(value, Sequence):
        return _as_str(value[0]) if len(value) else ""
    return str(value).strip()


def _header_from_dataset(dataset: object, n_slices: int) -> SeriesHeader:
    """Build a :class:`SeriesHeader` from one already-read pydicom dataset.

    ``spacing`` is ``PixelSpacing[0]``, the row spacing. In-plane spacing is
    isotropic in this corpus (the two values agree on every file surveyed), and
    the manifest column is a scalar, so keeping both would only invite a caller
    to pick one arbitrarily. ``thickness`` falls back to
    ``SpacingBetweenSlices`` when ``SliceThickness`` is absent: the two differ
    wherever there is a gap, and for the fingerprint's purpose either identifies
    the protocol.
    """
    return SeriesHeader(
        study_uid=_as_str(getattr(dataset, "StudyInstanceUID", "")),
        series_uid=_as_str(getattr(dataset, "SeriesInstanceUID", "")),
        patient_id=_as_str(getattr(dataset, "PatientID", "")),
        laterality=_as_str(_first(dataset, "Laterality", "ImageLaterality", default="")),
        manufacturer=_as_str(getattr(dataset, "Manufacturer", "")),
        model=_as_str(getattr(dataset, "ManufacturerModelName", "")),
        field_strength=_as_float(getattr(dataset, "MagneticFieldStrength", None)),
        rows=int(_as_float(getattr(dataset, "Rows", 0)) or 0),
        cols=int(_as_float(getattr(dataset, "Columns", 0)) or 0),
        spacing=_as_float(getattr(dataset, "PixelSpacing", None)),
        thickness=_as_float(
            _first(dataset, "SliceThickness", "SpacingBetweenSlices", default=None)
        ),
        n_slices=int(n_slices),
    )


def read_series_header(series_dir: Path | str) -> SeriesHeader:
    """Geometry and scanner fields for a series, reading **one** file and no pixels.

    ``n_slices`` is the file count, not a tag: no DICOM attribute records how
    many instances a series has, and ``ImagesInAcquisition`` is both optional and
    frequently wrong. The single file is the first by filename, which is
    arbitrary and adequate -- every field here is a series-level constant.

    Header-only because the manifest build reads all 24,371 series of 4,407
    studies. Decoding their pixels would be the entire cost of that pass for
    none of its value.
    """
    import pydicom

    files = series_files(series_dir)
    if not files:
        raise FileNotFoundError(f"no DICOM files under {Path(series_dir)}")
    dataset = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    return _header_from_dataset(dataset, n_slices=len(files))


def _slice_pixels(dataset: object) -> np.ndarray:
    """Rescaled float32 pixels of one slice, signed and stored-bit handling included.

    ``pixel_array`` already honours ``PixelRepresentation`` and ``BitsStored``,
    so signed data arrives as ``int16`` with the sign intact rather than as a
    huge unsigned value; the job here is the affine
    ``RescaleSlope``/``RescaleIntercept`` map, which is absent on much of this
    corpus and must not be assumed to be the identity when it is present.
    """
    pixels = np.asarray(dataset.pixel_array)  # type: ignore[attr-defined]
    if pixels.ndim != 2:
        raise ValueError(
            f"expected a single-frame 2-D image, got shape {pixels.shape}. A "
            "multi-frame instance needs its frames split into slices first, and "
            "silently taking frame 0 would drop the rest of the series."
        )
    slope = _as_float(getattr(dataset, "RescaleSlope", None))
    intercept = _as_float(getattr(dataset, "RescaleIntercept", None))
    out = pixels.astype(np.float32)
    if slope is not None and slope != 1.0:
        out = out * np.float32(slope)
    if intercept is not None and intercept != 0.0:
        out = out + np.float32(intercept)
    return out


def read_series(
    series_dir: Path | str,
    size: int = CACHE_SIZE,
    max_slices: int = MAX_SLICES,
    window: tuple[float, float] = WINDOW_PERCENTILES,
) -> SeriesVolume:
    """Decode a series to ``(S, size, size)`` ``uint8``, ordered by geometry.

    The order of operations is load-bearing and is the reason this is one
    function rather than four composable ones:

    1. read every header, project ``ImagePositionPatient`` onto the slice normal,
       and sort ascending -- **never** by ``InstanceNumber``;
    2. decimate to ``max_slices`` across the full extent;
    3. decode only the retained files;
    4. window over the whole retained stack at once;
    5. letterbox each plane to square, then resize.

    Windowing must precede padding, or the pad value joins the percentile
    population and shifts the window; and it must precede resizing, because the
    clip is non-linear and clipping after interpolation is not the same
    operation. Reading headers first means a 320-slice series decodes 64 files
    rather than 320.
    """
    import pydicom

    files = series_files(series_dir)
    if not files:
        raise FileNotFoundError(f"no DICOM files under {Path(series_dir)}")

    headers = [pydicom.dcmread(str(path), stop_before_pixels=True) for path in files]
    orientation = _first(headers[0], "ImageOrientationPatient")
    if orientation is None:
        raise ValueError(
            f"{files[0]} has no ImageOrientationPatient, so slice order cannot be "
            "derived from geometry. Falling back to InstanceNumber is not "
            "acceptable: it disagreed with physical order on 49 of 120 series probed "
            "(40.8%, measured over the local subset)."
        )
    positions = []
    reference = np.asarray([float(value) for value in orientation], dtype=np.float64)
    for path, dataset in zip(files, headers):
        origin = _first(dataset, "ImagePositionPatient")
        if origin is None:
            raise ValueError(f"{path} has no ImagePositionPatient, so it cannot be ordered")
        positions.append([float(value) for value in origin])

        # Every slice is projected onto the FIRST slice's normal, so a slice acquired
        # in a different plane is projected onto an axis that is not its own and lands
        # at a meaningless position. That reorders the volume silently -- no exception,
        # no shape mismatch, just a stack in the wrong order. Checked per slice rather
        # than assumed, because the mixed-matrix guard further down catches a different
        # thing (in-plane size) and would not notice this.
        own = _first(dataset, "ImageOrientationPatient")
        if own is None:
            raise ValueError(f"{path} has no ImageOrientationPatient")
        if not np.allclose(np.asarray([float(v) for v in own], dtype=np.float64),
                           reference, atol=1e-4):
            raise ValueError(
                f"{path} has ImageOrientationPatient {[float(v) for v in own]}, which "
                f"differs from the series' first slice {reference.tolist()}. Slices in "
                "two planes are not one volume, and projecting them onto a single "
                "normal orders them wrongly without failing."
            )
    projections = slice_projections(positions, orientation)

    # All-equal projections mean the geometry says nothing about order, and the stable
    # argsort below would quietly fall back to filename order -- which is the very
    # thing this function exists to avoid. Filenames are SOP UIDs here, so that order
    # is arbitrary, not merely a different convention.
    if len(projections) > 1 and float(np.ptp(projections)) == 0.0:
        raise ValueError(
            f"{Path(series_dir)}: all {len(projections)} slices project to the same "
            "point on the slice normal, so geometry cannot order them. Sorting by "
            "filename here would be an arbitrary order presented as an anatomical one."
        )

    # Stable sort so that slices sharing a z -- interleaved echoes, a repeated
    # acquisition -- keep filename order instead of shuffling between runs.
    order = np.argsort(projections, kind="stable")
    keep = order[select_slice_indices(len(order), max_slices)]

    invert = _as_str(getattr(headers[0], "PhotometricInterpretation", "")).upper() == (
        _INVERTED_PHOTOMETRIC
    )
    planes = [_slice_pixels(pydicom.dcmread(str(files[index]))) for index in keep]
    shapes = {plane.shape for plane in planes}
    if len(shapes) > 1:
        raise ValueError(
            f"{Path(series_dir)} mixes in-plane matrices {sorted(shapes)}; a series "
            "whose slices are different sizes is not one volume"
        )

    windowed = window_to_uint8(np.stack(planes), window, invert=invert)
    pixels = np.empty((len(planes), size, size), dtype=np.uint8)
    for position, plane in enumerate(windowed):
        resized = resize_plane(letterbox(plane), size)
        pixels[position] = np.clip(np.rint(resized), 0, 255).astype(np.uint8)

    header = _header_from_dataset(headers[0], n_slices=len(files))
    return SeriesVolume(
        pixels=pixels,
        slice_positions=projections[keep].astype(np.float64),
        sop_uids=tuple(_as_str(getattr(headers[index], "SOPInstanceUID", "")) for index in keep),
        header=header,
    )


def study_dirs(series_root: Path | str) -> list[Path]:
    """Study directories under ``<series_root>``, sorted, AppleDouble excluded.

    Sorted so that the shard arithmetic in :func:`build_cache` is a pure function
    of the directory contents. An array element that computed its share from an
    unsorted listing would process a different slice of the corpus on every
    resubmission, and the "already cached, skip it" resume would stop working.
    """
    root = Path(series_root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a series root")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(_APPLEDOUBLE_PREFIX)
    )


@dataclass(frozen=True)
class CacheReport:
    """What one cache-build pass actually did, for a log line and an exit code."""

    n_studies: int
    n_written: int
    n_skipped: int
    n_failed: int
    n_slices: int
    bytes_written: int
    seconds: float
    failures: tuple[tuple[str, str], ...] = ()

    @property
    def n_attempted(self) -> int:
        """Series this pass tried to decode, i.e. excluding the resumed ones."""
        return self.n_written + self.n_failed

    def summary(self) -> str:
        """One-line summary, including the per-series rate the next job is sized from."""
        rate = self.seconds / self.n_written if self.n_written else float("nan")
        return (
            f"{self.n_studies} studies: {self.n_written} written, "
            f"{self.n_skipped} already cached, {self.n_failed} failed; "
            f"{self.n_slices} slices, {self.bytes_written / 1e9:.2f} GB, "
            f"{self.seconds:.1f}s ({rate:.3f} s/series written)"
        )


def cache_series(
    series_dir: Path | str,
    out_path: Path | str,
    *,
    size: int = CACHE_SIZE,
    max_slices: int = MAX_SLICES,
    window: tuple[float, float] = WINDOW_PERCENTILES,
    overwrite: bool = False,
) -> int:
    """Decode one series to ``out_path``. Returns bytes written, or 0 if skipped.

    Written to a temporary name in the same directory and renamed into place.
    ``np.save`` on a killed process leaves a half-written ``.npy`` whose header
    still parses, and the resume check ("does the file exist?") would then treat
    a truncated volume as done -- an array job cancelled at walltime would
    poison exactly the studies it was working on. ``os.replace`` is atomic within
    a filesystem, so the file either is not there or is complete.
    """
    target = Path(out_path)
    if target.is_file() and not overwrite:
        return 0
    volume = read_series(series_dir, size=size, max_slices=max_slices, window=window)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.partial")
    try:
        with open(staging, "wb") as handle:
            np.save(handle, volume.pixels, allow_pickle=False)
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)
    return target.stat().st_size


def build_cache(
    series_root: Path | str,
    cache_root: Path | str,
    *,
    studies: Sequence[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    size: int = CACHE_SIZE,
    max_slices: int = MAX_SLICES,
    window: tuple[float, float] = WINDOW_PERCENTILES,
    overwrite: bool = False,
    limit: int | None = None,
    progress_every: int = 25,
) -> CacheReport:
    """Cache every series of a shard of studies, skipping what is already cached.

    Sharding is ``sorted(studies)[shard_index::shard_count]``, a stride rather
    than a contiguous block. A block split would put all of one site's studies in
    one element, and slice counts run 11-320, so the elements would finish hours
    apart and the job would be sized by its slowest one. A stride interleaves the
    corpus, so every element sees the same mix.

    A series that fails to decode is recorded and the pass continues. One
    unreadable series out of ~24,000 must not cost the other 23,999: the caller
    decides whether the failure count is acceptable, from
    :attr:`CacheReport.failures`.
    """
    root = Path(series_root)
    cache = Path(cache_root)
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}")

    if studies is None:
        selected = study_dirs(root)
    else:
        selected = [root / uid for uid in sorted(set(studies))]
    selected = selected[shard_index::shard_count]
    if limit is not None:
        selected = selected[:limit]

    written = skipped = failed = n_slices = 0
    total_bytes = 0
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    for position, study in enumerate(selected):
        if not study.is_dir():
            failures.append((study.name, "no such study directory"))
            failed += 1
            continue
        for series in sorted(
            path
            for path in study.iterdir()
            if path.is_dir() and not path.name.startswith(_APPLEDOUBLE_PREFIX)
        ):
            target = cache / study.name / f"{series.name}.npy"
            if target.is_file() and not overwrite:
                skipped += 1
                continue
            try:
                size_bytes = cache_series(
                    series,
                    target,
                    size=size,
                    max_slices=max_slices,
                    window=window,
                    overwrite=overwrite,
                )
            except Exception as exc:  # noqa: BLE001 - one bad series must not stop the pass
                failures.append((f"{study.name}/{series.name}", f"{type(exc).__name__}: {exc}"))
                failed += 1
                continue
            written += 1
            total_bytes += size_bytes
            # (S, size, size) uint8, so the slice count is recoverable from the
            # file size without re-reading the array.
            n_slices += size_bytes // (size * size)
        if progress_every and position and position % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {position}/{len(selected)} studies, {written} series written, "
                f"{elapsed:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    return CacheReport(
        n_studies=len(selected),
        n_written=written,
        n_skipped=skipped,
        n_failed=failed,
        n_slices=int(n_slices),
        bytes_written=int(total_bytes),
        seconds=time.perf_counter() - started,
        failures=tuple(failures),
    )
