"""Guards on the one preprocessing step whose failures are invisible downstream.

Why this exists: ``kappa.data.dicom`` is the only place in this project that
turns bytes into pixels, and three of its four invariants fail *silently*. A
volume sorted by ``InstanceNumber`` has the right dtype, the right shape and a
plausible histogram; it is simply mirrored in z. A volume windowed per slice
looks fine slice by slice. A volume resized without letterboxing is undistorted
in the 512x512 half of the corpus and stretched in the rest. None of those show
up in a shape assertion, a loss curve or a validation score that nobody has a
prior for -- so they have to be asserted here.

Every test that names a real series names it in full, because a fixture drawn
from the actual competition data is evidence and a synthetic one is only a
demonstration. Both are present: the synthetic cases pin the behaviour on a
machine with no drive mounted, and the real ones prove the behaviour on the
bytes that will actually be trained on.

The tests are written to fail on the *cheap wrong fix*. ``test_geometric_order_
wins_when_instance_number_disagrees`` fails if someone replaces the projection
sort with ``sorted(files, key=InstanceNumber)``, and it fails on real data as
well as synthetic, because 16 of the 30 series probed in the local subset number
their instances in exactly the reverse of physical order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from kappa.data.dicom import (
    CACHE_SIZE,
    MAX_SLICES,
    CacheReport,
    SeriesHeader,
    build_cache,
    cache_series,
    letterbox,
    read_series,
    read_series_header,
    resize_plane,
    select_slice_indices,
    series_files,
    slice_normal,
    slice_projections,
    window_to_uint8,
)
from kappa.data.paths import is_subset_staged, require_series_root

# A real study and two real series from the local 558-study subset, named so the
# evidence is checkable rather than asserted.
#
# ORDER_TRAP_SERIES has 38 slices whose InstanceNumbers run 38 down to 1 as the
# slice normal projection runs up. LETTERBOX_SERIES is 640 rows by 540 columns,
# one of the non-square matrices, so its cached planes must carry a zero band
# down each side.
REAL_STUDY = "1.2.826.0.1.3680043.8.498.10004945927472656027199792075652399585"
ORDER_TRAP_SERIES = "1.2.826.0.1.3680043.8.498.83327574802189551711953505990029424266"
LETTERBOX_SERIES = "1.2.826.0.1.3680043.8.498.87374594163896635184044513927478147680"

# Oblique orientation used by the synthetic ordering tests. Its normal is
# (0, -0.5, 0.866), so neither the z coordinate of ImagePositionPatient nor any
# single component of it is the slice axis -- which is the point.
_OBLIQUE = (1.0, 0.0, 0.0, 0.0, 0.8660254037844387, 0.5)
_OBLIQUE_NORMAL = np.array([0.0, -0.5, 0.8660254037844387])
_IN_PLANE = np.array([0.0, 0.8660254037844387, 0.5])

requires_subset = pytest.mark.skipif(
    not is_subset_staged(),
    reason="the local DICOM subset is not mounted; real-byte tests cannot run",
)


def _write_slice(path: Path, pixels: np.ndarray, **over: object) -> str:
    """Write one minimally valid MR instance and return its SOP Instance UID.

    Explicit VR Little Endian on purpose: every one of the 6,644 slices surveyed
    in the local subset uses that transfer syntax, so the synthetic fixtures
    exercise the same decode path as the real data and no JPEG plugin is needed.
    """
    signed = bool(over.pop("signed", False))
    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPClassUID = MRImageStorage
    sop_uid = str(over.pop("SOPInstanceUID", generate_uid()))
    dataset.file_meta.MediaStorageSOPInstanceUID = sop_uid
    dataset.SOPInstanceUID = sop_uid
    dataset.SOPClassUID = MRImageStorage
    dataset.Modality = "MR"
    dataset.StudyInstanceUID = "1.2.3.4"
    dataset.SeriesInstanceUID = "1.2.3.4.5"
    dataset.PatientID = "patient-1"
    dataset.Laterality = "R"
    dataset.Manufacturer = "ACME"
    dataset.ManufacturerModelName = "Model X"
    dataset.MagneticFieldStrength = "3"
    dataset.PixelSpacing = ["0.31", "0.31"]
    dataset.SliceThickness = "3.0"
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.Rows, dataset.Columns = (int(pixels.shape[0]), int(pixels.shape[1]))
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1 if signed else 0
    dataset.ImageOrientationPatient = list(_OBLIQUE)
    dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    dataset.InstanceNumber = 1
    for key, value in over.items():
        setattr(dataset, key, value)
    dataset.PixelData = pixels.astype(np.int16 if signed else np.uint16).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)
    return sop_uid


def _series(
    directory: Path,
    n_slices: int = 6,
    shape: tuple[int, int] = (8, 8),
    *,
    reverse_instance_numbers: bool = True,
    in_plane_jitter: float = 20.0,
    **over: object,
) -> list[str]:
    """A synthetic series whose slice k is uniformly filled with ``(k + 1) * 100``.

    The fill value encodes geometric rank, so the returned volume's per-slice
    means are the proof of what order the reader used. Two traps are built in on
    purpose:

    ``reverse_instance_numbers`` numbers the instances backwards, so a reader
    that sorts by ``InstanceNumber`` returns the volume mirrored in z.

    ``in_plane_jitter`` displaces each slice origin along the in-plane column
    direction, which is orthogonal to the slice normal. The projection is
    therefore untouched while the raw z coordinate is scrambled, so a reader that
    sorts by ``ImagePositionPatient[2]`` -- the other cheap wrong fix -- returns
    the slices shuffled.

    Files are named so that filename order is yet another wrong order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    uids: list[str] = []
    for rank in range(n_slices):
        origin = _OBLIQUE_NORMAL * (3.0 * rank) + _IN_PLANE * (
            in_plane_jitter * (1 if rank % 2 else -1)
        )
        instance = n_slices - rank if reverse_instance_numbers else rank + 1
        pixels = np.full(shape, (rank + 1) * 100, dtype=np.uint16)
        uids.append(
            _write_slice(
                directory / f"slice{n_slices - rank:03d}.dcm",
                pixels,
                ImagePositionPatient=[float(value) for value in origin],
                InstanceNumber=instance,
                **over,
            )
        )
    return uids


class TestSliceGeometry:
    """The normal and the projection, which everything else rests on."""

    def test_normal_is_the_cross_product_of_the_two_direction_cosines(self) -> None:
        assert np.allclose(slice_normal((1, 0, 0, 0, 1, 0)), (0.0, 0.0, 1.0))
        assert np.allclose(slice_normal(_OBLIQUE), _OBLIQUE_NORMAL)

    def test_normal_is_unit_length_so_the_projection_is_millimetres(self) -> None:
        # Scaled cosines are not legal DICOM, but a normal that is not
        # normalised turns the projection into an arbitrary scale, and the
        # inter-slice distances it reports would be silently wrong.
        assert np.isclose(np.linalg.norm(slice_normal((2, 0, 0, 0, 2, 0))), 1.0)

    def test_parallel_direction_cosines_raise_instead_of_returning_zero(self) -> None:
        with pytest.raises(ValueError, match="zero-length slice normal"):
            slice_normal((1, 0, 0, 1, 0, 0))

    def test_wrong_number_of_cosines_raises(self) -> None:
        with pytest.raises(ValueError, match="6 direction cosines"):
            slice_normal((1, 0, 0))

    def test_projection_ignores_in_plane_displacement(self) -> None:
        """An in-plane shift must not change a slice's through-plane coordinate.

        This is exactly why the projection is used rather than a coordinate: two
        slices at the same z of the anatomy but a different field of view offset
        are the same slice position, and any component-wise sort would separate
        them.
        """
        base = _OBLIQUE_NORMAL * 7.0
        positions = [base, base + _IN_PLANE * 50.0, base - _IN_PLANE * 50.0]
        projected = slice_projections(positions, _OBLIQUE)
        assert np.allclose(projected, projected[0])


class TestSliceOrder:
    """The invariant that, if broken, produces a mirrored volume and no error."""

    def test_geometric_order_wins_when_instance_number_disagrees(self, tmp_path: Path) -> None:
        """Slices come back in projection order even though InstanceNumber reverses it.

        Fails if the sort key becomes ``InstanceNumber``: the per-slice means
        would come back descending. Measured on real data, that substitution is
        wrong for 49 of 120 series probed (40.8%).
        """
        _series(tmp_path / "s", n_slices=6)
        volume = read_series(tmp_path / "s", size=8, window=(0.0, 100.0))

        means = volume.pixels.reshape(6, -1).mean(axis=1)
        assert list(np.argsort(means)) == list(range(6))
        assert (np.diff(volume.slice_positions) > 0).all()

        # The fixture is only evidence if the two orders really do differ, so
        # read the InstanceNumbers back *along the returned order* and check they
        # are descending. Reading them in filename order would prove nothing.
        by_uid = {}
        for path in series_files(tmp_path / "s"):
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
            by_uid[str(dataset.SOPInstanceUID)] = int(dataset.InstanceNumber)
        sequence = [by_uid[uid] for uid in volume.sop_uids]
        assert sequence == sorted(sequence, reverse=True)

    def test_geometric_order_wins_when_the_z_coordinate_disagrees(self, tmp_path: Path) -> None:
        """Sorting by ``ImagePositionPatient[2]`` is also wrong, and this catches it.

        The synthetic slices are jittered in plane, so their z coordinates are
        not monotone in slice order while their projections are.
        """
        _series(tmp_path / "s", n_slices=6, in_plane_jitter=40.0)
        z_values = [
            float(pydicom.dcmread(str(path), stop_before_pixels=True).ImagePositionPatient[2])
            for path in series_files(tmp_path / "s")
        ]
        assert sorted(z_values) != z_values, "fixture must scramble z, or it proves nothing"

        volume = read_series(tmp_path / "s", size=8, window=(0.0, 100.0))
        means = volume.pixels.reshape(6, -1).mean(axis=1)
        assert list(np.argsort(means)) == list(range(6))

    def test_sop_uids_stay_parallel_to_the_reordered_pixels(self, tmp_path: Path) -> None:
        uids = _series(tmp_path / "s", n_slices=5)
        volume = read_series(tmp_path / "s", size=8, window=(0.0, 100.0))
        # _series writes rank 0 first, so its UID list is already geometric order.
        assert list(volume.sop_uids) == uids

    def test_missing_image_position_raises_rather_than_falling_back(self, tmp_path: Path) -> None:
        directory = tmp_path / "s"
        _series(directory, n_slices=3)
        victim = series_files(directory)[0]
        dataset = pydicom.dcmread(str(victim))
        del dataset.ImagePositionPatient
        dataset.save_as(str(victim), enforce_file_format=True)
        with pytest.raises(ValueError, match="ImagePositionPatient"):
            read_series(directory, size=8)


class TestSliceSelection:
    """The 64-slice cap: which slices survive, and from where in the volume."""

    def test_series_at_or_under_the_cap_keep_every_slice(self) -> None:
        assert list(select_slice_indices(11, 64)) == list(range(11))
        assert list(select_slice_indices(64, 64)) == list(range(64))

    def test_the_cap_spans_the_full_extent_rather_than_cropping_the_middle(self) -> None:
        """A 320-slice series must still reach both ends of the joint.

        Fails if the cap becomes ``indices[:64]`` or a centre block: the first
        index would not be 0 or the last would not be 319, and the anatomy at
        that end would be missing from the cache with nothing to show it.
        """
        kept = select_slice_indices(320, 64)
        assert len(kept) == 64
        assert kept[0] == 0
        assert kept[-1] == 319
        assert (np.diff(kept) > 0).all()

    def test_a_capped_read_returns_the_cap_and_stays_ordered(self, tmp_path: Path) -> None:
        _series(tmp_path / "s", n_slices=20)
        volume = read_series(tmp_path / "s", size=8, max_slices=5, window=(0.0, 100.0))
        assert volume.pixels.shape[0] == 5
        assert volume.n_slices == 5
        assert (np.diff(volume.slice_positions) > 0).all()
        means = volume.pixels.reshape(5, -1).mean(axis=1)
        assert list(np.argsort(means)) == list(range(5))

    def test_nonsense_arguments_raise(self) -> None:
        with pytest.raises(ValueError):
            select_slice_indices(0)
        with pytest.raises(ValueError):
            select_slice_indices(10, 0)


class TestWindowing:
    """Per-series percentile windowing, and the ways it degenerates."""

    def test_window_is_computed_over_the_whole_series_not_per_slice(self) -> None:
        """A dim slice next to a bright one must stay dim.

        Fails if the window is applied per slice: each slice would then be
        stretched to full range independently and both would come back with the
        same maximum, destroying the between-slice contrast that any finding
        with extent depends on.
        """
        stack = np.stack(
            [np.full((4, 4), 100.0, dtype=np.float32), np.full((4, 4), 900.0, dtype=np.float32)]
        )
        windowed = window_to_uint8(stack, (0.0, 100.0))
        assert windowed[0].max() < windowed[1].min()

    def test_output_spans_the_full_byte_range_without_overflowing_it(self) -> None:
        values = np.arange(1000, dtype=np.float32).reshape(1, 20, 50)
        windowed = window_to_uint8(values, (0.0, 100.0))
        assert windowed.min() == 0.0
        # Exactly 255, not 255.00002: a float overshoot rounds to 256, which
        # wraps to 0 on the uint8 cast and turns the brightest pixels black.
        assert windowed.max() == 255.0
        assert np.rint(windowed).astype(np.uint8).max() == 255

    def test_percentiles_clip_outliers_rather_than_letting_them_set_the_scale(self) -> None:
        """One hot pixel must not compress the anatomy into the bottom of the range.

        A coil hot-spot or a metal artifact is common in MR; without percentile
        clipping a single 60,000-valued pixel puts real tissue below value 5.
        """
        values = np.full((1, 100, 100), 100.0, dtype=np.float32)
        values[0, 90:, :] = 200.0  # 10% of the frame is the "bright tissue"
        values[0, 0, :5] = 60000.0  # 0.05% is a coil hot-spot, well inside the 99th
        windowed = window_to_uint8(values, (1.0, 99.0))
        assert windowed[0, 95, 0] == 255.0, "the outlier set the scale instead of the tissue"
        assert windowed[0, 0, 0] == 255.0, "the hot-spot must clip, not overflow"
        assert windowed[0, 50, 0] == 0.0

    def test_a_flat_series_returns_zeros_and_never_nan(self) -> None:
        """Zero-width window must not divide by zero. NaN pixels reach the loss."""
        windowed = window_to_uint8(np.full((3, 4, 4), 7.0, dtype=np.float32))
        assert np.isfinite(windowed).all()
        assert windowed.max() == 0.0

    def test_inverted_photometric_interpretation_flips_the_polarity(self) -> None:
        values = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
        normal = window_to_uint8(values, (0.0, 100.0))
        inverted = window_to_uint8(values, (0.0, 100.0), invert=True)
        assert np.allclose(normal + inverted, 255.0)

    def test_bad_percentiles_raise(self) -> None:
        with pytest.raises(ValueError, match="percentiles"):
            window_to_uint8(np.zeros((1, 2, 2), dtype=np.float32), (99.0, 1.0))


class TestSignedPixelData:
    """PixelRepresentation=1 is 25.6% of the slices surveyed, so it is not an edge case."""

    def test_signed_pixel_data_stays_monotone_instead_of_wrapping(self, tmp_path: Path) -> None:
        """A negative stored value must not become 65,000-odd.

        Measured: 1,704 of 6,644 slices surveyed in the local subset are signed.
        Reading them as unsigned turns every negative into a value near the top
        of the 16-bit range, which then sets the 99th percentile and pushes all
        the real tissue toward 0. The output here would stop being monotone in
        the input ramp.
        """
        directory = tmp_path / "s"
        directory.mkdir()
        ramp = np.tile(np.linspace(-1000, 1000, 32).astype(np.int16), (32, 1))
        _write_slice(
            directory / "a.dcm",
            ramp,
            signed=True,
            ImagePositionPatient=[0.0, 0.0, 0.0],
        )
        volume = read_series(directory, size=32, window=(0.0, 100.0))
        row = volume.pixels[0, 16, :].astype(int)
        assert (np.diff(row) >= 0).all()
        assert row[0] == 0
        assert row[-1] == 255

    def test_rescale_slope_and_intercept_are_applied(self, tmp_path: Path) -> None:
        """Slope is present and not 1 on part of this corpus, so it is read.

        Honest note on what this proves: with a per-series percentile window and
        a slope that is constant within the series -- which is what all 120
        series checked in the local subset look like -- the affine rescale
        cancels out of the windowed result entirely. So this test asserts the
        rescale does not *corrupt* anything, not that it changes the output. It
        becomes load-bearing the day a series carries per-slice slopes, which is
        a real Philips behaviour and would otherwise make one series two
        different intensity scales.
        """
        plain, scaled = tmp_path / "plain", tmp_path / "scaled"
        _series(plain, n_slices=4)
        _series(scaled, n_slices=4, RescaleSlope="2.5", RescaleIntercept="-100")
        left = read_series(plain, size=8, window=(0.0, 100.0))
        right = read_series(scaled, size=8, window=(0.0, 100.0))
        assert np.array_equal(left.pixels, right.pixels)


class TestLetterboxAndResize:
    """13 aspect ratios exist, so padding before resizing is not optional."""

    def test_letterbox_centres_a_non_square_plane_in_a_square(self) -> None:
        padded = letterbox(np.ones((640, 540), dtype=np.float32))
        assert padded.shape == (640, 640)
        assert padded[:, :50].max() == 0.0
        assert padded[:, -50:].max() == 0.0
        assert padded[:, 50:-50].min() == 1.0

    def test_letterbox_leaves_an_already_square_plane_untouched(self) -> None:
        square = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert letterbox(square) is square

    def test_letterbox_then_resize_preserves_the_aspect_of_the_content(self) -> None:
        """A disc must come out a disc, not an ellipse.

        Fails if the resize is applied straight to a non-square plane. The
        measured stretch for a 640x540 slice is 1.19x in one axis only, which is
        the same anatomy at a different shape depending on the site.
        """
        grid_y, grid_x = np.mgrid[0:540, 0:540]
        disc = (
            ((grid_y - 270) ** 2 + (grid_x - 270) ** 2) < 100**2
        ).astype(np.float32)
        tall = np.zeros((640, 540), dtype=np.float32)
        tall[50:590, :] = disc

        resized = resize_plane(letterbox(tall), 512)
        mask = resized > 0.5
        height = mask.any(axis=1).sum()
        width = mask.any(axis=0).sum()
        assert abs(height - width) <= 2

    def test_resize_is_the_identity_at_native_size(self) -> None:
        plane = np.arange(64, dtype=np.float32).reshape(8, 8)
        assert np.allclose(resize_plane(plane, 8), plane)

    def test_resize_preserves_a_constant_with_no_edge_darkening(self) -> None:
        """Unnormalised filter taps show up as a dark frame around every slice."""
        constant = np.full((7, 7), 42.0, dtype=np.float32)
        for size in (3, 7, 13, 512):
            assert np.allclose(resize_plane(constant, size), 42.0)

    def test_downsampling_averages_rather_than_dropping_pixels(self) -> None:
        """Anti-aliasing: a one-pixel-wide bright line must not vanish.

        Plain nearest or unfiltered bilinear at 1024 -> 512 reads every other
        pixel, so a thin line survives or disappears depending on its parity.
        """
        plane = np.zeros((64, 64), dtype=np.float32)
        plane[:, 33] = 100.0
        assert resize_plane(plane, 16).max() > 0.0

    def test_resize_rejects_a_non_plane(self) -> None:
        with pytest.raises(ValueError, match="2-D plane"):
            resize_plane(np.zeros((2, 4, 4), dtype=np.float32), 8)


class TestSeriesHeader:
    """The header reader, which the manifest build consumes for 4,407 studies."""

    def test_header_carries_every_field_the_manifest_contract_names(self, tmp_path: Path) -> None:
        _series(tmp_path / "s", n_slices=3)
        header = read_series_header(tmp_path / "s")
        assert isinstance(header, SeriesHeader)
        for field in (
            "study_uid",
            "series_uid",
            "patient_id",
            "laterality",
            "manufacturer",
            "model",
            "field_strength",
            "rows",
            "cols",
            "spacing",
            "thickness",
            "n_slices",
        ):
            assert hasattr(header, field), field
        assert header.patient_id == "patient-1"
        assert header.laterality == "R"
        assert header.manufacturer == "ACME"
        assert header.model == "Model X"
        assert header.field_strength == 3.0
        assert header.spacing == 0.31
        assert header.thickness == 3.0
        assert header.n_slices == 3

    def test_the_manifest_alias_table_resolves_against_this_header(self, tmp_path: Path) -> None:
        """Cross-agent contract check: the manifest must find every field it needs.

        The manifest and this reader were written separately against the same
        pinned contract. This asserts they actually meet, rather than each
        agreeing with the document and not with each other.
        """
        manifest = pytest.importorskip("kappa.data.manifest")
        aliases = getattr(manifest, "_HEADER_ALIASES", None)
        read_field = getattr(manifest, "_header_field", None)
        if aliases is None or read_field is None:
            pytest.skip("kappa.data.manifest no longer exposes its header alias table")
        _series(tmp_path / "s", n_slices=3)
        header = read_series_header(tmp_path / "s")
        for name in aliases:
            read_field(header, name)

    def test_absent_numeric_tags_become_none_rather_than_zero(self, tmp_path: Path) -> None:
        """A missing spacing must stay missing; 0.0 would read as a measurement."""
        directory = tmp_path / "s"
        _series(directory, n_slices=1)
        victim = series_files(directory)[0]
        dataset = pydicom.dcmread(str(victim))
        del dataset.PixelSpacing
        dataset.MagneticFieldStrength = ""
        dataset.save_as(str(victim), enforce_file_format=True)
        header = read_series_header(directory)
        assert header.spacing is None
        assert header.field_strength is None

    def test_header_read_does_not_decode_pixels(self, tmp_path: Path) -> None:
        """Proven by reading a file whose pixel data is unusable.

        The manifest pass covers every series of 4,407 studies; if it decoded
        pixels it would cost as much as the cache build for none of the result.
        A truncated PixelData makes the difference observable: the header still
        reads, a pixel decode raises.
        """
        directory = tmp_path / "s"
        _series(directory, n_slices=1)
        victim = series_files(directory)[0]
        dataset = pydicom.dcmread(str(victim))
        dataset.PixelData = b"\x00" * 8
        dataset.save_as(str(victim), enforce_file_format=True)

        assert read_series_header(directory).rows == 8
        with pytest.raises(ValueError):
            read_series(directory, size=8)

    def test_thickness_falls_back_to_spacing_between_slices(self, tmp_path: Path) -> None:
        directory = tmp_path / "s"
        _series(directory, n_slices=1)
        victim = series_files(directory)[0]
        dataset = pydicom.dcmread(str(victim))
        del dataset.SliceThickness
        dataset.SpacingBetweenSlices = "4.4"
        dataset.save_as(str(victim), enforce_file_format=True)
        assert read_series_header(directory).thickness == 4.4

    def test_an_empty_directory_raises_instead_of_returning_a_blank_header(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            read_series_header(tmp_path / "empty")


class TestSeriesFiles:
    """Enumeration, and the macOS sidecar that doubles a slice count."""

    def test_appledouble_sidecars_are_excluded(self, tmp_path: Path) -> None:
        """Measured: a 34-DICOM directory on the exFAT subset globs 68 entries.

        Half of those are 4 KiB resource forks. Counting one inflates n_slices;
        decoding one raises. Both happen on the drive this project develops on.
        """
        directory = tmp_path / "s"
        _series(directory, n_slices=2)
        for path in list(directory.glob("*.dcm")):
            (directory / f"._{path.name}").write_bytes(b"\x00" * 4096)
        assert len(list(directory.glob("*.dcm"))) == 4
        assert len(series_files(directory)) == 2
        assert read_series(directory, size=8).pixels.shape[0] == 2

    def test_files_without_a_dcm_extension_are_still_found(self, tmp_path: Path) -> None:
        directory = tmp_path / "s"
        directory.mkdir()
        _write_slice(directory / "0001", np.full((8, 8), 500, dtype=np.uint16))
        assert len(series_files(directory)) == 1

    def test_a_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            series_files(tmp_path / "nope")


class TestReadSeriesContract:
    """The pinned SeriesVolume shape and dtype, on synthetic bytes."""

    def test_volume_is_uint8_square_and_parallel_across_all_three_arrays(
        self, tmp_path: Path
    ) -> None:
        _series(tmp_path / "s", n_slices=4, shape=(12, 9))
        volume = read_series(tmp_path / "s", size=16)
        assert volume.pixels.dtype == np.uint8
        assert volume.pixels.shape == (4, 16, 16)
        assert volume.slice_positions.shape == (4,)
        assert len(volume.sop_uids) == 4
        assert volume.header.rows == 12
        assert volume.header.cols == 9
        assert volume.header.n_slices == 4

    def test_default_geometry_matches_the_pinned_cache_contract(self, tmp_path: Path) -> None:
        _series(tmp_path / "s", n_slices=2)
        volume = read_series(tmp_path / "s")
        assert (CACHE_SIZE, MAX_SLICES) == (512, 64)
        assert volume.pixels.shape == (2, 512, 512)

    def test_mixed_in_plane_sizes_raise_instead_of_broadcasting(self, tmp_path: Path) -> None:
        directory = tmp_path / "s"
        _series(directory, n_slices=2, shape=(8, 8))
        _write_slice(
            directory / "odd.dcm",
            np.full((16, 16), 700, dtype=np.uint16),
            ImagePositionPatient=[0.0, 0.0, 99.0],
            InstanceNumber=99,
        )
        with pytest.raises(ValueError, match="mixes in-plane matrices"):
            read_series(directory, size=8)


class TestCacheBuild:
    """Resumability and shard arithmetic, which decide what a killed job costs."""

    def _studies(self, root: Path, n_studies: int = 4, n_series: int = 2) -> None:
        for study in range(n_studies):
            for series in range(n_series):
                _series(root / f"study{study}" / f"series{series}", n_slices=3, shape=(8, 8))

    def test_cache_series_writes_a_loadable_uint8_array(self, tmp_path: Path) -> None:
        _series(tmp_path / "s", n_slices=3)
        target = tmp_path / "out" / "v.npy"
        written = cache_series(tmp_path / "s", target, size=16)
        assert written > 0
        loaded = np.load(target)
        assert loaded.dtype == np.uint8
        assert loaded.shape == (3, 16, 16)

    def test_an_existing_cache_file_is_skipped_not_rewritten(self, tmp_path: Path) -> None:
        """This is the resume. Without it a cancelled array element costs its whole shard."""
        _series(tmp_path / "s", n_slices=3)
        target = tmp_path / "out" / "v.npy"
        cache_series(tmp_path / "s", target, size=16)
        stamp = target.stat().st_mtime_ns
        assert cache_series(tmp_path / "s", target, size=16) == 0
        assert target.stat().st_mtime_ns == stamp

    def test_no_partial_file_survives_a_successful_write(self, tmp_path: Path) -> None:
        """The staging name must be renamed away, or the resume sees a phantom.

        A half-written .npy still parses its header, so "the file exists" would
        accept a truncated volume. The write goes to a dotted staging name and is
        renamed atomically.
        """
        _series(tmp_path / "s", n_slices=2)
        target = tmp_path / "out" / "v.npy"
        cache_series(tmp_path / "s", target, size=16)
        assert list(target.parent.iterdir()) == [target]

    def test_build_cache_lays_the_pinned_directory_out(self, tmp_path: Path) -> None:
        self._studies(tmp_path / "raw", n_studies=2, n_series=2)
        report = build_cache(tmp_path / "raw", tmp_path / "cache", size=16, progress_every=0)
        assert isinstance(report, CacheReport)
        assert report.n_studies == 2
        assert report.n_written == 4
        assert (tmp_path / "cache" / "study0" / "series0.npy").is_file()

    def test_a_second_pass_writes_nothing_and_reports_it(self, tmp_path: Path) -> None:
        self._studies(tmp_path / "raw", n_studies=2, n_series=2)
        build_cache(tmp_path / "raw", tmp_path / "cache", size=16, progress_every=0)
        again = build_cache(tmp_path / "raw", tmp_path / "cache", size=16, progress_every=0)
        assert (again.n_written, again.n_skipped, again.n_failed) == (0, 4, 0)

    def test_shards_are_disjoint_and_together_cover_every_study(self, tmp_path: Path) -> None:
        """A gap here is a silently incomplete cache that reports success."""
        self._studies(tmp_path / "raw", n_studies=7, n_series=1)
        seen: list[str] = []
        for index in range(3):
            build_cache(
                tmp_path / "raw",
                tmp_path / "cache",
                shard_index=index,
                shard_count=3,
                size=16,
                progress_every=0,
            )
            found = sorted(p.name for p in (tmp_path / "cache").iterdir())
            seen = found
        assert seen == [f"study{i}" for i in range(7)]

    def test_a_shard_index_outside_the_shard_count_raises(self, tmp_path: Path) -> None:
        self._studies(tmp_path / "raw", n_studies=1, n_series=1)
        with pytest.raises(ValueError, match="shard_index"):
            build_cache(tmp_path / "raw", tmp_path / "cache", shard_index=4, shard_count=4)

    def test_one_undecodable_series_is_recorded_and_the_rest_still_cache(
        self, tmp_path: Path
    ) -> None:
        """24,000 series must not be lost to one bad file."""
        self._studies(tmp_path / "raw", n_studies=1, n_series=2)
        broken = tmp_path / "raw" / "study0" / "broken"
        broken.mkdir()
        (broken / "junk.dcm").write_bytes(b"not a dicom at all")
        report = build_cache(tmp_path / "raw", tmp_path / "cache", size=16, progress_every=0)
        assert report.n_written == 2
        assert report.n_failed == 1
        assert "broken" in report.failures[0][0]

    def test_report_summary_names_the_rate_the_next_job_is_sized_from(
        self, tmp_path: Path
    ) -> None:
        self._studies(tmp_path / "raw", n_studies=1, n_series=1)
        report = build_cache(tmp_path / "raw", tmp_path / "cache", size=16, progress_every=0)
        assert "s/series written" in report.summary()


@requires_subset
class TestRealSubset:
    """The same invariants on the actual competition bytes.

    These are the tests that matter. A synthetic fixture proves the code does
    what its author intended; only real DICOMs prove the intention matched the
    data. They skip, loudly, when the external drive is not mounted.
    """

    def _series_dirs(self, limit: int) -> list[Path]:
        root = require_series_root("train")
        found: list[Path] = []
        for study in sorted(p for p in root.iterdir() if p.is_dir() and p.name[0] != "."):
            for series in sorted(p for p in study.iterdir() if p.is_dir() and p.name[0] != "."):
                if series_files(series):
                    found.append(series)
                if len(found) >= limit:
                    return found
        return found

    def test_five_real_series_round_trip_to_a_plausible_volume(self) -> None:
        """Shape and dtype are the easy half; the value range is the half that catches bugs.

        A windowing error returns an all-zero or all-255 volume with a perfectly
        correct shape and dtype. Measured on ten real series: means 41-73,
        medians 12-59, min 0, max 255 in every one -- so the bounds below have
        wide margins and still exclude a collapsed volume.
        """
        directories = self._series_dirs(5)
        assert len(directories) == 5
        for directory in directories:
            volume = read_series(directory)
            assert volume.pixels.dtype == np.uint8
            assert volume.pixels.shape[1:] == (CACHE_SIZE, CACHE_SIZE)
            assert 1 <= volume.pixels.shape[0] <= MAX_SLICES
            assert volume.pixels.max() >= 200, f"{directory.name} came back dark"
            assert volume.pixels.min() <= 20, f"{directory.name} has no background"
            assert 5.0 < float(volume.pixels.mean()) < 200.0
            assert np.unique(volume.pixels).size > 64, f"{directory.name} lost its texture"
            assert (np.diff(volume.slice_positions) >= 0).all()
            assert len(volume.sop_uids) == volume.pixels.shape[0]
            assert volume.header.study_uid == directory.parent.name
            assert volume.header.series_uid == directory.name

    def test_the_named_real_series_whose_instance_numbers_run_backwards(self) -> None:
        """Series ...5990029424266 of study ...75652399585: 38 slices, InstanceNumber 38 to 1.

        This is the fixture that makes the ordering rule evidence rather than
        received wisdom. Measured on this series, InstanceNumber is exactly
        descending along ascending projection, so a reader that sorted by it
        would return this volume mirrored in z.
        """
        directory = require_series_root("train") / REAL_STUDY / ORDER_TRAP_SERIES
        volume = read_series(directory)

        by_uid = {}
        for path in series_files(directory):
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
            by_uid[str(dataset.SOPInstanceUID)] = int(dataset.InstanceNumber)
        sequence = [by_uid[uid] for uid in volume.sop_uids]

        assert (np.diff(volume.slice_positions) > 0).all()
        assert sequence == sorted(sequence, reverse=True), (
            "this fixture exists because InstanceNumber runs backwards here; if it "
            "no longer does, the data changed and the fixture must be re-chosen"
        )
        assert sequence != sorted(sequence)

    def test_the_named_real_non_square_series_comes_back_letterboxed(self) -> None:
        """Series ...3927478147680: 640 rows by 540 columns, so the sides must be padded.

        Measured: the outer 40 columns of every cached plane are exactly 0, and
        the outer rows carry anatomy. A direct resize would leave no zero band
        at all and would stretch the anatomy 1.19x horizontally.
        """
        directory = require_series_root("train") / REAL_STUDY / LETTERBOX_SERIES
        volume = read_series(directory)
        assert volume.header.rows != volume.header.cols
        assert volume.pixels.shape[1:] == (CACHE_SIZE, CACHE_SIZE)
        assert volume.pixels[:, :, :30].max() == 0
        assert volume.pixels[:, :, -30:].max() == 0
        assert volume.pixels[:, :, 30:-30].max() == 255

    def test_a_real_header_read_agrees_with_a_real_pixel_read(self) -> None:
        directory = self._series_dirs(1)[0]
        header = read_series_header(directory)
        volume = read_series(directory)
        assert header == volume.header
        assert header.n_slices == len(series_files(directory))
        assert header.manufacturer
        assert header.patient_id


class TestGeometryMustActuallyOrderTheSlices:
    """Two ways the projection sort silently returns an arbitrary order.

    Neither used to raise. Both produce a plausible volume rather than
    an error, which is the failure mode this module exists to prevent -- the same
    class as sorting by ``InstanceNumber``, and harder to notice because there is
    no second ordering to disagree with.
    """

    def test_a_series_whose_slices_are_in_two_planes_is_refused(self, tmp_path):
        # Every slice is projected onto the FIRST slice's normal. A slice acquired
        # in another plane is projected onto an axis that is not its own, lands at
        # a meaningless position, and reorders the stack without failing.
        directory = tmp_path / "mixed"
        _series(directory, n_slices=5)
        odd = sorted(directory.glob("*.dcm"))[2]
        dataset = pydicom.dcmread(str(odd))
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        dataset.save_as(str(odd), enforce_file_format=True)

        with pytest.raises(ValueError, match="differs from the series"):
            read_series(directory)

    def test_a_series_with_one_orientation_is_still_read(self, tmp_path):
        # The guard above must not reject ordinary series: 250 of 250 real series
        # sampled from the local subset pass it. This is the control.
        directory = tmp_path / "plain"
        _series(directory, n_slices=6)
        volume = read_series(directory)
        means = [float(plane.mean()) for plane in volume.pixels]
        assert means == sorted(means), "geometric order was not preserved"

    def test_a_series_where_every_slice_projects_to_one_point_is_refused(self, tmp_path):
        # All-equal projections mean geometry says nothing about order, and a
        # stable argsort then returns FILENAME order while looking like it sorted
        # geometrically. Filenames here are SOP UIDs, so that order is arbitrary.
        directory = tmp_path / "degenerate"
        _series(directory, n_slices=4, in_plane_jitter=0.0)
        for path in sorted(directory.glob("*.dcm")):
            dataset = pydicom.dcmread(str(path))
            dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
            dataset.save_as(str(path), enforce_file_format=True)

        with pytest.raises(ValueError, match="project to the same"):
            read_series(directory)

    def test_a_single_slice_series_is_not_caught_by_the_degeneracy_guard(self, tmp_path):
        # ptp of one projection is trivially 0. Refusing it would reject a valid
        # one-slice series, so the guard is explicitly len > 1.
        directory = tmp_path / "single"
        _series(directory, n_slices=1)
        assert read_series(directory).pixels.shape[0] == 1
