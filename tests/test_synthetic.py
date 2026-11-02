"""The fake corpus has to satisfy the real contracts, or every test above it is fiction.

Why this exists: this generator is the only thing that lets the training stack
run before the pixel cache and the derived labels are on disk, so it is load
bearing rather than decorative. Its whole value is that a run against it fails
for the same reasons a run against real data would. Two ways it could quietly
lose that value are checked here on purpose: drifting from the pinned manifest
schema or cache layout, and being so tidy that the hard cases never appear -- a
corpus where every study carries all six slots never exercises a presence mask,
and one where every label cell is usable never exercises a label mask.

The physical-scale tests are the reason the phantom is a disc: a disc has a
diameter, and a diameter in millimetres is the one thing a fixed-pixel crop
cannot preserve across two spacings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kappa.constants import ID_COLUMN, LABELS
from kappa.data.manifest import MANIFEST_COLUMNS, SLOT_ORDER, availability_patterns
from kappa.data.synthetic import (
    AVAILABILITY_PATTERNS,
    CACHE_SLICE_CAP,
    LABEL_MASK_RATE,
    label_table,
    make_synthetic_dataset,
    phantom_slices,
)


def _corpus(tmp_path, **over):
    """Small, cheap and pinned: 32 px cache, four slices, one slot unless asked."""
    options = {
        "n_studies": 3,
        "seed": 0,
        "cache_size": 32,
        "slice_counts": (4, 4),
        "patterns": ((1,),),
    }
    options.update(over)
    return make_synthetic_dataset(tmp_path, **options)


def _uids(n):
    return [f"1.2.3.{index}" for index in range(n)]


class TestCacheContract:
    def test_cache_path_matches_the_pinned_template(self, tmp_path):
        corpus = _corpus(tmp_path)
        for row in corpus.manifest.itertuples(index=False):
            expected = corpus.cache_dir / row.study_uid / f"{row.series_uid}.npy"
            assert expected.is_file(), f"{expected} missing"

    def test_cached_volumes_are_square_uint8_stacks(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=4)
        for path in sorted(corpus.cache_dir.rglob("*.npy")):
            volume = np.load(path)
            assert volume.dtype == np.uint8
            assert volume.ndim == 3
            assert volume.shape[1] == volume.shape[2] == corpus.cache_size

    def test_a_long_series_is_capped_in_the_cache_but_not_in_the_manifest(self, tmp_path):
        # The cache contract caps a series at 64 slices while the manifest keeps
        # the native count, so the two disagree by design. A loader that sizes
        # its slice sampling from the manifest indexes off the end of the array.
        corpus = _corpus(tmp_path, n_studies=1, slice_counts=(90, 90))
        assert int(corpus.manifest.loc[0, "n_slices"]) == 90
        cached = np.load(corpus.cache_dir / corpus.manifest.loc[0, "study_uid"]
                         / f"{corpus.manifest.loc[0, 'series_uid']}.npy")
        assert cached.shape[0] == CACHE_SLICE_CAP

    def test_cache_bytes_are_measured_from_the_files(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=2)
        measured = sum(path.stat().st_size for path in corpus.cache_dir.rglob("*.npy"))
        assert corpus.cache_bytes() == measured > 0


class TestManifest:
    def test_columns_are_exactly_the_contract_in_order(self, tmp_path):
        corpus = _corpus(tmp_path)
        assert tuple(corpus.manifest.columns) == tuple(MANIFEST_COLUMNS)
        assert tuple(pd.read_parquet(corpus.manifest_path).columns) == tuple(MANIFEST_COLUMNS)

    def test_slot_agrees_with_the_plane_and_contrast_it_encodes(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=12, patterns=None)
        for row in corpus.manifest.itertuples(index=False):
            assert SLOT_ORDER[int(row.slot)] == (row.plane, int(row.fs))

    def test_the_default_patterns_exercise_twelve_distinct_availability_patterns(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=12, patterns=None, cache_size=8, slice_counts=(2, 2))
        assert len(availability_patterns(corpus.manifest)) == len(AVAILABILITY_PATTERNS) == 12

    def test_axial_fs1_is_in_every_study_because_it_is_the_hundred_percent_slot(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=12, patterns=None, cache_size=8, slice_counts=(2, 2))
        per_study = corpus.manifest.groupby("study_uid")["slot"].apply(set)
        assert all(1 in slots for slots in per_study)

    def test_a_second_series_in_one_slot_is_the_thinner_one(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=1, slice_counts=(20, 20), duplicate_slot_studies=1)
        assert len(corpus.manifest) == 2
        counts = sorted(int(n) for n in corpus.manifest["n_slices"])
        assert counts == [10, 20]

    def test_patients_can_recur_across_studies(self, tmp_path):
        # Folds must be grouped by patient, and a corpus where every patient is
        # unique cannot tell a grouped split from a random one.
        corpus = _corpus(tmp_path, n_studies=6, n_patients=2)
        assert corpus.manifest["patient_id"].nunique() == 2

    def test_more_than_one_scanner_fingerprint_exists(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=6)
        assert corpus.manifest["scanner_fingerprint"].nunique() > 1

    def test_geometry_can_be_pinned_per_series(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=2, geometry=((0.4, 500, 500), (0.8, 250, 500)))
        assert list(corpus.manifest["spacing"]) == [0.4, 0.8]
        assert list(corpus.manifest["rows"]) == [500, 250]
        assert corpus.mm_per_pixel(corpus.manifest.loc[0, "series_uid"]) == 0.4 * 500 / 32


class TestLayoutMatchesThePathsModule:
    def test_the_corpus_is_a_drop_in_data_root(self, tmp_path, monkeypatch):
        # The point of laying it out this way is that KAPPA_DATA_ROOT can be
        # pointed at a synthetic corpus and no other module can tell.
        from kappa.data.paths import interim_dir, processed_dir

        corpus = _corpus(tmp_path)
        monkeypatch.setenv("KAPPA_DATA_ROOT", str(tmp_path))
        assert corpus.cache_dir.resolve() == (processed_dir() / "cache").resolve()
        assert corpus.manifest_path.resolve() == (interim_dir() / "manifest.parquet").resolve()
        assert corpus.labels_path.resolve() == (
            interim_dir() / "labels" / "derived.parquet"
        ).resolve()


class TestLabelTable:
    def test_columns_are_the_id_then_a_value_and_a_mask_for_each_label(self, tmp_path):
        corpus = _corpus(tmp_path)
        expected = [ID_COLUMN]
        for name in LABELS:
            expected += [name, f"{name}__mask"]
        assert list(corpus.labels.columns) == expected
        assert list(pd.read_parquet(corpus.labels_path).columns) == expected

    def test_one_row_per_study(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=5)
        assert len(corpus.labels) == 5
        assert corpus.labels[ID_COLUMN].is_unique

    def test_the_mask_columns_are_boolean_and_the_values_are_float(self, tmp_path):
        corpus = _corpus(tmp_path)
        for name in LABELS:
            assert corpus.labels[f"{name}__mask"].dtype == bool
            assert corpus.labels[name].dtype == np.float32

    def test_a_mask_rate_of_zero_makes_every_cell_usable(self):
        table = label_table(_uids(20), mask_rate=0.0)
        assert table[[f"{name}__mask" for name in LABELS]].to_numpy().all()

    def test_a_mask_rate_of_one_makes_no_cell_usable(self):
        table = label_table(_uids(20), mask_rate=1.0)
        assert not table[[f"{name}__mask" for name in LABELS]].to_numpy().any()

    def test_the_default_mask_rate_leaves_roughly_the_measured_one_percent_usable(self):
        # 98.7% of raw label cells are unknown. A generator that quietly defaults
        # to dense labels makes the masked loss look well conditioned when it is
        # the sparsest thing in the project.
        assert set(LABEL_MASK_RATE.values()) == {0.987}
        table = label_table(_uids(5000), seed=0)
        usable = table[[f"{name}__mask" for name in LABELS]].to_numpy().mean()
        assert 0.010 < usable < 0.017, usable

    def test_prevalence_is_settable_per_label(self):
        table = label_table(_uids(50), mask_rate=0.0, prevalence={"MCL": 1.0, "ACL": 0.0})
        assert (table["MCL"] == 1.0).all()
        assert (table["ACL"] == 0.0).all()

    def test_a_misspelled_label_name_raises_instead_of_being_ignored(self):
        # "Bakers" for "Baker's" would silently leave the default in place and
        # the test asserting the prevalence would pass for the wrong reason.
        with pytest.raises(KeyError, match="Bakers"):
            label_table(_uids(4), prevalence={"Bakers": 1.0})

    def test_a_rate_outside_the_unit_interval_raises(self):
        with pytest.raises(ValueError, match="mask_rate"):
            label_table(_uids(4), mask_rate=1.5)

    def test_masked_cells_are_finite_by_default(self):
        table = label_table(_uids(200), seed=1)
        assert np.isfinite(table[list(LABELS)].to_numpy()).all()

    def test_nan_masked_cells_is_opt_in_and_writes_nan(self):
        # The real table never does this. A loader still has to survive it,
        # because a NaN reaching the loss destroys a run without raising.
        table = label_table(_uids(200), seed=1, nan_masked_cells=True)
        values = table[list(LABELS)].to_numpy()
        usable = table[[f"{name}__mask" for name in LABELS]].to_numpy()
        assert np.isnan(values[~usable]).all()
        assert np.isfinite(values[usable]).all()

    def test_the_shared_default_rates_cannot_be_mutated_by_a_caller(self):
        with pytest.raises(TypeError):
            LABEL_MASK_RATE["ACL"] = 0.0  # type: ignore[index]


class TestPhantom:
    def test_the_disc_diameter_follows_millimetres_not_pixels(self):
        coarse = phantom_slices(1, 64, 6.25, diameter_mm=100.0)
        fine = phantom_slices(1, 64, 3.125, diameter_mm=100.0)
        assert int((coarse[0, 32] > 128).sum()) == 16
        assert int((fine[0, 32] > 128).sum()) == 32

    def test_every_slice_carries_its_own_geometric_index(self):
        volume = phantom_slices(9, 16, 1.0, diameter_mm=4.0)
        assert list(volume[:, 0, 0]) == list(range(9))

    def test_a_disc_wider_than_the_frame_is_clipped_not_rescaled(self):
        volume = phantom_slices(1, 16, 1.0, diameter_mm=1000.0)
        assert (volume[0, 1:] == 255).all()

    def test_noise_perturbs_the_pixels_but_not_the_index_marker(self):
        clean = phantom_slices(4, 16, 1.0, diameter_mm=8.0)
        noisy = phantom_slices(4, 16, 1.0, diameter_mm=8.0, noise=0.1,
                               rng=np.random.default_rng(0))
        assert not np.array_equal(clean[:, 8], noisy[:, 8])
        assert list(noisy[:, 0, 0]) == list(range(4))

    def test_a_nonpositive_spacing_raises(self):
        with pytest.raises(ValueError, match="mm_per_pixel"):
            phantom_slices(2, 8, 0.0)


class TestDeterminism:
    def test_the_same_seed_writes_identical_pixels_and_tables(self, tmp_path):
        first = _corpus(tmp_path / "a", seed=7, n_studies=4)
        second = _corpus(tmp_path / "b", seed=7, n_studies=4)
        pd.testing.assert_frame_equal(first.manifest, second.manifest)
        pd.testing.assert_frame_equal(first.labels, second.labels)
        for left in sorted(first.cache_dir.rglob("*.npy")):
            right = second.cache_dir / left.relative_to(first.cache_dir)
            assert left.read_bytes() == right.read_bytes()

    def test_a_different_seed_writes_a_different_corpus(self, tmp_path):
        first = _corpus(tmp_path / "a", seed=1, n_studies=4)
        second = _corpus(tmp_path / "b", seed=2, n_studies=4)
        assert list(first.manifest["spacing"]) != list(second.manifest["spacing"])

    def test_the_study_uid_carries_the_seed_so_two_seeds_can_share_a_root(self, tmp_path):
        first = _corpus(tmp_path, seed=1, n_studies=2)
        second = _corpus(tmp_path, seed=2, n_studies=2)
        assert not set(first.study_uids) & set(second.study_uids)


class TestValidation:
    def test_a_slot_outside_the_six_raises(self, tmp_path):
        with pytest.raises(ValueError, match="outside"):
            _corpus(tmp_path, patterns=((1, 6),))

    def test_an_empty_pattern_raises(self, tmp_path):
        with pytest.raises(ValueError, match="at least one slot"):
            _corpus(tmp_path, patterns=((),))

    def test_a_repeated_slot_in_one_pattern_raises(self, tmp_path):
        with pytest.raises(ValueError, match="repeats"):
            _corpus(tmp_path, patterns=((1, 1),))

    def test_zero_studies_raises(self, tmp_path):
        with pytest.raises(ValueError, match="n_studies"):
            _corpus(tmp_path, n_studies=0)

    def test_a_backwards_slice_count_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="slice_counts"):
            _corpus(tmp_path, slice_counts=(9, 2))

    def test_a_nonpositive_pinned_spacing_raises(self, tmp_path):
        with pytest.raises(ValueError, match="positive spacing"):
            _corpus(tmp_path, geometry=((0.0, 512, 512),))
