"""Four silent failures live in this loader, and each one still reports a number.

Why this exists: nothing in a training run tells you that the loader lied. The
loss goes down, the AUC comes out, and the four things that can be wrong here all
produce a plausible wrong answer rather than a traceback:

1. **A NaN label reaches the loss.** 98.7% of raw label cells are unknown. Every
   loss propagates NaN silently, so the run prints ``nan`` and finishes. The
   tests here forge a NaN past the sanitiser to prove the *assertion* is what
   catches it -- delete the assertion and they fail, which is the point.
2. **An absent slot is reported present.** Only Axial_fs1 is at 100% coverage;
   Axial_fs0 is at 19.4%. A zero-filled slot flagged present teaches the model
   that absence looks like black anatomy.
3. **A fixed pixel crop.** In-plane spacing spans 0.14-0.89 mm/px, so a fixed
   pixel crop makes the same knee a different size at every site. The physical
   crop tests assert two spacings agree *and* assert that the cached pixel
   counts they came from disagreed, so the test cannot pass by accident.
4. **Hardcoded ImageNet normalisation.** A greyscale-pretrained backbone fed
   ImageNet statistics still trains and still reports a score.

The phantom's per-slice index marker is what makes the geometry claims checkable
end to end: with a crop that happens to be the identity, the marker read out of
the batch *is* the cached slice index, so "equal intervals along the geometric
axis" and "previous, centre, next" become assertions instead of intentions.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

# kappa.data.dataset imports torch at module scope and torch is an optional
# extra, so the suite has to stay green on a machine that only installed the
# label pipeline.
pytest.importorskip("torch")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from kappa.constants import ID_COLUMN, LABELS, N_LABELS  # noqa: E402
from kappa.data.dataset import (  # noqa: E402
    CropSpec,
    Normalisation,
    StudyDataset,
    cache_mm_per_pixel,
    collate_studies,
    neighbour_indices,
    sample_slice_indices,
)
from kappa.data.synthetic import make_synthetic_dataset  # noqa: E402

# One pinned geometry for every series, so a physical-scale claim is arithmetic
# rather than a re-derivation of the generator: 0.5 mm/px over a 320 px matrix
# letterboxed into a 32 px cache is 5.0 mm per cached pixel, a 160 mm extent.
GEOMETRY = ((0.5, 320, 320),)
CACHE_MM_PER_PIXEL = 5.0


def _corpus(tmp_path, **over):
    options = {
        "n_studies": 3,
        "seed": 0,
        "cache_size": 32,
        "slice_counts": (8, 8),
        "patterns": ((1,),),
        "geometry": GEOMETRY,
        "mask_rate": 0.0,
        "phantom_mm": 100.0,
    }
    options.update(over)
    return make_synthetic_dataset(tmp_path, **options)


def _dataset(corpus, **over):
    options = {
        "normalisation": Normalisation.identity(),
        "labels": corpus.labels,
        "slots": (1,),
        "n_slices": 4,
        "crop": CropSpec(fov_mm=100.0, out_px=32),
    }
    options.update(over)
    return StudyDataset(corpus.manifest, corpus.cache_dir, **options)


def _identity_crop(corpus):
    """A crop that resamples nothing, so the cached slice markers survive to the batch."""
    return CropSpec(
        fov_mm=CACHE_MM_PER_PIXEL * corpus.cache_size, out_px=corpus.cache_size
    )


def _markers(volume, channel=1):
    """The phantom slice indices carried by one slot volume ``(n_slices, 3, H, W)``."""
    return (volume[:, channel, 0, 0] * 255.0).round().to(torch.int64).tolist()


def _disc_px(plane):
    """Width of the phantom disc across the centre row, at half maximum."""
    line = plane[plane.shape[0] // 2]
    threshold = 0.5 * float(line.max() + line.min())
    return int((line > threshold).sum())


class TestBatchContract:
    def test_one_item_carries_every_pinned_key_at_the_pinned_shape(self, tmp_path):
        corpus = _corpus(tmp_path)
        item = _dataset(corpus, slots=(1,), n_slices=4)[0]
        assert set(item) == {"pixels", "present", "slot_ids", "labels", "mask", "study_uid"}
        assert item["pixels"].shape == (1, 4, 3, 32, 32)
        assert item["present"].shape == (1,)
        assert item["slot_ids"].shape == (1,)
        assert item["labels"].shape == (N_LABELS,)
        assert item["mask"].shape == (N_LABELS,)
        assert isinstance(item["study_uid"], str)

    def test_every_tensor_has_the_pinned_dtype(self, tmp_path):
        item = _dataset(_corpus(tmp_path))[0]
        assert item["pixels"].dtype == torch.float32
        assert item["present"].dtype == torch.bool
        assert item["slot_ids"].dtype == torch.long
        assert item["labels"].dtype == torch.float32
        assert item["mask"].dtype == torch.bool

    def test_collate_stacks_into_the_batch_shape(self, tmp_path):
        corpus = _corpus(tmp_path)
        dataset = _dataset(corpus, n_slices=2)
        batch = collate_studies([dataset[i] for i in range(3)])
        assert batch["pixels"].shape == (3, 1, 2, 3, 32, 32)
        assert batch["present"].shape == (3, 1)
        assert batch["slot_ids"].shape == (3, 1)
        assert batch["labels"].shape == (3, N_LABELS)
        assert batch["mask"].shape == (3, N_LABELS)
        assert batch["study_uid"] == list(dataset.study_uids)

    def test_it_works_as_a_dataloader_collate_fn(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=4)
        dataset = _dataset(corpus, n_slices=2)
        loader = DataLoader(
            dataset, batch_size=2, num_workers=0, collate_fn=functools.partial(collate_studies)
        )
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0]["pixels"].shape == (2, 1, 2, 3, 32, 32)

    def test_an_empty_batch_raises_rather_than_returning_an_empty_tensor(self):
        with pytest.raises(ValueError, match="empty batch"):
            collate_studies([])


class TestLabelsNeverContainNaN:
    def test_a_nan_in_the_parquet_becomes_masked_not_nan(self, tmp_path):
        # The derived table writes a finite value plus a mask, but a loader that
        # only works against a well-formed table is a loader that fails on the
        # first bad file, eleven hours in.
        corpus = _corpus(tmp_path, mask_rate=0.5, nan_masked_cells=True, seed=4)
        dataset = _dataset(corpus, labels=corpus.labels)
        for index in range(len(dataset)):
            item = dataset[index]
            assert torch.isfinite(item["labels"]).all()
            assert (item["labels"][~item["mask"]] == 0.0).all()

    def test_a_value_present_with_a_nan_mask_cell_is_unusable(self, tmp_path):
        # bool(nan) is True, so a NaN mask cell promotes an unknown label to
        # usable and pairs it with the 0.0 fill: a fabricated negative arriving
        # through the mask rather than through the value.
        corpus = _corpus(tmp_path)
        labels = corpus.labels.copy()
        labels["ACL__mask"] = labels["ACL__mask"].astype(object)
        labels.loc[0, "ACL__mask"] = np.nan
        item = _dataset(corpus, labels=labels)[0]
        assert not bool(item["mask"][LABELS.index("ACL")])
        assert torch.isfinite(item["labels"]).all()

    def test_a_forged_nan_label_raises_out_of_getitem(self, tmp_path):
        # This is the test that fails if the assertion in __getitem__ is
        # deleted as redundant. It is not redundant: it is the last place a NaN
        # can be seen before it reaches a loss that will not complain.
        dataset = _dataset(_corpus(tmp_path))
        dataset._label_values[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            dataset[0]

    def test_a_forged_nan_label_raises_out_of_collate(self, tmp_path):
        dataset = _dataset(_corpus(tmp_path))
        sample = dataset[0]
        sample["labels"] = torch.full((N_LABELS,), float("nan"))
        with pytest.raises(ValueError, match="non-finite"):
            collate_studies([sample])

    def test_a_batch_augment_that_introduces_nan_raises(self, tmp_path):
        # The finiteness check runs after the hook precisely so a mixing bug in
        # CutMix cannot smuggle a NaN into the loss.
        dataset = _dataset(_corpus(tmp_path))

        def _broken_mix(batch, rng):
            batch["labels"] = batch["labels"] * float("nan")
            return batch

        with pytest.raises(ValueError, match="non-finite"):
            collate_studies([dataset[0], dataset[1]], batch_augment=_broken_mix)

    def test_an_unlabelled_study_is_masked_not_twelve_zeros(self, tmp_path):
        corpus = _corpus(tmp_path)
        item = _dataset(corpus, labels=None)[0]
        assert not item["mask"].any()
        assert (item["labels"] == 0.0).all()

    def test_a_study_missing_from_the_label_table_is_masked(self, tmp_path):
        corpus = _corpus(tmp_path)
        item = _dataset(corpus, labels=corpus.labels.iloc[1:])[0]
        assert not item["mask"].any()

    def test_a_duplicate_study_row_raises_instead_of_last_one_winning(self, tmp_path):
        # Two rows for one study means two derivations disagreeing, and taking
        # the later row makes the training labels depend on parquet row order.
        corpus = _corpus(tmp_path)
        doubled = pd.concat([corpus.labels, corpus.labels.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate study rows"):
            _dataset(corpus, labels=doubled)

    def test_a_label_table_missing_a_mask_column_raises(self, tmp_path):
        corpus = _corpus(tmp_path)
        labels = corpus.labels.drop(columns=["MCL__mask"])
        with pytest.raises(KeyError, match="MCL__mask"):
            _dataset(corpus, labels=labels)

    def test_mask_and_values_come_from_the_table_not_from_the_row_order(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=3, mask_rate=1.0)
        labels = corpus.labels.copy()
        labels.loc[1, "Baker's"] = 1.0
        labels.loc[1, "Baker's__mask"] = True
        dataset = _dataset(corpus, labels=labels.iloc[::-1].reset_index(drop=True))
        item = dataset[1]
        assert item["study_uid"] == str(labels.loc[1, ID_COLUMN])
        assert bool(item["mask"][LABELS.index("Baker's")])
        assert float(item["labels"][LABELS.index("Baker's")]) == 1.0


class TestPresenceIsHonest:
    def test_an_absent_slot_is_zero_filled_and_flagged_absent(self, tmp_path):
        corpus = _corpus(tmp_path, patterns=((1, 4),))
        item = _dataset(corpus, slots=(0, 1, 4))[0]
        assert item["present"].tolist() == [False, True, True]
        assert (item["pixels"][0] == 0.0).all()
        assert (item["pixels"][1] != 0.0).any()

    def test_the_hundred_percent_slot_is_present_in_every_study(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=6, patterns=None, cache_size=8, slice_counts=(2, 2))
        dataset = _dataset(
            corpus, slots=(1,), n_slices=2, crop=CropSpec(fov_mm=40.0, out_px=8)
        )
        assert all(bool(dataset[i]["present"][0]) for i in range(len(dataset)))

    def test_studies_with_no_cached_series_warn_and_are_listed(self, tmp_path):
        # A half-built cache is the normal state of this project, and a loader
        # that quietly emits black studies turns that into a training run.
        corpus = _corpus(tmp_path)
        manifest = corpus.manifest.copy()
        manifest.loc[manifest.index[0], "slot"] = 4
        with pytest.warns(UserWarning, match="no cached series"):
            dataset = StudyDataset(
                manifest,
                corpus.cache_dir,
                normalisation=Normalisation.identity(),
                labels=corpus.labels,
                slots=(1,),
                n_slices=2,
            )
        assert dataset.empty_study_uids == (corpus.study_uids[0],)
        assert not dataset[0]["present"].any()

    def test_a_fully_cached_single_slot_dataset_warns_about_nothing(self, tmp_path):
        corpus = _corpus(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset = _dataset(corpus)
        assert caught == []
        assert dataset.empty_study_uids == ()

    def test_multi_slot_training_without_series_dropout_warns(self, tmp_path):
        # Zero-fill, an honest presence flag and ~10% dropout are one design.
        # Enabling the first two without the third is the silent bug: the
        # encoder never meets an empty slot, so the flag carries no information.
        corpus = _corpus(tmp_path, patterns=((1, 4),))
        with pytest.warns(UserWarning, match="series_dropout"):
            _dataset(corpus, slots=(1, 4), training=True, series_dropout=0.0)

    def test_single_slot_training_without_series_dropout_does_not_warn(self, tmp_path):
        corpus = _corpus(tmp_path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _dataset(corpus, slots=(1,), training=True, series_dropout=0.0)
        assert caught == []

    def test_series_dropout_zeroes_and_unflags_the_slot_it_drops(self, tmp_path):
        corpus = _corpus(tmp_path, patterns=((0, 1, 2, 3, 4, 5),))
        dataset = _dataset(
            corpus, slots=(0, 1, 2, 3, 4, 5), training=True, series_dropout=1.0
        )
        item = dataset[0]
        dropped = ~item["present"]
        assert int(dropped.sum()) == 5
        assert (item["pixels"][dropped] == 0.0).all()

    def test_series_dropout_never_drops_every_present_slot(self, tmp_path):
        # A study with nothing present has no signal at all, so the loss would
        # be fitting a black volume to a real label.
        corpus = _corpus(tmp_path, n_studies=4, patterns=((0, 1, 2, 3, 4, 5),))
        dataset = _dataset(
            corpus, slots=(0, 1, 2, 3, 4, 5), training=True, series_dropout=1.0
        )
        for epoch in range(5):
            dataset.set_epoch(epoch)
            for index in range(len(dataset)):
                assert int(dataset[index]["present"].sum()) >= 1

    def test_dropout_is_inert_outside_training_mode(self, tmp_path):
        corpus = _corpus(tmp_path, patterns=((0, 1, 2, 3, 4, 5),))
        dataset = _dataset(
            corpus, slots=(0, 1, 2, 3, 4, 5), training=False, series_dropout=1.0
        )
        assert dataset[0]["present"].all()

    def test_an_out_of_range_dropout_probability_raises(self, tmp_path):
        with pytest.raises(ValueError, match="series_dropout"):
            _dataset(_corpus(tmp_path), series_dropout=1.5)


class TestSliceSampling:
    def test_indices_span_the_whole_extent_at_equal_intervals(self):
        indices = sample_slice_indices(30, 5)
        assert indices.tolist() == [0, 7, 14, 22, 29]

    def test_it_samples_a_fixed_count_not_every_slice(self):
        assert len(sample_slice_indices(320, 16)) == 16
        assert len(sample_slice_indices(11, 16)) == 16

    def test_a_short_series_repeats_slices_rather_than_padding_black(self, tmp_path):
        # A duplicated real slice is closer to the truth than a black one, and a
        # black one also teaches the encoder that absence looks like anatomy.
        indices = sample_slice_indices(3, 8)
        assert indices.min() == 0 and indices.max() == 2
        corpus = _corpus(tmp_path, slice_counts=(3, 3))
        volume = _dataset(corpus, n_slices=8)[0]["pixels"][0]
        assert all((volume[k] != 0.0).any() for k in range(8))

    def test_a_single_slice_series_is_sampled_without_dividing_by_zero(self):
        assert sample_slice_indices(1, 4).tolist() == [0, 0, 0, 0]

    def test_zero_slices_or_an_empty_volume_raises(self):
        with pytest.raises(ValueError, match="n_slices"):
            sample_slice_indices(10, 0)
        with pytest.raises(ValueError, match="n_available"):
            sample_slice_indices(0, 4)

    def test_the_sampled_slices_are_the_cached_geometric_order(self, tmp_path):
        # The cache is written in the order of the projection onto the slice
        # normal, so position in the array is geometric order. Reading the
        # per-slice marker back out of the batch proves nothing re-sorted it.
        corpus = _corpus(tmp_path, slice_counts=(30, 30))
        dataset = _dataset(corpus, n_slices=5, crop=_identity_crop(corpus))
        # Written out rather than compared against sample_slice_indices, so the
        # assertion cannot agree with a broken sampler by calling it twice.
        assert _markers(dataset[0]["pixels"][0]) == [0, 7, 14, 22, 29]

    def test_it_reads_the_slice_count_from_the_array_not_from_the_manifest(self, tmp_path):
        # The cache caps a series at 64 slices while the manifest keeps the
        # native count, so trusting the manifest indexes off the end -- or, as
        # here, silently samples a fraction of the volume.
        corpus = _corpus(tmp_path, slice_counts=(30, 30))
        manifest = corpus.manifest.copy()
        manifest.loc[manifest.index[0], "n_slices"] = 3
        dataset = StudyDataset(
            manifest,
            corpus.cache_dir,
            normalisation=Normalisation.identity(),
            labels=corpus.labels,
            slots=(1,),
            n_slices=5,
            crop=_identity_crop(corpus),
        )
        assert _markers(dataset[0]["pixels"][0]) == [0, 7, 14, 22, 29]


class TestNeighbourStacking:
    def test_an_interior_slice_stacks_previous_centre_next(self):
        assert neighbour_indices([4], 10)[0].tolist() == [3, 4, 5]

    def test_the_first_slice_replicates_its_missing_predecessor(self):
        # Replication, not zero padding: a black channel at the edge says the
        # anatomy ended, and what ended was the volume.
        assert neighbour_indices([0], 10)[0].tolist() == [0, 0, 1]

    def test_the_last_slice_replicates_its_missing_successor(self):
        assert neighbour_indices([9], 10)[0].tolist() == [8, 9, 9]

    def test_a_single_slice_volume_stacks_itself_three_times(self):
        assert neighbour_indices([0], 1)[0].tolist() == [0, 0, 0]

    def test_a_centre_outside_the_volume_raises(self):
        with pytest.raises(ValueError, match="outside"):
            neighbour_indices([10], 10)

    def test_both_edges_survive_end_to_end(self, tmp_path):
        corpus = _corpus(tmp_path, slice_counts=(12, 12))
        dataset = _dataset(corpus, n_slices=4, crop=_identity_crop(corpus))
        volume = dataset[0]["pixels"][0]
        previous, centre, following = (_markers(volume, channel=c) for c in (0, 1, 2))
        assert centre[0] == 0 and previous[0] == 0 and following[0] == 1
        assert centre[-1] == 11 and previous[-1] == 10 and following[-1] == 11

    def test_the_stack_axis_is_the_pinned_three_channels(self, tmp_path):
        assert _dataset(_corpus(tmp_path))[0]["pixels"].shape[2] == 3


class TestPhysicalCrop:
    def test_two_spacings_give_the_same_physical_scale(self, tmp_path):
        # 0.4 mm over 500 px and 0.8 mm over 500 px letterbox into 6.25 and
        # 12.5 mm per cached pixel, so the same 100 mm disc occupies 16 and 8
        # cached pixels. After a fixed 150 mm field of view they must agree.
        corpus = _corpus(
            tmp_path,
            n_studies=2,
            geometry=((0.4, 500, 500), (0.8, 500, 500)),
            phantom_mm=100.0,
        )
        dataset = _dataset(
            corpus, n_slices=1, crop=CropSpec(fov_mm=150.0, out_px=128)
        )
        widths = [_disc_px(dataset[i]["pixels"][0, 0, 1]) for i in range(2)]
        assert widths[0] == widths[1]
        assert widths[0] == pytest.approx(100.0 / (150.0 / 128), abs=2)

    def test_a_fixed_pixel_crop_would_have_disagreed(self, tmp_path):
        # The counterfactual that stops the test above passing by accident: the
        # two cached volumes really do hold the disc at different pixel sizes.
        corpus = _corpus(
            tmp_path, n_studies=2, geometry=((0.4, 500, 500), (0.8, 500, 500)), phantom_mm=100.0
        )
        cached = []
        for row in corpus.manifest.itertuples(index=False):
            volume = np.load(corpus.cache_dir / row.study_uid / f"{row.series_uid}.npy")
            cached.append(int((volume[0, corpus.cache_size // 2] > 128).sum()))
        assert cached == [16, 8]

    def test_the_letterbox_is_in_the_cached_pixel_size(self, tmp_path):
        # The cache squares each slice to max(rows, cols) before resizing, so
        # the cached pixel is not the DICOM spacing. Getting this wrong scales
        # every crop by the aspect ratio and is invisible in the images.
        assert cache_mm_per_pixel(0.5, 320, 640, 32) == 10.0
        assert cache_mm_per_pixel(0.5, 640, 320, 32) == 10.0
        assert cache_mm_per_pixel(0.31, 512, 512, 512) == pytest.approx(0.31)

    def test_bad_cache_geometry_raises(self):
        with pytest.raises(ValueError, match="bad geometry"):
            cache_mm_per_pixel(0.0, 512, 512, 512)
        with pytest.raises(ValueError, match="bad geometry"):
            cache_mm_per_pixel(0.31, 0, 512, 512)

    def test_a_field_of_view_wider_than_the_image_pads_instead_of_shrinking(self, tmp_path):
        # Shrinking to fit would restore the fixed-pixel-crop bug. A 10 mm
        # extent asked for a 150 mm field of view has to come back mostly black.
        corpus = _corpus(tmp_path, n_studies=1, geometry=((0.1, 100, 100),), phantom_mm=4.0)
        volume = _dataset(corpus, n_slices=1, crop=CropSpec(fov_mm=150.0, out_px=64))[0]["pixels"]
        assert float(volume[0, 0, 1, 0, 0]) == 0.0
        assert (volume[0, 0, 1] != 0.0).any()

    def test_the_output_pixel_size_is_the_same_for_every_study(self):
        assert CropSpec(fov_mm=150.0, out_px=384).mm_per_pixel == pytest.approx(0.390625)

    def test_a_nonpositive_field_of_view_raises(self):
        with pytest.raises(ValueError, match="fov_mm"):
            CropSpec(fov_mm=0.0, out_px=384)


class TestNormalisation:
    def test_normalisation_is_a_required_argument(self, tmp_path):
        # There is no ImageNet default on purpose: a greyscale-pretrained
        # backbone fed ImageNet statistics still trains and still reports a
        # number, and the number is wrong.
        corpus = _corpus(tmp_path)
        with pytest.raises(TypeError):
            StudyDataset(corpus.manifest, corpus.cache_dir)  # type: ignore[call-arg]

    def test_it_reads_mean_and_std_out_of_a_resolved_data_config(self):
        config = {"input_size": (3, 224, 224), "mean": (0.485, 0.456, 0.406),
                  "std": (0.229, 0.224, 0.225)}
        normalisation = Normalisation.from_data_config(config)
        assert normalisation.mean == (0.485, 0.456, 0.406)
        assert normalisation.std == (0.229, 0.224, 0.225)

    def test_a_greyscale_config_broadcasts_one_value_over_three_channels(self):
        # The shape a medical backbone reports, e.g. mean 0.5307 std 0.2583.
        mean, std = Normalisation.from_data_config({"mean": (0.5307,), "std": (0.2583,)}).tensors(3)
        assert mean.shape == (3, 1, 1) and std.shape == (3, 1, 1)
        assert torch.allclose(mean.flatten(), torch.full((3,), 0.5307))

    def test_a_config_without_mean_or_std_raises(self):
        with pytest.raises(KeyError, match="mean"):
            Normalisation.from_data_config({"input_size": (3, 224, 224)})

    def test_a_zero_standard_deviation_raises_instead_of_dividing(self):
        with pytest.raises(ValueError, match="std contains zero"):
            Normalisation((0.5,), (0.0,))

    def test_a_two_channel_statistic_raises_rather_than_broadcasting_wrongly(self):
        with pytest.raises(ValueError, match="need 1 or 3"):
            Normalisation((0.1, 0.2), (0.3, 0.4)).tensors(3)

    def test_the_given_statistics_are_actually_applied_to_the_pixels(self, tmp_path):
        corpus = _corpus(tmp_path)
        plain = _dataset(corpus)[0]["pixels"]
        scaled = _dataset(corpus, normalisation=Normalisation((0.5,), (0.25,)))[0]["pixels"]
        assert torch.allclose(scaled, (plain - 0.5) / 0.25, atol=1e-6)


class TestSlotSubset:
    def test_job_one_is_axial_fs1_only_and_needs_no_fusion(self, tmp_path):
        dataset = _dataset(_corpus(tmp_path), slots=(1,))
        item = dataset[0]
        assert dataset.slots == (1,)
        assert item["pixels"].shape[0] == 1
        assert item["slot_ids"].tolist() == [1]
        assert item["present"].tolist() == [True]

    def test_widening_to_six_slots_is_a_config_change(self, tmp_path):
        corpus = _corpus(tmp_path, patterns=((0, 1, 2, 3, 4, 5),))
        item = _dataset(corpus, slots=(0, 1, 2, 3, 4, 5))[0]
        assert item["pixels"].shape[0] == 6
        assert item["slot_ids"].tolist() == [0, 1, 2, 3, 4, 5]

    def test_slot_ids_stay_global_under_a_subset(self, tmp_path):
        # A slot embedding trained on one subset has to keep meaning the same
        # thing in the next job, so the ids are positions in SLOT_ORDER and not
        # positions in the subset.
        corpus = _corpus(tmp_path, patterns=((1, 4, 5),))
        assert _dataset(corpus, slots=(4, 5))[0]["slot_ids"].tolist() == [4, 5]

    def test_an_unknown_slot_id_raises(self, tmp_path):
        with pytest.raises(ValueError, match="outside"):
            _dataset(_corpus(tmp_path), slots=(6,))

    def test_a_repeated_slot_raises(self, tmp_path):
        with pytest.raises(ValueError, match="repeats"):
            _dataset(_corpus(tmp_path), slots=(1, 1))

    def test_an_empty_slot_selection_raises(self, tmp_path):
        with pytest.raises(ValueError, match="at least one slot"):
            _dataset(_corpus(tmp_path), slots=())


class TestAugmentationHooks:
    def test_the_per_study_hook_sees_the_cropped_normalised_volume(self, tmp_path):
        seen = []

        def _record(volume, rng):
            seen.append(tuple(volume.shape))
            return volume

        _dataset(_corpus(tmp_path), n_slices=4, augment=_record)[0]
        assert seen == [(4, 3, 32, 32)]

    def test_a_hook_that_changes_the_crop_raises(self, tmp_path):
        # A rotation or a random-resized crop that forgets to restore the size
        # would otherwise reach the collate and fail as a stack error somewhere
        # unrelated.
        def _shrink(volume, rng):
            return volume[..., :16, :16]

        with pytest.raises(ValueError, match="augment hook changed the crop"):
            _dataset(_corpus(tmp_path), augment=_shrink)[0]

    def test_the_hook_is_called_once_per_present_slot_only(self, tmp_path):
        calls = []

        def _count(volume, rng):
            calls.append(1)
            return volume

        corpus = _corpus(tmp_path, patterns=((1, 4),))
        _dataset(corpus, slots=(0, 1, 4), augment=_count)[0]
        assert len(calls) == 2

    def test_a_cutmix_shaped_batch_hook_may_mix_labels_and_mask_together(self, tmp_path):
        # CutMix mixes two studies, so it belongs in the collate where labels
        # and mask can be mixed with the pixels rather than after them.
        dataset = _dataset(_corpus(tmp_path), n_slices=2)

        def _cutmix(batch, rng):
            flipped = torch.flip(batch["pixels"], dims=(0,))
            batch["pixels"][..., :16, :16] = flipped[..., :16, :16]
            batch["labels"] = torch.maximum(batch["labels"], torch.flip(batch["labels"], (0,)))
            batch["mask"] = batch["mask"] & torch.flip(batch["mask"], (0,))
            return batch

        batch = collate_studies([dataset[0], dataset[1]], batch_augment=_cutmix)
        assert batch["pixels"].shape == (2, 1, 2, 3, 32, 32)
        assert torch.isfinite(batch["labels"]).all()


class TestSeriesResolution:
    def test_the_series_with_more_cached_slices_wins(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=1, slice_counts=(20, 20), duplicate_slot_studies=1)
        dataset = _dataset(corpus, n_slices=2, crop=_identity_crop(corpus))
        chosen = dataset._series[corpus.study_uids[0]][1]
        assert int(np.load(chosen.path, mmap_mode="r").shape[0]) == 20

    def test_the_choice_does_not_depend_on_manifest_row_order(self, tmp_path):
        # The same series has to be picked in a training run and in the
        # inference run that reads a differently ordered manifest.
        corpus = _corpus(tmp_path, n_studies=1, slice_counts=(20, 20), duplicate_slot_studies=1)
        forwards = _dataset(corpus, n_slices=2)._series[corpus.study_uids[0]][1]
        reversed_manifest = corpus.manifest.iloc[::-1].reset_index(drop=True)
        backwards = StudyDataset(
            reversed_manifest,
            corpus.cache_dir,
            normalisation=Normalisation.identity(),
            labels=corpus.labels,
            slots=(1,),
            n_slices=2,
        )._series[corpus.study_uids[0]][1]
        assert forwards.series_uid == backwards.series_uid

    def test_a_series_with_no_cache_file_is_absent_not_black(self, tmp_path):
        corpus = _corpus(tmp_path)
        manifest = corpus.manifest.copy()
        manifest.loc[manifest.index[0], "series_uid"] = "1.2.3.not.cached"
        with pytest.warns(UserWarning, match="no cached series"):
            dataset = StudyDataset(
                manifest,
                corpus.cache_dir,
                normalisation=Normalisation.identity(),
                labels=corpus.labels,
                slots=(1,),
                n_slices=2,
            )
        assert not dataset[0]["present"].any()

    def test_a_manifest_missing_a_required_column_raises(self, tmp_path):
        corpus = _corpus(tmp_path)
        with pytest.raises(KeyError, match="spacing"):
            StudyDataset(
                corpus.manifest.drop(columns=["spacing"]),
                corpus.cache_dir,
                normalisation=Normalisation.identity(),
            )

    def test_the_corpus_helper_builds_a_working_loader(self, tmp_path):
        # The entry point a training smoke run uses, so it has to be exercised
        # rather than assumed. Normalisation stays required even here.
        corpus = _corpus(tmp_path)
        dataset = corpus.study_dataset(
            normalisation=Normalisation.identity(), slots=(1,), n_slices=2
        )
        assert len(dataset) == 3
        assert dataset[0]["pixels"].shape[:3] == (1, 2, 3)
        with pytest.raises(TypeError):
            corpus.study_dataset(slots=(1,))

    def test_from_paths_reads_the_pinned_parquets(self, tmp_path):
        corpus = _corpus(tmp_path)
        dataset = StudyDataset.from_paths(
            corpus.manifest_path,
            corpus.cache_dir,
            corpus.labels_path,
            normalisation=Normalisation.identity(),
            slots=(1,),
            n_slices=2,
        )
        assert len(dataset) == 3
        assert dataset[0]["mask"].all()


class TestReproducibility:
    def test_two_datasets_with_the_same_seed_emit_the_same_pixels(self, tmp_path):
        corpus = _corpus(tmp_path, patterns=((0, 1, 2, 3, 4, 5),))
        options = dict(slots=(0, 1, 2, 3, 4, 5), training=True, series_dropout=0.5, seed=3)
        first = _dataset(corpus, **options)[0]
        second = _dataset(corpus, **options)[0]
        assert torch.equal(first["pixels"], second["pixels"])
        assert torch.equal(first["present"], second["present"])

    def test_set_epoch_moves_the_dropout_stream(self, tmp_path):
        corpus = _corpus(tmp_path, n_studies=8, patterns=((0, 1, 2, 3, 4, 5),))
        dataset = _dataset(
            corpus, slots=(0, 1, 2, 3, 4, 5), training=True, series_dropout=0.5, seed=1
        )
        first = [dataset[i]["present"].tolist() for i in range(8)]
        dataset.set_epoch(1)
        second = [dataset[i]["present"].tolist() for i in range(8)]
        assert first != second
