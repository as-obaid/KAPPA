"""Augmentation: shape, label/mask consistency, and the train-only guarantee.

Every failure this file is written against is silent. A CutMix that mixes pixels
and not labels trains on wrong targets and converges to a plausible curve; a
batch hook that reaches the validation loader mixes the labels the score is
computed against; a per-batch generator seeded from a constant draws one fixed
box forever. None of those raise, and none of them look wrong in a log.
"""

from __future__ import annotations

import numpy as np
import pytest

# The modelling extra is optional (pyproject [project.optional-dependencies]).
# Without this the module fails to COLLECT on a torch-free machine, which reads
# as a broken suite rather than an absent optional dependency.
pytest.importorskip("torch")

import torch  # noqa: E402

from kappa.config import AugmentConfig, ConfigError, loads_config
from kappa.data.augment import (
    CutMix,
    PerBatchStream,
    StudyAugment,
    batch_seed_from_uids,
    build_augment,
    build_batch_augment,
    build_collate,
    rotate,
)
from kappa.data.dataset import collate_studies

B, SLOTS, SLICES, PX = 6, 2, 4, 32


def _batch(*, present=None, mask=None, labels=None) -> dict:
    """A batch shaped like ``collate_studies`` emits, with per-study constant pixels.

    Constant-per-study pixels make a paste visible: any cell of study i holding
    study j's value is a pixel that moved.
    """
    pixels = (
        torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1, 1, 1).expand(B, SLOTS, SLICES, 3, PX, PX)
    )
    return {
        "pixels": pixels.clone(),
        "present": torch.ones(B, SLOTS, dtype=torch.bool) if present is None else present,
        "slot_ids": torch.zeros(B, SLOTS, dtype=torch.long),
        "labels": torch.rand(B, 12) if labels is None else labels,
        "mask": torch.ones(B, 12, dtype=torch.bool) if mask is None else mask,
        "study_uid": [f"study-{i}" for i in range(B)],
    }


class TestRotate:
    """Rotation must preserve the crop exactly: the crop is a physical measurement."""

    def test_the_crop_is_preserved(self):
        x = torch.randn(SLOTS, SLICES, 3, PX, PX)
        assert rotate(x, 13.0).shape == x.shape

    def test_zero_degrees_is_the_identity_not_an_interpolation(self):
        # An interpolating "identity" would blur every unaugmented run.
        x = torch.randn(2, 3, PX, PX)
        assert torch.equal(rotate(x, 0.0), x)

    def test_a_full_turn_returns_the_image(self):
        x = torch.randn(2, 3, PX, PX)
        assert torch.allclose(rotate(x, 360.0), x, atol=1e-4)

    def test_rotation_actually_moves_pixels(self):
        x = torch.zeros(1, 1, PX, PX)
        x[..., 4, 8] = 1.0
        assert not torch.allclose(rotate(x, 30.0), x)

    def test_vacated_corners_are_filled_with_the_normalised_mean(self):
        # 0.0 downstream of normalisation is the dataset mean, not black. Filling
        # with a real intensity would hand the model an edge artefact to learn.
        x = torch.full((1, 1, PX, PX), 5.0)
        assert float(rotate(x, 45.0)[0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-5)


class TestStudyAugment:
    """The per-study hook honours the contract ``StudyDataset`` enforces."""

    def test_it_never_changes_the_crop(self):
        # StudyDataset raises if it does; this catches it one layer earlier.
        x = torch.randn(SLOTS, SLICES, 3, PX, PX)
        aug = StudyAugment(rotate_deg=15.0, brightness=0.2, contrast=0.2)
        assert aug(x, np.random.default_rng(0)).shape == x.shape

    def test_it_is_deterministic_given_the_generator(self):
        x = torch.randn(2, 3, PX, PX)
        aug = StudyAugment(rotate_deg=15.0, brightness=0.2, contrast=0.2)
        first = aug(x, np.random.default_rng(7))
        second = aug(x, np.random.default_rng(7))
        assert torch.equal(first, second)

    def test_different_draws_differ(self):
        x = torch.randn(2, 3, PX, PX)
        aug = StudyAugment(rotate_deg=15.0)
        rng = np.random.default_rng(0)
        assert not torch.equal(aug(x, rng), aug(x, rng))

    def test_an_all_zero_config_leaves_the_pixels_untouched(self):
        x = torch.randn(2, 3, PX, PX)
        assert torch.equal(StudyAugment()(x, np.random.default_rng(0)), x)
        assert not StudyAugment().enabled

    def test_a_negative_half_width_is_refused(self):
        with pytest.raises(ValueError):
            StudyAugment(rotate_deg=-1.0)

    def test_output_stays_finite(self):
        x = torch.randn(2, 3, PX, PX)
        out = StudyAugment(rotate_deg=20.0, brightness=1.0, contrast=1.0)(
            x, np.random.default_rng(1)
        )
        assert bool(torch.isfinite(out).all())


class TestCutMixMovesLabelsWithPixels:
    """The silent corruption this hook exists to avoid."""

    def test_pixels_move(self):
        batch = _batch()
        out = CutMix(prob=1.0)(batch, np.random.default_rng(3))
        assert not torch.equal(out["pixels"], batch["pixels"])
        assert out["pixels"].shape == batch["pixels"].shape

    def test_labels_move_when_pixels_do(self):
        batch = _batch()
        out = CutMix(prob=1.0)(batch, np.random.default_rng(3))
        assert not torch.equal(out["labels"], batch["labels"])

    def test_the_label_weight_matches_the_area_that_was_actually_pasted(self):
        # The box is clipped at the frame edge, so the lambda drawn from the Beta
        # and the fraction of pixels that actually moved are different numbers.
        # Mixing labels by the drawn one describes a mix that did not happen, and
        # that is a wrong target rather than an error.
        #
        # Pixels are constant per study, so the value sitting in the pasted region
        # names the donor and the changed fraction measures the area. Both are read
        # back off the output rather than assumed.
        labels = torch.zeros(B, 12)
        labels[:, 0] = (torch.arange(B, dtype=torch.float32) + 1.0) / (B + 1.0)
        batch = _batch(labels=labels.clone())
        out = CutMix(prob=1.0)(batch, np.random.default_rng(11))
        frame = out["pixels"][:, 0, 0, 0]
        mixed_any = False
        for study in range(B):
            changed = frame[study] != float(study)
            if not bool(changed.any()):
                # Self-mix: the donor was the study itself, so nothing moved and the
                # label must be untouched.
                assert float(out["labels"][study, 0]) == pytest.approx(
                    float(labels[study, 0]), abs=1e-6
                )
                continue
            mixed_any = True
            area = float(changed.to(torch.float32).mean())
            donor = int(frame[study][changed].reshape(-1)[0].item())
            expected = (1.0 - area) * float(labels[study, 0]) + area * float(labels[donor, 0])
            assert float(out["labels"][study, 0]) == pytest.approx(expected, abs=1e-6), study
        assert mixed_any, "this batch mixed nothing; the test proved nothing"

    def test_labels_stay_in_range_and_finite(self):
        out = CutMix(prob=1.0)(_batch(), np.random.default_rng(5))
        assert bool(torch.isfinite(out["labels"]).all())
        assert bool(((out["labels"] >= 0.0) & (out["labels"] <= 1.0)).all())

    def test_the_mask_only_ever_shrinks(self):
        # A cell one study does not know cannot become known by mixing.
        mask = torch.ones(B, 12, dtype=torch.bool)
        mask[0, 0] = False
        mask[3, 5] = False
        batch = _batch(mask=mask)
        out = CutMix(prob=1.0)(batch, np.random.default_rng(3))
        assert bool((out["mask"] <= batch["mask"]).all())
        assert out["mask"].dtype == torch.bool

    def test_a_cell_either_study_does_not_know_is_masked_out(self):
        mask = torch.ones(B, 12, dtype=torch.bool)
        mask[:, 4] = False  # nobody knows column 4
        out = CutMix(prob=1.0)(_batch(mask=mask), np.random.default_rng(3))
        assert not bool(out["mask"][:, 4].any())

    def test_masked_cells_carry_exactly_zero(self):
        # The batch contract: labels are finite everywhere and the mask alone
        # decides. A stale value under a False mask is the bug that survives every
        # NaN check in the project.
        mask = torch.ones(B, 12, dtype=torch.bool)
        mask[0, :] = False
        out = CutMix(prob=1.0)(_batch(mask=mask), np.random.default_rng(3))
        assert bool((out["labels"][~out["mask"]] == 0.0).all())

    def test_the_result_survives_the_collate_finiteness_gate(self):
        samples = [
            {
                "pixels": torch.randn(SLOTS, SLICES, 3, PX, PX),
                "present": torch.ones(SLOTS, dtype=torch.bool),
                "slot_ids": torch.zeros(SLOTS, dtype=torch.long),
                "labels": torch.rand(12),
                "mask": torch.ones(12, dtype=torch.bool),
                "study_uid": f"s{i}",
            }
            for i in range(B)
        ]
        out = collate_studies(samples, batch_augment=CutMix(prob=1.0), seed=0)
        assert bool(torch.isfinite(out["labels"]).all())


class TestCutMixRespectsThePresenceMask:
    """An absent slot is not a black slot, and must not be painted into one."""

    def test_an_absent_slot_is_never_pasted_into(self):
        present = torch.ones(B, SLOTS, dtype=torch.bool)
        present[:, 1] = False
        batch = _batch(present=present)
        out = CutMix(prob=1.0)(batch, np.random.default_rng(3))
        assert torch.equal(out["pixels"][:, 1], batch["pixels"][:, 1])
        assert not torch.equal(out["pixels"][:, 0], batch["pixels"][:, 0])

    def test_presence_itself_is_never_rewritten(self):
        batch = _batch()
        out = CutMix(prob=1.0)(batch, np.random.default_rng(3))
        assert torch.equal(out["present"], batch["present"])

    def test_a_pair_sharing_no_slot_keeps_its_labels(self):
        # Its pixels were untouched, so its labels must be too -- a weight that did
        # not match the pixels would be the same corruption in the other direction.
        present = torch.zeros(B, SLOTS, dtype=torch.bool)
        present[:, 0] = True
        present[1:, 0] = False  # only study 0 has anything at all
        batch = _batch(present=present)
        out = CutMix(prob=1.0)(batch, np.random.default_rng(3))
        assert torch.equal(out["labels"], batch["labels"])
        assert torch.equal(out["pixels"], batch["pixels"])


class TestCutMixProbability:
    def test_probability_zero_is_an_exact_no_op(self):
        batch = _batch()
        out = CutMix(prob=0.0)(batch, np.random.default_rng(0))
        assert torch.equal(out["pixels"], batch["pixels"])
        assert torch.equal(out["labels"], batch["labels"])

    def test_a_single_study_batch_is_left_alone(self):
        batch = {k: (v[:1] if hasattr(v, "__getitem__") else v) for k, v in _batch().items()}
        batch["study_uid"] = ["only"]
        out = CutMix(prob=1.0)(batch, np.random.default_rng(0))
        assert torch.equal(out["pixels"], batch["pixels"])

    def test_bad_parameters_are_refused(self):
        with pytest.raises(ValueError):
            CutMix(prob=1.5)
        with pytest.raises(ValueError):
            CutMix(alpha=0.0)


class TestPerBatchStream:
    """A constant seed per batch is the failure that looks exactly like success."""

    def test_two_different_batches_get_different_mixes(self):
        # collate_studies builds default_rng(seed) fresh on every call, so a hook
        # using it directly draws the identical box forever.
        hook = PerBatchStream(CutMix(prob=1.0), seed=0)
        first = _batch()
        second = _batch()
        second["study_uid"] = [f"other-{i}" for i in range(B)]
        constant = np.random.default_rng(0)
        out_a = hook(first, constant)
        out_b = hook(second, constant)
        assert not torch.equal(out_a["pixels"][0, 0, 0, 0], out_b["pixels"][0, 0, 0, 0])

    def test_the_same_batch_and_seed_reproduce(self):
        hook = PerBatchStream(CutMix(prob=1.0), seed=0)
        a = hook(_batch(), np.random.default_rng(0))
        b = hook(_batch(), np.random.default_rng(999))  # passed rng is ignored by design
        assert torch.equal(a["pixels"], b["pixels"])

    def test_the_run_seed_changes_the_stream(self):
        a = PerBatchStream(CutMix(prob=1.0), seed=0)(_batch(), np.random.default_rng(0))
        b = PerBatchStream(CutMix(prob=1.0), seed=1)(_batch(), np.random.default_rng(0))
        assert not torch.equal(a["pixels"], b["pixels"])

    def test_the_seed_is_stable_across_processes(self):
        # str.__hash__ is salted per process, so the builtin hash would give a
        # different stream on every run while claiming to be reproducible.
        assert np.array_equal(
            batch_seed_from_uids(["a", "b"], 0).random(4),
            batch_seed_from_uids(["a", "b"], 0).random(4),
        )
        assert not np.array_equal(
            batch_seed_from_uids(["a", "b"], 0).random(4),
            batch_seed_from_uids(["a", "c"], 0).random(4),
        )


class TestBuildersDefaultOff:
    """Default off is the property the queued baseline jobs depend on."""

    def test_a_default_config_builds_no_hooks_at_all(self):
        cfg = loads_config("name: t")
        assert build_augment(cfg) is None
        assert build_batch_augment(cfg, seed=0) is None

    def test_a_default_config_gets_the_bare_collate_function(self):
        # Identity-by-object, so an unaugmented run is the same code path it was
        # before this module existed.
        assert build_collate(loads_config("name: t"), seed=0) is collate_studies

    def test_the_validation_loader_never_gets_a_batch_hook(self):
        cfg = loads_config("name: t\naugment: {cutmix_prob: 1.0}\n")
        assert build_collate(cfg, seed=0, training=False) is collate_studies
        assert build_collate(cfg, seed=0, training=True) is not collate_studies

    def test_each_hook_is_built_only_when_its_own_keys_are_set(self):
        rotate_only = loads_config("name: t\naugment: {rotate_deg: 5}\n")
        assert build_augment(rotate_only) is not None
        assert build_batch_augment(rotate_only, seed=0) is None
        cutmix_only = loads_config("name: t\naugment: {cutmix_prob: 1.0}\n")
        assert build_augment(cutmix_only) is None
        assert build_batch_augment(cutmix_only, seed=0) is not None

    def test_the_builders_accept_a_bare_augment_section(self):
        assert build_augment(AugmentConfig(rotate_deg=5.0)) is not None

    def test_horizontal_flip_cannot_be_reached_through_a_config(self):
        with pytest.raises(ConfigError):
            loads_config("name: t\naugment: {hflip: true}\n")
