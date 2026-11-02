"""The presence mask must be real, and the backbone must be named.

Two of these tests are the reason the file exists.

**The garbage-fill test.** An absent ``plane x contrast`` slot is zero-filled to keep
the batch rectangular. The cheap implementation runs it through the encoder and relies
on a zero attention weight to cancel it, which is not the same thing: ``0.0 * nan`` is
``nan``, and with a BatchNorm backbone in training mode the absent slot's content
moves the batch statistics of every present slot. So the test fills an absent slot
with garbage -- large finite values, then ``nan`` and ``inf`` -- and asserts the study
output is *bit-identical*. If that assertion can be removed without a failure
elsewhere, the mask is decoration.

**The backbone-identity tests.** A run that does not print which backbone it loaded is
not evidence about that backbone, and the most expensive form of that mistake is
normalisation: hardcoded ImageNet constants against weights that want ``mean=0.5``
produce a real number supporting a false conclusion. Measured with
``timm.data.resolve_data_config`` this session, ``convnext_small`` wants
``mean=(0.485, 0.456, 0.406)`` at 224 px and ``maxvit_tiny_tf_384`` wants
``mean=(0.5, 0.5, 0.5)`` at 384 px, so there is no safe constant to hardcode.

Everything runs on CPU with ``pretrained=False`` and tiny tensors, because there is no
guaranteed network and timm will try to download.
"""

from __future__ import annotations

import inspect
import warnings as _warnings

import pytest

# The modelling extra is optional (pyproject [project.optional-dependencies]).
# Without this the module fails to COLLECT on a torch-free machine, which reads
# as a broken suite rather than an absent optional dependency.
pytest.importorskip("torch")
pytest.importorskip("timm")

import timm  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from kappa.constants import N_LABELS
from kappa.models.aggregator import (
    GatedAttentionPool,
    LabelQueryHead,
    SeriesAggregator,
    StudyAggregator,
)
from kappa.models.encoder import (
    N_SLOTS,
    PLANES,
    RECOMMENDED_BACKBONES,
    SLOT_ORDER,
    SliceEncoder,
    backbone_data_config,
    create_backbone,
)
from kappa.models.study_classifier import ModelOutput, StudyClassifier

#: A tiny real timm model. 64-dim features, no BatchNorm, and its resolved mean/std is
#: (0.5, 0.5, 0.5) -- deliberately *not* ImageNet, so a hardcoded constant would show.
TINY = "test_convnext"

#: A tiny real timm model with BatchNorm (13 layers, measured), which is the family
#: where masking after the encoder instead of gathering before it silently fails.
TINY_BN = "test_efficientnet"

SIZE = 32
N_SLICES = 4


def _model(**over) -> StudyClassifier:
    kwargs = dict(
        backbone=TINY, n_slices=N_SLICES, pretrained=False, drop_path_rate=0.0, aux_dim=0
    )
    kwargs.update(over)
    return StudyClassifier(**kwargs)


def _batch(batch: int = 2, slots: int = N_SLOTS, n_slices: int = N_SLICES, size: int = SIZE):
    generator = torch.Generator().manual_seed(0)
    pixels = torch.randn(batch, slots, n_slices, 3, size, size, generator=generator)
    present = torch.ones(batch, slots, dtype=torch.bool)
    slot_ids = torch.arange(slots, dtype=torch.long).expand(batch, slots).contiguous()
    return pixels, present, slot_ids


def _series(batch: int = 2, slots: int = N_SLOTS, dim: int = 8):
    generator = torch.Generator().manual_seed(1)
    vectors = torch.randn(batch, slots, dim, generator=generator)
    present = torch.ones(batch, slots, dtype=torch.bool)
    slot_ids = torch.arange(slots, dtype=torch.long).expand(batch, slots).contiguous()
    return vectors, present, slot_ids


def _gen() -> torch.Generator:
    """Seeded generator, so a failure is reproducible rather than flaky."""
    return torch.Generator().manual_seed(7)


class warnings_as_errors:
    """Turn every warning into an error for the duration of the block."""

    def __enter__(self) -> warnings_as_errors:
        self._ctx = _warnings.catch_warnings()
        self._ctx.__enter__()
        _warnings.simplefilter("error")
        return self

    def __exit__(self, *exc) -> bool | None:
        return self._ctx.__exit__(*exc)


class TestBackboneFactory:
    def test_every_recommended_backbone_is_a_real_timm_name(self):
        unknown = [name for name in RECOMMENDED_BACKBONES if name not in timm.list_models()]
        assert unknown == []

    def test_every_recommended_backbone_has_published_weights(self):
        # The whole point of naming tf_efficientnetv2_s rather than efficientnetv2_s.
        missing = [name for name in RECOMMENDED_BACKBONES if not timm.list_pretrained(name)]
        assert missing == []

    def test_asking_for_weights_that_do_not_exist_raises_instead_of_randomising(self):
        # Measured this session: timm.list_pretrained("efficientnetv2_s") == [].
        assert timm.list_pretrained("efficientnetv2_s") == []
        with pytest.raises(ValueError, match="no pretrained weights"):
            create_backbone("efficientnetv2_s", pretrained=True)

    def test_the_error_names_the_spelling_that_does_have_weights(self):
        with pytest.raises(ValueError, match="tf_efficientnetv2_s"):
            create_backbone("efficientnetv2_s", pretrained=True)

    def test_a_scaled_up_backbone_warns_with_the_measured_null(self):
        with pytest.warns(UserWarning, match=r"0\.0011"):
            create_backbone("convnext_base", pretrained=False, drop_path_rate=0.0)

    def test_a_small_backbone_does_not_warn(self):
        with warnings_as_errors():
            create_backbone(TINY, pretrained=False, drop_path_rate=0.0)

    def test_pretrained_false_is_recorded_as_random_init(self):
        _, info = create_backbone(TINY, pretrained=False, drop_path_rate=0.0)
        assert info.weights_loaded is False
        assert "RANDOM INIT" in info.render()
        # A checkpoint name must not appear next to randomly initialised weights.
        assert "source=n/a" in info.render()

    def test_drop_path_rate_is_recorded_as_applied(self):
        _, info = create_backbone(TINY, pretrained=False, drop_path_rate=0.15)
        assert info.drop_path_applied == pytest.approx(0.15)



class TestPreprocessingComesFromTheBackbone:
    def test_normalisation_is_read_from_timm_not_hardcoded(self):
        tiny, _ = create_backbone(TINY, pretrained=False, drop_path_rate=0.0)
        tiny_bn, _ = create_backbone(TINY_BN, pretrained=False, drop_path_rate=0.0)
        assert backbone_data_config(tiny)["mean"] != backbone_data_config(tiny_bn)["mean"]

    def test_a_backbone_that_wants_half_does_not_report_imagenet(self):
        # test_convnext resolves to mean/std of 0.5. If anything in this package
        # substituted ImageNet constants, this is where it would surface.
        model = _model()
        assert tuple(model.data_config["mean"]) == (0.5, 0.5, 0.5)
        assert tuple(model.data_config["std"]) == (0.5, 0.5, 0.5)

    def test_the_model_exposes_the_config_the_dataset_needs(self):
        config = _model().data_config
        assert {"mean", "std", "input_size"} <= set(config)


class TestInputGeometryIsBuiltIntoAVit:
    """For a ViT the crop is a *construction* argument, and it has one right value.

    ``convnext_small`` is fully convolutional and eats any size, which is why
    ``create_backbone`` never passed ``img_size`` and nothing noticed. A ViT bakes its
    patch grid into the position embedding: ``vit_base_patch14_dinov2`` is published at
    518 px, this project feeds 392 (= 14 x 28; 384 / 14 = 27.43 is not an integer), and
    the mismatch is not adapted away at forward time.

    The load-bearing pair here is "created at 392 accepts 392" together with "created
    at 518 refuses 392". Without the second, the first would also pass for a model that
    ignored ``img_size`` entirely, and the argument would be decoration.
    """

    #: A patch-14 ViT, so 392 divides by the patch size, and specifically the small
    #: sibling of the ``vit_base_patch14_dinov2`` this project runs: same family, same
    #: patch 14, same native 518 px, so it exercises the same 518 -> 28 x 28
    #: position-embedding interpolation. Measured on this machine: 21.8M random-init
    #: parameters, 0.23 s to build and 0.08 s for one 392 px forward. The ``test_vit*``
    #: stubs would be faster and cannot be used -- all four are patch 16, and 392 / 16
    #: is 24.5, so they would only ever test a size this project cannot feed.
    VIT = "vit_small_patch14_dinov2"
    IMG_SIZE = 392

    def test_convnext_small_with_no_img_size_is_built_exactly_as_it_was_before(
        self, monkeypatch
    ):
        # The measured 0.7280 convnext_small baseline stays comparable only while its
        # construction is untouched, so this pins the timm call rather than the model
        # that comes out of it: one call, and no img_size anywhere in it. Resolving the
        # native size and passing it explicitly would look harmless and would fail
        # here, which is the point.
        calls: list[tuple[str, dict]] = []
        real = timm.create_model

        def spy(name, **kwargs):
            calls.append((name, kwargs))
            return real(name, **kwargs)

        monkeypatch.setattr(timm, "create_model", spy)
        model, info = create_backbone("convnext_small", pretrained=False, drop_path_rate=0.0)
        assert calls == [
            (
                "convnext_small",
                {
                    "drop_path_rate": 0.0,
                    "pretrained": False,
                    "num_classes": 0,
                    "global_pool": "avg",
                },
            )
        ]
        assert info.img_size_requested is None
        assert info.img_size_applied is None
        model.eval()
        with torch.no_grad():
            pooled = model(torch.randn(2, 3, SIZE, SIZE, generator=_gen()))
        assert pooled.shape == (2, info.num_features)

    def test_a_vit_created_at_392_reports_392_and_accepts_a_392_batch(self):
        model, info = create_backbone(
            self.VIT, pretrained=False, drop_path_rate=0.0, img_size=self.IMG_SIZE
        )
        assert info.img_size_applied == self.IMG_SIZE
        assert info.img_size_requested == self.IMG_SIZE
        model.eval()
        with torch.no_grad():
            pooled = model(torch.randn(1, 3, self.IMG_SIZE, self.IMG_SIZE, generator=_gen()))
        assert pooled.shape == (1, info.num_features)
        assert torch.isfinite(pooled).all()

    def test_the_same_vit_left_at_its_native_size_refuses_a_392_batch(self):
        # The other half of the pair. Measured this session: timm raises
        # "Input height (392) doesn't match model (518)". The message is what is
        # asserted on, not the exception class, which differs between timm's asserts
        # and its size-check branches.
        model, info = create_backbone(self.VIT, pretrained=False, drop_path_rate=0.0)
        assert info.img_size_applied is None
        model.eval()
        with pytest.raises((AssertionError, RuntimeError, ValueError), match="518"):
            with torch.no_grad():
                model(torch.randn(1, 3, self.IMG_SIZE, self.IMG_SIZE, generator=_gen()))

    def test_the_printed_summary_states_the_size_the_model_was_built_at(self):
        _, info = create_backbone(
            self.VIT, pretrained=False, drop_path_rate=0.0, img_size=self.IMG_SIZE
        )
        rendered = info.render()
        # input_size comes from timm's pretrained_cfg and still says 518 for a model
        # built at 392 (measured), so the summary has to carry both numbers or a
        # 392 px run gets filed as a 518 px one.
        assert "input_size=(3, 518, 518)" in rendered
        assert f"img_size   : {self.IMG_SIZE}" in rendered

    def test_a_family_that_refuses_img_size_still_builds_and_records_the_drop(self):
        # test_convnext is the same ConvNeXt class as convnext_small and raises
        # TypeError on img_size (measured: "ConvNeXt.__init__() got an unexpected
        # keyword argument 'img_size'"). Refusing to build would take the whole
        # convolutional half of RECOMMENDED_BACKBONES out of service for a kwarg it
        # does not need; reporting 392 anyway would be worse, because then the only
        # thing wrong would be the log.
        model, info = create_backbone(
            TINY, pretrained=False, drop_path_rate=0.1, img_size=self.IMG_SIZE
        )
        assert info.img_size_requested == self.IMG_SIZE
        assert info.img_size_applied is None
        assert "DROPPED" in info.render()
        # The kwarg this family does accept must survive the narrowing: dropping the
        # whole optional set on the first TypeError would silently lose drop_path too.
        assert info.drop_path_applied == pytest.approx(0.1)
        model.eval()
        with torch.no_grad():
            assert model(torch.randn(1, 3, SIZE, SIZE, generator=_gen())).shape == (
                1,
                info.num_features,
            )

    def test_a_backbone_with_no_img_size_requested_does_not_claim_one(self):
        _, info = create_backbone(TINY, pretrained=False, drop_path_rate=0.0)
        assert "not requested" in info.render()
        assert "DROPPED" not in info.render()

    def test_the_encoder_forwards_img_size_and_embeds_392_px_slices(self):
        encoder = SliceEncoder(
            self.VIT, pretrained=False, drop_path_rate=0.0, img_size=self.IMG_SIZE
        )
        assert encoder.info.img_size_applied == self.IMG_SIZE
        encoder.eval()
        # One slot, one slice: the geometry is what is under test, and the pinned
        # 6-slot batch would put 24 x 392 px images through a 22M ViT per assertion.
        pixels = torch.randn(1, 1, 1, 3, self.IMG_SIZE, self.IMG_SIZE, generator=_gen())
        with torch.no_grad():
            embeddings = encoder(pixels, torch.zeros(1, 1, dtype=torch.long))
        assert embeddings.shape == (1, 1, 1, encoder.embed_dim)
        assert torch.isfinite(embeddings).all()

    def test_the_study_classifier_forwards_img_size_and_prints_what_happened(self, capsys):
        # StudyClassifier is the only object the training loop, the evaluation loop and
        # the Kaggle notebook hold, so a size that cannot be set here cannot be set.
        model = _model(img_size=self.IMG_SIZE)
        assert model.encoder.info.img_size_requested == self.IMG_SIZE
        assert "DROPPED" in capsys.readouterr().out


class TestForwardShapes:
    def test_logits_are_one_row_of_twelve_per_study(self):
        model = _model()
        out = model(*_batch(batch=3))
        assert isinstance(out, ModelOutput)
        assert out.logits.shape == (3, N_LABELS)

    def test_the_auxiliary_head_is_absent_by_default(self):
        assert _model()(*_batch()).aux_logits is None

    def test_the_auxiliary_head_is_wired_when_aux_dim_is_set(self):
        out = _model(aux_dim=17)(*_batch())
        assert out.aux_logits is not None
        assert out.aux_logits.shape == (2, 17)

    def test_a_single_slot_study_works(self):
        pixels, present, slot_ids = _batch(slots=1)
        assert _model()(pixels, present, slot_ids).logits.shape == (2, N_LABELS)

    def test_more_slices_than_configured_are_interpolated_not_refused(self):
        model = _model(n_slices=N_SLICES)
        pixels, present, slot_ids = _batch(n_slices=N_SLICES + 3)
        assert model(pixels, present, slot_ids).logits.shape == (2, N_LABELS)

    def test_logits_are_finite(self):
        assert torch.isfinite(_model()(*_batch()).logits).all()

    def test_a_wrong_channel_count_is_refused_by_name(self):
        model = _model()
        pixels = torch.randn(1, N_SLOTS, N_SLICES, 5, SIZE, SIZE)
        present = torch.ones(1, N_SLOTS, dtype=torch.bool)
        slot_ids = torch.arange(N_SLOTS).expand(1, N_SLOTS)
        with pytest.raises(ValueError, match="pinned at 3"):
            model(pixels, present, slot_ids)

    def test_a_five_dimensional_batch_is_refused(self):
        with pytest.raises(ValueError, match="pinned batch"):
            SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0)(
                torch.randn(1, N_SLOTS, N_SLICES, SIZE, SIZE),
                torch.zeros(1, N_SLOTS, dtype=torch.long),
            )



class TestPresenceMaskIsLoadBearing:
    """An absent slot must be unable to change the output. Bit-identical, not close."""

    @staticmethod
    def _two_runs(model, fill, absent=(0, 3), train_mode=False):
        pixels, present, slot_ids = _batch()
        present[:, list(absent)] = False
        model.train(train_mode)
        torch.manual_seed(0)
        clean = model(pixels.clone(), present, slot_ids).logits
        poisoned = pixels.clone()
        poisoned[:, list(absent)] = fill
        torch.manual_seed(0)
        dirty = model(poisoned, present, slot_ids).logits
        return clean, dirty

    def test_large_garbage_in_an_absent_slot_cannot_change_the_study_output(self):
        clean, dirty = self._two_runs(_model(), 1e6)
        assert torch.equal(clean, dirty)

    def test_non_finite_garbage_in_an_absent_slot_cannot_change_the_output(self):
        # 0.0 * nan is nan, so a zero attention weight is not enough on its own.
        clean, dirty = self._two_runs(_model(), float("nan"))
        assert torch.isfinite(dirty).all()
        assert torch.equal(clean, dirty)

    def test_infinite_garbage_in_an_absent_slot_cannot_change_the_output(self):
        clean, dirty = self._two_runs(_model(), float("inf"))
        assert torch.equal(clean, dirty)

    def test_a_batchnorm_backbone_in_training_mode_is_also_unaffected(self):
        # This is the case that fails if absent slots are encoded and masked later:
        # BatchNorm pools statistics across the flattened slice axis, so the absent
        # slot's content would move every present slot's embedding.
        model = _model(backbone=TINY_BN)
        assert model.encoder.info.n_batchnorm > 0
        clean, dirty = self._two_runs(model, 1e6, train_mode=True)
        assert torch.equal(clean, dirty)

    def test_the_absent_slot_embedding_is_exactly_zero(self):
        encoder = SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0)
        pixels, present, slot_ids = _batch()
        present[0, 2] = False
        embeddings = encoder(pixels, slot_ids, present)
        assert torch.equal(embeddings[0, 2], torch.zeros_like(embeddings[0, 2]))
        assert not torch.equal(embeddings[0, 1], torch.zeros_like(embeddings[0, 1]))

    def test_marking_a_present_slot_absent_does_change_the_output(self):
        # The mirror of the garbage test: the mask must not be ignored in the other
        # direction either, or "absent" and "present" would be indistinguishable.
        model = _model()
        model.eval()
        pixels, present, slot_ids = _batch()
        with torch.no_grad():
            full = model(pixels, present, slot_ids).logits
            fewer = present.clone()
            fewer[:, 0] = False
            reduced = model(pixels, fewer, slot_ids).logits
        assert not torch.allclose(full, reduced)

    def test_a_study_with_no_present_slot_stays_finite(self):
        model = _model()
        pixels, present, slot_ids = _batch()
        present[:] = False
        assert torch.isfinite(model(pixels, present, slot_ids).logits).all()

    def test_gradients_stay_finite_with_a_non_finite_absent_slot(self):
        # torch.where kills nan in the forward pass; this asserts it also keeps the
        # backward pass clean, which multiplying by a zero weight would not.
        model = _model()
        pixels, present, slot_ids = _batch()
        present[0, 0] = False
        pixels[0, 0] = float("nan")
        model(pixels, present, slot_ids).logits.sum().backward()
        bad = [n for n, p in model.named_parameters() if p.grad is not None
               and not torch.isfinite(p.grad).all()]
        assert bad == []

    def test_the_study_pool_requires_the_mask_rather_than_defaulting_to_present(self):
        pool = StudyAggregator(dim=8)
        with pytest.raises(TypeError):
            pool(torch.zeros(1, N_SLOTS, 8))  # type: ignore[call-arg]


class TestStudyAggregatorDirectly:
    def test_garbage_in_an_absent_series_vector_is_bit_identical(self):
        pool = StudyAggregator(dim=8)
        pool.eval()
        vectors, present, slot_ids = _series(dim=8)
        present[:, 1] = False
        with torch.no_grad():
            clean, _ = pool(vectors.clone(), present, slot_ids)
            poisoned = vectors.clone()
            poisoned[:, 1] = float("nan")
            dirty, _ = pool(poisoned, present, slot_ids)
        assert torch.equal(clean, dirty)

    def test_a_non_finite_absent_series_vector_keeps_the_backward_pass_clean(self):
        # Measured: torch.where leaves the parameter gradients and the gradient at
        # the poisoned input finite for nan and inf alike, which multiplying by a
        # zero attention weight does not.
        for fill in (float("nan"), float("inf"), 1e6):
            pool = StudyAggregator(dim=8)
            vectors, present, slot_ids = _series(dim=8)
            present[:, 1] = False
            vectors[:, 1] = fill
            vectors.requires_grad_(True)
            pooled, _ = pool(vectors, present, slot_ids)
            pooled.sum().backward()
            assert torch.isfinite(pooled).all()
            assert torch.isfinite(vectors.grad).all()
            assert all(
                torch.isfinite(p.grad).all() for p in pool.parameters() if p.grad is not None
            )

    def test_absent_slots_get_zero_attention_weight(self):
        pool = StudyAggregator(dim=8)
        vectors, present, slot_ids = _series(dim=8)
        present[0, 2] = False
        _, weights = pool(vectors, present, slot_ids)
        assert weights[0, 2].item() == 0.0
        assert weights[0].sum().item() == pytest.approx(1.0)

    def test_weights_are_all_zero_when_nothing_is_present(self):
        pool = StudyAggregator(dim=8)
        vectors, present, slot_ids = _series(dim=8)
        present[:] = False
        pooled, weights = pool(vectors, present, slot_ids)
        assert weights.abs().sum().item() == 0.0
        assert torch.isfinite(pooled).all()

    def test_a_mismatched_mask_shape_is_refused(self):
        pool = StudyAggregator(dim=8)
        vectors, _, slot_ids = _series(dim=8)
        with pytest.raises(ValueError, match="present has shape"):
            pool(vectors, torch.ones(2, 3, dtype=torch.bool), slot_ids)


class TestSlotConditioning:
    def test_the_same_pixels_under_a_different_slot_embed_differently(self):
        encoder = SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0)
        encoder.eval()
        pixels = torch.randn(1, 2, N_SLICES, 3, SIZE, SIZE, generator=_gen())
        pixels[:, 1] = pixels[:, 0]
        slot_ids = torch.tensor([[0, 4]])
        with torch.no_grad():
            embeddings = encoder(pixels, slot_ids)
        assert not torch.allclose(embeddings[0, 0], embeddings[0, 1])

    def test_conditioning_is_live_at_initialisation(self):
        # A zero-initialised FiLM would make a fresh encoder provably ignore the slot,
        # which reads exactly like broken wiring. It is initialised small, not zero.
        encoder = SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0)
        assert encoder.plane_film.weight.abs().sum().item() > 0.0
        assert encoder.contrast_film.weight.abs().sum().item() > 0.0

    def test_conditioning_starts_close_to_the_unconditioned_backbone(self):
        # Live, but not so loud that it scrambles the pretrained features on step 0.
        encoder = SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0)
        encoder.eval()
        pixels = torch.randn(1, 1, 1, 3, SIZE, SIZE, generator=_gen())
        with torch.no_grad():
            conditioned = encoder(pixels, torch.tensor([[0]]))[0, 0, 0]
            raw = encoder.backbone(pixels.view(1, 3, SIZE, SIZE))[0]
        assert torch.allclose(conditioned, raw, atol=0.5)

    def test_swapping_the_slot_ids_changes_the_study_logits(self):
        model = _model()
        model.eval()
        pixels, present, slot_ids = _batch()
        with torch.no_grad():
            first = model(pixels, present, slot_ids).logits
            second = model(pixels, present, slot_ids.flip(-1)).logits
        assert not torch.allclose(first, second)

    def test_plane_and_contrast_are_factorised_from_slot_order(self):
        encoder = SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0)
        planes = [PLANES[int(i)] for i in encoder.slot_plane]
        contrast = [int(i) for i in encoder.slot_contrast]
        assert list(zip(planes, contrast)) == [tuple(pair) for pair in SLOT_ORDER]

    def test_more_slots_than_slot_order_defines_is_refused(self):
        # slot_ids stay global even when a job selects fewer slots, so a shorter
        # conditioning table would silently condition on the wrong plane.
        with pytest.raises(ValueError, match=f"outside 1..{N_SLOTS}"):
            SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0, n_slots=N_SLOTS + 1)

    def test_a_slot_id_outside_the_table_is_refused(self):
        encoder = SliceEncoder(TINY, pretrained=False, drop_path_rate=0.0)
        pixels, present, _ = _batch(slots=1)
        with pytest.raises(ValueError, match="slot_ids outside"):
            encoder(pixels, torch.full((2, 1), N_SLOTS, dtype=torch.long), present)



class TestTheBackboneIsPrinted:
    def test_construction_prints_the_backbone_name(self, capsys):
        _model()
        assert TINY in capsys.readouterr().out

    def test_construction_states_that_the_weights_are_random(self, capsys):
        _model()
        assert "RANDOM INIT" in capsys.readouterr().out

    def test_construction_prints_the_preprocessing_the_backbone_expects(self, capsys):
        _model()
        printed = capsys.readouterr().out
        assert "mean=(0.5, 0.5, 0.5)" in printed
        assert "input_size=(3, 160, 160)" in printed

    def test_describe_names_the_presence_masked_pool_and_the_twelve_heads(self):
        described = _model().describe()
        assert "presence-masked" in described
        assert f"{N_LABELS} labels" in described

    def test_parameter_count_is_reported(self):
        model = _model()
        assert model.n_parameters == sum(p.numel() for p in model.parameters())
        assert f"{model.n_parameters:,}" in model.describe()


class TestNoScalingKnob:
    def test_the_constructor_signature_is_exactly_the_pinned_one(self):
        # Guards the measured null: scaling the encoder bought +0.0011 against a
        # 0.0020 noise floor, so there must be no width, depth or multiplier knob.
        # `head` and `head_num_heads` shape the READOUT, not the encoder, and the
        # published ablation puts a bespoke head at ~0.006 -- small, and about
        # twenty times less than where the encoder's weights come from.
        # `img_size` is admitted because input geometry is the axis this project is
        # told to spend on rather than parameters, and because for a patch-14 ViT it
        # is not a knob at all: 518-px weights fed the 392 px this project crops to
        # assert on the size instead of adapting. It changes what the model sees, not
        # how much model there is, so the forbidden set below still holds.
        # `freeze_backbone_bn` is admitted for the same reason `head`/`head_num_heads`
        # are: it changes how the encoder NORMALISES, not its width, depth or
        # multiplier, and is refused by config validation on every backbone but the
        # volumetric one, where it exists to fix a train/eval statistics divergence.
        # `per_label_slots` and `head_residual` are admitted on the same ground and
        # are the cheapest entries on this list: between them they add 72 slot scores
        # and one LayerNorm, against ~50 M in the encoder, and they change WHO READS
        # the pooled slots -- twelve labels sharing one set of plane weights, or each
        # holding its own -- not how much model there is. Neither is a width, a depth
        # or a multiplier, and `test_no_parameter_scales_the_encoder` below is the
        # assertion that this stays true rather than being argued for in a comment.
        parameters = list(inspect.signature(StudyClassifier.__init__).parameters)
        assert parameters == [
            "self",
            "backbone",
            "n_slices",
            "pretrained",
            "drop_path_rate",
            "aux_dim",
            "head",
            "head_num_heads",
            "freeze_backbone_bn",
            "img_size",
            "per_label_slots",
            "head_residual",
        ]

    def test_no_parameter_scales_the_encoder(self):
        # The property the pinned list stands in for, asserted directly so that
        # updating the list for a readout knob cannot quietly admit a scaling one.
        parameters = set(inspect.signature(StudyClassifier.__init__).parameters)
        forbidden = {
            "width", "depth", "width_mult", "depth_mult", "scale", "multiplier",
            "embed_dim", "hidden_dim", "layers", "n_layers", "blocks", "n_blocks",
        }
        assert not parameters & forbidden, sorted(parameters & forbidden)

    def test_the_default_backbone_is_the_small_one(self):
        defaults = inspect.signature(StudyClassifier.__init__).parameters
        assert defaults["backbone"].default == "convnext_small"
        assert defaults["n_slices"].default == 16
        assert defaults["drop_path_rate"].default == pytest.approx(0.2)


class TestNoBatchNormInThePoolingStack:
    def test_the_pools_contain_no_batch_normalisation(self):
        # BatchNorm anywhere here would couple studies inside a batch and make the
        # presence guarantee false however carefully the mask is applied.
        modules = list(SeriesAggregator(dim=8, n_slices=4).modules())
        modules += list(StudyAggregator(dim=8).modules())
        modules += list(GatedAttentionPool(8).modules())
        offenders = [m for m in modules if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        assert offenders == []


class TestGatedAttentionPool:
    def test_weights_sum_to_one_over_the_sequence(self):
        pooled, weights = GatedAttentionPool(8)(torch.randn(3, 5, 8, generator=_gen()))
        assert pooled.shape == (3, 8)
        assert torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-5)

    def test_an_excluded_element_cannot_reach_the_pooled_vector(self):
        pool = GatedAttentionPool(8)
        pool.eval()
        x = torch.randn(1, 4, 8, generator=_gen())
        mask = torch.tensor([[True, True, False, True]])
        with torch.no_grad():
            clean, _ = pool(x.clone(), mask)
            poisoned = x.clone()
            poisoned[0, 2] = float("nan")
            dirty, _ = pool(poisoned, mask)
        assert torch.equal(clean, dirty)


class TestSeriesAggregator:
    def test_slice_order_matters(self):
        # Permutation invariance would throw away the axis a tear propagates along.
        pool = SeriesAggregator(dim=8, n_slices=4)
        pool.eval()
        x = torch.randn(1, 4, 8, generator=_gen())
        with torch.no_grad():
            assert not torch.allclose(pool(x), pool(x.flip(1)))

    def test_the_exact_length_path_uses_the_table_untouched(self):
        pool = SeriesAggregator(dim=8, n_slices=4)
        assert pool.positional(4) is pool.position

    def test_a_different_length_is_interpolated_to_that_length(self):
        pool = SeriesAggregator(dim=8, n_slices=4)
        assert pool.positional(7).shape == (1, 7, 8)

    def test_a_wrong_width_is_refused_by_name(self):
        pool = SeriesAggregator(dim=8, n_slices=4)
        with pytest.raises(ValueError, match="expected 8"):
            pool(torch.zeros(1, 4, 16))


class TestLabelQueryHead:
    """Twelve queries, and the presence guarantee the shared Linear also has to keep."""

    @staticmethod
    def _head(dim: int = 16, **over):
        kwargs = {"dim": dim, "n_labels": N_LABELS, "num_heads": 4}
        kwargs.update(over)
        return LabelQueryHead(**kwargs)

    def test_it_emits_one_logit_per_label(self):
        head = self._head()
        out = head(torch.randn(3, 20, 16), torch.ones(3, 20, dtype=torch.bool))
        assert out.shape == (3, N_LABELS)
        assert bool(torch.isfinite(out).all())

    @pytest.mark.parametrize("poison", [1e4, float("nan"), float("inf")])
    def test_an_excluded_token_cannot_change_a_logit(self, poison):
        # NaN and inf are the cases that matter, and they are why the tokens are
        # zeroed rather than only masked out of the softmax: on torch 2.14 a nan in
        # a key excluded by key_padding_mask still reaches the output,
        # because the masked weight is exactly 0 and 0 * nan is nan.
        head = self._head().eval()
        tokens = torch.randn(2, 20, 16)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[0, 10:] = False
        before = head(tokens, mask)
        poisoned = tokens.clone()
        poisoned[0, 10:] = poison
        after = head(poisoned, mask)
        assert bool(torch.isfinite(after).all())
        assert torch.allclose(before, after, atol=1e-4)

    def test_the_result_does_not_depend_on_how_many_tokens_are_excluded(self):
        # This is what the softmax mask buys, separately from the zeroing: without
        # it the excluded tokens still take softmax mass, so the same study would
        # score differently with four absent slots than with five.
        head = self._head().eval()
        tokens = torch.randn(1, 12, 16)
        kept = torch.ones(1, 12, dtype=torch.bool)
        padded = torch.cat([tokens, torch.randn(1, 8, 16)], dim=1)
        padded_mask = torch.cat([kept, torch.zeros(1, 8, dtype=torch.bool)], dim=1)
        assert torch.allclose(head(tokens, kept), head(padded, padded_mask), atol=1e-5)

    def test_a_study_with_nothing_present_yields_the_bias_finitely(self):
        # A contract test, not a mutation detector: on torch 2.14 an all-masked
        # key_padding_mask already returns zeros rather than the NaN older versions
        # produced, so this passes with or without the explicit guard. It is here so
        # that a torch upgrade which reintroduces the NaN is caught by this suite.
        head = self._head().eval()
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[0] = False
        out = head(torch.randn(2, 20, 16), mask)
        assert bool(torch.isfinite(out).all())
        assert torch.allclose(out[0], head.bias, atol=1e-6)

    def test_an_empty_row_does_not_poison_the_gradient(self):
        head = self._head()
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[0] = False
        head(torch.randn(2, 20, 16), mask).sum().backward()
        assert all(
            bool(torch.isfinite(p.grad).all()) for p in head.parameters() if p.grad is not None
        )

    def test_each_label_has_its_own_query_and_readout(self):
        # The whole point: a shared readout would put the twelve back on one vector.
        head = self._head()
        assert head.queries.shape == (N_LABELS, 16)
        assert head.weight.shape == (N_LABELS, 16)

    @pytest.mark.parametrize("label", [0, 1, N_LABELS - 1])
    def test_each_label_s_logit_uses_that_label_s_own_readout_row(self, label):
        # Shape alone does not prove independence -- a broadcast row has the same
        # shape after expand(), and a head that routed every label through row 0
        # would still pass a check written against label 0 only. So differentiate
        # each label in turn: its own row must move and no other row may.
        head = self._head()
        logits = head(torch.randn(2, 20, 16), torch.ones(2, 20, dtype=torch.bool))
        logits[:, label].sum().backward()
        rows = head.weight.grad
        assert not bool(torch.allclose(rows[label], torch.zeros(16))), "own row got no gradient"
        others = [i for i in range(N_LABELS) if i != label]
        assert bool(torch.allclose(rows[others], torch.zeros(len(others), 16))), "leaked"
        assert bool(
            torch.allclose(head.queries.grad[others], torch.zeros(len(others), 16))
        ), "a query other than this label's moved"

    def test_the_twelve_queries_do_not_start_identical(self):
        # Twelve identical queries are twelve copies of one head that happen to have
        # twelve rows; they would attend to exactly the same tokens at step zero.
        head = self._head()
        assert not torch.allclose(head.queries[0], head.queries[1])

    def test_the_head_count_is_reduced_to_something_that_divides_the_width(self):
        # A width that does not divide would otherwise raise deep inside
        # nn.MultiheadAttention rather than here.
        assert self._head(dim=12, num_heads=8).num_heads in (4, 2, 1)
        assert 12 % self._head(dim=12, num_heads=8).num_heads == 0

    def test_mismatched_shapes_are_refused(self):
        head = self._head()
        with pytest.raises(ValueError):
            head(torch.randn(2, 20, 8), torch.ones(2, 20, dtype=torch.bool))
        with pytest.raises(ValueError):
            head(torch.randn(2, 20, 16), torch.ones(2, 19, dtype=torch.bool))


class TestHeadSelection:
    """The head is a config choice, and the linear default must not move."""

    def test_the_default_head_is_the_shared_linear_one(self):
        model = _model()
        assert model.head_kind == "linear"
        assert isinstance(model.head, torch.nn.Linear)

    def test_the_query_head_is_selectable_and_changes_the_parameter_count(self):
        model = _model(head="query", head_num_heads=4)
        assert isinstance(model.head, LabelQueryHead)
        assert model.n_parameters > _model().n_parameters

    def test_both_heads_produce_the_same_output_shape(self):
        pixels, present, slot_ids = _batch()
        for kind in ("linear", "query"):
            out = _model(head=kind, head_num_heads=4)(pixels, present, slot_ids)
            assert out.logits.shape == (pixels.shape[0], N_LABELS)
            assert bool(torch.isfinite(out.logits).all())

    def test_the_query_head_keeps_the_absent_slot_guarantee(self):
        # This is the property that rules out BatchNorm backbones; a new head must
        # not be the thing that breaks it.
        model = _model(head="query", head_num_heads=4).eval()
        pixels, present, slot_ids = _batch()
        present = present.clone()
        present[0, 2:] = False
        before = model(pixels, present, slot_ids).logits
        poisoned = pixels.clone()
        poisoned[0, 2:] = 1e3
        assert torch.allclose(before, model(poisoned, present, slot_ids).logits, atol=1e-3)

    def test_the_aux_head_still_works_under_the_query_head(self):
        out = _model(head="query", head_num_heads=4, aux_dim=8)(*_batch())
        assert out.aux_logits is not None and out.aux_logits.shape[-1] == 8

    def test_the_description_names_which_head_ran(self):
        # describe() is printed at startup and is the run's only record of it.
        assert "linear" in _model().describe()
        assert "query" in _model(head="query", head_num_heads=4).describe()

    def test_an_unknown_head_is_refused(self):
        with pytest.raises(ValueError, match="unknown head"):
            _model(head="transformer")


#: Every parameter and buffer :class:`StudyAggregator` has ever carried. Pinned as a
#: literal rather than compared against a rebuilt module, because the property is
#: "identical to the module BEFORE `per_label` existed" and there is no longer a
#: version of the class that can be rebuilt to check against. Twenty checkpoints on
#: the cluster hold exactly these eleven keys under `study_pool.*` and load with
#: `strict=True`; an addition here that is not conditional on `per_label` makes every
#: one of them an unexpected-key failure, which is a two-hour discovery inside a
#: queued job rather than a test failure here.
STUDY_POOL_KEYS_BEFORE_PER_LABEL = [
    "norm.bias",
    "norm.weight",
    "out_norm.bias",
    "out_norm.weight",
    "pool.gate.bias",
    "pool.gate.weight",
    "pool.score.bias",
    "pool.score.weight",
    "pool.value.bias",
    "pool.value.weight",
    "slot_embedding.weight",
]


class TestPerLabelSlotsIsStructurallyInvisibleWhenOff:
    """`per_label=0` must be the module that already exists, not an equivalent of it."""

    def test_the_state_dict_keys_are_exactly_the_ones_every_checkpoint_holds(self):
        assert sorted(StudyAggregator(dim=8).state_dict()) == STUDY_POOL_KEYS_BEFORE_PER_LABEL

    def test_the_per_label_table_is_not_built_at_all_when_it_is_off(self):
        # Not "built and unused": an unconditionally registered parameter would add
        # two unexpected keys to all twenty existing checkpoints.
        pool = StudyAggregator(dim=8)
        assert not hasattr(pool, "label_slot_logits")
        assert not hasattr(pool, "label_norm")

    def test_turning_it_on_adds_the_slot_table_and_nothing_else_of_substance(self):
        off = set(StudyAggregator(dim=8).state_dict())
        on = set(StudyAggregator(dim=8, per_label=N_LABELS).state_dict())
        assert off < on
        assert on - off == {"label_slot_logits", "label_norm.weight", "label_norm.bias"}

    def test_the_table_is_twelve_by_six_and_not_a_projection_of_the_features(self):
        # The arm's cost claim -- 72 numbers -- and its meaning: a preference over
        # SLOTS, not a content-dependent scorer, which would answer a different
        # question and would not be readable as "which plane does this label use".
        pool = StudyAggregator(dim=64, n_slots=N_SLOTS, per_label=N_LABELS)
        assert tuple(pool.label_slot_logits.shape) == (N_LABELS, N_SLOTS)
        assert pool.label_slot_logits.numel() == 72

    def test_the_forward_contract_stays_a_pair_at_both_settings(self):
        # The return arity must not depend on a constructor argument; the third output
        # lives on `forward_with_per_label`, which `forward` delegates to.
        vectors, present, slot_ids = _series(dim=8)
        for per_label in (0, N_LABELS):
            pooled = StudyAggregator(dim=8, per_label=per_label)(vectors, present, slot_ids)
            assert len(pooled) == 2

    def test_the_third_output_is_none_when_the_arm_is_off(self):
        vectors, present, slot_ids = _series(dim=8)
        pool = StudyAggregator(dim=8)
        assert pool.forward_with_per_label(vectors, present, slot_ids)[2] is None

    def test_the_shared_pool_is_untouched_by_the_per_label_path(self):
        # The residual claim, at the module level: the shared vector the head has
        # always read must be the same number whether or not the arm is on. Measured
        # by copying the shared weights across rather than by reseeding, so nothing
        # depends on the order the two modules drew their initialisation.
        off = StudyAggregator(dim=8).eval()
        on = StudyAggregator(dim=8, per_label=N_LABELS).eval()
        on.load_state_dict(off.state_dict(), strict=False)
        vectors, present, slot_ids = _series(dim=8)
        present[0, 3] = False
        with torch.no_grad():
            shared, weights = off(vectors, present, slot_ids)
            also_shared, also_weights, _ = on.forward_with_per_label(vectors, present, slot_ids)
        assert torch.equal(shared, also_shared)
        assert torch.equal(weights, also_weights)


class TestThePerLabelPathKeepsThePresenceGuarantee:
    """The guarantee the whole module exists for, asserted again on the new path.

    A second softmax over the slots is a second chance to read an absent one. Each
    test here is the per-label twin of one in `TestStudyAggregatorDirectly`.
    """

    def _pool(self):
        return StudyAggregator(dim=8, per_label=N_LABELS)

    @pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf"), 1e9])
    def test_garbage_in_an_absent_slot_leaves_the_per_label_vectors_bit_identical(
        self, poison
    ):
        pool = self._pool().eval()
        vectors, present, slot_ids = _series(dim=8)
        present[:, 1] = False
        with torch.no_grad():
            _, _, clean = pool.forward_with_per_label(vectors.clone(), present, slot_ids)
            poisoned = vectors.clone()
            poisoned[:, 1] = poison
            _, _, dirty = pool.forward_with_per_label(poisoned, present, slot_ids)
        assert clean is not None and dirty is not None
        assert torch.equal(clean, dirty)

    @pytest.mark.parametrize("fill", [float("nan"), float("inf"), 1e6])
    def test_a_non_finite_absent_slot_keeps_the_backward_pass_clean(self, fill):
        # The forward being bit-identical is not enough: `0.0 * nan` is `nan` and the
        # gradient is where that would show up first.
        pool = self._pool()
        vectors, present, slot_ids = _series(dim=8)
        present[:, 1] = False
        vectors[:, 1] = fill
        vectors.requires_grad_(True)
        _, _, per_label = pool.forward_with_per_label(vectors, present, slot_ids)
        per_label.sum().backward()
        assert torch.isfinite(per_label).all()
        assert torch.isfinite(vectors.grad).all()
        assert all(
            torch.isfinite(p.grad).all() for p in pool.parameters() if p.grad is not None
        )

    def test_a_study_with_no_present_slot_stays_finite(self):
        pool = self._pool()
        vectors, present, slot_ids = _series(dim=8)
        present[:] = False
        pooled, weights, per_label = pool.forward_with_per_label(vectors, present, slot_ids)
        assert torch.isfinite(pooled).all()
        assert torch.isfinite(per_label).all()
        assert weights.abs().sum().item() == 0.0

    def test_an_absent_slot_takes_no_softmax_mass_from_any_label(self):
        # The other half of the guarantee, and the half the zeroing cannot supply:
        # without it the result would depend on HOW MANY slots are absent.
        pool = self._pool()
        vectors, present, slot_ids = _series(dim=8)
        present[0, 2] = False
        weights = pool._per_label_weights(present, slot_ids)
        assert weights.shape == (vectors.shape[0], N_LABELS, vectors.shape[1])
        assert weights[0, :, 2].abs().sum().item() == 0.0
        assert torch.allclose(weights[0].sum(-1), torch.ones(N_LABELS), atol=1e-5)

    def test_each_label_weights_the_slots_differently(self):
        # The entire point of the arm. Twelve identical rows would be the shared pool
        # with twelve times the arithmetic.
        pool = self._pool()
        _, present, slot_ids = _series(dim=8)
        weights = pool._per_label_weights(present, slot_ids)
        assert not torch.allclose(weights[0, 0], weights[0, 1])

    def test_the_table_is_read_by_slot_identity_not_by_column_order(self):
        # A table read by column would mean a different PLANE in the one-slot job than
        # in the six-slot job, and the two could not be compared.
        pool = self._pool()
        _, present, slot_ids = _series(dim=8)
        straight = pool._per_label_weights(present, slot_ids)
        swapped_ids = slot_ids.clone()
        swapped_ids[:, [0, 1]] = swapped_ids[:, [1, 0]]
        swapped = pool._per_label_weights(present, swapped_ids)
        assert torch.allclose(straight[:, :, 0], swapped[:, :, 1])

    def test_no_batch_normalisation_reached_the_new_path(self):
        # BatchNorm here would couple studies inside a batch and make every assertion
        # above false however carefully the mask is applied.
        modules = list(StudyAggregator(dim=8, per_label=N_LABELS).modules())
        assert [m for m in modules if isinstance(m, nn.modules.batchnorm._BatchNorm)] == []

    def test_a_negative_per_label_is_refused_rather_than_silently_disabling_the_arm(self):
        with pytest.raises(ValueError, match="per_label"):
            StudyAggregator(dim=8, per_label=-1)


class TestThePerLabelHeadArithmetic:
    """Each label reads its own slot weighting, and the shared vector stays under it."""

    def test_the_flags_default_off_and_leave_the_state_dict_alone(self):
        plain = _model()
        assert plain.per_label_slots is False and plain.head_residual is False
        assert set(plain.state_dict()) == set(_model(per_label_slots=False).state_dict())

    def test_turning_it_on_adds_only_the_slot_table_and_its_norm(self):
        added = set(_model(per_label_slots=True).state_dict()) - set(_model().state_dict())
        assert added == {
            "study_pool.label_slot_logits",
            "study_pool.label_norm.weight",
            "study_pool.label_norm.bias",
        }

    def test_the_readout_is_row_wise_and_not_a_shared_linear(self):
        # `logits[b, l] = (study[b] + per_label[b, l]) . W[l] + bias[l]`, recomputed
        # here from the model's own pieces. `self.head(vector)` would score every
        # label's vector against every label's row and return (B, 12, 12); this
        # asserts the row-wise contraction actually happened.
        model = _model(per_label_slots=True).eval()
        pixels, present, slot_ids = _batch()
        with torch.no_grad():
            out = model(pixels, present, slot_ids)
            embeddings = model.encoder(pixels, slot_ids, present)
            b, sl, ns, d = embeddings.shape
            series = model.series_pool(embeddings.reshape(b * sl, ns, d))
            study, _, per_label = model.study_pool.forward_with_per_label(
                series.view(b, sl, d), present, slot_ids
            )
            vector = study.unsqueeze(1) + per_label
            expected = (vector * model.head.weight).sum(-1) + model.head.bias
        assert torch.equal(out.logits, expected)

    def test_a_zero_per_label_increment_reproduces_the_baseline(self):
        # The degenerate case, and the reason the residual shape can be measured
        # against the baseline rather than merely compared to it: with the per-label
        # vector identically zero, `v[b, l] = study[b]` for all twelve labels and the
        # row-wise dot product IS `F.linear(study, W, bias)` as an expression.
        #
        # NOT bit-identical, and the tolerance is measured rather than guessed. The
        # two contract the `dim` axis in different orders, so they differ by float32
        # reduction order alone: max |difference| 3.0e-8 at dim 64 and 6.6e-7 at dim
        # 768, measured on random inputs. `torch.equal` here would be a test that
        # fails for a reason that is not about this model. The `per_label_slots=False`
        # path does not take this branch at all and IS bit-identical -- that is
        # `test_the_flags_default_off_and_leave_the_state_dict_alone`'s half.
        base = _model().eval()
        arm = _model(per_label_slots=True).eval()
        arm.load_state_dict(base.state_dict(), strict=False)
        with torch.no_grad():
            arm.study_pool.label_norm.weight.zero_()
            arm.study_pool.label_norm.bias.zero_()
            pixels, present, slot_ids = _batch()
            gap = (
                base(pixels, present, slot_ids).logits - arm(pixels, present, slot_ids).logits
            ).abs().max()
        assert gap < 1e-6, gap

    def test_a_non_zero_increment_does_move_the_logits(self):
        # The companion to the test above: if the increment could not move anything,
        # the arm would be a no-op wearing a config flag.
        base = _model().eval()
        arm = _model(per_label_slots=True).eval()
        arm.load_state_dict(base.state_dict(), strict=False)
        pixels, present, slot_ids = _batch()
        with torch.no_grad():
            assert not torch.equal(
                base(pixels, present, slot_ids).logits, arm(pixels, present, slot_ids).logits
            )

    def test_the_absent_slot_guarantee_survives_the_whole_model(self):
        model = _model(per_label_slots=True).eval()
        pixels, present, slot_ids = _batch()
        present = present.clone()
        present[0, 2:] = False
        before = model(pixels, present, slot_ids).logits
        poisoned = pixels.clone()
        poisoned[0, 2:] = 1e3
        assert torch.allclose(before, model(poisoned, present, slot_ids).logits, atol=1e-3)

    def test_the_logits_stay_finite_when_a_study_has_nothing_present(self):
        model = _model(per_label_slots=True).eval()
        pixels, present, slot_ids = _batch()
        present = torch.zeros_like(present)
        assert bool(torch.isfinite(model(pixels, present, slot_ids).logits).all())

    def test_it_works_under_the_query_head_too(self):
        out = _model(per_label_slots=True, head="query", head_num_heads=4)(*_batch())
        assert out.logits.shape == (2, N_LABELS)
        assert bool(torch.isfinite(out.logits).all())


class TestTheHeadResidual:
    """Without it, `head="query"` with `aux_dim=0` trains neither pool at all."""

    def test_it_is_refused_on_the_linear_head_rather_than_ignored(self):
        with pytest.raises(ValueError, match="head_residual"):
            _model(head_residual=True)

    def test_the_query_head_leaves_the_pools_untrained_without_it(self):
        # The documented behaviour this flag exists to change, asserted rather than
        # asserted-about: nothing downstream consumes the pooled vector, so the
        # gradient never reaches either pool and both sit in the checkpoint at their
        # initial values.
        model = _model(head="query", head_num_heads=4)
        model(*_batch()).logits.sum().backward()
        assert model.study_pool.out_norm.weight.grad is None
        assert model.series_pool.pool.score.weight.grad is None

    def test_the_residual_puts_the_gradient_back_into_both_pools(self):
        model = _model(head="query", head_num_heads=4, head_residual=True)
        model(*_batch()).logits.sum().backward()
        for tensor in (
            model.study_pool.out_norm.weight,
            model.study_pool.slot_embedding.weight,
            model.series_pool.pool.score.weight,
        ):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()

    def test_the_per_label_arm_trains_the_pools_without_the_residual(self):
        # The two flags are independent and neither implies the other: the per-label
        # vector runs through the same slot embedding and norms, so it carries the
        # gradient on its own.
        model = _model(head="query", head_num_heads=4, per_label_slots=True)
        model(*_batch()).logits.sum().backward()
        assert model.study_pool.label_slot_logits.grad is not None
        assert model.study_pool.slot_embedding.weight.grad is not None

    def test_the_residual_is_added_and_does_not_replace_what_the_head_reads(self):
        # Replacing is the losing arrangement in the published ablation (82.1 / 84.2 /
        # 85.3), so this asserts the attended vector is still in the sum.
        head = LabelQueryHead(dim=8, n_labels=N_LABELS, num_heads=4).eval()
        tokens = torch.randn(2, 5, 8, generator=_gen())
        mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            plain = head(tokens, mask)
            zero = head(tokens, mask, torch.zeros(2, 1, 8))
            shifted = head(tokens, mask, torch.ones(2, 1, 8))
        assert torch.equal(plain, zero), "a zero residual must be the untouched head"
        assert not torch.equal(plain, shifted)

    def test_a_residual_of_the_wrong_shape_is_refused_by_name(self):
        head = LabelQueryHead(dim=8, n_labels=N_LABELS, num_heads=4)
        tokens = torch.randn(2, 5, 8, generator=_gen())
        mask = torch.ones(2, 5, dtype=torch.bool)
        with pytest.raises(ValueError, match="residual has shape"):
            head(tokens, mask, torch.zeros(2, 1, 7))
        with pytest.raises(ValueError, match="residual has 3 label rows"):
            head(tokens, mask, torch.zeros(2, 3, 8))

    def test_the_absent_slot_guarantee_survives_the_residual(self):
        model = _model(head="query", head_num_heads=4, head_residual=True).eval()
        pixels, present, slot_ids = _batch()
        present = present.clone()
        present[0, 2:] = False
        before = model(pixels, present, slot_ids).logits
        poisoned = pixels.clone()
        poisoned[0, 2:] = 1e3
        assert torch.allclose(before, model(poisoned, present, slot_ids).logits, atol=1e-3)

    def test_the_description_states_which_pooling_ran(self):
        # describe() is printed at startup and is the run's only record of it. The
        # DEFAULT has to say so too, because `pool` is dropped from config_hash there.
        assert "shared pool" in _model().describe()
        assert "per-label" in _model(per_label_slots=True).describe()
        assert "no gradient" in _model(head="query", head_num_heads=4).describe()
        assert "pools trained" in _model(
            head="query", head_num_heads=4, head_residual=True
        ).describe()

