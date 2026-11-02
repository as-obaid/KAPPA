"""Test-time augmentation: the views are label-safe, the default is off, the cost is stated.

Every failure this file is written against is silent, and three of them would produce a
submission that scores worse for reasons nobody could see in a log.

* **A flipped knee.** A horizontal flip swaps medial and lateral. Medial/Lateral
  Meniscus and Medial/Lateral OA could be swapped back, but MCL has no LCL column and
  PF OA is not lateralised, so no relabelling makes a flipped study correct on all
  twelve. The legal set contains no flip and :class:`TestTheLegalSet` is what keeps it
  that way as names are added.
* **A view that resized the crop.** The crop is a physical measurement,
  ``physical_crop_mm / size`` mm/px. A zoom implemented as a resize would return a
  different pixel grid, every shape check downstream would still pass, and the run
  would report an mm/px it no longer has. :class:`TestTheViewsPreserveTheGrid`.
* **A default that quietly changed.** ``tta=()`` has to be the code that ran before TTA
  existed, not merely something that agrees with it to five decimals -- otherwise every
  number this project has already measured moves by an unrecorded amount.
  :class:`TestTheDefaultIsOff` asserts bit-identity, not closeness.
* **A rate that describes one view.** TTA multiplies the forward cost and the
  efficiency track divides by 32,400 s. :class:`TestTheCostIsReported`.

:class:`TestTheViewsArePooledInLogitSpace` is the other half: this is the one place in
the project that averages by mean rather than by rank, and the tests there pin the two
properties that make that correct -- the pool is exactly the mean of the per-view
logits, and a study's score does not depend on which other studies were scored with it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from kappa import config as config_module  # noqa: E402
from kappa.constants import N_LABELS  # noqa: E402
from kappa.data.dataset import CropSpec, Normalisation, StudyDataset  # noqa: E402
from kappa.data.paths import repo_root  # noqa: E402
from kappa.data.synthetic import make_synthetic_dataset  # noqa: E402
from kappa.inference.predict import (  # noqa: E402
    TTA_ROTATE_DEGREES,
    TTA_TRANSFORMS,
    TTA_ZOOM_FACTOR,
    _tta_views,
    _zoom,
    predict_studies,
)
from kappa.models import StudyClassifier  # noqa: E402

#: A tiny real timm model: 64-dim features, no BatchNorm, no published weights needed.
#: Matches ``tests/test_loop.py`` so the two files exercise the same architecture.
TINY = "test_convnext"

N_STUDIES = 6
N_SLICES = 2
CACHE_PX = 32
CROP_PX = 32

#: Two views, and deliberately the symmetric rotation pair rather than one signed
#: rotation: a single signed view biases every study the same way, which is a different
#: model rather than an averaged one.
PAIR = ("rotate_cw", "rotate_ccw")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("tta-corpus")
    return make_synthetic_dataset(
        root,
        n_studies=N_STUDIES,
        seed=0,
        patterns=((1,),),
        cache_size=CACHE_PX,
        slice_counts=(4, 6),
        mask_rate=0.0,
        prevalence=0.5,
    )


@pytest.fixture(scope="module")
def model() -> StudyClassifier:
    """A fixed-weight model in eval mode. Random weights are fine and the point.

    Nothing here asserts anything about a score's *value*; the claims are about
    identity, determinism and arithmetic, all of which a randomly initialised network
    exercises exactly as well as a trained one. The seed is fixed so a failure can be
    reproduced rather than re-rolled.
    """
    torch.manual_seed(0)
    net = StudyClassifier(
        backbone=TINY, n_slices=N_SLICES, pretrained=False, drop_path_rate=0.0
    )
    return net.eval()


def _dataset(corpus, *, study_uids=None) -> StudyDataset:
    return StudyDataset(
        corpus.manifest,
        corpus.cache_dir,
        normalisation=Normalisation.identity(),
        labels=None,
        study_uids=study_uids,
        slots=(1,),
        n_slices=N_SLICES,
        crop=CropSpec(fov_mm=150.0, out_px=CROP_PX),
        training=False,
        series_dropout=0.0,
    )


def _predict(model, corpus, *, tta=(), study_uids=None):
    return predict_studies(
        model,
        _dataset(corpus, study_uids=study_uids),
        device="cpu",
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        amp=False,
        tta=tta,
    )


class TestTheLegalSet:
    """No flip, no drift from the config module, no silently doubled view."""

    def test_the_legal_set_is_identical_to_the_config_modules(self):
        # Restated in predict.py so a scoring notebook need not import the
        # training-side config module. Restating is only safe while the two agree: a
        # config-legal `eval.tta` that this module refused would raise at the end of a
        # nine-hour run rather than at parse time.
        assert TTA_TRANSFORMS == config_module.TTA_TRANSFORMS

    def test_no_flip_or_mirror_is_reachable_through_the_legal_set(self):
        # This is a knee. MCL has no LCL column and PF OA is not lateralised, so a
        # mirrored study is mislabelled on at least two of the twelve and no
        # relabelling repairs it.
        banned = ("flip", "mirror", "hflip", "lr", "transpose")
        assert not [n for n in TTA_TRANSFORMS if any(b in n.lower() for b in banned)]

    def test_asking_for_a_flip_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown tta"):
            _tta_views(("identity", "hflip"))

    def test_an_unknown_view_is_refused_and_the_message_lists_the_legal_ones(self):
        with pytest.raises(ValueError, match=r"unknown tta.*legal names"):
            _tta_views(("rotate_180",))

    def test_a_repeated_view_is_refused(self):
        # Not a harmless no-op: it double-weights one view in the mean and pays a full
        # extra forward pass for the privilege. kappa.config refuses the same thing.
        with pytest.raises(ValueError, match="duplicate tta"):
            _tta_views(("identity", "identity"))

    def test_every_legal_name_resolves_to_a_view(self):
        assert len(_tta_views(TTA_TRANSFORMS)) == len(TTA_TRANSFORMS)

    def test_the_empty_set_resolves_to_no_views_at_all(self):
        assert _tta_views(()) == ()

    def test_a_bad_name_is_refused_before_any_pixel_is_read(self, model, corpus):
        # The check has to be cheap and early: discovering it after the loader has
        # decoded a test set is an hour of a nine-hour budget spent on a typo.
        with pytest.raises(ValueError, match="unknown tta"):
            _predict(model, corpus, tta=("identity", "sideways"))


class TestTheViewsPreserveTheGrid:
    """The crop is a physical measurement; a view that resized it would falsify it."""

    @pytest.mark.parametrize("name", TTA_TRANSFORMS)
    def test_every_view_returns_the_shape_it_was_given(self, name):
        pixels = torch.randn(2, N_SLICES, 3, CROP_PX, CROP_PX)
        (view,) = _tta_views((name,))
        assert view(pixels).shape == pixels.shape

    def test_zoom_preserves_the_shape_it_was_given(self):
        pixels = torch.randn(2, 3, CROP_PX, CROP_PX)
        assert _zoom(pixels, 1.4).shape == pixels.shape
        assert _zoom(pixels, 1.0 / 1.4).shape == pixels.shape

    def test_a_unit_zoom_returns_the_tensor_rather_than_resampling_it(self):
        # `is`, not `torch.equal`: a unit-scale grid_sample happens to round-trip
        # bit-exactly at this geometry, so equality would pass with the early return
        # deleted and the guarantee would rest on luck. The early return is what makes
        # "a unit zoom does not interpolate" true at every size, and object identity is
        # the only observable that distinguishes it.
        pixels = torch.randn(2, 3, CROP_PX, CROP_PX)
        assert _zoom(pixels, 1.0) is pixels

    def test_identity_returns_the_very_same_tensor(self):
        pixels = torch.randn(2, 3, CROP_PX, CROP_PX)
        (view,) = _tta_views(("identity",))
        assert view(pixels) is pixels

    def test_a_non_positive_zoom_is_refused(self):
        with pytest.raises(ValueError, match="zoom factor must be positive"):
            _zoom(torch.zeros(1, 1, 4, 4), 0.0)

    def test_zoom_in_magnifies_and_zoom_out_shrinks(self):
        # A filled central square: magnifying it must cover more of the frame and
        # shrinking it less. Direction matters -- a sign error here would make the two
        # views identical to each other and the ensemble a doubled single view.
        pixels = torch.zeros(1, 1, CROP_PX, CROP_PX)
        pixels[..., 8:24, 8:24] = 1.0
        area = lambda x: float((x > 0.5).sum())  # noqa: E731
        assert area(_zoom(pixels, TTA_ZOOM_FACTOR)) > area(pixels)
        assert area(_zoom(pixels, 1.0 / TTA_ZOOM_FACTOR)) < area(pixels)

    def test_zoom_out_fills_the_vacated_border_with_the_normalised_mean(self):
        # 0.0 downstream of the backbone's normalisation is the mean pixel, not black.
        # Black is an intensity a scanner can produce and would read as an edge.
        pixels = torch.full((1, 1, CROP_PX, CROP_PX), 5.0)
        out = _zoom(pixels, 1.0 / 2.0)
        assert float(out[0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-5)

    def test_the_rotations_are_opposite_and_neither_is_the_identity(self):
        pixels = torch.zeros(1, 1, CROP_PX, CROP_PX)
        pixels[..., 4, CROP_PX // 2] = 1.0
        cw, ccw = (view(pixels) for view in _tta_views(("rotate_cw", "rotate_ccw")))
        assert not torch.equal(cw, pixels) and not torch.equal(ccw, pixels)
        # Clockwise moves a mark at the top to the right, anticlockwise to the left.
        column = lambda x: int(torch.argmax(x)) % CROP_PX  # noqa: E731
        assert column(cw) > CROP_PX // 2 > column(ccw)

    def test_the_rotation_magnitude_is_the_one_training_uses(self):
        # A TTA view drawn from a distribution the model never trained on tests an
        # invariance nobody asked for. 10 deg is configs/ablate_rotate.yaml's
        # augment.rotate_deg.
        assert TTA_ROTATE_DEGREES == 10.0

    @pytest.mark.parametrize("name", TTA_TRANSFORMS)
    def test_an_absent_slot_stays_exactly_zero_under_every_view(self, name):
        # An absent slot is zero-filled and flagged present=False. A view that left a
        # non-zero smear there would paint signal into a slot the study does not have,
        # which is the exact confusion series_dropout exists to prevent -- and it would
        # be invisible, because the shape and the presence flag would both still agree.
        absent = torch.zeros(1, N_SLICES, 3, CROP_PX, CROP_PX)
        (view,) = _tta_views((name,))
        assert torch.equal(view(absent), absent)


class TestTheDefaultIsOff:
    """With no TTA the result is not merely close to the pre-TTA result. It is it."""

    def test_no_tta_takes_a_single_plain_pass(self, model, corpus):
        result = _predict(model, corpus)
        assert result.tta == ()
        assert result.n_passes == 1
        assert result.n_forward_batches == result.n_batches
        assert result.scores.shape == (N_STUDIES, N_LABELS)

    def test_a_single_identity_view_is_bit_identical_to_no_tta(self, model, corpus):
        # Bit-identical, not allclose: the mean over one view divides by 1.0, which is
        # exact, so anything less than equality means an extra operation crept onto the
        # path. This is the strongest available statement that the default did not move.
        plain = _predict(model, corpus)
        identity = _predict(model, corpus, tta=("identity",))
        assert np.array_equal(plain.scores, identity.scores)
        assert plain.study_uids == identity.study_uids

    def test_the_timing_report_says_nothing_about_tta_when_there_is_none(
        self, model, corpus
    ):
        text = _predict(model, corpus).timing_report(n_studies=1_000)
        assert "tta" not in text
        assert "MEASURED" in text and "not measured at that size" in text

    def test_importing_the_scoring_path_still_pulls_in_no_augmentation_or_config(self):
        # The default-off promise reaches the import graph too. kappa.data.augment
        # reaches kappa.config, and this module's header commits to keeping the
        # training-side config module out of a scoring notebook, so the rotate import
        # is inside the function that needs it.
        code = (
            "import sys, json;"
            "import kappa.inference.predict;"
            "print(json.dumps(sorted("
            "m for m in ('kappa.config', 'kappa.data.augment') if m in sys.modules"
            ")))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo_root() / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]", (
            f"importing the scoring path loaded {done.stdout.strip()} before any TTA was "
            "asked for; with tta=() the notebook's import graph must be what it was "
            "before TTA existed"
        )


class TestTheInferenceBoundaryStillHolds:
    """A1: submissions are image-only, and TTA added an import. Check it went nowhere near."""

    def test_predict_names_no_report_reading_module_anywhere(self):
        # ast.walk, not tree.body: the TTA path imports inside a function, which the
        # module-scope check in tests/test_loop.py would not see.
        tree = ast.parse(
            (repo_root() / "src/kappa/inference/predict.py").read_text(encoding="utf-8")
        )
        modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        modules += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not [name for name in modules if name.startswith("kappa.labels")]

    def test_resolving_every_view_loads_no_report_reading_module(self):
        # The prefix match, not the FORBIDDEN_MODULES list: the list was a floor once
        # and it failed, because the one module inference actually pulled in was the
        # one nobody had named.
        code = (
            "import sys, json;"
            "import kappa.inference.predict as p;"
            "p._tta_views(p.TTA_TRANSFORMS);"
            "print(json.dumps(sorted("
            "m for m in sys.modules"
            " if m == p.FORBIDDEN_PREFIX or m.startswith(p.FORBIDDEN_PREFIX + '.')"
            ")))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo_root() / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]", (
            f"resolving the TTA views loaded report-reading module(s): "
            f"{done.stdout.strip()}. Submissions are image-only (constraint A1)."
        )


class TestTtaActuallyRuns:
    """A view that changed nothing would buy a full extra forward pass with nothing."""

    def test_the_views_move_the_scores(self, model, corpus):
        plain = _predict(model, corpus)
        turned = _predict(model, corpus, tta=PAIR)
        assert not np.array_equal(plain.scores, turned.scores)
        assert np.abs(plain.scores - turned.scores).max() > 1e-6

    def test_tta_changes_neither_the_studies_nor_the_shape(self, model, corpus):
        plain = _predict(model, corpus)
        turned = _predict(model, corpus, tta=PAIR)
        assert turned.study_uids == plain.study_uids
        assert turned.scores.shape == plain.scores.shape
        assert np.isfinite(turned.scores).all()

    def test_the_same_request_scores_the_same_twice(self, model, corpus):
        # Training augmentation draws from an RNG; a TTA view that did the same would
        # give one checkpoint a different macro on every evaluation, and the gap
        # between working and top-10 here is about 0.006.
        first = _predict(model, corpus, tta=PAIR)
        second = _predict(model, corpus, tta=PAIR)
        assert np.array_equal(first.scores, second.scores)

    def test_the_result_records_which_views_ran_and_in_what_order(self, model, corpus):
        result = _predict(model, corpus, tta=("zoom_out", "identity"))
        assert result.tta == ("zoom_out", "identity")
        assert result.n_passes == 2


class TestTheViewsArePooledInLogitSpace:
    """The one mean-not-rank pool in the project. These are the two reasons it is safe."""

    def test_the_pool_is_exactly_the_mean_of_the_per_view_logits(self, model, corpus):
        # Not "correlates with" -- exactly. If this ever became a rank pool the numbers
        # would still be well formed, still be finite, still write a valid submission,
        # and would silently be a different combination.
        views = ("identity", "rotate_cw", "zoom_in")
        singles = [_predict(model, corpus, tta=(name,)).scores for name in views]
        pooled = _predict(model, corpus, tta=views).scores
        # atol at float32 resolution, not bitwise: the pass accumulates in float32 on
        # the device while the reference mean is taken in float64. Nothing weaker would
        # be true, and nothing looser would catch a rank pool, which differs by O(1).
        np.testing.assert_allclose(pooled, np.mean(singles, axis=0), rtol=0, atol=1e-6)

    def test_the_pool_is_not_a_rank_average_of_the_views(self, model, corpus):
        # The two are genuinely different combinations, so the ruling above is a choice
        # with consequences rather than a restatement of the same arithmetic.
        #
        # Stated plainly: this is a claim about the two operators, not about this
        # module, so no mutation of predict.py can make it fail -- it is documentation
        # with an assertion attached, and the ruling rests on the three tests around it.
        # It earns its place by failing if `rank_average` ever degenerated into a mean,
        # which would quietly make those three ambiguous.
        from kappa.evaluation.metrics import rank_average

        views = ("identity", "rotate_cw", "zoom_in")
        singles = [_predict(model, corpus, tta=(name,)).scores for name in views]
        pooled = _predict(model, corpus, tta=views).scores
        assert not np.allclose(pooled, rank_average(singles))

    def test_a_studys_score_does_not_depend_on_who_was_scored_with_it(
        self, model, corpus
    ):
        # This is the property a rank pool would destroy. Rank-normalising inside the
        # pass makes a score a function of its batch-mates: score one study alone and
        # every column collapses to 0.5, in silence, in a perfectly valid submission.
        # The notebook is free to chunk the test set, and the loop scores fold by fold.
        everyone = _predict(model, corpus, tta=PAIR)
        alone = _predict(
            model, corpus, tta=PAIR, study_uids=[everyone.study_uids[0]]
        )
        # atol at float32 resolution: a batch of one and a batch of four reduce the
        # convolutions in a different order, so the two agree to float32 and not
        # bitwise. The claim under test is that the score does not depend on the
        # *cohort*; a rank pool would put this study at 0.5, off by O(1).
        np.testing.assert_allclose(
            alone.scores[0], everyone.scores[0], rtol=0, atol=1e-6
        )

    def test_the_pooled_scores_are_still_logits_and_not_bounded_ranks(
        self, model, corpus
    ):
        # PredictionResult.scores is contractually raw logits: kappa.training.loop
        # stacks each fold's scores into one OOF table, and fold-internal ranks would
        # make that concatenation five incomparable scales with a plausible AUC.
        pooled = _predict(model, corpus, tta=PAIR).scores
        ranks = (np.arange(N_STUDIES) + 0.5) / N_STUDIES
        for column in range(N_LABELS):
            assert not np.allclose(np.sort(pooled[:, column]), ranks)


class TestTheCostIsReported:
    """TTA multiplies the forward cost and the efficiency track divides by 32,400 s."""

    def test_the_forward_batch_count_carries_the_multiplier(self, model, corpus):
        plain = _predict(model, corpus)
        turned = _predict(model, corpus, tta=("identity", "rotate_cw", "rotate_ccw"))
        assert turned.n_batches == plain.n_batches
        assert turned.n_forward_batches == plain.n_batches * 3

    def test_the_measured_rate_still_covers_the_whole_pass(self, model, corpus):
        # The timer wraps the loader as well as every view, because timing only the
        # forward call underestimated a fold's wall clock by 1.7x three times running.
        result = _predict(model, corpus, tta=PAIR)
        assert result.seconds > 0.0
        assert result.seconds_per_study == pytest.approx(result.seconds / N_STUDIES)

    def test_the_timing_report_names_the_views_and_says_the_rate_includes_them(
        self, model, corpus
    ):
        text = _predict(model, corpus, tta=PAIR).timing_report(n_studies=1_000)
        assert "rotate_cw" in text and "rotate_ccw" in text
        assert "2 view(s)" in text
        assert "already includes all of them" in text
        assert "runtime term" in text

    def test_the_projection_multiplies_a_multi_view_rate_not_a_single_view_one(
        self, model, corpus
    ):
        result = _predict(model, corpus, tta=PAIR)
        seconds, term = result.projection(1_000)
        assert seconds == pytest.approx(result.seconds_per_study * 1_000)
        assert term == pytest.approx(seconds / 32_400)


class TestTheSourceOnDisk:
    """Two claims the runtime cannot make about itself."""

    def test_the_default_off_branch_is_the_pre_tta_code(self):
        # `if not views:` exists so the unaugmented path is the old code rather than
        # the new code specialised to one view. Folding it away is exactly the change
        # that would move every number this project has already measured.
        source = (repo_root() / "src/kappa/inference/predict.py").read_text(
            encoding="utf-8"
        )
        assert "if not views:" in source

    def test_predict_studies_takes_tta_as_a_keyword_only_argument(self):
        import inspect

        parameter = inspect.signature(predict_studies).parameters["tta"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == ()


def test_the_module_docstring_states_the_no_flip_ruling():
    # A wrong or missing reason here is what gets a flip added by the next reader.
    from kappa.inference import predict as module

    assert "flip" in (module.__doc__ or "")
    assert "MCL" in (module.__doc__ or "")


def test_the_repo_has_no_flip_named_anywhere_in_the_tta_surface():
    # Cheap, and it catches the one-line "helpful" addition that no other test would.
    source = Path(repo_root() / "src/kappa/inference/predict.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not [
        text for text in literals if text in ("hflip", "flip_lr", "horizontal_flip")
    ]
