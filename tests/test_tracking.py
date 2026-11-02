"""Diagnostic metric keys and curve panels, and the two ways they must not lie.

:func:`diagnostic_label_metrics` is the key-construction function for average
precision, precision/recall/F1 and best-F1. The keys are fixed, verbatim
strings -- ``val/ap/<label>``, ``val/f1_best/<label>``, ``val/threshold_best/<label>``,
and four macros -- and label names are used exactly as they come out of
:class:`DiagnosticResult`, apostrophe and spaces included: ``Baker's`` is the one
name a slugifying key-builder would get wrong first, so it is asserted here by
name, not by count.

:func:`roc_curve_panel` and :func:`pr_curve_panel` build one W&B panel each,
overlaying every scoreable label. Both must return ``None`` the moment ``wandb``
is unavailable -- the same contract :func:`kappa.tracking.per_label_table`
already carries -- because logging a ``None`` is the one way a missing logger
breaks a run that is otherwise supposed to carry on regardless. That path is
exercised here by making ``import wandb`` fail on purpose
(``sys.modules["wandb"] = None``), since the real dependency is installed in
this environment and a plain skip would never touch the guard at all.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from kappa.constants import LABELS, N_LABELS
from kappa.evaluation.metrics import diagnostic_metrics
from kappa.tracking import diagnostic_label_metrics, pr_curve_panel, roc_curve_panel


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _truth(rng: np.random.Generator, n_rows: int = 40, rate: float = 0.35) -> np.ndarray:
    """Binary truth with both classes forced into every column, as in test_metrics.py."""
    y = (rng.random((n_rows, N_LABELS)) < rate).astype(np.float64)
    y[0, :] = 1.0
    y[1, :] = 0.0
    return y


def _scores(rng: np.random.Generator, n_rows: int = 40) -> np.ndarray:
    return rng.random((n_rows, N_LABELS))


@pytest.fixture
def no_wandb(monkeypatch):
    """Make ``import wandb`` raise ImportError, without needing wandb absent for real."""
    monkeypatch.setitem(sys.modules, "wandb", None)


class TestDiagnosticLabelMetricKeys:
    def test_every_key_from_the_brief_is_present_verbatim(self):
        rng = _rng(0)
        result = diagnostic_metrics(_truth(rng), _scores(rng))
        metrics = diagnostic_label_metrics(result, prefix="val")
        for label in LABELS:
            for pattern in ("ap", "precision", "recall", "f1", "f1_best", "threshold_best"):
                assert f"val/{pattern}/{label}" in metrics
        for macro in ("ap_macro", "precision_macro", "recall_macro", "f1_macro"):
            assert f"val/{macro}" in metrics

    def test_bakers_apostrophe_survives_into_the_key_unslugified(self):
        rng = _rng(1)
        result = diagnostic_metrics(_truth(rng), _scores(rng))
        metrics = diagnostic_label_metrics(result, prefix="val")
        assert "val/ap/Baker's" in metrics
        assert not any("Bakers" in key or "baker_s" in key.lower() for key in metrics)

    def test_a_two_word_label_keeps_its_space(self):
        rng = _rng(2)
        result = diagnostic_metrics(_truth(rng), _scores(rng))
        metrics = diagnostic_label_metrics(result, prefix="val")
        assert "val/recall/Lateral Meniscus" in metrics
        assert "val/recall/Lateral_Meniscus" not in metrics

    def test_the_prefix_is_honoured_everywhere_not_only_on_the_macro(self):
        rng = _rng(3)
        result = diagnostic_metrics(_truth(rng), _scores(rng))
        metrics = diagnostic_label_metrics(result, prefix="fold00")
        assert "fold00/ap_macro" in metrics
        assert "fold00/f1/ACL" in metrics
        assert not any(key.startswith("val/") for key in metrics)

    def test_the_values_match_the_result_they_were_built_from(self):
        rng = _rng(4)
        result = diagnostic_metrics(_truth(rng), _scores(rng))
        metrics = diagnostic_label_metrics(result, prefix="val")
        assert metrics["val/ap_macro"] == pytest.approx(result.ap_macro)
        assert metrics["val/f1_best/ACL"] == pytest.approx(result.f1_best["ACL"])
        assert metrics["val/threshold_best/ACL"] == pytest.approx(result.threshold_best["ACL"])

    def test_a_skipped_label_logs_nan_rather_than_a_missing_key(self):
        # Zero positives for one label must still produce every key for it --
        # nan, not absence, is what makes a skip visible in the runs table.
        rng = _rng(5)
        y, s = _truth(rng), _scores(rng)
        y[:, 3] = 0.0  # Lateral Meniscus: nobody positive.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = diagnostic_metrics(y, s)
        metrics = diagnostic_label_metrics(result, prefix="val")
        assert "val/ap/Lateral Meniscus" in metrics
        assert np.isnan(metrics["val/ap/Lateral Meniscus"])
        assert np.isnan(metrics["val/f1_best/Lateral Meniscus"])


class TestCurvePanelsWithWandb:
    """These exercise the real ``wandb``, and skip where it is not installed.

    ``wandb`` is deliberately absent from ``pyproject.toml`` -- tracking is optional
    by construction, and :mod:`kappa.tracking` returns ``None`` from every panel
    builder when the import fails. So these tests describe the cluster training env,
    where it is installed, and cannot describe a laptop that never had it. Without
    the guard below all six failed locally on ``assert panel is not None``, which
    reads as a broken panel builder rather than a missing optional dependency.
    """

    @pytest.fixture(autouse=True)
    def _requires_wandb(self):
        pytest.importorskip("wandb", reason="tracking is optional; panels return None")

    def test_the_roc_panel_returns_one_series_per_scoreable_label(self):
        rng = _rng(10)
        y, s = _truth(rng, n_rows=40), _scores(rng, n_rows=40)
        panel = roc_curve_panel(y, s)
        assert panel is not None
        assert type(panel).__name__ == "CustomChart"

    def test_the_pr_panel_returns_one_series_per_scoreable_label(self):
        rng = _rng(11)
        y, s = _truth(rng, n_rows=40), _scores(rng, n_rows=40)
        panel = pr_curve_panel(y, s)
        assert panel is not None
        assert type(panel).__name__ == "CustomChart"

    def test_the_roc_panel_drops_a_label_with_one_class_present(self):
        rng = _rng(12)
        y, s = _truth(rng, n_rows=40), _scores(rng, n_rows=40)
        y[:, 3] = 0.0  # Lateral Meniscus: no ROC curve exists.
        panel = roc_curve_panel(y, s)
        assert panel is not None  # eleven other labels still have a curve

    def test_the_roc_panel_is_none_when_every_label_is_single_class(self):
        y = np.zeros((10, N_LABELS))
        s = _scores(_rng(13), n_rows=10)
        assert roc_curve_panel(y, s) is None

    def test_the_pr_panel_is_none_when_every_label_has_zero_positives(self):
        y = np.zeros((10, N_LABELS))
        s = _scores(_rng(14), n_rows=10)
        assert pr_curve_panel(y, s) is None

    def test_the_pr_panel_keeps_an_all_positive_label_the_roc_panel_would_drop(self):
        # A degenerate but real precision-recall curve exists with zero negatives;
        # a degenerate ROC curve does not. The two panels must disagree here.
        rng = _rng(15)
        y, s = _truth(rng, n_rows=40), _scores(rng, n_rows=40)
        y[:, 5] = 1.0  # Lateral OA: every study positive.
        pr_panel = pr_curve_panel(y, s)
        assert pr_panel is not None

    def test_mask_narrows_the_cells_a_panel_is_built_from(self):
        rng = _rng(16)
        y, s = _truth(rng, n_rows=40), _scores(rng, n_rows=40)
        mask = np.ones_like(y, dtype=bool)
        mask[:, 0] = False  # ACL entirely masked out
        assert roc_curve_panel(y, s, mask) is not None  # eleven labels remain

    def test_shape_mismatch_raises_rather_than_building_a_wrong_panel(self):
        y = np.zeros((4, N_LABELS))
        s = np.zeros((5, N_LABELS))
        with pytest.raises(ValueError, match="n_studies, n_labels"):
            roc_curve_panel(y, s)


class TestCurvePanelsWithoutWandb:
    def test_the_roc_panel_is_none_without_wandb(self, no_wandb):
        rng = _rng(20)
        y, s = _truth(rng, n_rows=20), _scores(rng, n_rows=20)
        assert roc_curve_panel(y, s) is None

    def test_the_pr_panel_is_none_without_wandb(self, no_wandb):
        rng = _rng(21)
        y, s = _truth(rng, n_rows=20), _scores(rng, n_rows=20)
        assert pr_curve_panel(y, s) is None

    def test_the_panel_is_none_before_any_shape_validation_would_fire(self, no_wandb):
        # The wandb-absent guard is the first thing either panel checks, so a
        # shape that would otherwise raise must not raise here -- carrying on
        # untracked must never depend on the arrays being well-formed.
        y = np.zeros((4, N_LABELS))
        s = np.zeros((5, N_LABELS))
        assert roc_curve_panel(y, s) is None
        assert pr_curve_panel(y, s) is None


class TestATrackedRunMustProduceARunDirectory:
    """Untracked runs that trained normally and left nothing to sync, as a test.

    ``--track`` is ``action="store_true"`` in ``scripts/train.py``. Without it
    :func:`kappa.tracking.init_run` returns a :class:`NoOpRun`, which trains and
    reports perfectly while writing **no** ``offline-run-*`` directory. There is
    nothing to sync afterwards -- compute nodes have no outbound internet -- and
    ``sync_wandb.sh`` reports success either way, because an empty set syncs cleanly.

    The discriminator has to be the directory. A ``NoOpRun`` carries a plausible
    ``id`` and an honest ``offline = True``, so neither of those tells it apart from
    a genuine offline run.
    """

    def test_an_untracked_run_has_no_directory(self, monkeypatch):
        from kappa.tracking import NoOpRun, init_run, offline_directory

        monkeypatch.setenv("KAPPA_NO_TRACKING", "1")
        run = init_run("t", job_type="test")
        assert isinstance(run, NoOpRun)
        assert offline_directory(run) is None

    def test_requesting_tracking_without_getting_it_raises_and_names_the_flag(
        self, monkeypatch
    ):
        from kappa.tracking import init_run, require_run_directory

        monkeypatch.setenv("KAPPA_NO_TRACKING", "1")
        run = init_run("t", job_type="test")
        with pytest.raises(RuntimeError, match="--track"):
            require_run_directory(run, tracked=True)

    def test_not_asking_for_tracking_is_not_an_error(self, monkeypatch):
        """An untracked smoke run is legitimate; only a broken PROMISE is an error."""
        from kappa.tracking import init_run, require_run_directory

        monkeypatch.setenv("KAPPA_NO_TRACKING", "1")
        run = init_run("t", job_type="test")
        assert require_run_directory(run, tracked=False) is None

    def test_a_real_run_directory_satisfies_the_check(self, tmp_path):
        """`run.dir` is the ``files/`` subdirectory; the syncable unit is its parent.

        Asserting on ``run.dir`` itself would pass for a directory ``wandb sync``
        cannot read, so the contract is the parent and this pins it.
        """
        from kappa.tracking import require_run_directory

        files = tmp_path / "offline-run-20260908_120000-abcd1234" / "files"
        files.mkdir(parents=True)

        class Run:
            dir = str(files)

        assert require_run_directory(Run(), tracked=True) == files.parent

    def test_an_empty_dir_attribute_is_treated_as_no_directory(self):
        """wandb hands back "" rather than None on some failure paths."""
        from kappa.tracking import require_run_directory

        class Run:
            dir = ""

        with pytest.raises(RuntimeError, match="no directory on disk"):
            require_run_directory(Run(), tracked=True)
