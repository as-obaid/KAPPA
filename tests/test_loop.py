"""The wiring is provable today, and three of these tests are the only proof of it.

Why this exists: a training stack that has never been executed is not a training
stack -- every wiring bug is still in it, waiting to surface on a cluster node that
costs an hour of walltime and has no internet. So :class:`TestEndToEndOnCpu` runs an
actual experiment over a generated corpus from :mod:`kappa.data.synthetic`, on CPU,
with ``pretrained=False``, and asserts that it produces an OOF table, a macro AUC and
a ``submission.csv`` whose header matches ``SUBMISSION_COLUMNS`` byte for byte. That
run is the whole point of the synthetic table existing.

Four classes here exist because of a specific mistake that has already been paid for.

* **A run that does not print what it ran.** :class:`TestTheRunStatesWhatItRan`.
  A fold trained against a misspelled key uses the default and gets filed under the
  wrong hypothesis; a run whose log does not name its backbone, config hash, git SHA
  and seed is not evidence about any of them.
* **Timing the inner step.** :class:`TestTimingIsWholeLoop`. Partial timing
  underestimated a fold's wall clock by 1.7x three times running, so the total and
  the inner-step sum are both reported and the total must be the larger.
* **Reading a delta with no floor.** :class:`TestNoiseFloor`. The gap between
  working and top-10 on this task is about 0.006 macro. Until the seed-to-seed spread
  of one config has been measured, no comparison of two runs is readable, so it is a
  supported mode and a tested one.
* **A report on the inference path.** :class:`TestTheInferenceBoundary`. Submissions
  are image-only (constraint A1), so a report-derived import is not a slow path -- it
  is a notebook that cannot run. That test spawns a clean interpreter and checks
  ``sys.modules``; it is a competition-legality check, not a style preference.

Everything runs on CPU with a tiny real timm model, ``mask_rate`` raised so batches
carry usable cells, and a fixed seed, because the point is the plumbing and not the
number. The macros produced here are meaningless by construction and no test asserts
anything about their value.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from kappa.config import TrainConfig, config_hash  # noqa: E402
from kappa.constants import ID_COLUMN, LABELS, N_LABELS, SUBMISSION_COLUMNS  # noqa: E402
from kappa.data.dataset import Normalisation  # noqa: E402
from kappa.data.folds import (  # noqa: E402
    FOLD_COLUMNS,
    FOLD_COLUMNS_V2,
    TEST_FOLD,
    build_folds,
)
from kappa.data.paths import repo_root  # noqa: E402
from kappa.data.synthetic import make_synthetic_dataset  # noqa: E402
from kappa.inference.predict import (  # noqa: E402
    FORBIDDEN_MODULES,
    FORBIDDEN_PREFIX,
    NOTEBOOK_LIMIT_SECONDS,
    CropSpec,
    load_checkpoint,
    predict_studies,
    save_inference_checkpoint,
    transform_scores,
    write_predictions,
)
from kappa.inference.submission import expected_header_bytes, validate_submission  # noqa: E402
from kappa.models.study_classifier import StudyClassifier  # noqa: E402
from kappa.training.losses import AsymmetricLoss, MaskedBCE  # noqa: E402
from kappa.training.loop import (  # noqa: E402
    OOF_COLUMNS,
    _PARAM_GROUP_NAMES,
    DataBundle,
    build_criterion,
    build_model,
    build_optimiser,
    build_scheduler,
    measure_noise_floor,
    run_experiment,
    seed_everything,
    slot_indices,
    train_fold,
)

#: A tiny real timm model: 64-dim features, no BatchNorm, no published weights needed.
TINY = "test_convnext"

#: Nine studies over three folds is the smallest shape that gives every fold a
#: validation set of more than one study, which every AUC needs.
N_STUDIES = 9
N_FOLDS = 3


def _cfg(**over) -> TrainConfig:
    """A CPU-sized config. Every field that makes this fast is set explicitly."""
    options: dict = {
        "name": "wiring",
        "seed": 0,
        "data": {
            "slots": ("Axial_fs1",),
            "n_slices": 2,
            "size": 64,
            "physical_crop_mm": 150.0,
            "num_workers": 0,
            "pin_memory": False,
        },
        "model": {"backbone": TINY, "pretrained": False, "drop_path_rate": 0.0},
        # cosine rather than the baseline's onecycle: a two-epoch run over nine studies
        # is four optimiser steps, which is too short for OneCycleLR to have a warmup
        # phase at all. TestOptimiserAndSchedule covers both branches deliberately.
        "optim": {
            "epochs": 2,
            "batch_size": 4,
            "grad_accum": 1,
            "lr": 1e-3,
            "schedule": "cosine",
        },
        "folds": {"n_folds": N_FOLDS},
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(options.get(key), dict):
            options[key] = {**options[key], **value}
        else:
            options[key] = value
    return TrainConfig(**options)


def _bundle(root: Path, *, n_studies: int = N_STUDIES, seed: int = 0, **over) -> DataBundle:
    """A synthetic corpus plus a fold table over it, as the loop's ``DataBundle``.

    ``mask_rate=0.0`` here rather than the measured 0.987: at the realistic rate only
    1.32% of cells are usable and most folds would have no positives at all, so every
    label would be skipped and the assertions would be about nothing. The sparsity is
    a real property and the loss is tested against it elsewhere; this fixture is
    testing the loop.
    """
    options = {
        "n_studies": n_studies,
        "seed": seed,
        "patterns": ((1,),),
        "cache_size": 32,
        "slice_counts": (4, 6),
        "mask_rate": 0.0,
        "prevalence": 0.5,
    }
    options.update(over)
    corpus = make_synthetic_dataset(root, **options)
    study_table = corpus.labels.merge(
        corpus.manifest.drop_duplicates("study_uid")[
            ["study_uid", "patient_id", "scanner_fingerprint"]
        ].rename(columns={"study_uid": ID_COLUMN}),
        on=ID_COLUMN,
        how="left",
    )
    # The synthetic corpus has no report text, so each study is its own report
    # group. Real folds get this column from train.csv in scripts/build_folds.py.
    study_table["report_group"] = study_table[ID_COLUMN]
    folds = build_folds(study_table, n_folds=N_FOLDS, strata_key=None, seed=seed)
    return DataBundle(
        manifest=corpus.manifest,
        labels=corpus.labels,
        folds=folds,
        cache_dir=corpus.cache_dir,
    )


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("corpus")


@pytest.fixture(scope="module")
def bundle(corpus_root) -> DataBundle:
    return _bundle(corpus_root)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory, bundle):
    """One real experiment over every fold. Module-scoped: it is the expensive one."""
    out = tmp_path_factory.mktemp("run")
    return run_experiment(
        _cfg(),
        bundle,
        device="cpu",
        out_dir=out,
        normalisation=Normalisation.identity(),
        verbose=False,
    )


class TestEndToEndOnCpu:
    """One real run, on CPU, producing the three artifacts the pipeline is judged by."""

    def test_the_experiment_produces_a_finite_macro_auc(self, experiment):
        assert np.isfinite(experiment.summary.macro_mean)
        assert 0.0 <= experiment.summary.macro_mean <= 1.0

    def test_it_reports_a_per_label_mean_and_a_fold_std_for_all_twelve(self, experiment):
        assert list(experiment.summary.per_label_mean) == list(LABELS)
        assert list(experiment.summary.per_label_std) == list(LABELS)
        assert set(experiment.summary.fold_skips) == set(LABELS)

    def test_it_writes_an_oof_parquet_that_reads_back(self, experiment):
        assert experiment.oof_path is not None and experiment.oof_path.is_file()
        assert experiment.oof_path.suffix == ".parquet"
        assert tuple(pd.read_parquet(experiment.oof_path).columns) == OOF_COLUMNS

    def test_a_whole_experiment_keeps_the_unsuffixed_oof_name(self, experiment):
        # The complement of the test below. A run that trained every fold owns the
        # whole out-of-fold table, so its filename stays exactly as it was and no
        # existing artifact or path is invalidated by the suffixing rule.
        assert experiment.oof_path.name == "oof_seed0.parquet"

    def test_a_partial_run_puts_its_folds_in_the_oof_filename(self, bundle, tmp_path):
        # The sweep runs one config as five parallel array tasks, one fold each.
        # Every one of them used to write `oof_seed<seed>.parquet` -- a path with no
        # fold in it -- holding only its own fold's rows. They do not merge: the last
        # writer wins, four folds of predictions are gone, and because the tasks
        # overlap in time it is also a torn write. Nothing raises, the per-fold AUCs
        # still print, and the damage only shows up as a rank ensemble built from one
        # fold of a five-fold table.
        first = run_experiment(
            _cfg(), bundle, device="cpu", out_dir=tmp_path,
            normalisation=Normalisation.identity(), verbose=False, folds=[0],
        )
        second = run_experiment(
            _cfg(), bundle, device="cpu", out_dir=tmp_path,
            normalisation=Normalisation.identity(), verbose=False, folds=[1],
        )
        assert first.oof_path != second.oof_path, (
            "two folds of one config wrote the same OOF path; the second overwrote "
            "the first and three folds' predictions would be silently lost"
        )
        assert "fold0" in first.oof_path.name and "fold1" in second.oof_path.name
        assert first.oof_path.is_file() and second.oof_path.is_file()
        # And both survive, which is the property the ensemble actually needs.
        assert len(pd.read_parquet(first.oof_path)) > 0
        assert len(pd.read_parquet(second.oof_path)) > 0

    def test_the_oof_holds_every_study_exactly_once(self, experiment):
        oof = pd.read_parquet(experiment.oof_path)
        assert len(oof) == N_STUDIES
        assert oof[ID_COLUMN].nunique() == N_STUDIES

    def test_the_oof_carries_the_truth_and_the_mask_beside_every_score(self, experiment):
        oof = pd.read_parquet(experiment.oof_path)
        for name in LABELS:
            assert name in oof
            assert f"{name}__true" in oof
            assert f"{name}__mask" in oof

    def test_the_oof_scores_are_finite_everywhere(self, experiment):
        oof = pd.read_parquet(experiment.oof_path)
        assert np.isfinite(oof[list(LABELS)].to_numpy()).all()

    def test_every_fold_wrote_a_checkpoint(self, experiment):
        for fold in experiment.folds:
            assert fold.checkpoint is not None and fold.checkpoint.is_file()

    def test_a_submission_written_from_the_checkpoints_has_the_exact_header(
        self, experiment, bundle, tmp_path
    ):
        # The end of the wiring claim: weights on disk to a submission whose header is
        # byte-identical to SUBMISSION_COLUMNS, with no step in between reading a
        # report or a config file.
        results = []
        for fold in experiment.folds:
            loaded = load_checkpoint(fold.checkpoint, device="cpu")
            dataset = loaded.dataset(bundle.manifest, bundle.cache_dir)
            results.append(predict_studies(loaded.model, dataset, device="cpu", batch_size=4))
        path = write_predictions(tmp_path / "submission.csv", results)

        written = path.read_bytes().split(b"\n", 1)[0] + b"\n"
        assert written == expected_header_bytes()
        assert written.decode().rstrip("\n").split(",") == list(SUBMISSION_COLUMNS)
        assert b"\r" not in path.read_bytes()
        assert validate_submission(path).ok

    def test_the_submission_covers_every_study_with_finite_scores(
        self, experiment, bundle, tmp_path
    ):
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        result = predict_studies(
            loaded.model, loaded.dataset(bundle.manifest, bundle.cache_dir), device="cpu"
        )
        path = write_predictions(tmp_path / "s.csv", [result])
        check = validate_submission(path, expected_uids=result.study_uids)
        assert check.n_passed == 7, check.summary()


class TestTheRunStatesWhatItRan:
    """A run that does not name its backbone is not evidence about that backbone."""

    def test_the_startup_banner_names_the_backbone_and_the_random_init(
        self, bundle, tmp_path, capsys
    ):
        train_fold(
            _cfg(optim={"epochs": 1}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert TINY in out
        # pretrained=False has to be stated, or a random-init run reads as a
        # pretrained one -- which is the most expensive version of this mistake.
        assert "RANDOM INIT" in out or "pretrained=False" in out

    def test_the_startup_banner_carries_the_config_hash_git_sha_seed_and_device(
        self, bundle, tmp_path, capsys
    ):
        cfg = _cfg(optim={"epochs": 1}, seed=7)
        train_fold(
            cfg,
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert config_hash(cfg) in out
        assert "git sha" in out
        assert "seed       : 7" in out
        assert "device     : cpu" in out

    def test_the_banner_states_the_normalisation_the_backbone_wants(
        self, bundle, tmp_path, capsys
    ):
        # The single most expensive silent failure: a backbone fed the wrong mean/std
        # still trains and still reports a number.
        train_fold(
            _cfg(optim={"epochs": 1}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert "mean=" in out and "std=" in out

    def test_the_fold_result_carries_the_hash_sha_seed_and_backbone(self, experiment):
        for fold in experiment.folds:
            assert fold.config_hash == experiment.config_hash
            assert fold.git_sha == experiment.git_sha
            assert fold.backbone == TINY
            assert fold.seed == 0

    def test_the_report_names_the_fold_column_used(self, bundle, tmp_path, capsys):
        # `fold` groups by patient, `fold_scanner` also by scanner. Two different
        # numbers, so the log has to say which one produced this one.
        train_fold(
            _cfg(optim={"epochs": 1}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            fold_column="fold_scanner",
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=True,
        )
        assert "(fold_scanner)" in capsys.readouterr().out

    def test_the_fold_column_defaults_from_the_config_not_a_literal(
        self, bundle, tmp_path, capsys
    ):
        train_fold(
            _cfg(optim={"epochs": 1}, folds={"column": "fold_scanner"}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=True,
        )
        assert "(fold_scanner)" in capsys.readouterr().out

    def test_the_per_label_table_prints_its_own_denominator(self, experiment):
        text = experiment.folds[0].auc.table()
        assert "n" in text and "pos" in text
        assert "macro" in text

    def test_a_skipped_label_is_named_rather_than_silently_dropped(self, corpus_root, tmp_path):
        # A label with one class present has no AUC. Averaging eleven and calling it
        # macro-12 is a different quantity, so the skip has to be visible.
        bundle = _bundle(tmp_path / "single", prevalence=0.0, seed=3)
        result = train_fold(
            _cfg(optim={"epochs": 1}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=False,
        )
        assert len(result.auc.skipped) == N_LABELS
        assert not result.auc.is_complete
        assert set(result.auc.skipped) == set(LABELS)


class TestTimingIsWholeLoop:
    """Partial timing underestimated a fold by 1.7x three times. Both numbers, always."""

    def test_the_total_is_at_least_the_sum_of_the_inner_steps(self, experiment):
        for fold in experiment.folds:
            assert fold.wall_seconds >= fold.step_seconds

    def test_the_total_is_strictly_greater_because_it_includes_validation(self, experiment):
        # Not merely >=: dataset construction, validation on every epoch and the
        # checkpoint write are all inside the total and outside the step sum.
        for fold in experiment.folds:
            assert fold.wall_seconds > fold.step_seconds

    def test_the_ratio_between_them_is_reported_as_a_number(self, experiment):
        fold = experiment.folds[0]
        assert fold.partial_timing_ratio >= 1.0
        assert np.isfinite(fold.partial_timing_ratio)

    def test_the_experiment_total_covers_every_fold(self, experiment):
        assert experiment.wall_seconds >= sum(f.wall_seconds for f in experiment.folds) * 0.99

    def test_the_fold_report_says_the_total_is_the_number_to_quote(self, experiment):
        text = experiment.folds[0].report()
        assert "TOTAL" in text
        assert "1.7x" in text

    def test_every_epoch_records_its_own_train_and_validation_seconds(self, experiment):
        for record in experiment.folds[0].epochs:
            assert record.train_seconds > 0.0
            assert record.val_seconds > 0.0
            assert record.step_seconds <= record.train_seconds


class TestMeasuredInferenceCost:
    """The efficiency term is T / 32400, so the rate has to be measured, not guessed."""

    def test_every_fold_reports_measured_seconds_per_study(self, experiment):
        for fold in experiment.folds:
            assert fold.seconds_per_study > 0.0
            assert fold.inference.n_studies == fold.n_val

    def test_the_rate_is_the_measured_total_over_the_study_count(self, experiment):
        result = experiment.folds[0].inference
        assert result.seconds_per_study == pytest.approx(result.seconds / result.n_studies)

    def test_the_runtime_term_divides_by_the_nine_hour_notebook_limit(self, experiment):
        seconds, term = experiment.folds[0].inference.projection(1_000)
        assert term == pytest.approx(seconds / NOTEBOOK_LIMIT_SECONDS)
        assert NOTEBOOK_LIMIT_SECONDS == 32_400

    def test_the_timing_report_marks_the_measurement_and_the_projection_apart(
        self, experiment
    ):
        text = experiment.folds[0].inference.timing_report(n_studies=1_000)
        assert "MEASURED" in text
        assert "not measured at that size" in text

    def test_the_experiment_report_states_the_runtime_term(self, experiment):
        text = experiment.report()
        assert "s/study MEASURED" in text
        assert "runtime term" in text
        assert "32400" in text

    def test_the_validation_pass_is_the_same_function_the_notebook_calls(self):
        # Two implementations would drift, and the one that drifted would be the one
        # nobody times. The loop imports predict_studies; it does not have its own.
        import inspect

        from kappa.training import loop

        source = inspect.getsource(loop.train_fold)
        assert "predict_studies(" in source
        assert loop.predict_studies is predict_studies


class TestNoiseFloor:
    """A first-class mode, because a floor nobody runs is a floor nobody has."""

    def test_two_seeds_of_one_config_share_one_hash(self, bundle):
        result = measure_noise_floor(
            _cfg(optim={"epochs": 1}),
            bundle,
            seeds=(0, 1),
            device="cpu",
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        assert len({e.config_hash for e in result.experiments}) == 1
        assert result.config_hash == config_hash(_cfg(optim={"epochs": 1}))

    def test_it_reports_a_spread_and_a_standard_deviation(self, bundle):
        result = measure_noise_floor(
            _cfg(optim={"epochs": 1}),
            bundle,
            seeds=(0, 1, 2),
            device="cpu",
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        assert len(result.macros) == 3
        assert np.isfinite(result.spread)
        assert np.isfinite(result.std)
        assert result.spread >= 0.0

    def test_one_seed_is_refused_because_one_run_has_no_spread(self, bundle):
        with pytest.raises(ValueError, match="at least two seeds"):
            measure_noise_floor(_cfg(), bundle, seeds=(0,), device="cpu", verbose=False)

    def test_a_repeated_seed_is_refused(self, bundle):
        with pytest.raises(ValueError, match="seeds repeat"):
            measure_noise_floor(_cfg(), bundle, seeds=(0, 0), device="cpu", verbose=False)

    def test_the_report_says_a_delta_below_the_spread_is_the_same_number(self, bundle):
        result = measure_noise_floor(
            _cfg(optim={"epochs": 1}),
            bundle,
            seeds=(0, 1),
            device="cpu",
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        text = result.report()
        assert "spread" in text
        assert "0.006" in text
        assert "not a measurement from this repository" in text

    def test_a_delta_under_the_measured_spread_is_not_evidence(self, bundle):
        result = measure_noise_floor(
            _cfg(optim={"epochs": 1}),
            bundle,
            seeds=(0, 1),
            device="cpu",
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        floor = result.spread if np.isfinite(result.spread) else 0.0020
        assert not result.is_readable(floor * 0.5)
        assert result.is_readable(floor * 2.0 + 0.01)

    def test_the_mode_writes_no_checkpoints_by_default(self, bundle, tmp_path):
        out = tmp_path / "floor"
        measure_noise_floor(
            _cfg(optim={"epochs": 1}),
            bundle,
            seeds=(0, 1),
            device="cpu",
            out_dir=out,
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        assert not list(out.rglob("*.pt"))
        assert not list(out.rglob("*.parquet"))

    def test_the_noise_floor_flag_is_a_real_flag_on_the_entrypoint(self):
        # A supported mode, not a manual exercise: the claim is checked against the
        # script's own argument parser rather than against a docstring.
        text = (repo_root() / "scripts" / "train.py").read_text(encoding="utf-8")
        assert '"--noise-floor"' in text
        assert '"--noise-floor-seeds"' in text


class TestTrackingIsWiredToTheOneConvention:
    """The tracking path must actually execute, and it must not be a second convention.

    ``track=True`` is the branch nothing else in this file touches, so a rename inside
    :mod:`kappa.tracking` breaks it silently: the whole point of that module is that a
    run which cannot be read a week later is not evidence, and a NameError on the
    logging path only shows up on the cluster, hours in.
    """

    def test_a_tracked_run_completes_with_tracking_disabled(
        self, bundle, tmp_path, monkeypatch
    ):
        # KAPPA_NO_TRACKING makes init_run return a NoOpRun, so this exercises every
        # call site on the tracking path without needing wandb or a network.
        monkeypatch.setenv("KAPPA_NO_TRACKING", "1")
        result = run_experiment(
            _cfg(optim={"epochs": 1}),
            bundle,
            folds=(0,),
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            track=True,
            verbose=False,
        )
        assert np.isfinite(result.summary.macro_mean)

    def test_a_tracked_multi_fold_run_completes(self, bundle, tmp_path, monkeypatch):
        monkeypatch.setenv("KAPPA_NO_TRACKING", "1")
        result = run_experiment(
            _cfg(optim={"epochs": 1}),
            bundle,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            track=True,
            verbose=False,
        )
        assert len(result.folds) == N_FOLDS

    def test_a_tracked_noise_floor_completes(self, bundle, tmp_path, monkeypatch):
        monkeypatch.setenv("KAPPA_NO_TRACKING", "1")
        result = measure_noise_floor(
            _cfg(optim={"epochs": 1}),
            bundle,
            seeds=(0, 1),
            folds=(0,),
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            track=True,
            verbose=False,
        )
        assert len(result.macros) == 2

    def test_the_loop_names_no_metric_keys_of_its_own(self):
        # Every metric name comes from kappa.tracking. A call site that spells its own
        # is how two folds end up under two namespaces and stop being comparable.
        from kappa.training import loop

        assert loop.label_metrics is not None
        assert loop.epoch_metrics is not None
        source = Path(loop.__file__).read_text(encoding="utf-8")
        assert "macro_mean\":" not in source
        assert '"auc/' not in source

    def test_the_loop_uses_the_shared_run_starter_not_its_own_init(self):
        from kappa.training import loop

        source = Path(loop.__file__).read_text(encoding="utf-8")
        assert "start_training_run(" in source
        # init_run is the low-level door; going through it here would bypass the name,
        # group and config-entity convention that makes a seed sweep readable.
        assert "init_run(" not in source

    def test_every_tracking_name_the_loop_imports_exists(self):
        # Cheap insurance against exactly the failure this class was added for: a
        # helper renamed in kappa.tracking and left dangling in an import list.
        import ast

        from kappa import tracking
        from kappa.training import loop

        tree = ast.parse(Path(loop.__file__).read_text(encoding="utf-8"))
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "kappa.tracking"
            for alias in node.names
        ]
        assert imported
        missing = [name for name in imported if not hasattr(tracking, name)]
        assert not missing, f"kappa.tracking has no {missing}"

    def test_a_run_without_a_directory_raises_before_any_fold_trains(
        self, bundle, tmp_path, monkeypatch
    ):
        """A tracked run with nowhere to write must fail before it trains.

        ``--track`` used to be able to fail silently into a :class:`NoOpRun` --
        ``WANDB_API_KEY`` unset, wandb not installed, ``wandb.init`` raising -- and
        the job trained for hours, reported a macro, and left nothing on disk to
        sync. ``require_run_directory`` must now fire BEFORE the fold loop spends
        any GPU time, not merely exist as a function nothing calls.
        """
        from kappa.tracking import NoOpRun
        from kappa.training import loop as loop_module

        monkeypatch.setattr(loop_module, "start_training_run", lambda *a, **k: NoOpRun())

        def _must_not_train(*_a, **_k):
            raise AssertionError("train_fold ran despite a missing run directory")

        monkeypatch.setattr(loop_module, "train_fold", _must_not_train)
        with pytest.raises(RuntimeError, match="--track"):
            run_experiment(
                _cfg(optim={"epochs": 1}),
                bundle,
                folds=(0,),
                device="cpu",
                out_dir=tmp_path,
                normalisation=Normalisation.identity(),
                track=True,
                verbose=False,
            )

    def test_kappa_no_tracking_still_overrides_track_without_raising(
        self, bundle, tmp_path, monkeypatch
    ):
        """KAPPA_NO_TRACKING=1 is a deliberate, different intent from --track failing.

        kappa.tracking's own docstring names it "what tests should do" to force a
        NoOpRun regardless of --track, so it must stay usable for a wandb-free smoke
        test and must not trip the same guard that catches a genuinely broken
        --track.
        """
        monkeypatch.setenv("KAPPA_NO_TRACKING", "1")
        result = run_experiment(
            _cfg(optim={"epochs": 1}),
            bundle,
            folds=(0,),
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            track=True,
            verbose=False,
        )
        assert np.isfinite(result.summary.macro_mean)

    def test_a_crash_during_training_still_commits_the_tracked_run(
        self, bundle, tmp_path, monkeypatch
    ):
        """A crash mid-fold must not leave the same 7-byte stub a walltime kill does.

        wandb's offline transaction log is block-buffered at 32 KiB with no
        time-based flush; only ``run.finish()`` commits it.
        ``install_walltime_guard`` only fires on SIGUSR1/SIGTERM -- a CUDA OOM or a
        data bug raises a plain Python exception instead, and without a
        try/except around the fold loop the process would unwind with the run
        never finished.
        """
        from kappa.training import loop as loop_module

        class FakeRun:
            def __init__(self) -> None:
                self.finished = False
                self.summary_calls: list[dict] = []

        run = FakeRun()

        def _boom(*_a, **_k):
            raise RuntimeError("synthetic CUDA OOM")

        monkeypatch.setattr(loop_module, "start_training_run", lambda *a, **k: run)
        monkeypatch.setattr(loop_module, "require_run_directory", lambda *a, **k: None)
        monkeypatch.setattr(loop_module, "install_walltime_guard", lambda *a, **k: None)
        monkeypatch.setattr(loop_module, "finish_run", lambda r, **k: setattr(r, "finished", True))
        monkeypatch.setattr(
            loop_module, "summarise", lambda r, metrics: r.summary_calls.append(dict(metrics))
        )
        monkeypatch.setattr(loop_module, "train_fold", _boom)

        with pytest.raises(RuntimeError, match="synthetic CUDA OOM"):
            run_experiment(
                _cfg(optim={"epochs": 1}),
                bundle,
                folds=(0,),
                device="cpu",
                out_dir=tmp_path,
                normalisation=Normalisation.identity(),
                track=True,
                verbose=False,
            )

        assert run.finished, "a crash must still commit the wandb run, not leave a stub"
        assert any(call.get("crashed") is True for call in run.summary_calls)


class TestAmpIsNotSilentlyOn:
    """A CPU run must not claim to have exercised the AMP path."""

    def test_amp_is_off_on_cpu_and_says_why(self, experiment):
        for fold in experiment.folds:
            assert fold.amp.startswith("off")
            assert "cuda" in fold.amp

    def test_amp_off_in_the_config_is_reported_as_a_config_choice(self, bundle, tmp_path):
        result = train_fold(
            _cfg(optim={"epochs": 1, "amp": False}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=False,
        )
        assert "disabled in config" in result.amp

    def test_the_inference_pass_also_reports_amp_off_on_cpu(self, experiment):
        assert experiment.folds[0].inference.amp == "off"

    def test_the_amp_label_is_carried_into_the_fold_report(self, experiment):
        assert "amp off" in experiment.folds[0].report()


class TestTheLossIsAsl:
    """Plain BCE and symmetric focal are both wrong here, so the choice is pinned."""

    def test_the_criterion_built_from_a_config_is_the_asymmetric_loss(self):
        criterion = build_criterion(_cfg())
        assert isinstance(criterion, AsymmetricLoss)
        assert not isinstance(criterion, MaskedBCE)

    def test_the_asymmetric_parameters_come_from_the_config(self):
        criterion = build_criterion(_cfg(loss={"gamma_neg": 2.0, "gamma_pos": 1.0, "clip": 0.1}))
        assert (criterion.gamma_neg, criterion.gamma_pos, criterion.clip) == (2.0, 1.0, 0.1)

    def test_the_loop_imports_the_loss_rather_than_defining_one(self):
        from kappa.training import loop

        assert loop.AsymmetricLoss is AsymmetricLoss
        source = Path(loop.__file__).read_text(encoding="utf-8")
        assert "class AsymmetricLoss" not in source
        assert "logsigmoid" not in source

    def test_the_loop_does_not_hand_roll_an_auc(self):
        from kappa.evaluation.metrics import macro_roc_auc as canonical
        from kappa.training import loop

        assert loop.macro_roc_auc is canonical
        source = Path(loop.__file__).read_text(encoding="utf-8")
        assert "roc_auc_score" not in source
        assert "nanmean" not in source

    def test_a_nan_label_would_be_refused_before_it_reached_the_loss(self):
        # The guarantee the loop depends on: 0.0 * nan is nan, so a zero mask does not
        # neutralise a NaN target and the check has to be unconditional.
        criterion = build_criterion(_cfg())
        logits = torch.zeros(2, N_LABELS)
        labels = torch.zeros(2, N_LABELS)
        labels[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            criterion(logits, labels, torch.zeros(2, N_LABELS, dtype=torch.bool))

    def test_a_fully_masked_batch_gives_a_finite_zero_loss(self):
        # It happens: a batch drawn from the unlabelled majority has every cell masked.
        criterion = build_criterion(_cfg())
        loss = criterion(
            torch.zeros(2, N_LABELS),
            torch.zeros(2, N_LABELS),
            torch.zeros(2, N_LABELS, dtype=torch.bool),
        )
        assert torch.isfinite(loss) and float(loss) == 0.0

    def test_the_training_loss_stayed_finite_on_every_epoch(self, experiment):
        for fold in experiment.folds:
            for record in fold.epochs:
                assert np.isfinite(record.train_loss)


class TestOptimiserAndSchedule:
    """AdamW with decay off the norms, and a schedule stepped per optimiser step."""

    def test_the_optimiser_is_adamw(self):
        model = build_model(_cfg())
        assert isinstance(build_optimiser(_cfg(), model), torch.optim.AdamW)

    def test_biases_and_norms_are_excluded_from_weight_decay(self):
        # Four groups since the backbone/head learning-rate split: the decay rule is
        # applied inside each half, so the pattern is decay/no-decay twice over.
        model = build_model(_cfg(optim={"weight_decay": 0.05}))
        groups = build_optimiser(_cfg(optim={"weight_decay": 0.05}), model).param_groups
        assert [group["name"] for group in groups] == list(_PARAM_GROUP_NAMES)
        assert [group["weight_decay"] for group in groups] == [0.05, 0.0, 0.05, 0.0]
        for group in groups:
            if group["name"].endswith("no-decay"):
                assert all(p.ndim <= 1 for p in group["params"]), group["name"]
            else:
                assert all(p.ndim > 1 for p in group["params"]), group["name"]

    def test_weight_decay_membership_is_the_rank_rule_and_nothing_else(self):
        # The invariant that makes moving the backbone boundary safe. `_BACKBONE_PREFIX`
        # was tightened from `encoder.` to `encoder.backbone.` so the freshly
        # initialised FiLM tables stop being treated as transferred weights, which
        # moves them from backbone-decay to head-decay -- both decay groups, so no
        # parameter's weight decay may change. Asserted as a property of the whole
        # partition rather than as a before/after comparison, so it also holds for the
        # next time that boundary moves.
        cfg = _cfg(optim={"weight_decay": 0.05})
        model = build_model(cfg)
        groups = build_optimiser(cfg, model).param_groups
        decayed = {id(p) for g in groups if g["weight_decay"] > 0.0 for p in g["params"]}
        by_rank = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not (parameter.ndim <= 1 or name.endswith(".bias"))
        }
        assert decayed == by_rank

    def test_group_zero_is_the_head_because_group_zero_is_what_the_loop_logs(self):
        # `run.log(lr=...)` and `EpochRecord.lr_last` both read param_groups[0]. If
        # group 0 were the backbone, the logged series would silently change scale the
        # first time a run set backbone_lr_mult, and an old uniform run could no longer
        # be overlaid on a new differential one.
        cfg = _cfg(optim={"lr": 1e-3, "backbone_lr_mult": 0.2})
        groups = build_optimiser(cfg, build_model(cfg)).param_groups
        assert groups[0]["name"] == "head-decay"
        assert groups[0]["lr"] == pytest.approx(1e-3)

    def test_every_trainable_parameter_lands_in_exactly_one_group(self):
        # Exhaustive AND disjoint, by identity rather than by count: a parameter listed
        # in two groups would be stepped twice per optimiser step, and one listed in
        # none would never be trained at all. Counting alone cannot tell either from a
        # correct partition once there are four groups to mis-route between.
        cfg = _cfg()
        model = build_model(cfg)
        groups = build_optimiser(cfg, model).param_groups
        assigned = [id(p) for group in groups for p in group["params"]]
        assert len(assigned) == len(set(assigned))
        assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
        assert all(group["params"] for group in groups), "an empty group here is a bug"

    def test_onecycle_is_used_when_the_schedule_is_long_enough(self):
        cfg = _cfg(optim={"schedule": "onecycle"})
        scheduler = build_scheduler(cfg, build_optimiser(cfg, build_model(cfg)), 100)
        assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_a_four_step_onecycle_falls_back_to_cosine_with_a_warning(self):
        # OneCycleLR divides by the width of its warmup phase and raises
        # ZeroDivisionError on a schedule this short, which is exactly what a
        # two-epoch smoke run produces. A wiring run must not die there.
        cfg = _cfg(optim={"schedule": "onecycle"})
        with pytest.warns(UserWarning, match="no room for a warmup"):
            scheduler = build_scheduler(cfg, build_optimiser(cfg, build_model(cfg)), 2)
        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_a_long_onecycle_schedule_trains_end_to_end(self, bundle, tmp_path):
        # The baseline config uses onecycle, so the branch that a short smoke run
        # cannot reach has to be exercised somewhere. batch_size=1 over 6 train
        # studies x 2 epochs is 12 optimiser steps, which is enough for a warmup.
        result = train_fold(
            _cfg(optim={"schedule": "onecycle", "batch_size": 1, "epochs": 2}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=False,
        )
        assert len(result.epochs) == 2
        assert all(np.isfinite(record.train_loss) for record in result.epochs)
        # The learning rate moved, which is what distinguishes a live schedule from a
        # constant one that happens to run.
        assert result.epochs[0].lr_last != result.epochs[1].lr_last

    def test_cosine_with_warmup_rises_then_falls(self):
        cfg = _cfg(optim={"schedule": "cosine", "warmup_pct": 0.1, "lr": 1e-3})
        optimiser = build_optimiser(cfg, build_model(cfg))
        scheduler = build_scheduler(cfg, optimiser, 100)
        rates = []
        with warnings.catch_warnings():
            # Stepping the scheduler without a real optimiser step warns about the
            # order; irrelevant here, where the schedule's shape is the whole claim.
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(100):
                rates.append(optimiser.param_groups[0]["lr"])
                scheduler.step()
        assert rates[0] < rates[10]  # warmup
        assert rates[10] > rates[99]  # decay
        assert rates[99] < rates[0] + 1e-9

    def test_the_warmup_never_starts_at_exactly_zero(self):
        # A zero learning rate on step 0 is a wasted batch, and with gradient
        # accumulation a wasted accumulation group.
        cfg = _cfg(optim={"schedule": "cosine", "warmup_pct": 0.1})
        optimiser = build_optimiser(cfg, build_model(cfg))
        build_scheduler(cfg, optimiser, 100)
        assert optimiser.param_groups[0]["lr"] > 0.0

    def test_a_zero_step_schedule_is_refused(self):
        cfg = _cfg()
        with pytest.raises(ValueError, match="total_steps"):
            build_scheduler(cfg, build_optimiser(cfg, build_model(cfg)), 0)

    def test_the_schedule_length_counts_optimiser_steps_not_batches(self, bundle, tmp_path):
        # Changing grad_accum must not silently change the shape of the schedule.
        result = train_fold(
            _cfg(optim={"epochs": 1, "batch_size": 2, "grad_accum": 2}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=False,
        )
        record = result.epochs[0]
        assert record.n_optimiser_steps < record.n_batches

    def test_gradient_accumulation_still_runs_at_least_one_step(self, bundle, tmp_path):
        result = train_fold(
            _cfg(optim={"epochs": 1, "batch_size": 4, "grad_accum": 64}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=False,
        )
        assert result.epochs[0].n_optimiser_steps >= 1


def _as_v2(folds: pd.DataFrame, *, held: int = 1) -> pd.DataFrame:
    """The bundle's v1 fold table re-shaped as folds_v2, holding out ``held`` studies.

    Hand-built rather than cut with ``build_folds_v2``: the point here is what
    :meth:`DataBundle.split` does with a row marked held out, so the marking has to
    be exact and independent of how a real holdout would have chosen it.
    """
    v2 = pd.DataFrame(
        {
            ID_COLUMN: folds[ID_COLUMN].astype(str),
            "split": "dev",
            "fold": folds["fold"].astype(int),
            "fold_scanner": folds["fold_scanner"].astype(int),
            "report_group": folds["patient_group"].astype(int),
            "scanner_group": folds["scanner_group"].astype(int),
            "language": folds["language"].astype(str),
        },
        columns=list(FOLD_COLUMNS_V2),
    )
    for row in range(held):
        v2.loc[row, "split"] = "test"
        v2.loc[row, "fold"] = TEST_FOLD
        v2.loc[row, "fold_scanner"] = TEST_FOLD
    return v2


class TestTheBackboneCanTrainSlowerThanTheHead:
    """``optim.backbone_lr_mult``, asserted AFTER the scheduler has stepped.

    That last part is the whole point of this class. ``OneCycleLR`` accepts ``max_lr``
    as a scalar or as one value per param group, and given a scalar it broadcasts that
    number over EVERY group -- so a backbone/head split built correctly in the
    optimiser is erased at the first ``.step()`` and the run trains uniformly while its
    config, its hash and its log all say it did not. Every assertion below is taken
    after stepping; taken on the freshly built optimiser they all pass with the bug in
    place, which is why a construction-time test is not enough.
    """

    @staticmethod
    def _stepped_rates(cfg, *, total_steps: int = 100, steps: int = 5):
        """The per-group rates after each of ``steps`` scheduler steps, by group name."""
        optimiser = build_optimiser(cfg, build_model(cfg))
        scheduler = build_scheduler(cfg, optimiser, total_steps)
        history = []
        with warnings.catch_warnings():
            # Stepping the scheduler without a real optimiser step warns about the
            # order; irrelevant here, where the rates themselves are the whole claim.
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(steps):
                scheduler.step()
                history.append({group["name"]: group["lr"] for group in optimiser.param_groups})
        return history

    @pytest.mark.parametrize("schedule", ["onecycle", "cosine"])
    def test_the_default_multiplier_leaves_every_group_on_one_rate(self, schedule):
        # The default has to be numerically the single-rate optimiser this repo ran
        # before the split existed, at every point of the schedule and not just at the
        # peak -- otherwise every config already on disk quietly became a new
        # experiment in arithmetic as well as in hash.
        cfg = _cfg(optim={"schedule": schedule})
        assert cfg.optim.backbone_lr_mult == 1.0
        for rates in self._stepped_rates(cfg):
            assert len(set(rates.values())) == 1, rates

    @pytest.mark.parametrize("schedule", ["onecycle", "cosine"])
    def test_a_slower_backbone_holds_its_ratio_through_the_schedule(self, schedule):
        cfg = _cfg(optim={"schedule": schedule, "lr": 1e-3, "backbone_lr_mult": 0.2})
        for rates in self._stepped_rates(cfg):
            head = rates["head-decay"]
            assert head > 0.0
            assert rates["head-no-decay"] == pytest.approx(head)
            assert rates["backbone-decay"] == pytest.approx(head * 0.2, rel=1e-9)
            assert rates["backbone-no-decay"] == pytest.approx(head * 0.2, rel=1e-9)

    @pytest.mark.parametrize("schedule", ["onecycle", "cosine"])
    def test_the_fresh_film_tables_run_at_the_head_rate_not_the_trunks(self, schedule):
        # `plane_film` and `contrast_film` live on the encoder but are randomly
        # initialised -- `nn.init.normal_` in SliceEncoder, `torch.randn * std` in
        # VolumeEncoder -- so they are head-like in every way that matters here. Under
        # the naive `encoder.` prefix they were filed with the trunk and would have
        # trained at the slowed rate: a fresh table handed a pretrained table's
        # learning rate, which is this whole mechanism applied backwards. Read after a
        # scheduler step, because that is where a rate stops being a construction
        # detail and becomes the rate a parameter is actually updated with.
        cfg = _cfg(optim={"schedule": schedule, "lr": 1e-3, "backbone_lr_mult": 0.2})
        model = build_model(cfg)
        film = {id(p) for name, p in model.named_parameters() if "film" in name}
        assert len(film) == 2, "expected plane_film and contrast_film"
        optimiser = build_optimiser(cfg, model)
        scheduler = build_scheduler(cfg, optimiser, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scheduler.step()
        holders = {
            group["name"]
            for group in optimiser.param_groups
            if any(id(p) in film for p in group["params"])
        }
        assert holders == {"head-decay"}
        rates = {group["name"]: group["lr"] for group in optimiser.param_groups}
        assert rates["head-decay"] == pytest.approx(rates["backbone-decay"] / 0.2)
        # And the trunk itself did not follow them out: the split still exists.
        assert rates["backbone-decay"] < rates["head-decay"]

    def test_only_the_transferred_trunk_is_in_the_backbone_groups(self):
        # The prefix is asserted against the module it is meant to name, not against a
        # hand-copied list of parameter names that would rot the next time either
        # encoder gains a submodule.
        cfg = _cfg()
        model = build_model(cfg)
        groups = build_optimiser(cfg, model).param_groups
        in_backbone_groups = {
            id(p)
            for group in groups
            if group["name"].startswith("backbone")
            for p in group["params"]
        }
        assert in_backbone_groups == {
            id(p) for p in model.encoder.backbone.parameters() if p.requires_grad
        }

    def test_a_scalar_max_lr_would_have_flattened_the_split(self):
        # The failure mode itself, demonstrated against torch rather than described in
        # a comment: this is what `build_scheduler` would do if `max_lr` were ever
        # written back as `cfg.optim.lr`. If torch's broadcasting behaviour changes,
        # this test says so instead of the guard above quietly becoming unnecessary.
        cfg = _cfg(optim={"schedule": "onecycle", "lr": 1e-3, "backbone_lr_mult": 0.2})
        optimiser = build_optimiser(cfg, build_model(cfg))
        assert len({group["lr"] for group in optimiser.param_groups}) == 2
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=cfg.optim.lr,
            total_steps=100,
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scheduler.step()
        assert len({group["lr"] for group in optimiser.param_groups}) == 1

    def test_an_already_scheduled_optimiser_is_refused(self):
        # `build_scheduler` reads each group's current lr as that group's peak, which
        # is only true before anything has stepped. A resume path that restored a
        # mid-run optimiser would otherwise get a schedule peaking at whatever the old
        # one had annealed to -- quietly under-trained, and on a differential run a
        # quietly different ratio as well. `initial_lr` is the stamp every scheduler
        # leaves behind, so its presence is the exact signal.
        cfg = _cfg(optim={"schedule": "cosine"})
        optimiser = build_optimiser(cfg, build_model(cfg))
        build_scheduler(cfg, optimiser, 100)
        with pytest.raises(ValueError, match="already carry 'initial_lr'"):
            build_scheduler(cfg, optimiser, 100)

    def test_a_multiplier_that_matches_no_backbone_parameter_is_refused(self):
        # The prefix is `encoder.backbone.`, and `TeacherClassifier` nests its trunk one
        # level down at `image.encoder.backbone.`. Handed a model the prefix does not reach, a
        # multiplier applies to nothing: the run trains uniformly, reports a null
        # effect and the null gets read as evidence about differential learning rates.
        # Crash instead.
        class NoTrunkPrefix(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.image = torch.nn.Linear(4, 4)
                # FiLM-shaped, and deliberately NOT enough: a module under `encoder.`
                # that carries no `encoder.backbone.` is a conditioning table with no
                # trunk behind it, and a multiplier still applies to nothing.
                self.encoder = torch.nn.Module()
                self.encoder.plane_film = torch.nn.Parameter(torch.randn(2, 4))

        with pytest.raises(ValueError, match="would apply to nothing"):
            build_optimiser(_cfg(optim={"backbone_lr_mult": 0.5}), NoTrunkPrefix())
        # Legal at 1.0: there is no split to erase, so nothing is being misreported.
        groups = build_optimiser(_cfg(), NoTrunkPrefix()).param_groups
        assert not groups[2]["params"] and not groups[3]["params"]

    def test_a_differential_run_trains_end_to_end_and_says_so_in_its_banner(
        self, bundle, tmp_path, capsys
    ):
        # The wiring, not the arithmetic: a config with a multiplier has to survive
        # `train_fold` -- optimiser, scheduler, checkpoint-free run -- and the header
        # has to distinguish it from a uniform run, because the Slurm log is the only
        # place a reader finds out which of the two ran.
        cfg = _cfg(optim={"backbone_lr_mult": 0.1, "lr": 1e-3})
        result = train_fold(
            cfg,
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            verbose=True,
        )
        assert all(np.isfinite(record.train_loss) for record in result.epochs)
        # `lr_last` reads group 0, which is the HEAD, so the logged series stays on the
        # unscaled rate. Compared against the backbone's rate rather than against a
        # literal: the cosine tail reaches exactly zero on the last step of the
        # schedule, so only the ORDERING of the two rates is a stable claim here.
        assert result.epochs[0].lr_last > cfg.optim.lr * cfg.optim.backbone_lr_mult
        assert all(record.lr_last >= 0.0 for record in result.epochs)
        out = capsys.readouterr().out
        assert "lr=0.001 head / 0.0001 backbone" in out
        assert "backbone x0.1" in out


class TestTheHeldOutTestSetIsInNeitherSide:
    """``train = uids[assignment != fold]`` silently trains on the whole test set.

    folds_v2 marks its frozen held-out 15% with ``split == "test"`` and ``fold ==
    -1``. Composed with an inequality against one fold index, every fold's
    *training* set then contains all 661 held-out studies. Nothing raises, no shape
    changes, and the only symptom is a cross-validation number that is quietly too
    high -- contradicted by the private leaderboard in late October, when nothing
    can be changed. This is the most expensive bug in the fold layer, so it is
    asserted for every fold and for both fold columns.
    """

    def _bundle_with(self, bundle: DataBundle, folds: pd.DataFrame) -> DataBundle:
        return DataBundle(
            manifest=bundle.manifest,
            labels=bundle.labels,
            folds=folds,
            cache_dir=bundle.cache_dir,
        )

    def test_a_held_out_study_is_in_neither_the_train_nor_the_val_side(self, bundle):
        folds = _as_v2(bundle.folds, held=2)
        held = set(folds.loc[folds["split"] == "test", ID_COLUMN])
        broken = self._bundle_with(bundle, folds)
        for column in ("fold", "fold_scanner"):
            for fold in range(N_FOLDS):
                train, val = broken.split(fold, column=column)
                assert not held & set(train), f"{column} fold {fold} trains on the test set"
                assert not held & set(val), f"{column} fold {fold} validates on the test set"

    def test_the_two_sides_sum_to_the_dev_pool_and_not_to_the_corpus(self, bundle):
        folds = _as_v2(bundle.folds, held=2)
        broken = self._bundle_with(bundle, folds)
        for fold in range(N_FOLDS):
            train, val = broken.split(fold)
            assert len(train) + len(val) == N_STUDIES - 2

    def test_a_split_marked_test_is_excluded_even_with_a_real_fold_index(self, bundle):
        # The hand edit: someone sets split=test and leaves the fold index alone.
        folds = _as_v2(bundle.folds, held=0)
        folds.loc[0, "split"] = "test"
        held = str(folds.loc[0, ID_COLUMN])
        broken = self._bundle_with(bundle, folds)
        for fold in range(N_FOLDS):
            train, val = broken.split(fold)
            assert held not in train and held not in val

    def test_a_negative_fold_is_refused_rather_than_validated_on(self, bundle):
        # Otherwise `--fold -1` hands back the frozen test set as a validation
        # fold, scores it, and logs the number.
        broken = self._bundle_with(bundle, _as_v2(bundle.folds, held=2))
        for fold in (-1, -2):
            with pytest.raises(ValueError, match="negative"):
                broken.split(fold)

    def test_a_negative_fold_is_refused_on_a_v1_table_too(self, bundle):
        # v1 has no -1 rows, so the old code took this path to "fold -1 does not
        # appear". Refusing by name says why, and says it identically either way.
        with pytest.raises(ValueError, match="negative"):
            bundle.split(-1)

    def test_a_folds_v2_table_is_accepted_by_the_bundle(self, bundle):
        folds = _as_v2(bundle.folds, held=1)
        assert tuple(self._bundle_with(bundle, folds).folds.columns) == FOLD_COLUMNS_V2

    def test_a_folds_v1_table_is_still_accepted_by_the_bundle(self, bundle):
        # v1 is frozen, not deprecated: a bundle that refuses it makes every number
        # already taken against data/folds/folds_v1.csv unreproducible.
        assert tuple(bundle.folds.columns) == FOLD_COLUMNS
        assert sum(len(bundle.split(fold)[1]) for fold in range(N_FOLDS)) == N_STUDIES

    def test_a_fold_present_only_in_the_test_rows_does_not_count_as_present(self, bundle):
        # `fold not in set(assignment)` would accept a fold index that exists only
        # on held-out rows, and then validate on those rows alone.
        folds = _as_v2(bundle.folds, held=0)
        folds.loc[folds["fold"] == 2, "split"] = "test"
        folds.loc[folds["fold"] == 2, "fold"] = TEST_FOLD
        broken = self._bundle_with(bundle, folds)
        with pytest.raises(ValueError, match="does not appear"):
            broken.split(2)


class TestFoldSplits:
    """A leaked study inflates the macro by an amount that looks exactly like progress."""

    def test_the_split_is_disjoint(self, bundle):
        for fold in range(N_FOLDS):
            train, val = bundle.split(fold)
            assert not set(train) & set(val)

    def test_every_study_is_in_exactly_one_validation_fold(self, bundle):
        seen: list[str] = []
        for fold in range(N_FOLDS):
            seen.extend(bundle.split(fold)[1])
        assert len(seen) == len(set(seen)) == N_STUDIES

    def test_a_leaked_fold_table_is_refused_by_name(self, bundle):
        # Force the leak the assertion exists for: the same study on both sides.
        leaked = bundle.folds.copy()
        leaked.loc[:, "fold"] = 0
        leaked.loc[leaked.index[0], "fold"] = 1
        broken = DataBundle(
            manifest=bundle.manifest,
            labels=bundle.labels,
            folds=pd.concat([leaked, leaked.iloc[[0]].assign(fold=0)], ignore_index=True),
            cache_dir=bundle.cache_dir,
        )
        with pytest.raises(ValueError, match="both the train and validation"):
            broken.split(0)

    def test_a_fold_table_with_the_wrong_columns_is_refused(self, bundle):
        with pytest.raises(ValueError, match="contract requires"):
            DataBundle(
                manifest=bundle.manifest,
                labels=bundle.labels,
                folds=bundle.folds.drop(columns=["language"]),
                cache_dir=bundle.cache_dir,
            )

    def test_the_fold_table_carries_both_grouping_columns(self, bundle):
        assert tuple(bundle.folds.columns) == FOLD_COLUMNS
        assert "fold" in bundle.folds and "fold_scanner" in bundle.folds

    def test_an_unknown_fold_column_is_refused(self, bundle):
        with pytest.raises(KeyError, match="fold_patient"):
            bundle.split(0, column="fold_patient")

    def test_a_fold_that_does_not_exist_is_refused(self, bundle):
        with pytest.raises(ValueError, match="does not appear"):
            bundle.split(99)

    def test_a_duplicated_oof_study_is_refused_by_run_experiment(self, bundle, tmp_path):
        # An out-of-fold table with a study scored twice is not out of fold. Duplicated
        # inside one fold on purpose: a duplicate across two folds is caught earlier,
        # by the disjointness assertion in split().
        duplicated = pd.concat(
            [bundle.folds, bundle.folds[bundle.folds["fold"] == 0].iloc[[0]]],
            ignore_index=True,
        )
        broken = DataBundle(
            manifest=bundle.manifest,
            labels=bundle.labels,
            folds=duplicated,
            cache_dir=bundle.cache_dir,
        )
        with pytest.raises(ValueError, match="not out of fold"):
            run_experiment(
                _cfg(optim={"epochs": 1}),
                broken,
                folds=(0,),
                device="cpu",
                out_dir=tmp_path,
                normalisation=Normalisation.identity(),
                save_checkpoint=False,
                verbose=False,
            )

    def test_a_fold_table_that_puts_every_study_in_one_fold_is_refused(
        self, bundle, tmp_path
    ):
        # Surfaces as "num_samples should be a positive integer" out of the DataLoader
        # otherwise, which reads as a torch bug rather than a fold-table mistake.
        collapsed = bundle.folds.copy()
        collapsed["fold"] = 0
        broken = DataBundle(
            manifest=bundle.manifest,
            labels=bundle.labels,
            folds=collapsed,
            cache_dir=bundle.cache_dir,
        )
        with pytest.raises(ValueError, match="both sides need at least one"):
            train_fold(
                _cfg(optim={"epochs": 1}),
                broken,
                0,
                device="cpu",
                out_dir=tmp_path,
                normalisation=Normalisation.identity(),
                save_checkpoint=False,
                verbose=False,
            )

    def test_the_usable_cell_count_is_reported_per_fold(self, experiment):
        # The denominator of everything downstream: a table barely denser than the
        # 98.7% unknown rate is training on almost nothing and the loss curve is mute.
        for fold in experiment.folds:
            assert fold.usable_train_cells > 0
            assert fold.usable_val_cells > 0


class TestSlotsAndSeeding:
    def test_slot_names_map_to_global_slot_ids(self):
        assert slot_indices(("Axial_fs1",)) == (1,)
        assert slot_indices(("Axial_fs0", "Sagittal_fs1")) == (0, 5)

    def test_an_unknown_slot_name_is_refused(self):
        with pytest.raises(ValueError, match="unknown slot"):
            slot_indices(("Axial_FS1",))

    def test_the_selected_slot_keeps_its_global_id_not_a_local_one(self, experiment, bundle):
        # A slot embedding trained on job 1 has to stay valid in job 2, so widening
        # the run is a config change rather than a reindexing.
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        assert loaded.slots == (1,)

    def test_seeding_is_reproducible_for_the_model_initialisation(self):
        seed_everything(11)
        first = build_model(_cfg()).head.weight.detach().clone()
        seed_everything(11)
        second = build_model(_cfg()).head.weight.detach().clone()
        assert torch.equal(first, second)

    def test_two_seeds_initialise_differently(self):
        seed_everything(1)
        first = build_model(_cfg()).head.weight.detach().clone()
        seed_everything(2)
        second = build_model(_cfg()).head.weight.detach().clone()
        assert not torch.equal(first, second)


class TestCheckpointsCarryTheirPreprocessing:
    """Re-resolving mean/std at inference needs the hub and can silently disagree."""

    def test_the_checkpoint_stores_the_normalisation_it_trained_with(self, experiment):
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        assert loaded.normalisation == Normalisation.identity()

    def test_the_checkpoint_stores_the_physical_crop(self, experiment):
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        assert loaded.crop.fov_mm == pytest.approx(150.0)
        assert loaded.crop.out_px == 64
        assert loaded.crop.mm_per_pixel == pytest.approx(150.0 / 64.0)

    def test_the_checkpoint_names_the_backbone_the_hash_the_sha_and_the_fold(
        self, experiment
    ):
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        assert loaded.backbone == TINY
        assert loaded.meta["config_hash"] == experiment.config_hash
        assert loaded.meta["git_sha"] == experiment.git_sha
        assert loaded.meta["fold"] == 0
        assert loaded.meta["seed"] == 0

    def test_loading_never_asks_for_pretrained_weights(self):
        # A hub download inside a no-internet notebook fails after the architecture is
        # already built, so the traceback blames the backbone rather than the network.
        source = Path(sys.modules["kappa.inference.predict"].__file__).read_text()
        assert "pretrained=False" in source
        assert "pretrained=True" not in source

    def test_a_checkpoint_from_another_version_is_refused_by_name(self, experiment, tmp_path):
        payload = torch.load(experiment.folds[0].checkpoint, map_location="cpu", weights_only=True)
        payload["kappa_checkpoint_version"] = 99
        path = tmp_path / "old.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="normalisation"):
            load_checkpoint(path)

    def test_a_checkpoint_missing_its_normalisation_is_refused(self, experiment, tmp_path):
        payload = torch.load(experiment.folds[0].checkpoint, map_location="cpu", weights_only=True)
        del payload["normalisation"]
        path = tmp_path / "nonorm.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match="'normalisation'"):
            load_checkpoint(path)

    def test_the_reloaded_model_reproduces_the_training_run_predictions(
        self, experiment, bundle
    ):
        # The wiring claim that matters most for a submission: the weights on disk
        # score the same as the weights in memory did.
        fold = experiment.folds[0]
        loaded = load_checkpoint(fold.checkpoint, device="cpu")
        _, val_uids = bundle.split(fold.fold)
        dataset = loaded.dataset(bundle.manifest, bundle.cache_dir, study_uids=list(val_uids))
        again = predict_studies(loaded.model, dataset, device="cpu", batch_size=4)
        assert again.study_uids == fold.inference.study_uids
        np.testing.assert_allclose(again.scores, fold.inference.scores, atol=1e-5)

    def test_the_checkpoint_carries_no_optimiser_state(self, experiment):
        # This file is read by a notebook that only scores; optimiser moments would
        # double its size for a tensor nobody loads.
        payload = torch.load(experiment.folds[0].checkpoint, map_location="cpu", weights_only=True)
        assert "optimizer" not in payload and "optimiser" not in payload


class TestScoreTransforms:
    """Both write options are AUC-identical, and neither is calibration."""

    def test_rank_and_raw_logits_score_identically(self, experiment):
        oof = pd.read_parquet(experiment.oof_path)
        truth = oof[[f"{n}__true" for n in LABELS]].to_numpy(dtype=float)
        mask = oof[[f"{n}__mask" for n in LABELS]].to_numpy(dtype=bool)
        scores = oof[list(LABELS)].to_numpy(dtype=float)
        from kappa.evaluation.metrics import macro_roc_auc

        y_true = np.where(mask, truth, np.nan)
        assert macro_roc_auc(y_true, transform_scores(scores, "none")).macro == pytest.approx(
            macro_roc_auc(y_true, transform_scores(scores, "rank")).macro
        )

    def test_the_rank_transform_lands_in_the_unit_interval(self, experiment):
        scores = pd.read_parquet(experiment.oof_path)[list(LABELS)].to_numpy(dtype=float)
        ranked = transform_scores(scores, "rank")
        assert ranked.min() > 0.0 and ranked.max() < 1.0

    def test_raw_logits_are_left_untouched(self, experiment):
        scores = pd.read_parquet(experiment.oof_path)[list(LABELS)].to_numpy(dtype=float)
        np.testing.assert_allclose(transform_scores(scores, "none"), scores)

    def test_an_unknown_transform_is_refused(self):
        with pytest.raises(ValueError, match="unknown transform"):
            transform_scores(np.zeros((2, N_LABELS)), "sigmoid")  # type: ignore[arg-type]

    def test_members_in_a_different_order_are_refused_rather_than_misaligned(
        self, experiment, bundle, tmp_path
    ):
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        uids = sorted(str(u) for u in bundle.manifest["study_uid"].unique())
        a = predict_studies(
            loaded.model,
            loaded.dataset(bundle.manifest, bundle.cache_dir, study_uids=uids),
            device="cpu",
        )
        b = predict_studies(
            loaded.model,
            loaded.dataset(bundle.manifest, bundle.cache_dir, study_uids=uids[::-1]),
            device="cpu",
        )
        with pytest.raises(ValueError, match="disagree on their study order"):
            write_predictions(tmp_path / "s.csv", [a, b])

    def test_a_missing_expected_uid_is_refused(self, experiment, bundle, tmp_path):
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        result = predict_studies(
            loaded.model, loaded.dataset(bundle.manifest, bundle.cache_dir), device="cpu"
        )
        with pytest.raises(ValueError, match="were not scored"):
            write_predictions(
                tmp_path / "s.csv", [result], expected_uids=[*result.study_uids, "missing"]
            )

    def test_expected_uids_reorder_the_rows(self, experiment, bundle, tmp_path):
        loaded = load_checkpoint(experiment.folds[0].checkpoint, device="cpu")
        result = predict_studies(
            loaded.model, loaded.dataset(bundle.manifest, bundle.cache_dir), device="cpu"
        )
        wanted = list(result.study_uids)[::-1]
        path = write_predictions(tmp_path / "s.csv", [result], expected_uids=wanted)
        assert validate_submission(path, expected_uids=wanted).n_passed == 7


# ---------------------------------------------------------------------------
# The competition-legality boundary
# ---------------------------------------------------------------------------

#: Every module a Kaggle inference notebook loads. Read from the package rather than
#: listed by hand where possible; these are the ones the notebook imports directly.
INFERENCE_PATH = (
    "kappa/inference/__init__.py",
    "kappa/inference/predict.py",
    "kappa/inference/submission.py",
    "kappa/data/dataset.py",
    # dataset.py imports SLOT_ORDER from here, so manifest.py is on the path even
    # though no notebook imports it directly. Leaving it out is what let a
    # `kappa.labels.normalise` import reach inference unnoticed.
    "kappa/data/manifest.py",
    # predict.py lazily imports `rotate` from here to build a TTA view, so
    # augment.py is on the inference path whenever eval.tta is non-empty. It is
    # listed for the same reason manifest.py is: a module reachable from a
    # notebook that never faces these guards is where the next A1 breach lands.
    "kappa/data/augment.py",
    "kappa/models/__init__.py",
    "kappa/models/encoder.py",
    "kappa/models/aggregator.py",
    "kappa/models/study_classifier.py",
    "kappa/evaluation/metrics.py",
    "kappa/constants.py",
)

#: Path shapes that only exist on a machine this code will not run on.
_ABSOLUTE_PATH = re.compile(r"(/Users/|/Volumes/|/scratch/|/home/|/projects/|[A-Za-z]:\\\\)")


def _code_strings(path: Path) -> list[str]:
    """Every string literal in ``path`` that is not a docstring.

    Docstrings are excluded deliberately: a docstring may name a machine-specific
    path in order to explain why it must not appear in code, and a grep-based
    check would flag the explanation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc_nodes = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in doc_nodes
    ]


class TestTheInferenceBoundary:
    """Submissions are image-only. This is a competition rule, not a preference."""

    def test_importing_the_inference_package_loads_no_report_reading_module(self):
        # A clean interpreter, because the rest of this test session has already
        # imported half the project and sys.modules would be useless.
        code = (
            "import sys, json;"
            "import kappa.inference.predict as p;"
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
            f"the inference path loaded report-reading module(s): {done.stdout.strip()}. "
            "Submissions are image-only (constraint A1), so this is a notebook that "
            "cannot run rather than a slow one."
        )

    @pytest.mark.parametrize(
        "module", ["kappa.inference.submission", "kappa.inference", "kappa.evaluation"]
    )
    def test_the_torch_free_half_imports_with_torch_blocked(self, module):
        # pyproject.toml states that only kappa.models / kappa.training may import
        # torch at module scope, and kappa.evaluation states that scoring and
        # blending work with the `train` extra uninstalled. Both were false: a
        # package __init__ runs before any submodule, so eagerly importing
        # kappa.inference.predict made `import kappa.inference.submission` require
        # torch even though the submission writer needs only numpy.
        #
        # torch is installed in this environment, so absence has to be simulated --
        # a plain `import` here would pass for the wrong reason.
        code = (
            "import sys, importlib.abc\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('No module named torch (simulated)')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            f"import {module}\n"
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
            "print('OK')\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo_root() / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert done.returncode == 0, (
            f"{module} cannot be imported without torch:\n{done.stderr}"
        )
        assert done.stdout.strip() == "OK"

    def test_only_the_declared_modules_import_torch_at_module_scope(self):
        # pyproject.toml names the exact set. Stated as "kappa.models /
        # kappa.training only", it was false for three modules and nothing
        # noticed, because the rule lived in a comment and not in a test.
        allowed = {
            "kappa/data/augment.py",
            "kappa/data/dataset.py",
            "kappa/inference/predict.py",
        }
        found = set()
        for path in sorted((repo_root() / "src" / "kappa").rglob("*.py")):
            if "_recovered" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module scope only
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n == "torch" or n.startswith("torch.") for n in names):
                    found.add(path.relative_to(repo_root() / "src").as_posix())
                    break
        packages = {p for p in found if p.startswith(("kappa/models/", "kappa/training/"))}
        assert found - packages == allowed, (
            f"module-scope torch imports outside kappa.models/kappa.training changed: "
            f"{sorted(found - packages)} != {sorted(allowed)}. Update pyproject.toml's "
            f"boundary comment with it, or move the import inside a function."
        )

    def test_the_forbidden_list_names_the_modules_that_actually_read_reports(self):
        # A list that named nothing real would pass the test above vacuously.
        assert len(FORBIDDEN_MODULES) >= 8
        for name in FORBIDDEN_MODULES:
            assert name.startswith("kappa.labels.")
        for reader in ("kappa.labels.engine", "kappa.labels.discover", "kappa.labels.polarity"):
            assert reader in FORBIDDEN_MODULES

    @pytest.mark.parametrize("relative", INFERENCE_PATH)
    def test_no_module_on_the_inference_path_imports_a_report_reader(self, relative):
        # tree.body, not ast.walk: a module-level import runs when the notebook
        # imports kappa.inference, and that is the thing constraint A1 forbids. An
        # import inside a function body only runs if that function is called, and
        # test_importing_the_inference_package_loads_no_report_reading_module is
        # what proves none of them is.
        tree = ast.parse((repo_root() / "src" / relative).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        bad = [
            name for name in imported
            if name == FORBIDDEN_PREFIX or name.startswith(FORBIDDEN_PREFIX + ".")
        ]
        assert not bad, f"{relative} imports {bad} at module scope"

    def test_predict_does_not_import_kappa_labels_at_all(self):
        tree = ast.parse((repo_root() / "src/kappa/inference/predict.py").read_text())
        modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert not [name for name in modules if name.startswith("kappa.labels")]

    @pytest.mark.parametrize("relative", INFERENCE_PATH)
    def test_no_module_on_the_inference_path_holds_an_absolute_local_path(self, relative):
        # The notebook imports src/kappa/ and runs where none of this machine's paths
        # exist, so a hardcoded root is a run that fails on the graded machine only.
        strings = _code_strings(repo_root() / "src" / relative)
        offenders = [s for s in strings if _ABSOLUTE_PATH.search(s)]
        assert not offenders, f"{relative} holds absolute path literal(s) {offenders}"

    def test_the_absolute_path_check_would_catch_one(self, tmp_path):
        # The scanner has to be able to fail, or the parametrised test above is
        # asserting that a broken regex found nothing.
        planted = tmp_path / "planted.py"
        planted.write_text('"""A docstring naming /home/x is fine."""\nROOT = "/home/x"\n')
        assert [s for s in _code_strings(planted) if _ABSOLUTE_PATH.search(s)] == ["/home/x"]

    def test_the_predict_entrypoint_takes_every_path_as_an_argument(self):
        text = (repo_root() / "scripts" / "predict.py").read_text(encoding="utf-8")
        for flag in ("--checkpoint", "--manifest", "--cache-dir", "--out"):
            assert flag in text
        assert not _ABSOLUTE_PATH.search(
            "".join(_code_strings(repo_root() / "scripts" / "predict.py"))
        )

    def test_the_inference_path_never_reads_a_report_column(self):
        # Belt and braces on the import check: no module here may name the report
        # column or the file the reports live in.
        for relative in INFERENCE_PATH:
            strings = _code_strings(repo_root() / "src" / relative)
            for value in strings:
                assert value != "report"
                assert not value.endswith("train.csv")


@pytest.fixture(scope="module")
def sbatch() -> str:
    return (repo_root() / "slurm" / "train.sbatch").read_text(encoding="utf-8")


class TestTheSlurmScript:
    """The QOS is the binding constraint, and a job over the cap never starts."""

    def test_it_requests_exactly_one_gpu(self, sbatch):
        # The gpu QOS caps a job at 1 GPU and a job over the cap is PERMANENTLY
        # ineligible (Reason=QOSMaxGRESPerJob), not slow.
        assert "--gres=gpu:h200:1" in sbatch
        # Every mention of it, including the ones in the comments that explain the
        # cap, so a "raise it later" note cannot creep in beside the real directive.
        assert set(re.findall(r"--gres=gpu:\w+:(\d+)", sbatch)) == {"1"}

    def test_it_runs_on_the_eight_hour_gpu_partition(self, sbatch):
        assert "--partition=gpu\n" in sbatch
        assert "--time=8:00:00" in sbatch

    def test_folds_are_an_array_not_ranks_of_one_job(self, sbatch):
        assert "--array=" in sbatch
        assert "srun" not in sbatch
        assert "--ntasks=" not in sbatch

    def test_it_honours_kappa_train_venv_and_defaults_to_the_cu126_train_env(self, sbatch):
        # The GPU nodes' driver is too old for cu130 wheels, so the vLLM env cannot
        # train here at all.
        #
        # The variable read is KAPPA_TRAIN_VENV, not KAPPA_VENV. KAPPA_VENV defaults
        # to the cu130 vLLM env and sbatch propagates the submitting shell's
        # environment, so reading it would hand this job the wrong interpreter
        # whenever the shell had sourced env_common.sh.
        assert 'KAPPA_VENV="${KAPPA_TRAIN_VENV:-' in sbatch
        assert '"${KAPPA_VENV:-' not in sbatch
        assert "kappa-train" in sbatch

    def test_it_refuses_to_train_on_cpu_after_asking_for_a_gpu(self, sbatch):
        # The failure this guard exists for: cu130 against a 570 driver imports fine,
        # reports cuda unavailable, trains on CPU for eight hours and still prints a
        # macro.
        assert "torch.cuda.is_available()" in sbatch
        assert "SystemExit" in sbatch

    def test_it_sets_the_offline_environment_compute_nodes_need(self, sbatch):
        assert "HF_HUB_OFFLINE=1" in sbatch
        assert "WANDB_MODE=offline" in sbatch

    def test_no_node_is_excluded_or_pinned(self, sbatch):
        # Every H200 node in the partition runs the same driver, so excluding one
        # only shrinks the pool. Checked against the directive lines only, so prose
        # in the header that mentions the flags cannot satisfy or break it.
        directives = [line for line in sbatch.splitlines() if line.startswith("#SBATCH")]
        assert not [line for line in directives if "--exclude" in line or "--nodelist" in line]


class TestTheWalltimeGuardCommitsTheRun:
    """A run killed at the 8-hour walltime must not vanish.

    wandb block-buffers the offline transaction log at 32 KiB with no time-based
    flush, and only ``run.finish()`` commits it, so a walltime kill before the first
    flush leaves a stub file that still looks syncable. Under SIGUSR1 an unguarded
    run stays a 7-byte stub, while a guarded one commits its config, metrics and
    the signal markers.
    """

    def _fake_run(self):
        class FakeRun:
            def __init__(self):
                self.finished = False
                self.summary = {}
                self.dir = "/tmp/does-not-matter/files"

            def finish(self):
                self.finished = True

        return FakeRun()

    def test_a_trapped_signal_finishes_the_run(self, monkeypatch):
        import signal as signal_module

        from kappa import tracking

        monkeypatch.setattr(tracking, "_GUARDED_RUNS", [])
        monkeypatch.setattr(tracking, "summarise", lambda run, metrics: None)
        installed: dict[int, object] = {}
        monkeypatch.setattr(
            tracking.signal, "signal", lambda number, handler: installed.setdefault(number, handler)
        )
        monkeypatch.setattr(tracking.os, "kill", lambda pid, number: None)

        run = self._fake_run()
        tracking.install_walltime_guard(run)
        assert signal_module.SIGUSR1 in installed, "SIGUSR1 was never trapped"
        assert signal_module.SIGTERM in installed, "SIGTERM (scancel) was never trapped"

        tracking._commit_and_die(int(signal_module.SIGUSR1), None)
        assert run.finished, "the signal did not commit the run, so the run is lost"

    def test_the_guard_ignores_a_disabled_run(self, monkeypatch):
        # KAPPA_NO_TRACKING=1 and a missing WANDB_API_KEY both yield a NoOpRun.
        # Trapping signals for one would change how the process dies for no gain.
        from kappa import tracking

        monkeypatch.setattr(tracking, "_GUARDED_RUNS", [])
        trapped: list[int] = []
        monkeypatch.setattr(
            tracking.signal, "signal", lambda number, handler: trapped.append(number)
        )
        tracking.install_walltime_guard(tracking.NoOpRun())
        tracking.install_walltime_guard(None)
        assert trapped == []
        assert tracking._GUARDED_RUNS == []

    def test_the_loop_arms_the_guard_on_the_run_it_creates(self):
        # A guard nobody installs is the same as no guard. This asserts the call
        # site exists rather than trusting that it was remembered.
        source = (repo_root() / "src/kappa/training/loop.py").read_text(encoding="utf-8")
        assert "install_walltime_guard(run)" in source

    def test_the_sbatch_asks_slurm_for_the_signal(self):
        # The handler can only fire if Slurm is asked to send SIGUSR1 before the
        # kill, and asked to send it to the batch script (B:) rather than the step.
        sbatch = (repo_root() / "slurm/train.sbatch").read_text(encoding="utf-8")
        assert "--signal=B:USR1@" in sbatch
        assert "kill -USR1" in sbatch, "the trap never forwards the signal to python"


class TestBf16IsRefusedOnCardsThatCannotDoIt:
    """The `gpu` partition is heterogeneous and only some of it does bf16.

    T4 (sm_75), V100 (sm_70), A100 (sm_80) and H200 (sm_90) all live in it, and
    bf16 needs sm_80 or newer. On a Tesla T4, asking for bf16 does not raise an
    unsupported-dtype error. It fails several frames deep inside ConvNeXt's
    depthwise convolution with "GET was unable to find an engine to execute this
    computation", which reads like a broken CUDA install rather than the wrong card.
    """

    def _cfg(self, dtype: str):
        from kappa.config import load_config

        return load_config(
            repo_root() / "configs/baseline_axial_fs1.yaml",
            overrides=[f"optim.amp_dtype={dtype}"],
        )

    def test_bf16_on_an_unsupporting_card_names_the_card(self, monkeypatch):
        import torch

        from kappa.training import loop as loop_module

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "Tesla T4")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))

        with pytest.raises(RuntimeError) as excinfo:
            loop_module._amp_setup(self._cfg("bf16"), torch.device("cuda"))
        message = str(excinfo.value)
        assert "Tesla T4" in message and "sm_75" in message
        assert "h200" in message, "the message must name the fix, not just the fault"

    def test_bf16_is_allowed_where_it_is_supported(self, monkeypatch):
        import torch

        from kappa.training import loop as loop_module

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
        monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **k: None)
        dtype, scaler, label = loop_module._amp_setup(self._cfg("bf16"), torch.device("cuda"))
        assert dtype is torch.bfloat16 and label == "cuda/bf16"
        assert scaler is None, "bf16 has fp32 exponent range and needs no scaler"

    def test_fp16_is_not_gated_by_the_bf16_check(self, monkeypatch):
        # fp16 runs on sm_70 upward, so the guard must not refuse it on a V100.
        import torch

        from kappa.training import loop as loop_module

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
        monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **k: "scaler")
        dtype, scaler, label = loop_module._amp_setup(self._cfg("fp16"), torch.device("cuda"))
        assert dtype is torch.float16 and label == "cuda/fp16"
        assert scaler == "scaler", "fp16 does need a grad scaler"

    def test_the_sbatch_pins_a_card_that_can_do_bf16(self):
        # The guard above is the safety net; pinning the card is the actual fix.
        sbatch = (repo_root() / "slurm/train.sbatch").read_text(encoding="utf-8")
        assert "--gres=gpu:h200:1" in sbatch


class TestTheDiagnosticsReachTheRun:
    """AP, precision, recall and F1 are logged per epoch -- and never select anything.

    The metrics themselves are tested in ``tests/test_metrics.py``; what is tested
    here is only that the training loop actually *emits* them. A metric that exists
    in a module and is never called is invisible in exactly the way a missing metric
    is, and the whole point of these is to make a weak rare label visible during a
    run rather than after the leaderboard.
    """

    class RecordingRun:
        """The one method ``train_fold`` calls on a run, plus the log it kept."""

        def __init__(self) -> None:
            self.logged: list[dict] = []

        def log(self, metrics: dict) -> None:
            self.logged.append(dict(metrics))

    def _train(self, bundle, tmp_path, *, epochs: int = 2):
        run = self.RecordingRun()
        result = train_fold(
            _cfg(optim={"epochs": epochs}),
            bundle,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            run=run,
            verbose=False,
        )
        return result, run

    def test_a_fold_carries_the_selected_epochs_diagnostics(self, experiment):
        diagnostics = experiment.folds[0].diagnostics
        assert set(diagnostics.ap) == set(LABELS), "every label gets a row, scoreable or not"
        assert diagnostics.threshold == pytest.approx(0.5)

    def test_the_diagnostics_read_exactly_the_cells_the_auc_reads(self, experiment):
        # Both scorers are handed the same masked truth by `_masked_truth`. If they
        # ever disagree about which cells exist, the two macros are over different
        # denominators and comparing them says nothing.
        fold = experiment.folds[0]
        assert fold.diagnostics.n_used == fold.auc.n_used
        assert fold.diagnostics.n_pos == fold.auc.n_pos

    def test_every_epoch_logs_per_label_ap_precision_recall_and_f1(self, bundle, tmp_path):
        _, run = self._train(bundle, tmp_path, epochs=2)
        epochs = [entry for entry in run.logged if "fold00/macro" in entry]
        assert len(epochs) == 2, "one metric row per epoch"
        for entry in epochs:
            for label in LABELS:
                # Verbatim label names, apostrophe and spaces included. Baker's is
                # the one that breaks anything that rebuilds names by splitting.
                assert f"fold00/ap/{label}" in entry
                assert f"fold00/precision/{label}" in entry
                assert f"fold00/recall/{label}" in entry
                assert f"fold00/f1/{label}" in entry
            for macro in ("ap_macro", "precision_macro", "recall_macro", "f1_macro"):
                assert f"fold00/{macro}" in entry

    def test_the_diagnostics_share_the_epoch_namespace_with_the_auc(self, bundle, tmp_path):
        # One row per epoch, not two. Split across two log() calls they land on
        # different steps and no chart puts AUC and AP on the same x-axis.
        _, run = self._train(bundle, tmp_path, epochs=1)
        together = [
            entry
            for entry in run.logged
            if "fold00/macro" in entry and "fold00/ap_macro" in entry
        ]
        assert len(together) == 1

    def test_the_best_f1_threshold_is_logged_and_labelled_as_fitted(self, bundle, tmp_path):
        # It is logged because it is informative, and it must never be carried
        # forward: a threshold chosen on a fold fits that fold.
        _, run = self._train(bundle, tmp_path, epochs=1)
        entry = next(entry for entry in run.logged if "fold00/macro" in entry)
        for label in LABELS:
            assert f"fold00/f1_best/{label}" in entry
            assert f"fold00/threshold_best/{label}" in entry

    def test_the_curve_panels_are_logged_once_per_fold_not_once_per_epoch(
        self, bundle, tmp_path
    ):
        pytest.importorskip("wandb")
        _, run = self._train(bundle, tmp_path, epochs=2)
        roc = [entry for entry in run.logged if "fold00/roc_curves" in entry]
        pr = [entry for entry in run.logged if "fold00/pr_curves" in entry]
        assert len(roc) == 1 and len(pr) == 1, "the panels are images; one per fold"

    def test_the_experiment_summary_carries_the_cv_level_diagnostics(
        self, bundle, tmp_path, monkeypatch
    ):
        # Into the run SUMMARY, not only the step history: the runs table sorts and
        # filters on the summary, so a number that exists only as the last point of a
        # series is invisible in the one view used to compare a hundred runs.
        from kappa.training import loop as loop_module

        captured: dict = {}
        run = self.RecordingRun()
        monkeypatch.setattr(loop_module, "start_training_run", lambda *a, **k: run)
        # This test's RecordingRun is a bare stand-in with no ``.dir``, and it is not
        # supposed to exercise require_run_directory's own contract -- that lives in
        # tests/test_tracking.py's TestATrackedRunMustProduceARunDirectory. Patched out
        # here for the same reason install_walltime_guard and finish_run are.
        monkeypatch.setattr(loop_module, "require_run_directory", lambda *a, **k: None)
        monkeypatch.setattr(loop_module, "install_walltime_guard", lambda *a, **k: None)
        monkeypatch.setattr(loop_module, "log_standard_artifacts", lambda *a, **k: None)
        monkeypatch.setattr(loop_module, "finish_run", lambda *a, **k: None)
        monkeypatch.setattr(
            loop_module, "summarise", lambda _run, metrics: captured.update(metrics)
        )
        run_experiment(
            _cfg(optim={"epochs": 1}),
            bundle,
            folds=[0],
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            save_checkpoint=False,
            write_oof=False,
            track=True,
            verbose=False,
        )
        assert "cv/macro" in captured, "the competition metric is still the headline"
        for macro in ("ap_macro", "precision_macro", "recall_macro", "f1_macro"):
            assert f"cv/{macro}" in captured
        for label in LABELS:
            assert f"cv/ap/{label}" in captured

    def test_the_fold_report_marks_the_diagnostics_as_never_a_selection_signal(
        self, experiment
    ):
        text = experiment.folds[0].report()
        assert "DIAGNOSTIC ONLY" in text
        assert "AP macro" in text

    def test_selection_still_reads_the_auc_and_not_the_f1(self):
        # The guard is structural: assert the selection expression names `macro` and
        # mentions no diagnostic quantity, so a later edit cannot quietly swap them.
        source = (repo_root() / "src/kappa/training/loop.py").read_text(encoding="utf-8")
        selection = source[source.index("if best is None or (") :][:220]
        assert "auc.macro" in selection
        for banned in ("f1", "ap_macro", "precision", "recall", "threshold"):
            assert banned not in selection, f"{banned} must never choose a checkpoint"


class TestAugmentationIsTrainOnly:
    """The val set must never be augmented, for the same reason it is never dropped.

    A validation pass that saw CutMix would be scored against labels CutMix had
    already mixed, and the resulting AUC would be a number about a pipeline that
    does not exist. Nothing about it would look wrong.
    """

    @staticmethod
    def _spy(monkeypatch):
        """Capture the ``augment`` and ``training`` kwargs of every dataset built.

        The patched name is ``build_study_dataset``, which is what the loop calls.
        It used to be ``StudyDataset``, and when the teacher work moved dataset
        construction behind ``kappa.training.report_dataset.build_study_dataset``
        this spy stopped intercepting anything at all. One test then failed loudly
        (an empty capture cannot hold a train/val pair) but its sibling asserted
        ``all(... is None for ...)`` over the empty list and passed **vacuously** --
        it would have gone on passing with the validation set fully augmented. That
        is why every test below asserts the capture is non-empty first.
        """
        import kappa.training.loop as loop_module

        built: list[dict] = []
        real = loop_module.build_study_dataset

        def recording(*args, **kwargs):
            built.append(
                {"training": kwargs.get("training"), "augment": kwargs.get("augment")}
            )
            return real(*args, **kwargs)

        monkeypatch.setattr(loop_module, "build_study_dataset", recording)
        return built

    def test_only_the_training_set_receives_an_augment(self, bundle, tmp_path, monkeypatch):
        built = self._spy(monkeypatch)
        train_fold(
            _cfg(augment={"rotate_deg": 10.0, "cutmix_prob": 1.0}, optim={"epochs": 1}),
            bundle,
            fold=0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        by_split = {record["training"]: record["augment"] for record in built}
        assert set(by_split) == {True, False}, "expected exactly one train and one val set"
        assert by_split[True] is not None, "the training set was not augmented"
        assert by_split[False] is None, "the validation set was handed an augment"

    def test_the_baseline_augments_neither_side(self, bundle, tmp_path, monkeypatch):
        built = self._spy(monkeypatch)
        train_fold(
            _cfg(optim={"epochs": 1}),
            bundle,
            fold=0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        assert built, "the dataset spy captured nothing, so this asserts over an empty list"
        assert all(record["augment"] is None for record in built)

    def test_the_validation_collate_is_the_unaugmented_one(self):
        # predict_studies builds its own loader with bare collate_studies, and
        # build_collate refuses to bind a batch hook when training is False. Both
        # paths are checked, because either one alone would let CutMix through.
        from kappa.data.augment import build_collate
        from kappa.data.dataset import collate_studies

        cfg = _cfg(augment={"cutmix_prob": 1.0})
        assert build_collate(cfg, seed=0, training=False) is collate_studies
        assert build_collate(cfg, seed=0, training=True) is not collate_studies


class TestWeightAveragingIsAnArmNotAReplacement:
    """EMA and last-N are scored beside the raw weights, never instead of them."""

    def _fold(self, bundle, tmp_path, **over):
        options = {"optim": {"epochs": 2, **over.pop("optim", {})}}
        options.update(over)
        return train_fold(
            _cfg(**options),
            bundle,
            fold=0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            verbose=False,
        )

    def test_the_baseline_reports_no_averaged_arm(self, bundle, tmp_path):
        result = self._fold(bundle, tmp_path)
        assert result.averaged is None
        assert result.averaging is None
        assert result.averaged_checkpoint is None

    def test_an_ema_run_reports_both_numbers(self, bundle, tmp_path):
        result = self._fold(bundle, tmp_path, optim={"ema_decay": 0.9})
        assert result.averaging == "ema"
        assert result.averaged is not None
        # The raw number must survive: it is what the arm is compared against.
        assert np.isfinite(result.auc.macro)
        assert np.isfinite(result.averaged.macro)

    def test_a_last_n_run_reports_both_numbers(self, bundle, tmp_path):
        result = self._fold(bundle, tmp_path, optim={"swa_last_n": 2})
        assert result.averaging == "swa"
        assert result.averaged is not None

    def test_the_averaged_weights_go_to_their_own_file(self, bundle, tmp_path):
        # Overwriting the raw checkpoint would make the arm unfalsifiable afterwards.
        result = self._fold(bundle, tmp_path, optim={"ema_decay": 0.9})
        assert result.checkpoint is not None and result.checkpoint.is_file()
        assert result.averaged_checkpoint is not None
        assert result.averaged_checkpoint.is_file()
        assert result.averaged_checkpoint != result.checkpoint

    def test_averaging_does_not_disturb_the_raw_weights(self, bundle, tmp_path):
        # The swap is reversible, so the checkpointed raw weights must be the ones
        # the trajectory actually ended on -- identical to an unaveraged run.
        plain = self._fold(bundle, tmp_path / "plain")
        averaged = self._fold(bundle, tmp_path / "ema", optim={"ema_decay": 0.9})
        assert averaged.auc.macro == pytest.approx(plain.auc.macro, abs=1e-9)
        assert averaged.selected_epoch == plain.selected_epoch

    def test_the_diagnostics_field_is_still_populated(self, bundle, tmp_path):
        # Called out in the handoff: train_fold must keep carrying it.
        assert self._fold(bundle, tmp_path, optim={"ema_decay": 0.9}).diagnostics is not None


class TestEvalTtaReachesTheValidationPass:
    """A config field that changes the hash but not the run is a config that lies.

    ``eval.tta`` reaches ``config_hash`` and the startup banner. If the loop did
    not pass it to ``predict_studies``, setting it would produce a differently
    named, differently hashed run that behaved identically to the baseline -- and
    the CV would be quoted for a pipeline nobody ships.
    """

    @staticmethod
    def _spy(monkeypatch):
        import kappa.training.loop as loop_module

        seen: list[tuple[str, ...]] = []
        real = loop_module.predict_studies

        def recording(*args, **kwargs):
            seen.append(tuple(kwargs.get("tta", ())))
            return real(*args, **kwargs)

        monkeypatch.setattr(loop_module, "predict_studies", recording)
        return seen

    def _fold(self, bundle, tmp_path, **over):
        optim = {"epochs": 1, **over.pop("optim", {})}
        return train_fold(
            _cfg(optim=optim, **over),
            bundle,
            fold=0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            verbose=False,
        )

    def test_the_configured_transforms_reach_every_scoring_pass(
        self, bundle, tmp_path, monkeypatch
    ):
        seen = self._spy(monkeypatch)
        self._fold(bundle, tmp_path, eval={"tta": ("identity", "rotate_cw")})
        assert seen, "predict_studies was never called"
        assert all(passed == ("identity", "rotate_cw") for passed in seen), seen

    def test_the_baseline_still_scores_a_single_plain_pass(
        self, bundle, tmp_path, monkeypatch
    ):
        seen = self._spy(monkeypatch)
        self._fold(bundle, tmp_path)
        assert all(passed == () for passed in seen), seen

    def test_the_averaged_arm_is_scored_the_same_way_as_the_raw_one(
        self, bundle, tmp_path, monkeypatch
    ):
        # Two passes scored under different TTA would not be comparable, which is
        # the one thing the averaged arm exists to be.
        seen = self._spy(monkeypatch)
        self._fold(
            bundle, tmp_path, optim={"ema_decay": 0.9}, eval={"tta": ("identity",)}
        )
        assert len(set(seen)) == 1, seen


class TestTheCheckpointDescribesTheReadoutItWasTrainedWith:
    """A shape that is not in the file is a shape the notebook cannot rebuild.

    `load_checkpoint` builds the architecture from `meta` and then loads weights into
    it, so an unrecorded readout is not a cosmetic omission: `strict=True` fails on
    `study_pool.label_slot_logits`, and `strict=False` succeeds and scores the
    submission with an untrained table. Checkpoints written before these four keys
    existed carry none of them, which is why every default here is the value the
    architecture had before the key existed.
    """

    def _write(self, tmp_path, model, name="ckpt.pt"):
        return save_inference_checkpoint(
            tmp_path / name,
            model,
            normalisation=Normalisation.identity(),
            crop=CropSpec(fov_mm=150.0, out_px=64),
            slots=(1,),
            n_slices=2,
            meta={
                "backbone": TINY,
                "drop_path_rate": 0.0,
                "head": model.head_kind,
                "head_num_heads": 4,
                "per_label_slots": model.per_label_slots,
                "head_residual": model.head_residual,
            },
        )

    def _model(self, **over):
        kwargs = dict(backbone=TINY, n_slices=2, pretrained=False, drop_path_rate=0.0)
        kwargs.update(over)
        return StudyClassifier(**kwargs)

    @pytest.mark.parametrize(
        "flags",
        [
            {},
            {"per_label_slots": True},
            {"head": "query", "head_num_heads": 4},
            {"head": "query", "head_num_heads": 4, "head_residual": True},
            {"head": "query", "head_num_heads": 4, "per_label_slots": True},
        ],
    )
    def test_every_readout_round_trips_strictly(self, tmp_path, flags):
        model = self._model(**flags).eval()
        loaded = load_checkpoint(self._write(tmp_path, model), device="cpu")
        assert set(loaded.model.state_dict()) == set(model.state_dict())
        assert loaded.model.per_label_slots == model.per_label_slots
        assert loaded.model.head_residual == model.head_residual
        assert loaded.model.head_kind == model.head_kind

    def test_the_reloaded_per_label_model_scores_identically(self, tmp_path):
        # The claim that matters for a submission: the file on disk is the model that
        # was validated, not an architecture that merely accepted its weights.
        model = self._model(per_label_slots=True).eval()
        loaded = load_checkpoint(self._write(tmp_path, model), device="cpu")
        pixels = torch.randn(2, 6, 2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
        present = torch.ones(2, 6, dtype=torch.bool)
        slot_ids = torch.arange(6).expand(2, 6).contiguous()
        with torch.no_grad():
            assert torch.equal(
                model(pixels, present, slot_ids).logits,
                loaded.model(pixels, present, slot_ids).logits,
            )

    def test_a_checkpoint_predating_the_pool_section_still_loads(self, tmp_path):
        # Older checkpoints. Their `meta` has no `head`, no `per_label_slots`
        # and no `head_residual`, and every one of them was trained with the shared
        # linear readout the defaults rebuild. Written here as a checkpoint whose meta
        # is stripped back to exactly what that generation wrote.
        model = self._model().eval()
        path = self._write(tmp_path, model, name="old.pt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["meta"] = {"backbone": TINY, "drop_path_rate": 0.0, "aux_dim": 0}
        torch.save(payload, path)
        loaded = load_checkpoint(path, device="cpu")
        assert loaded.model.head_kind == "linear"
        assert loaded.model.per_label_slots is False
        assert loaded.model.head_residual is False
        assert set(loaded.model.state_dict()) == set(model.state_dict())

    def test_the_training_loop_writes_all_four_keys(self, experiment):
        # The other half: a default run must record the readout too, or the next
        # generation of checkpoints inherits exactly the gap this test closes.
        meta = load_checkpoint(experiment.folds[0].checkpoint, device="cpu").meta
        assert meta["head"] == "linear"
        assert meta["per_label_slots"] is False
        assert meta["head_residual"] is False
        assert meta["head_num_heads"] == 8


class TestGradCheckpointingIsAMemoryKnobAndNotAnExperiment:
    """``--grad-checkpoint`` buys activation memory and must cost nothing else.

    It exists because a 24-slice config ran an H200 out of memory in every fold,
    inside the ConvNeXt shortcut add, within about a minute of starting. The batch is
    ``(B, 6, n_slices, 3, H, W)``, so at ``n_slices=24`` one step pushes
    ``8*6*24 = 1152`` images at 384 px through the backbone and activations rather
    than weights exhaust the card.

    The design claim these tests pin is that it is a runtime knob and not a config
    field. ``_HASH_EXCLUDED`` is only name/out_dir/seed/seeds, so a ``TrainConfig``
    field would have moved every hash in the repository and renamed the very run it
    was added to rescue, turning a fix into a new experiment.
    """

    @staticmethod
    def _spy(monkeypatch):
        """Record every ``set_grad_checkpointing`` call made on the built model.

        Patched at ``build_model``, which is the loop's own constructor call, so the
        spy sees the real model rather than a stand-in and cannot pass vacuously
        against a refactor that stops building one.
        """
        import kappa.training.loop as loop_module

        calls: list[bool] = []
        real = loop_module.build_model

        def recording(cfg):
            model = real(cfg)
            original = model.set_grad_checkpointing

            def capture(enable: bool = True):
                calls.append(bool(enable))
                return original(enable)

            model.set_grad_checkpointing = capture
            return model

        monkeypatch.setattr(loop_module, "build_model", recording)
        return calls

    def test_the_flag_turns_backbone_checkpointing_on(self, bundle, tmp_path, monkeypatch):
        calls = self._spy(monkeypatch)
        train_fold(
            _cfg(optim={"epochs": 1}),
            bundle,
            0,
            out_dir=tmp_path,
            save_checkpoint=False,
            normalisation=Normalisation.identity(),
            grad_checkpoint=True,
        )
        assert calls == [True], f"expected one enabling call, got {calls}"

    def test_it_is_off_unless_asked_for(self, bundle, tmp_path, monkeypatch):
        # The default is the behaviour every completed run trained under. If it
        # flipped, those runs would stop being comparable to new ones, and the
        # extra recompute per fold could push a run past its walltime.
        calls = self._spy(monkeypatch)
        train_fold(
            _cfg(optim={"epochs": 1}),
            bundle,
            0,
            out_dir=tmp_path,
            save_checkpoint=False,
            normalisation=Normalisation.identity(),
        )
        assert calls == [], f"checkpointing was enabled without being asked: {calls}"

    def test_it_moves_no_config_hash(self):
        # The whole reason it is an argument: a config rerun with the flag must
        # keep its hash, or the rescued run becomes a different experiment.
        cfg = _cfg()
        assert config_hash(cfg) == config_hash(_cfg())
        # And it is genuinely absent from the config surface, not merely defaulted.
        assert not hasattr(cfg, "grad_checkpoint")
        assert "grad_checkpoint" not in cfg.model_dump(mode="json")

    def test_the_cli_exposes_it_as_a_store_true(self):
        # Batch scripts forward their arguments verbatim to scripts/train.py, so
        # they can only reach this as a bare flag.
        source = (repo_root() / "scripts" / "train.py").read_text(encoding="utf-8")
        assert '"--grad-checkpoint"' in source
        assert 'ap.add_argument("--grad-checkpoint", action="store_true"' in source
        assert "grad_checkpoint=args.grad_checkpoint" in source
