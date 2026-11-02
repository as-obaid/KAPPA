"""The resume path, and the five claims ``kappa/training/resume.py`` cites this file for.

**Why this file exists:** ``resume.py`` was written complete, self-consistent
and with a docstring that cites ``tests/test_resume.py`` five times as proof of a
round-trip, a load order, a sidecar contract and an averager coupling -- and the file
did not exist, and the module had zero call sites. A resume module nothing calls and
nothing tests is not a resume path; it is a plan for one, and every one of its bugs is
still in it, waiting for a cluster node that costs GPU-hours and has no internet.

Four of the classes here are each about one failure a resume can produce while looking
entirely healthy:

* :class:`TestTheRoundTrip` -- a field that quietly does not survive. An EMA restarted
  empty, an optimiser without its per-group rates, a ``best`` tracker that came back as
  ``None``: each of those trains on and reports a number for an arm it did not run.
* :class:`TestARefusalIsTheCorrectOutcome` -- ``--select`` is **not** in
  ``config_hash``, so two runs differing only in selection policy hash identically.
  Without the four-field identity check a resume would continue a ``last``-policy run
  under ``best`` and manufacture a fold whose epochs were selected under two
  different policies.
* :class:`TestTheLoadOrder` -- ``build_scheduler`` reads each param group's current
  ``lr`` as its intended peak. Install the optimiser state before the scheduler exists
  and the new schedule peaks at wherever the old one had annealed to: a silently
  under-trained run with a completely correct-looking config.
* :class:`TestAnUnresumedRunIsUnchanged` -- jobs already queued when this path was
  wired pick this code up when they start. Without ``resume=True`` nothing may be read,
  written, or even constructed.

:class:`TestCpuParity` is the one that matters and the one whose limits are worth
reading: it trains a real fold over a synthetic corpus on CPU, kills it at an epoch
boundary the way a walltime kill does, resumes it, and compares the finished halves. It
proves parity **on this machine, on CPU, in one process**, which is a strictly weaker
claim than parity on a GPU over the real cache -- the one-fold check for that is
written out in ``resume.py``'s module docstring and has to be run by hand. Read that
class's own docstring before quoting this test as parity.

Everything runs on CPU with the tiny ``test_convnext`` backbone and a generated corpus
whose labels mean nothing, because the point is the state and not the score.
"""

from __future__ import annotations

import dataclasses
import inspect
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from kappa.constants import ID_COLUMN, N_LABELS  # noqa: E402
from kappa.data.dataset import Normalisation  # noqa: E402
from kappa.inference.predict import PredictionResult  # noqa: E402
from kappa.training import loop as loop_module  # noqa: E402
from kappa.training.averaging import ExponentialMovingAverage, LastNAverage  # noqa: E402
from kappa.training.loop import (  # noqa: E402
    _PARAM_GROUP_NAMES,
    EpochRecord,
    build_model,
    build_optimiser,
    build_scheduler,
    train_fold,
)
from kappa.training.resume import (  # noqa: E402
    RESUME_VERSION,
    ResumeIdentity,
    ResumeMismatch,
    capture_rng,
    checkpoint_of,
    clear_resume,
    install_averager,
    load_resume,
    read_sidecar,
    restore_rng,
    save_resume,
    sidecar_of,
)

# The corpus and the config come from test_loop rather than being re-derived here: a
# second, subtly different synthetic fixture is how two test files end up disagreeing
# about what "the loop" is.
from test_loop import _bundle, _cfg  # noqa: E402

#: The keys :func:`sidecar_of`'s docstring promises an external scheduler will find.
#: Pinned as a set rather than checked one at a time so that an ADDED key fails here
#: too -- a scheduler that greps line-anchored patterns can ignore a key it does not
#: expect, but a key it expects and cannot find costs a whole job.
SIDECAR_KEYS = frozenset(
    {
        "kappa_resume_version",
        "config_hash",
        "git_sha",
        "select",
        "epochs",
        "next_epoch",
        "seed",
        "fold",
        "name",
        "stamp",
        "checkpoint",
        "wandb_run_id",
    }
)

#: The parity budget. Three epochs with the kill after the first is the smallest shape
#: that leaves MORE THAN ONE epoch on the far side of the resume -- with two epochs and
#: a kill after one, a loader generator that came back wrong would still be asked for
#: exactly one permutation and could agree by luck.
PARITY_EPOCHS = 3
PARITY_KILL_AFTER = 1


class _Walltime(RuntimeError):
    """Stands in for the Slurm walltime SIGKILL, which cannot be raised for real."""


def _identity(**over) -> ResumeIdentity:
    fields = {
        "config_hash": "fc8ee481925e",
        "git_sha": "1178950",
        "select": "best",
        "epochs": 8,
        "seed": 0,
        "fold": 3,
        "name": "wiring",
    }
    fields.update(over)
    return ResumeIdentity(**fields)


def _prediction(n: int = 5, *, offset: float = 0.0) -> PredictionResult:
    """A ``PredictionResult`` whose every field is distinguishable from its neighbours."""
    scores = np.arange(n * N_LABELS, dtype=np.float32).reshape(n, N_LABELS) / 100.0 + offset
    return PredictionResult(
        study_uids=tuple(f"study-{index}" for index in range(n)),
        scores=scores,
        seconds=12.5 + offset,
        n_batches=3,
        device="cpu",
        amp="off (cpu is not a cuda device)",
        tta=("plain", "hflip"),
    )


def _record(epoch: int, *, skipped: tuple[str, ...] = ()) -> EpochRecord:
    return EpochRecord(
        epoch=epoch,
        train_loss=0.5 - 0.01 * epoch,
        n_batches=7,
        n_optimiser_steps=7,
        lr_last=1e-3 / (epoch + 1),
        macro=0.70 + 0.01 * epoch,
        n_scored=N_LABELS - len(skipped),
        skipped=skipped,
        train_seconds=83.0,
        val_seconds=17.0,
        step_seconds=61.0,
    )


def _trained_triple(cfg=None, *, steps: int = 3):
    """A model, optimiser and scheduler that have actually moved.

    Saving a freshly built triple would round-trip trivially: AdamW's ``exp_avg`` does
    not exist until the first ``step()`` and the scheduler's ``last_epoch`` is still at
    its initial value, so a save/load that dropped both would compare equal anyway.
    Three steps is what makes the comparison mean something.
    """
    cfg = cfg or _cfg()
    model = build_model(cfg)
    optimiser = build_optimiser(cfg, model)
    scheduler = build_scheduler(cfg, optimiser, max(steps * 2, 4))
    for _ in range(steps):
        loss = sum(param.square().sum() for param in model.parameters())
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        scheduler.step()
    return model, optimiser, scheduler


def _write(out, identity, **over):
    """:func:`save_resume` with the fields a given caller does not care about filled in."""
    model, optimiser, scheduler = over.pop("triple", None) or _trained_triple()
    payload = {
        "next_epoch": 1,
        "global_step": 7,
        "step_total": 61.0,
        "scaler": None,
        "loader_generator": torch.Generator().manual_seed(0),
        "epochs": [_record(0)],
        "best": None,
        "last": None,
        "averager": None,
    }
    payload.update(over)
    return save_resume(
        out, identity=identity, model=model, optimiser=optimiser, scheduler=scheduler, **payload
    )


# The expensive fixtures are module-scoped and live at module level rather than being
# class-scoped instance methods, which pytest deprecates -- and which tests/test_loop.py
# already avoids for the same reason.


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One populated save/load round trip. Shared, because building the model is the cost."""
    out = tmp_path_factory.mktemp("resume-round-trip")
    cfg = _cfg()
    model, optimiser, scheduler = _trained_triple(cfg)
    averager = ExponentialMovingAverage(model, decay=0.99)
    averager.after_step(model)
    averager.after_step(model)
    generator = torch.Generator().manual_seed(1234)
    torch.randperm(9, generator=generator)  # advance it off its freshly seeded state
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    identity = _identity()
    path = save_resume(
        out,
        identity=identity,
        next_epoch=5,
        global_step=41,
        step_total=613.5,
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        scaler=None,
        loader_generator=generator,
        epochs=[_record(0), _record(1, skipped=("MCL",))],
        best=(0.8445, 1, _prediction(offset=1.0)),
        last=(1, _prediction(offset=2.0)),
        averager=averager,
        wandb_run_id="abc123",
    )
    return {
        "out": out,
        "cfg": cfg,
        "path": path,
        "model": model,
        "optimiser": optimiser,
        "scheduler": scheduler,
        "averager": averager,
        "generator": generator,
        "identity": identity,
        "state": load_resume(out, expect=identity),
    }


@pytest.fixture(scope="module")
def refusals(tmp_path_factory):
    out = tmp_path_factory.mktemp("resume-refusals")
    _write(
        out, _identity(), next_epoch=4, global_step=28, step_total=400.0,
        last=(0, _prediction()),
    )
    return out


@pytest.fixture(scope="module")
def sidecar_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("resume-sidecar")
    _write(
        out, _identity(), next_epoch=6, global_step=42, step_total=500.0,
        wandb_run_id="run-xyz",
    )
    return out


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """One synthetic corpus for every test here that trains a real fold."""
    return _bundle(tmp_path_factory.mktemp("resume-corpus"))


@pytest.fixture(scope="module")
def parity_cfg():
    """Every stochastic path the loop has, switched ON.

    A resume that lost one of the streams has to diverge rather than accidentally
    agree, and with the ``_cfg`` defaults -- ``drop_path_rate=0.0``, no augmentation, no
    series dropout -- most of them draw nothing at all and the parity claim would be a
    claim about a deterministic pipeline.
    """
    return _cfg(
        name="parity",
        optim={"epochs": PARITY_EPOCHS},
        model={"drop_path_rate": 0.1},
        data={"series_dropout": 0.25},
        # No `hflip`: TrainConfig REFUSES it, because flipping a knee horizontally
        # swaps medial and lateral and no relabelling makes a flipped study correct for
        # all twelve labels. Rotation is the larger measured effect anyway.
        augment={"rotate_deg": 8.0, "brightness": 0.2, "contrast": 0.2, "cutmix_prob": 0.5},
    )


@pytest.fixture(scope="module")
def halves(parity_cfg, corpus, tmp_path_factory):
    """``(reference, resumed, run_dir)`` -- the uninterrupted fold and the resumed one."""
    reference_dir = tmp_path_factory.mktemp("parity-ref")
    run_dir = tmp_path_factory.mktemp("parity-run")
    common = dict(
        device="cpu",
        fold_column="fold",
        # `select="best"` on BOTH halves. `select` is not in config_hash, so a policy
        # difference between them would be invisible in every artifact -- and it is the
        # path that exercises the serialised best-so-far tracker, since the selected
        # epoch can be one that was trained before the kill and deserialised after it.
        select="best",
        normalisation=Normalisation.identity(),
        verbose=False,
    )
    reference = train_fold(parity_cfg, corpus, 0, out_dir=reference_dir, resume=False, **common)

    # THE KILL. A walltime SIGKILL cannot be raised for real, so the stand-in is an
    # exception thrown immediately after the epoch-boundary save that a killed run would
    # have completed. That is exactly the state Slurm leaves behind: the file published,
    # the process gone, no artifact, and nothing in the log that looks wrong.
    real_save = loop_module.save_resume

    def _save_then_die(*args, **kwargs):
        path = real_save(*args, **kwargs)
        if int(kwargs["next_epoch"]) >= PARITY_KILL_AFTER:
            raise _Walltime(f"killed after publishing next_epoch={kwargs['next_epoch']}")
        return path

    loop_module.save_resume = _save_then_die
    try:
        with pytest.raises(_Walltime):
            train_fold(parity_cfg, corpus, 0, out_dir=run_dir, resume=True, **common)
    finally:
        loop_module.save_resume = real_save

    assert read_sidecar(run_dir, 0, 0)["next_epoch"] == PARITY_KILL_AFTER
    resumed = train_fold(parity_cfg, corpus, 0, out_dir=run_dir, resume=True, **common)
    return reference, resumed, run_dir


class TestTheRoundTrip:
    """Every field goes in and comes back. The ones that would not are the expensive ones."""

    def test_the_checkpoint_lands_on_the_documented_path(self, saved):
        assert saved["path"] == checkpoint_of(saved["out"], 3, 0)
        assert saved["path"].name == "resume_fold3_seed0.pt"

    def test_the_identity_comes_back_field_for_field(self, saved):
        assert saved["state"].identity == saved["identity"]

    def test_the_progress_counters_come_back(self, saved):
        state = saved["state"]
        assert (state.next_epoch, state.global_step, state.step_total) == (5, 41, 613.5)

    def test_every_model_tensor_comes_back_bit_exact(self, saved):
        source = saved["model"].state_dict()
        restored = saved["state"].model
        assert set(restored) == set(source)
        assert all(torch.equal(restored[k].cpu(), v.cpu()) for k, v in source.items())

    def test_the_optimiser_moments_come_back(self, saved):
        fresh = build_optimiser(saved["cfg"], build_model(saved["cfg"]))
        assert not fresh.state_dict()["state"], "a fresh AdamW has no moments to compare"
        fresh.load_state_dict(saved["state"].optimiser)
        source = saved["optimiser"].state_dict()["state"]
        landed = fresh.state_dict()["state"]
        assert set(landed) == set(source) and source
        for key, entry in source.items():
            assert int(landed[key]["step"]) == int(entry["step"])
            assert torch.equal(landed[key]["exp_avg"], entry["exp_avg"])
            assert torch.equal(landed[key]["exp_avg_sq"], entry["exp_avg_sq"])

    def test_the_four_param_groups_keep_their_order_and_their_rates(self, saved):
        # The ORDER is load-bearing: group 0 is head-decay and the loop logs
        # `param_groups[0]["lr"]` as *the* learning rate of the run. Without the
        # per-group rates the `backbone_lr_mult` ratio dies silently and the run becomes
        # a uniform-LR run wearing a differential-LR config.
        groups = saved["state"].optimiser["param_groups"]
        assert tuple(group["name"] for group in groups) == _PARAM_GROUP_NAMES
        source = saved["optimiser"].param_groups
        assert [group["lr"] for group in groups] == [group["lr"] for group in source]
        assert [group["weight_decay"] for group in groups] == [
            group["weight_decay"] for group in source
        ]

    def test_the_scheduler_comes_back_where_it_had_annealed_to(self, saved):
        source = saved["scheduler"].state_dict()
        restored = saved["state"].scheduler
        assert restored["last_epoch"] == source["last_epoch"] > 0
        assert restored["base_lrs"] == source["base_lrs"]
        assert restored["_last_lr"] == source["_last_lr"]

    def test_the_loader_generator_draws_the_same_next_permutation(self, saved):
        # The claim is not that the bytes match, it is that the NEXT DRAW matches --
        # that is what reproduces the permutation the uninterrupted run would have used
        # for the epoch after the one that was saved.
        wanted = torch.randperm(9, generator=saved["generator"])
        fresh = torch.Generator()
        fresh.set_state(saved["state"].loader_generator)
        assert torch.equal(torch.randperm(9, generator=fresh), wanted)

    def test_the_epoch_history_comes_back_whole(self, saved):
        # Whole, because `result.report()` is built from it: a resumed fold whose
        # history came back short describes fewer epochs than it trained, and nothing in
        # the log says so.
        rows = saved["state"].epochs
        assert len(rows) == 2
        rebuilt = [EpochRecord(**{**row, "skipped": tuple(row["skipped"])}) for row in rows]
        assert rebuilt == [_record(0), _record(1, skipped=("MCL",))]
        assert rebuilt[1].skipped == ("MCL",)

    def test_the_epoch_history_survives_a_new_epochrecord_field(self, saved):
        # `save_resume` uses `dataclasses.asdict` rather than a hand-written field list,
        # so a field added to EpochRecord -- it gained `step_seconds` once already --
        # cannot go quietly missing. This pins the two halves together.
        assert set(saved["state"].epochs[0]) == {
            field.name for field in dataclasses.fields(EpochRecord)
        }

    def test_the_best_tracker_comes_back_with_its_predictions(self, saved):
        # SERIALISED, not recomputed: recomputing means a validation pass per candidate
        # epoch at ~17 s of steady GPU time each, to avoid writing about 42 KB.
        macro, epoch, prediction = saved["state"].best
        assert (round(macro, 4), epoch) == (0.8445, 1)
        wanted = _prediction(offset=1.0)
        assert prediction.study_uids == wanted.study_uids
        assert np.array_equal(prediction.scores, wanted.scores)
        assert prediction.scores.dtype == np.float32
        assert (prediction.seconds, prediction.n_batches) == (wanted.seconds, wanted.n_batches)
        assert (prediction.device, prediction.amp, prediction.tta) == (
            wanted.device,
            wanted.amp,
            wanted.tta,
        )

    def test_the_last_epochs_predictions_come_back_too(self, saved):
        epoch, prediction = saved["state"].last
        assert epoch == 1
        assert np.array_equal(prediction.scores, _prediction(offset=2.0).scores)

    def test_the_ema_comes_back_with_its_step_count_and_its_shadow(self, saved):
        # An EMA restarted empty is an expensive copy of the mid-run weights, which is
        # not the quantity the averaging arm is asking about.
        fresh = ExponentialMovingAverage(build_model(saved["cfg"]), decay=0.99)
        assert fresh.steps == 0
        install_averager(fresh, saved["state"].averager)
        assert fresh.steps == saved["averager"].steps == 2
        source, landed = saved["averager"].state_dict(), fresh.state_dict()
        assert set(landed) == set(source) and source
        assert all(torch.equal(landed[k], source[k]) for k in source)

    def test_the_wandb_run_id_comes_back(self, saved):
        # Carried so that a resumed fold can continue ONE run rather than appearing
        # twice in the runs table as what looks like an extra fold.
        assert saved["state"].wandb_run_id == "abc123"

    def test_a_bf16_run_writes_no_scaler_key_and_that_is_not_an_error(self, saved):
        # Every config in this repository sets amp_dtype: bf16 and `_amp_setup` returns
        # scaler=None for bf16, so this is the path every real run takes.
        assert saved["state"].scaler is None
        raw = torch.load(saved["path"], map_location="cpu", weights_only=True)
        assert "scaler" not in raw

    def test_an_fp16_scaler_is_written_and_comes_back(self, tmp_path):
        scaler = torch.amp.GradScaler("cpu", init_scale=2.0**12, enabled=True)
        identity = _identity(fold=0)
        _write(tmp_path, identity, scaler=scaler)
        state = load_resume(tmp_path, expect=identity)
        assert state.scaler is not None
        assert state.scaler["scale"] == scaler.state_dict()["scale"] == 2.0**12
        assert state.best is None and state.last is None and state.averager is None

    def test_a_swa_averager_round_trips_its_snapshots(self, tmp_path):
        cfg = _cfg()
        triple = _trained_triple(cfg)
        averager = LastNAverage(triple[0], n=2)
        averager.after_epoch(triple[0])
        averager.after_epoch(triple[0])
        identity = _identity(fold=1)
        _write(tmp_path, identity, triple=triple, averager=averager, next_epoch=2, epochs=[])
        state = load_resume(tmp_path, expect=identity)
        fresh = LastNAverage(build_model(cfg), n=2)
        assert not fresh.ready()
        install_averager(fresh, state.averager)
        assert fresh.observed == averager.observed == 2
        source, landed = averager.state_dict(), fresh.state_dict()
        assert all(torch.equal(landed[k], source[k]) for k in source)

    def test_an_averager_of_the_wrong_kind_is_refused_rather_than_half_installed(self, saved):
        # The config that chooses the averager (`optim.ema_decay`/`swa_last_n`) is
        # inside config_hash, so `load_resume` should already have refused this. It is
        # checked anyway because the failure it would otherwise produce is an averaging
        # arm reporting a number for weights that were never averaged.
        swa = LastNAverage(build_model(saved["cfg"]), n=2)
        with pytest.raises(ResumeMismatch, match="'ema'"):
            install_averager(swa, saved["state"].averager)

    def test_an_averager_present_on_one_side_only_is_refused(self, saved):
        with pytest.raises(ResumeMismatch, match="averager"):
            install_averager(None, saved["state"].averager)
        with pytest.raises(ResumeMismatch, match="averager"):
            install_averager(ExponentialMovingAverage(saved["model"], decay=0.99), None)

    def test_clear_resume_takes_both_files_and_is_idempotent(self, tmp_path):
        _write(tmp_path, _identity(fold=2))
        assert checkpoint_of(tmp_path, 2, 0).exists()
        assert sidecar_of(tmp_path, 2, 0).exists()
        clear_resume(tmp_path, 2, 0)
        assert not checkpoint_of(tmp_path, 2, 0).exists()
        assert not sidecar_of(tmp_path, 2, 0).exists()
        # Idempotent: a driver pass or a second call must not trip over a clean
        # directory, which is the state every finished fold leaves.
        clear_resume(tmp_path, 2, 0)

    def test_no_tmp_file_is_left_behind(self, saved):
        # The `.tmp` + `os.replace` publish means the visible path is either the
        # previous complete checkpoint or the new one. A leftover `.tmp` beside a
        # ~700 MB file is both debris and a sign the replace did not happen.
        assert not list(Path(saved["out"]).glob("*.tmp"))


class TestARefusalIsTheCorrectOutcome:
    """Each of the four CHECKED fields refuses on its own, and so does a version bump."""

    def test_the_matching_identity_loads(self, refusals):
        assert load_resume(refusals, expect=_identity()).next_epoch == 4

    @pytest.mark.parametrize(
        "field, value",
        [
            # A real config_hash pair: v1_axial_fs1_convnext hashes fc8ee481925e at
            # 8 epochs and e296f5db8835 at 10.
            ("config_hash", "e296f5db8835"),
            ("git_sha", "da14c80"),
            ("select", "last"),
            ("epochs", 10),
        ],
    )
    def test_each_checked_field_refuses_on_its_own(self, refusals, field, value):
        with pytest.raises(ResumeMismatch) as excinfo:
            load_resume(refusals, expect=_identity(**{field: value}))
        # The message has to name WHICH field and BOTH values, because the operator
        # decision that follows -- restart the fold, or fix the submit line -- depends
        # on which one moved.
        message = str(excinfo.value)
        assert field in message and str(value) in message
        assert "REFUSING" in message

    def test_the_checked_set_is_exactly_the_four_fields_that_can_differ(self):
        assert ResumeIdentity.CHECKED == ("config_hash", "git_sha", "select", "epochs")

    def test_a_disagreement_names_the_policy_hazard_it_is_preventing(self, refusals):
        with pytest.raises(ResumeMismatch, match="policy-mixed fold"):
            load_resume(refusals, expect=_identity(select="last"))

    @pytest.mark.parametrize("field, value", [("seed", 1), ("fold", 4)])
    def test_seed_and_fold_are_not_checked_because_the_filename_encodes_them(
        self, refusals, field, value
    ):
        # Checking them would test the filesystem rather than the experiment: the path
        # IS `resume_fold<k>_seed<s>.pt`, so a disagreement cannot reach the identity
        # comparison -- it looks for a file that is not there.
        with pytest.raises(FileNotFoundError):
            load_resume(refusals, expect=_identity(**{field: value}))

    def test_name_is_recorded_not_checked_and_nothing_here_catches_a_collision(self, refusals):
        # `name` is not in config_hash but IS the output path, so two configs sharing a
        # name share this directory and therefore this checkpoint. Nothing in resume.py
        # can detect that; the one-name-per-config test is what prevents it, and this
        # asserts the limitation rather than pretending it is covered.
        found = load_resume(refusals, expect=_identity(name="a-different-cell")).identity
        assert found.name == "wiring"

    def test_a_version_this_build_does_not_read_is_refused_not_partially_read(
        self, refusals, monkeypatch
    ):
        # Refused rather than best-effort: a partially understood payload is how an
        # averager or an RNG stream goes quietly missing from a resumed run.
        monkeypatch.setattr("kappa.training.resume.RESUME_VERSION", RESUME_VERSION + 1)
        with pytest.raises(ResumeMismatch, match="resume version"):
            load_resume(refusals, expect=_identity())

    def test_a_payload_that_is_not_a_mapping_is_refused(self, tmp_path):
        torch.save(torch.zeros(3), checkpoint_of(tmp_path, 3, 0))
        with pytest.raises(ResumeMismatch, match="does not hold a KAPPA resume mapping"):
            load_resume(tmp_path, expect=_identity())


class TestTheSidecar:
    """A few hundred bytes the DRIVER reads, so nothing torch.loads 700 MB to schedule a job."""

    def test_it_lands_on_the_documented_path(self, sidecar_dir):
        assert sidecar_of(sidecar_dir, 3, 0).name == "resume_fold3_seed0.json"

    def test_its_keys_are_exactly_what_is_documented(self, sidecar_dir):
        assert set(read_sidecar(sidecar_dir, 3, 0)) == SIDECAR_KEYS

    def test_it_carries_the_fields_a_scheduler_needs_without_opening_the_big_file(
        self, sidecar_dir
    ):
        payload = read_sidecar(sidecar_dir, 3, 0)
        assert payload["next_epoch"] == 6
        assert payload["epochs"] == 8
        assert payload["select"] == "best"
        assert payload["config_hash"] == "fc8ee481925e"
        assert payload["git_sha"] == "1178950"
        assert payload["kappa_resume_version"] == RESUME_VERSION
        assert payload["wandb_run_id"] == "run-xyz"
        assert payload["checkpoint"] == "resume_fold3_seed0.pt"

    def test_it_stays_small_enough_that_reading_it_is_never_the_wrong_tool(self, sidecar_dir):
        # The whole point of the file. Reading a checkpoint of hundreds of megabytes to
        # decide what to run is the wrong tool, and on a memory-capped login node it is
        # also how a process gets OOM-killed with no output at all.
        assert sidecar_of(sidecar_dir, 3, 0).stat().st_size < 1024

    def test_every_key_sits_alone_on_its_own_line_because_sed_reads_it(self, sidecar_dir):
        # This is a contract with shell schedulers that extract next_epoch, epochs and
        # stamp with a line-anchored `sed`. A compact one-line dump would still be valid
        # JSON and would silently defeat that extraction.
        lines = sidecar_of(sidecar_dir, 3, 0).read_text(encoding="utf-8").splitlines()
        keyed = [line for line in lines if '":' in line]
        assert len(keyed) == len(SIDECAR_KEYS)
        names = [line.strip().split('"')[1] for line in keyed]
        assert names == sorted(names), "sort_keys=True is part of the contract"
        assert lines[0] == "{" and lines[-1] == "}"

    def test_the_sidecar_is_published_after_the_checkpoint(self, sidecar_dir):
        # ORDER IS LOAD-BEARING. The sidecar is what the driver reads as "a resumable
        # checkpoint exists here", so publishing it before the .pt is complete would let
        # a scheduling pass believe in a checkpoint a GPU job then cannot load. A .pt
        # with no sidecar reads as not started and costs an epoch; a sidecar with no
        # .pt costs a whole job.
        assert (
            sidecar_of(sidecar_dir, 3, 0).stat().st_mtime_ns
            >= checkpoint_of(sidecar_dir, 3, 0).stat().st_mtime_ns
        )

    def test_a_missing_sidecar_reads_as_none_rather_than_raising(self, tmp_path):
        assert read_sidecar(tmp_path, 0, 0) is None

    def test_a_torn_write_reads_as_none_rather_than_crashing_the_driver(self, tmp_path):
        # None rather than an exception because this is the CHEAP scheduling read, and a
        # scheduler that crashes on a half-written JSON is worse than one that treats
        # the fold as not-yet-resumable and starts it from scratch. The TRAINING path
        # has no such tolerance -- see load_resume.
        sidecar_of(tmp_path, 0, 0).write_text('{"next_epoch": 3, "conf', encoding="utf-8")
        assert read_sidecar(tmp_path, 0, 0) is None

    def test_a_directory_in_place_of_the_sidecar_reads_as_none(self, tmp_path):
        sidecar_of(tmp_path, 1, 0).mkdir(parents=True)
        assert read_sidecar(tmp_path, 1, 0) is None


class TestTheRngStreams:
    """Restore puts back every stream the loop can draw from, and the next draw proves it."""

    def test_the_torch_cpu_stream_draws_the_same_next_number(self):
        torch.manual_seed(11)
        state = capture_rng()
        wanted = torch.randn(4)
        torch.manual_seed(999)
        restore_rng(state)
        assert torch.equal(torch.randn(4), wanted)

    def test_the_python_stream_draws_the_same_next_number(self):
        random.seed(11)
        state = capture_rng()
        wanted = [random.random() for _ in range(4)]
        random.seed(999)
        restore_rng(state)
        assert [random.random() for _ in range(4)] == wanted

    def test_the_numpy_stream_draws_the_same_next_number(self):
        np.random.seed(11)
        state = capture_rng()
        wanted = np.random.rand(4)
        np.random.seed(999)
        restore_rng(state)
        assert np.array_equal(np.random.rand(4), wanted)

    def test_the_streams_survive_the_torch_save_round_trip(self, tmp_path):
        # The states are flattened to ints and strings rather than stored as the objects
        # the stdlib hands back, because `torch.load(weights_only=True)` refuses
        # arbitrary reconstructors -- and `np.random.get_state()` returns a tuple
        # containing an ndarray while `random.getstate()` returns nested tuples. This is
        # the test that the flattening is lossless.
        torch.manual_seed(3)
        random.seed(3)
        np.random.seed(3)
        state = capture_rng()
        wanted = (torch.randn(2), random.random(), np.random.rand(2))
        path = tmp_path / "rng.pt"
        torch.save(state, path)
        torch.manual_seed(4)
        random.seed(4)
        np.random.seed(4)
        restore_rng(torch.load(path, map_location="cpu", weights_only=True))
        assert torch.equal(torch.randn(2), wanted[0])
        assert random.random() == wanted[1]
        assert np.array_equal(np.random.rand(2), wanted[2])

    def test_a_cuda_payload_restored_where_there_is_no_cuda_does_not_raise(self):
        # Tolerated one way only: a payload written on a CUDA node and restored on a CPU
        # one skips the device streams rather than failing. The reverse leaves the device
        # generator where seed_everything put it, which is what a fresh run has.
        state = dict(capture_rng())
        state["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]
        restore_rng(state)

    def test_missing_keys_are_skipped_rather_than_raising(self):
        restore_rng({})


class TestTheLoadOrder:
    """``build_scheduler`` is what enforces it, and it enforces it by REFUSING."""

    def test_the_mandated_order_works(self):
        # optimiser -> scheduler -> optimiser.load_state_dict -> scheduler.load_state_dict
        cfg = _cfg()
        _, source_opt, source_sch = _trained_triple(cfg)
        optimiser = build_optimiser(cfg, build_model(cfg))
        scheduler = build_scheduler(cfg, optimiser, 6)
        optimiser.load_state_dict(source_opt.state_dict())
        scheduler.load_state_dict(source_sch.state_dict())
        assert scheduler.last_epoch == source_sch.last_epoch > 0
        assert scheduler.get_last_lr() == source_sch.get_last_lr()

    def test_the_reverse_order_is_refused_by_the_guard_that_exists_for_this(self):
        # Without the guard, a scheduler built on annealed rates peaks wherever the
        # old schedule had got to, so the run is silently under-trained with a
        # completely correct-looking config -- and on a differential-LR run the
        # backbone/head ratio is silently different too. The cause is that
        # build_scheduler reads each group's current `lr` as its intended peak, and
        # `load_state_dict` stamps `initial_lr` into the groups. With the guard, the
        # load fails loudly instead of the results table being wrong.
        cfg = _cfg()
        _, source_opt, _ = _trained_triple(cfg)
        optimiser = build_optimiser(cfg, build_model(cfg))
        optimiser.load_state_dict(source_opt.state_dict())
        assert all("initial_lr" in group for group in optimiser.param_groups)
        with pytest.raises(ValueError, match="initial_lr"):
            build_scheduler(cfg, optimiser, 6)

    def test_the_loop_installs_them_in_the_mandated_order(self):
        # Pinned by reading the source, because the failure is an ORDER and no assertion
        # on a returned value can see it. If train_fold is ever refactored so that the
        # optimiser state lands before the scheduler is built, the guard above fires at
        # runtime only after a GPU has been allocated; this fails here instead.
        source = inspect.getsource(loop_module.train_fold)
        order = [
            source.index("optimiser = build_optimiser("),
            source.index("scheduler = build_scheduler("),
            source.index("optimiser.load_state_dict("),
            source.index("scheduler.load_state_dict("),
        ]
        assert order == sorted(order)


class TestTheAveragerCoupling:
    """resume.py reaches into private attributes by name, so the names are pinned here."""

    def test_the_ema_attributes_resume_reads_still_exist(self):
        averager = ExponentialMovingAverage(build_model(_cfg()), decay=0.99)
        assert isinstance(averager._steps, int)
        assert averager._shadow

    def test_the_lastn_attribute_resume_reads_still_exists(self):
        assert LastNAverage(build_model(_cfg()), n=2)._snapshots == []

    def test_state_dict_still_means_the_averaged_weights_and_not_resume_state(self):
        # The reason resume state does not live on `WeightAverager.state_dict`: that
        # name already means the AVERAGED WEIGHTS, the thing that gets scored and saved,
        # and one name cannot mean both without making a call site wrong.
        model = build_model(_cfg())
        averager = ExponentialMovingAverage(model, decay=0.99)
        averager.after_step(model)
        keys = set(averager.state_dict())
        assert keys and keys <= set(dict(model.named_parameters()))


class TestAnUnresumedRunIsUnchanged:
    """Jobs queued before resume existed must train exactly as they did before."""

    def test_a_fold_trained_without_the_flag_writes_no_resume_file(self, corpus, tmp_path):
        result = train_fold(
            _cfg(optim={"epochs": 2}),
            corpus,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            verbose=False,
        )
        assert result.checkpoint is not None
        assert not list(tmp_path.glob("resume_*")), "an unresumed run wrote resume state"

    def test_a_fold_trained_without_the_flag_does_not_read_one_that_is_there(
        self, corpus, tmp_path, capsys
    ):
        # A stale file beside the output directory must be INVISIBLE, not "refused": a
        # refusal printed by a run that never asked to resume is noise in the log an
        # operator reads for the fold's identity, and it would also mean the file was
        # opened -- which on a 700 MB checkpoint is not free.
        _write(tmp_path, _identity(fold=0, epochs=2, select="last", config_hash="stale"))
        result = train_fold(
            _cfg(optim={"epochs": 2}),
            corpus,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            verbose=True,
        )
        printed = capsys.readouterr().out
        assert not [line for line in printed.splitlines() if line.startswith("resume")]
        assert [record.epoch for record in result.epochs] == [0, 1]
        # And the file is still there: an unresumed run does not clear what it never read.
        assert checkpoint_of(tmp_path, 0, 0).exists()

    def test_the_default_does_not_reach_the_resume_module_at_all(
        self, corpus, tmp_path, monkeypatch
    ):
        # The strongest form of the claim, and the one the in-flight cohort needs: not
        # "it wrote nothing" but "it did not take the path". Anything that so much as
        # constructed a ResumeIdentity fails here.
        def _must_not_run(*_args, **_kwargs):
            raise AssertionError("an unresumed fold reached the resume path")

        for name in (
            "ResumeIdentity",
            "read_sidecar",
            "load_resume",
            "save_resume",
            "clear_resume",
            "restore_rng",
            "install_averager",
        ):
            monkeypatch.setattr(loop_module, name, _must_not_run)
        train_fold(
            _cfg(optim={"epochs": 1}),
            corpus,
            1,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            verbose=False,
        )

    def test_the_entrypoint_defaults_the_flag_off(self):
        assert inspect.signature(train_fold).parameters["resume"].default is False
        assert inspect.signature(loop_module.run_experiment).parameters["resume"].default is False
        source = (Path(__file__).resolve().parents[1] / "scripts" / "train.py").read_text()
        assert '"--resume", dest="resume", action="store_true", default=False' in source
        assert '"--no-resume", dest="resume", action="store_false"' in source
        assert "resume=args.resume" in source

    def test_resume_is_not_a_config_field_and_moves_no_hash(self):
        # Rule 1 of resume.py, asserted rather than trusted. `canonical_dump` hashes the
        # whole model_dump minus _HASH_EXCLUDED = (name, out_dir, seed, seeds), so a
        # single new top-level field would move EVERY existing config_hash -- including
        # the one the first measured baseline (macro 0.7280) is filed under.
        from kappa.config import TrainConfig, config_hash

        assert "resume" not in TrainConfig.model_fields
        assert "resume" not in _cfg().model_dump()
        with pytest.raises(Exception):
            _cfg(resume=True)
        assert config_hash(_cfg()) == config_hash(_cfg())


class TestCpuParity:
    """A fold killed at an epoch boundary and resumed, against the uninterrupted fold.

    **WHAT THIS PROVES.** On CPU, in one process, with ``num_workers=0``, a fold stopped
    after epoch 0 and resumed for epochs 1-2 produces the **same weights bit for bit**
    and the same per-epoch train loss, learning rate and macro as the fold that ran all
    three epochs uninterrupted. That covers the four things a resume can silently get
    wrong: the AdamW moments and the per-group rates, the scheduler's position in its
    anneal, the shuffle permutation the DataLoader draws next, and the global torch CPU
    stream that ``drop_path`` and the validation loader's base-seed draw consume.
    ``drop_path_rate``, the per-sample augmentations, ``series_dropout`` and CutMix are
    all switched ON in ``parity_cfg``, so the claim that the per-sample and per-batch
    streams are stateless by construction -- ``StudyDataset`` derives from
    ``(seed, epoch, index)``, ``PerBatchStream`` from the batch's own uids -- is
    exercised here rather than assumed.

    **WHAT THIS DOES NOT PROVE.** Not parity on a GPU: CUDA has its own RNG (captured
    here, but there is no device to restore it on), reduction kernels that are not
    deterministic and that ``seed_everything`` deliberately does not disable, and cuDNN
    algorithm selection. Not parity across processes: a second process re-imports torch
    and may pick a different thread count, and ``num_workers > 0`` adds per-worker
    seeding this fixture never exercises. Not parity across a mid-epoch kill -- only the
    epoch boundary is saved, by design. And not parity across a code change: ``git_sha``
    is checked, so that case is a refusal rather than a comparison.

    The GPU check is the one-fold, 4-dp recipe in ``kappa/training/resume.py``'s module
    docstring, and this test is not a substitute for it.
    """

    def test_the_kill_left_a_resume_point_in_the_middle_of_the_budget(self, halves):
        # If next_epoch were 0 the kill came too early to prove anything, and if it were
        # PARITY_EPOCHS the resumed half trained nothing. Either way the rest of this
        # class would be vacuous, so it is asserted before anything is compared.
        _, resumed, _ = halves
        assert 0 < PARITY_KILL_AFTER < PARITY_EPOCHS
        assert len(resumed.epochs) == PARITY_EPOCHS

    def test_the_per_epoch_history_agrees_exactly(self, halves):
        reference, resumed, _ = halves

        def numbers(record):
            # Timings are excluded and nothing else is: a wall clock cannot be expected
            # to repeat, and asserting on one would make this test flaky for the single
            # reason that is not about resume.
            return (
                record.epoch,
                record.train_loss,
                record.n_batches,
                record.n_optimiser_steps,
                record.lr_last,
                record.macro,
                record.n_scored,
                record.skipped,
            )

        assert [numbers(r) for r in resumed.epochs] == [numbers(r) for r in reference.epochs]

    def test_the_selected_epoch_and_the_macro_agree(self, halves):
        reference, resumed, _ = halves
        assert resumed.selected_epoch == reference.selected_epoch
        assert resumed.macro == reference.macro

    def test_every_one_of_the_twelve_label_aucs_agrees_to_four_decimals(self, halves):
        # Four decimal places on all twelve, not the macro alone: a macro can agree
        # while two labels move in opposite directions, and it is the per-label table
        # the write-ups quote. At fixed seed this pipeline is bit-exact -- the 8 Sept and
        # 12 Sept runs of one fold agree on all twelve to 4 dp across a different
        # config_hash, git_sha and node -- so 4 dp is the right bar, not a generous one.
        reference, resumed, _ = halves
        assert set(resumed.auc.per_label) == set(reference.auc.per_label)
        assert resumed.auc.skipped == reference.auc.skipped
        for label, value in reference.auc.per_label.items():
            got = resumed.auc.per_label[label]
            # `per_label` holds nan for a skipped label rather than omitting the key,
            # and nan != nan -- so an unguarded equality here would fail on exactly the
            # labels a tiny synthetic fold makes single-class.
            if np.isnan(value):
                assert np.isnan(got), label
            else:
                assert round(got, 4) == round(value, 4), label

    def test_the_weights_are_bit_identical(self, halves):
        # The strongest available form of the claim, and the one that localises a
        # failure: macro ROC-AUC only ranks, so a resume that lost the optimiser moments
        # could still score identically on nine synthetic studies and be wrong. Weights
        # cannot agree by luck.
        reference, resumed, _ = halves
        left = torch.load(reference.checkpoint, map_location="cpu", weights_only=True)
        right = torch.load(resumed.checkpoint, map_location="cpu", weights_only=True)
        assert set(left["state_dict"]) == set(right["state_dict"])
        differing = [
            name
            for name, tensor in left["state_dict"].items()
            if not torch.equal(tensor, right["state_dict"][name])
        ]
        assert not differing, f"{len(differing)} tensors differ after a resume: {differing[:5]}"

    def test_the_oof_rows_agree(self, halves):
        reference, resumed, _ = halves
        left = reference.oof.sort_values(ID_COLUMN).reset_index(drop=True)
        right = resumed.oof.sort_values(ID_COLUMN).reset_index(drop=True)
        assert list(left.columns) == list(right.columns)
        assert left[ID_COLUMN].tolist() == right[ID_COLUMN].tolist()
        for column in left.columns:
            if left[column].dtype.kind == "f":
                # rtol=atol=0: this is an equality, not a closeness. Anything looser
                # would let a resume that restarted the optimiser pass.
                assert np.allclose(left[column], right[column], atol=0, rtol=0), column

    def test_the_resumed_half_reports_every_epoch_it_trained_and_the_ones_it_did_not(
        self, halves
    ):
        # Without the serialised EpochRecord history `result.report()` comes back
        # truncated and a resumed fold's log silently describes fewer epochs than the
        # fold trained -- the same class of mistake as quoting a 1-fold mean.
        _, resumed, _ = halves
        assert [record.epoch for record in resumed.epochs] == list(range(PARITY_EPOCHS))
        assert resumed.step_seconds > 0

    def test_a_finished_fold_leaves_no_resume_file_behind(self, halves):
        # Scaffolding that outlives the build is debris, and at ~700 MB a file it adds
        # up fast across many folds. It is also what a later scheduling pass would read
        # as "this fold is mid-flight".
        _, _, run_dir = halves
        assert not list(Path(run_dir).glob("resume_*"))
        assert read_sidecar(run_dir, 0, 0) is None

    def test_a_resume_across_a_changed_epoch_budget_starts_over_rather_than_mixing(
        self, corpus, tmp_path, capsys
    ):
        # The non-fatal half of rule 3, end to end through train_fold. `optim.epochs` IS
        # in config_hash, so the identity disagrees, and the correct outcome is a loud
        # line and epoch 0 -- not an abort, which the driver would simply re-place, and
        # not a silent continuation onto a schedule of a different length.
        _write(
            tmp_path,
            _identity(fold=0, seed=0, name="parity", epochs=99, select="best"),
            next_epoch=2,
            epochs=[_record(0), _record(1)],
        )
        result = train_fold(
            _cfg(name="parity", optim={"epochs": 1}),
            corpus,
            0,
            device="cpu",
            out_dir=tmp_path,
            normalisation=Normalisation.identity(),
            select="best",
            resume=True,
            verbose=True,
        )
        printed = capsys.readouterr().out
        assert "resume     : REFUSED" in printed
        assert "from epoch 0 instead" in printed
        assert [record.epoch for record in result.epochs] == [0]
        # The finished fold cleared the stale file rather than leaving it for the next
        # pass to refuse all over again.
        assert not list(tmp_path.glob("resume_*"))
