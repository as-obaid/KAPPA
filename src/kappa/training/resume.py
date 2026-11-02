"""Epoch-boundary resume: everything a killed fold needs to carry on, and a sidecar to schedule by.

Without this module, a fold killed at the walltime limit leaves no artifact, so a
scheduler that derives outstanding work from artifacts on disk submits it again and
it dies the same way. :func:`kappa.training.loop.train_fold` held its entire state
in local variables, and the only file it wrote was the inference checkpoint, which
deliberately carries no optimiser or scheduler state: an offline scoring notebook
reads it, and widening it would double its size for tensors nobody loads.

So training state goes into a second, separate file, written at every epoch
boundary and **deleted on success**, plus a small JSON sidecar for the scheduler to
read. The two files are not interchangeable and neither replaces the other.

---

## The four rules this module is built around

**1. Nothing here may reach ``TrainConfig``.** ``canonical_dump`` hashes the whole
``model_dump`` minus ``_HASH_EXCLUDED = ("name", "out_dir", "seed", "seeds")``
(``config.py:1033``). A single new top-level field moves **every** existing
``config_hash``, including the one the first measured baseline (macro 0.7280) is
filed under, and a baseline that cannot be hash-matched to its own config file
stops being a reference point. Resume is therefore driven by **CLI flags only**,
exactly like ``--select``, and this module never sees a config field it invented.
Adding a section to ``_DEFAULTED_SECTIONS`` is not an escape hatch either: that
table is a promise that the default is a no-op *training* behaviour, and a resume
policy is not a training section.

**2. A scheduler must never ``torch.load`` a ~700 MB file to decide what to run.**
Loading a checkpoint that size is slow, and on a memory-capped login node it is
how a process gets OOM-killed with ``rc=137``. So every field a scheduler needs --
``next_epoch``, ``epochs``, ``select``, ``config_hash``, ``git_sha`` -- is
duplicated into a **JSON sidecar** of a few hundred bytes beside the big file.
:func:`sidecar_of` is the whole contract, and an external scheduler can read it
with ``sed``.

**3. Resuming across a selection-policy change is a structural hazard.**
``--select`` is **not** in ``config_hash`` -- it is a CLI argument that never
reaches ``TrainConfig`` -- so two runs differing only in selection policy hash
**identically** and nothing downstream can tell them apart. Without a refusal
here, a resume would happily continue a ``last``-policy run under ``best`` and
produce a policy-mixed fold that no artifact can distinguish from a clean one.
:func:`load_resume` therefore refuses on any disagreement of
``config_hash``, ``git_sha``, ``select`` or ``epochs``, by raising
:class:`ResumeMismatch`. **Restarting from epoch 0 is the correct outcome of a
mismatch**; continuing is not.

**4. ``name:`` is not in the hash but is the output path.** ``out_root =
cfg.out_dir / cfg.name``, so two configs sharing a ``name`` share a directory --
and would therefore share a resume checkpoint, resuming one run's weights into
another run's optimiser with no error anywhere. The OOF parquet has the same
collision. Nothing here can detect it; the one-name-per-config test is what
prevents it, and it must not be weakened.

---

## What is saved, and the two things that are deliberately not

Saved, because each of them is state that an uninterrupted run would have and a
naive restart would not:

* model ``state_dict``;
* optimiser ``state_dict`` -- AdamW's ``exp_avg``/``exp_avg_sq``/``step``, **and**
  the per-group ``lr``/``initial_lr``/``weight_decay``. There are **four** named
  groups and their ORDER is load-bearing (``loop.py:468``); without the per-group
  rates the ``backbone_lr_mult`` ratio dies silently and the run becomes a
  uniform-LR run wearing a differential-LR config;
* the scheduler ``state_dict`` -- ``last_epoch``, ``_schedule_phases``,
  ``base_lrs``. This is what puts the OneCycle anneal back where it was;
* the ``GradScaler``, **conditionally**. Every config in this repository is bf16
  and ``_amp_setup`` returns ``scaler=None`` for bf16 (``loop.py:702``), because
  bf16 has fp32's exponent range and there is no underflow to rescue. So the key
  is written only when a scaler exists, and an absent key is not an error;
* ``next_epoch``, as a **distinct field**. ``meta["epoch"]`` in the inference
  ``.pt`` is the *selected* epoch -- an output of the selection policy -- not a
  measure of progress. Conflating the two would resume a ``--select best`` run at
  whichever epoch happened to score highest;
* the DataLoader shuffle ``torch.Generator`` (``loop.py:1384``), which is the
  **only stateful RNG in the loop**. Restoring it reproduces the exact
  permutation the uninterrupted run would have drawn for the next epoch;
* the global torch RNG, CPU **and** CUDA. ``drop_path`` and ``nn.Dropout`` draw
  from the *device* generator, so a CUDA run that restored only the CPU stream
  would take different masks after a resume. ``random`` and ``np.random`` go with
  them for completeness;
* the ``--select best`` best-so-far tracker, ``(macro, epoch, PredictionResult)``.
  It is **serialised, not recomputed**: recomputing would mean re-running a
  validation pass per candidate epoch, and the whole object is ~880x12 float32
  plus uids, about **42 KB**. ``last`` is carried beside it for the same price,
  which additionally lets a fold killed *after* its final epoch's save finish
  without retraining anything;
* the per-epoch ``EpochRecord`` history (``FoldResult.epochs``); without it
  ``result.report()`` comes back truncated and a resumed fold's log silently
  describes fewer epochs than it trained;
* the averager -- ``ExponentialMovingAverage._shadow`` and ``_steps``, or
  ``LastNAverage``'s snapshots. An EMA restarted empty is an expensive copy of
  the mid-run weights, which is not the quantity the arm is about;
* ``step_total`` (the cumulative step-seconds that
  ``FoldResult.partial_timing_ratio`` is computed from), and the **W&B run id**,
  so a resumed fold continues one run rather than creating a second one that
  looks like an extra fold in the runs table.

**Not saved, and this is deliberate -- do not "fix" it:**

* **the per-sample augmentation RNG.** ``StudyDataset.__getitem__`` draws from
  ``np.random.default_rng((self.seed, self._epoch, index))`` (``dataset.py:506``):
  a pure function of ``(seed, epoch, index)`` with no carried state. Restoring
  ``set_epoch(epoch)`` restores it exactly.
* **the CutMix / batch-augment RNG.** ``PerBatchStream.__call__`` explicitly
  discards the stream it is handed and derives its generator from
  ``batch_seed_from_uids(batch["study_uid"], self.seed)`` (``augment.py:313``) --
  a pure function of ``(study_uids, seed)``. The batch's own ids are the entropy.

Both are stateless by construction, so there is nothing to serialise. Saving
something for them would not make a resumed run more faithful; it would only add
a field that a future reader has to verify.

## Atomicity, and why the order of the two writes matters

Both files are written ``.tmp`` first and then ``os.replace``d, which is atomic on
local and network filesystems alike. The **big file is published first and the
sidecar second**, and that order is load-bearing: the sidecar is what a scheduler
reads as "a resumable checkpoint exists here", so publishing it before the ``.pt``
is complete would let a scheduling pass believe in a checkpoint that a training
job then cannot load. A ``.pt`` with no sidecar simply reads as ``todo``, which
costs an epoch; a sidecar with no ``.pt`` costs a whole job.

## A GPU parity check, run by hand

``tests/test_resume.py`` proves the round-trip and an end-to-end equality on a
tiny synthetic setup on CPU. That is not the same claim as parity on a GPU over
the real cache, so the check below is worth running once by hand; nothing here
runs it. At fixed seed this pipeline is bit-exact -- two runs of one fold agree on
all 12 label AUCs to 4 dp across a different ``config_hash``, ``git_sha`` and
node -- so 4 dp is the right bar, not a generous one.

.. code-block:: console

   # One config, one fold, one seed. Both runs must use --select best: `--select`
   # is not in config_hash, so a policy difference between the two halves is
   # invisible in every artifact.
   CFG=configs/<config>.yaml
   OUT=$KAPPA_DATA_ROOT/parity

   # (1) The reference: uninterrupted, resume off, so it cannot read a stale file.
   python scripts/train.py --config $CFG --fold 0 --select best --no-resume --out-dir $OUT/ref

   # (2) The interrupted half, part one. Do not shorten optim.epochs: it is in
   #     config_hash, so changing it changes the experiment. Kill the job mid-run
   #     instead, e.g. with a scheduler walltime of about half the fold's runtime.
   python scripts/train.py --config $CFG --fold 0 --select best --out-dir $OUT/run

   # (3) Confirm it died mid-fold and left a resume point. next_epoch must be
   #     >= 1 and < epochs; if it is 0 the kill came too early to prove anything.
   cat $OUT/run/resume_fold0_seed0.json

   # (4) Resume it. Same config, same --select, same commit. A different git SHA
   #     is refused by load_resume rather than silently mixed.
   python scripts/train.py --config $CFG --fold 0 --select best --resume --out-dir $OUT/run

   # (5) Compare all twelve label AUCs to 4 dp. Not the macro alone: a macro can
   #     agree while two labels move in opposite directions.
   python - <<'PY'
   import pandas as pd
   from sklearn.metrics import roc_auc_score
   from kappa.constants import LABELS
   def aucs(p):
       d = pd.read_parquet(p)
       out = {}
       for lab in LABELS:
           m = d[f"mask_{lab}"].astype(bool)
           out[lab] = round(float(roc_auc_score(d.loc[m, f"y_{lab}"], d.loc[m, lab])), 4)
       return out
   a, b = aucs("REF/oof_seed0_fold0.parquet"), aucs("RUN/oof_seed0_fold0.parquet")
   bad = {k: (a[k], b[k]) for k in LABELS if a[k] != b[k]}
   print("PARITY OK" if not bad else f"PARITY FAILED on {len(bad)}: {bad}")
   PY

**Read the result honestly.** A disagreement at 4 dp is a *finding about this
module*, not noise -- the seed-variance floor of 0.0060 is between *seeds*, and
these two runs share one. If it fails, the first three things to check are the
load order (see :func:`load_resume`), whether the killed half wrote its sidecar at
an epoch boundary or mid-epoch, and whether ``num_workers`` differed between the
two submissions.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kappa.inference.predict import PredictionResult
from kappa.training.averaging import ExponentialMovingAverage, LastNAverage, WeightAverager

__all__ = [
    "RESUME_VERSION",
    "ResumeIdentity",
    "ResumeMismatch",
    "ResumeState",
    "capture_rng",
    "checkpoint_of",
    "clear_resume",
    "load_resume",
    "read_sidecar",
    "restore_rng",
    "save_resume",
    "sidecar_of",
]

#: Bumped whenever the payload's shape changes. A version this build does not
#: recognise is REFUSED rather than partially read: a resume that silently dropped,
#: say, the averager would train on and report a number for an arm it never ran.
RESUME_VERSION = 1


class ResumeMismatch(ValueError):
    """The checkpoint on disk belongs to a different experiment than the one asking.

    Raised rather than returned, and never downgraded to a warning inside this
    module. The caller decides what to do -- ``train_fold`` starts from epoch 0 and
    says so loudly -- but the decision must be taken somewhere that a human reads,
    not absorbed by a ``try: ... except: pass``.
    """


@dataclass(frozen=True)
class ResumeIdentity:
    """The seven fields that decide whether a checkpoint may be resumed into this run.

    Four of them are CHECKED (``config_hash``, ``git_sha``, ``select``, ``epochs``)
    and three are RECORDED (``seed``, ``fold``, ``name``). The split is not
    arbitrary:

    * ``config_hash`` covers ``data``, ``augment``, ``model``, ``loss``, ``optim``,
      ``folds`` and ``eval``, so a disagreement means the two runs are different
      experiments and their optimiser states are not interchangeable.
    * ``select`` is checked precisely BECAUSE it is absent from ``config_hash``.
      It is the one axis on which two runs can be genuinely different and hash
      identically, which makes it the one axis a hash check cannot cover.
    * ``epochs`` is redundant with ``config_hash`` today -- ``optim.epochs`` is
      hashed, so one config hashes differently at 8 epochs and at 10 -- and it is
      checked anyway, because the schedule
      length is what ``last_epoch`` is meaningful relative to, and a redundant
      check that costs nothing is worth more than a comment saying it is redundant.
    * ``git_sha`` is checked because the code between two epochs of one fold must
      be one generation. A resume across a commit is not a faster run; it is two
      halves of two different runs reported as one.
    * ``seed``, ``fold`` and ``name`` are recorded rather than checked because the
      *path* already encodes them -- ``<out_root>/resume_fold<k>_seed<s>.pt`` under
      a directory that IS ``cfg.name``. Checking them would test the filesystem,
      not the experiment. They are written so a file found on its own can say what
      it is.
    """

    config_hash: str
    git_sha: str
    select: str
    epochs: int
    seed: int
    fold: int
    name: str

    #: The subset compared by :func:`load_resume`. A table rather than four
    #: hand-written comparisons so that a field added above cannot be added to the
    #: checked set and forgotten here, or vice versa -- the omission would be silent
    #: and would let exactly the wrong checkpoint through.
    CHECKED = ("config_hash", "git_sha", "select", "epochs")

    def disagreements(self, other: ResumeIdentity) -> dict[str, tuple[Any, Any]]:
        """``{field: (on_disk, wanted)}`` for every checked field that differs."""
        return {
            field: (getattr(self, field), getattr(other, field))
            for field in self.CHECKED
            if getattr(self, field) != getattr(other, field)
        }


@dataclass
class ResumeState:
    """One fold, mid-flight. Everything :func:`kappa.training.loop.train_fold` needs back.

    ``epochs`` is a tuple of plain dicts rather than ``EpochRecord`` objects, and
    that is an import-cycle constraint rather than a taste: ``EpochRecord`` lives in
    ``loop.py``, ``loop.py`` imports this module, so this module must not import
    ``loop.py``. The two-line reconstruction happens at the call site, where the
    type is in scope.
    """

    identity: ResumeIdentity
    #: The epoch the loop should START at. Distinct from any "selected" epoch.
    next_epoch: int
    #: Cumulative optimiser steps over the WHOLE fold. See :func:`save_resume` on
    #: why this is not the loop's ``n_steps``.
    global_step: int
    #: Cumulative step-seconds, which ``FoldResult.partial_timing_ratio`` divides.
    step_total: float
    model: dict[str, torch.Tensor]
    optimiser: dict[str, Any]
    scheduler: dict[str, Any]
    #: ``None`` on every bf16 config in this repository, which is all of them.
    scaler: dict[str, Any] | None
    #: ``torch.Generator.get_state()`` of the DataLoader's shuffle generator.
    loader_generator: torch.Tensor
    rng: dict[str, Any]
    epochs: tuple[dict[str, Any], ...]
    best: tuple[float, int, PredictionResult] | None
    last: tuple[int, PredictionResult] | None
    averager: dict[str, Any] | None
    wandb_run_id: str | None


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# Named beside the inference checkpoint (`fold<k>_seed<s>.pt`) and the OOF parquet
# (`oof_seed<s>_fold<k>.parquet`), with a `resume_` prefix that is the ONLY thing
# distinguishing them. The prefix matters twice over: a scheduler globs for it, and
# `log_standard_artifacts` uploads `FoldResult.checkpoint` by path -- so a resume
# file that shared the inference naming would be a 700 MB W&B artifact per fold.


def checkpoint_of(out_root: Path | str, fold: int, seed: int) -> Path:
    """The big file: model, optimiser, scheduler, RNG, predictions. ~700 MB."""
    return Path(out_root) / f"resume_fold{int(fold)}_seed{int(seed)}.pt"


def sidecar_of(out_root: Path | str, fold: int, seed: int) -> Path:
    """The small file a scheduler reads. A few hundred bytes, and never torch.load-ed.

    **Its layout is a contract with external schedulers**, which can extract
    ``next_epoch``, ``epochs`` and ``stamp`` with ``sed`` on a line-anchored
    pattern. It is written with ``indent=2`` and ``sort_keys=True`` so that every
    key sits alone on its own line; a compact one-line dump would still be valid
    JSON and would silently defeat that extraction.
    ``tests/test_resume.py`` pins the two halves against each other.
    """
    return Path(out_root) / f"resume_fold{int(fold)}_seed{int(seed)}.json"


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
# Everything below is converted to plain ints/floats/strings rather than stored as
# the objects the stdlib hands back. `np.random.get_state()` returns a tuple
# containing an ndarray and `random.getstate()` returns nested tuples; both pickle,
# but `torch.load(weights_only=True)` -- which this module uses, for the same reason
# `predict.load_checkpoint` does -- refuses arbitrary reconstructors. Flattening
# them here means the resume file is tensors and plain containers by construction
# and the safe loader is sufficient.


def capture_rng(device: torch.device | None = None) -> dict[str, Any]:
    """Every RNG stream the training loop can draw from, in loadable form.

    The CUDA stream is the one that is easy to forget and expensive to omit:
    ``drop_path`` and ``nn.Dropout`` draw from the *device* generator, not the CPU
    one, so a resumed run that restored only ``torch.get_rng_state()`` would take a
    different set of masks from the epoch it resumed into.

    ``get_rng_state_all()`` rather than ``get_rng_state()`` because it is
    device-count-shaped and therefore survives a one-GPU job being restored on a
    node whose visible device count differs -- :func:`restore_rng` puts back only as
    many as both sides have, rather than raising.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    captured: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python": [int(py_state[0]), [int(v) for v in py_state[1]], py_state[2]],
        # ('MT19937', keys[624] uint32, pos, has_gauss, cached_gaussian)
        "numpy": [
            str(np_state[0]),
            [int(v) for v in np_state[1]],
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ],
    }
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in this environment
        captured["torch_cuda"] = list(torch.cuda.get_rng_state_all())
    del device  # accepted for symmetry with restore_rng; CUDA is captured wholesale
    return captured


def restore_rng(state: Mapping[str, Any]) -> None:
    """Put every stream :func:`capture_rng` took back where it was.

    Missing keys are tolerated one way only: a payload written on a CUDA node and
    restored on a CPU one simply skips the device streams. The reverse -- a CPU
    payload restored onto CUDA -- leaves the device generator at whatever
    ``seed_everything`` set it to, which is the same state a fresh run would have.
    """
    cpu = state.get("torch_cpu")
    if cpu is not None:
        # A uint8 CPU ByteTensor is what set_rng_state demands; torch.load may hand
        # it back on another device if map_location said so, so force it here.
        torch.set_rng_state(torch.as_tensor(cpu, dtype=torch.uint8).cpu())

    python = state.get("python")
    if python is not None:
        random.setstate((int(python[0]), tuple(int(v) for v in python[1]), python[2]))

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(
            (
                str(numpy_state[0]),
                np.asarray(numpy_state[1], dtype=np.uint32),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            )
        )

    cuda = state.get("torch_cuda")
    if cuda and torch.cuda.is_available():  # pragma: no cover - no CUDA here
        have = torch.cuda.device_count()
        for index, blob in enumerate(cuda[:have]):
            torch.cuda.set_rng_state(torch.as_tensor(blob, dtype=torch.uint8).cpu(), index)


# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------


def _dump_prediction(prediction: PredictionResult) -> dict[str, Any]:
    """A ``PredictionResult`` as tensors and plain containers.

    ``scores`` goes in as a torch tensor rather than a numpy array so the whole
    payload loads under ``weights_only=True``. About 880 x 12 float32 -- **42 KB**
    -- which is why the best-so-far tracker is serialised rather than recomputed:
    recomputing it means a validation pass per candidate epoch, at ~17 s of steady
    GPU time each, to avoid writing 42 KB.
    """
    return {
        "study_uids": [str(uid) for uid in prediction.study_uids],
        "scores": torch.from_numpy(np.ascontiguousarray(prediction.scores, dtype=np.float32)),
        "seconds": float(prediction.seconds),
        "n_batches": int(prediction.n_batches),
        "device": str(prediction.device),
        "amp": str(prediction.amp),
        "tta": [str(view) for view in prediction.tta],
    }


def _load_prediction(payload: Mapping[str, Any]) -> PredictionResult:
    return PredictionResult(
        study_uids=tuple(str(uid) for uid in payload["study_uids"]),
        scores=np.asarray(payload["scores"], dtype=np.float32),
        seconds=float(payload["seconds"]),
        n_batches=int(payload["n_batches"]),
        device=str(payload["device"]),
        amp=str(payload["amp"]),
        tta=tuple(str(view) for view in payload.get("tta", ())),
    )


# ---------------------------------------------------------------------------
# AVERAGERS
# ---------------------------------------------------------------------------
# `WeightAverager.state_dict()` already means something else -- the AVERAGED
# WEIGHTS, the thing that gets scored and saved -- so resume state cannot reuse that
# name without making one of the two meanings wrong at every call site. It lives
# here instead, reaching into `_shadow`/`_steps`/`_snapshots` by name.
#
# That coupling is real and is pinned by a test: `tests/test_resume.py` asserts the
# attribute names exist, so renaming one in averaging.py fails there rather than
# producing a resumed run whose EMA silently restarted from the mid-run weights --
# which is not the quantity an EMA run is meant to measure.


def _dump_averager(averager: WeightAverager | None) -> dict[str, Any] | None:
    if averager is None:
        return None
    if isinstance(averager, ExponentialMovingAverage):
        return {
            "kind": "ema",
            "steps": int(averager._steps),
            "shadow": {name: t.detach().cpu() for name, t in averager._shadow.items()},
        }
    if isinstance(averager, LastNAverage):
        return {
            "kind": "swa",
            # Already on the CPU in fp32 by construction (see LastNAverage), and
            # bounded at n -- so this is n x ~200 MB and not the whole history.
            "snapshots": [
                {name: t.detach().cpu() for name, t in snapshot.items()}
                for snapshot in averager._snapshots
            ],
        }
    raise TypeError(  # pragma: no cover - there are exactly two averagers
        f"no resume representation for {type(averager).__name__}; add one here rather "
        "than letting a resumed run report a number for an averaging arm it did not run"
    )


def _load_averager(averager: WeightAverager | None, state: Mapping[str, Any] | None) -> None:
    """Restore an averager, refusing a payload of the wrong kind.

    A mismatch here means the config changed ``optim.ema_decay``/``swa_last_n``
    without moving ``config_hash`` -- which cannot happen, both are hashed -- or
    that the file belongs to another experiment, which :func:`load_resume` should
    already have refused. It is checked anyway because the failure it would
    otherwise produce is an averaging arm reporting a number for weights that were
    never averaged.
    """
    if averager is None or state is None:
        if averager is not None or state is not None:
            raise ResumeMismatch(
                f"averager is {type(averager).__name__ if averager else 'off'} in this run "
                f"but {'present' if state else 'absent'} in the checkpoint. The averaging "
                "config is inside config_hash, so this should have been caught earlier; "
                "do not resume across it."
            )
        return
    kind = str(state.get("kind"))
    if isinstance(averager, ExponentialMovingAverage):
        if kind != "ema":
            raise ResumeMismatch(f"checkpoint holds a {kind!r} averager, this run runs an EMA")
        averager._steps = int(state["steps"])
        with torch.no_grad():
            for name, tensor in state["shadow"].items():
                target = averager._shadow[name]
                target.copy_(tensor.to(target.device, target.dtype))
        return
    if isinstance(averager, LastNAverage):
        if kind != "swa":
            raise ResumeMismatch(f"checkpoint holds a {kind!r} averager, this run runs a LastN")
        averager._snapshots = [
            {name: tensor.detach().to("cpu", torch.float32).clone() for name, tensor in snap.items()}
            for snap in state["snapshots"]
        ]
        return
    raise TypeError(  # pragma: no cover - there are exactly two averagers
        f"no resume representation for {type(averager).__name__}"
    )


# ---------------------------------------------------------------------------
# WRITE
# ---------------------------------------------------------------------------


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """``.tmp`` then ``os.replace``, which is atomic within one filesystem.

    The failure this closes is not hypothetical for a file this size: a walltime
    kill or a node failure part-way through a ~700 MB write leaves a truncated
    ``.pt`` that still exists, still matches a scheduler's glob, and fails to
    unpickle only once a GPU has been allocated to it. ``os.replace`` means the
    published path is either the previous complete checkpoint or the new complete
    one, never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_resume(
    out_root: Path | str,
    *,
    identity: ResumeIdentity,
    next_epoch: int,
    global_step: int,
    step_total: float,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any | None,
    loader_generator: torch.Generator,
    epochs: Sequence[Any],
    best: tuple[float, int, PredictionResult] | None,
    last: tuple[int, PredictionResult] | None,
    averager: WeightAverager | None,
    wandb_run_id: str | None = None,
    device: torch.device | None = None,
) -> Path:
    """Write the checkpoint and then the sidecar. Returns the checkpoint path.

    **``global_step`` is NOT the loop's ``n_steps``, and that separation is the
    point.** ``n_steps`` is re-zeroed at the top of every epoch (``loop.py:1522``),
    which makes the guard three lines below it --
    ``if n_steps < total_steps: scheduler.step()``, where
    ``total_steps = steps_per_epoch * epochs`` -- dead code for every budget this
    repository runs: a per-epoch counter cannot reach a whole-run total once
    ``epochs >= 2``. The comment on that guard describes behaviour it does not
    have. The *natural* way to make the counter resumable is to make ``n_steps``
    cumulative -- and that would ACTIVATE the dead guard, so the scheduler would
    silently stop stepping near the end of training and the run would anneal to the
    wrong place while looking entirely healthy. So ``n_steps`` is left exactly as it
    is and a separate cumulative counter is carried here. **Do not merge them.**

    ``scaler`` may be ``None`` and usually is: every config in this repository sets
    ``amp_dtype: bf16``, and ``_amp_setup`` builds a ``GradScaler`` only for fp16.
    An absent scaler writes no key, and a resume that finds no key builds none.
    """
    identity_payload = dataclasses.asdict(identity)
    checkpoint = checkpoint_of(out_root, identity.fold, identity.seed)
    sidecar = sidecar_of(out_root, identity.fold, identity.seed)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    payload: dict[str, Any] = {
        "kappa_resume_version": RESUME_VERSION,
        "identity": identity_payload,
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "step_total": float(step_total),
        "stamp": stamp,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loader_generator": loader_generator.get_state(),
        "rng": capture_rng(device),
        # `dataclasses.asdict` rather than a hand-written field list: EpochRecord
        # gains fields (it gained `step_seconds`), and a list here would go stale
        # silently, truncating `result.report()` on exactly the runs that were
        # interrupted.
        "epochs": [dataclasses.asdict(record) for record in epochs],
        "averager": _dump_averager(averager),
        "wandb_run_id": None if wandb_run_id is None else str(wandb_run_id),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if best is not None:
        payload["best"] = {
            "macro": float(best[0]),
            "epoch": int(best[1]),
            "prediction": _dump_prediction(best[2]),
        }
    if last is not None:
        payload["last"] = {"epoch": int(last[0]), "prediction": _dump_prediction(last[1])}

    # Order is load-bearing. Big file first, sidecar second -- see the module
    # docstring. A sidecar published ahead of its checkpoint would let a scheduling
    # pass believe in a checkpoint that a GPU job then cannot load.
    _atomic_torch_save(payload, checkpoint)
    _atomic_text(
        json.dumps(
            {
                "kappa_resume_version": RESUME_VERSION,
                "config_hash": identity.config_hash,
                "git_sha": identity.git_sha,
                "select": identity.select,
                "epochs": int(identity.epochs),
                "next_epoch": int(next_epoch),
                "seed": int(identity.seed),
                "fold": int(identity.fold),
                "name": identity.name,
                "stamp": stamp,
                "checkpoint": checkpoint.name,
                # In the sidecar as well as the big file, deliberately. The point of
                # carrying a run id at all is that a resumed fold continues one W&B
                # run rather than appearing twice in the runs table, and the decision
                # to reattach has to be taken BEFORE `wandb.init`, which is before
                # `train_fold` exists -- see `run_experiment`. Reading it out of the
                # ~700 MB checkpoint to make that decision would be the same "wrong
                # tool" the sidecar exists to avoid; 36 bytes of JSON is not.
                "wandb_run_id": None if wandb_run_id is None else str(wandb_run_id),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        sidecar,
    )
    return checkpoint


def clear_resume(out_root: Path | str, fold: int, seed: int) -> None:
    """Delete both files. Called on SUCCESS only, and the size is why.

    A resume checkpoint is roughly **700 MB** -- 227 MiB of model plus ~454 MiB of
    AdamW moments, and another 227 MiB when an EMA is running. Left behind across
    every fold of a sweep, they would fill a scratch filesystem. The inference
    checkpoint and the OOF parquet are the artifacts; this file is scaffolding, and
    scaffolding that outlives the build is just debris.

    The sidecar goes first. It is what a scheduler reads, so removing it first means
    the intermediate state visible to a scheduling pass is "a ``.pt`` with no
    sidecar" -- which reads as ``todo``, i.e. the pre-resume behaviour -- rather
    than "a sidecar with no ``.pt``", which would promise a checkpoint that is gone.
    """
    sidecar_of(out_root, fold, seed).unlink(missing_ok=True)
    checkpoint_of(out_root, fold, seed).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------


def read_sidecar(out_root: Path | str, fold: int, seed: int) -> dict[str, Any] | None:
    """The sidecar as a dict, or ``None`` if it is absent or unreadable.

    ``None`` rather than an exception for a malformed file: this is the cheap
    scheduling read, and a scheduler that crashes on a torn JSON write is worse
    than one that treats the fold as not-yet-resumable and runs it from the start.
    The *training* path has no such tolerance -- see :func:`load_resume`.
    """
    path = sidecar_of(out_root, fold, seed)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_resume(
    out_root: Path | str,
    *,
    expect: ResumeIdentity,
    map_location: str | torch.device = "cpu",
) -> ResumeState:
    """Read the checkpoint, refusing anything that is not this exact experiment.

    **The load order is the trap, and it is enforced in ``build_scheduler``, not
    here.** This function returns state dicts; it does not install
    them, because the order they must be installed in involves objects it does not
    own. That order is:

    .. code-block:: python

        optimiser = build_optimiser(cfg, model)          # 1. no initial_lr yet
        scheduler = build_scheduler(cfg, optimiser, n)   # 2. stamps initial_lr in
        optimiser.load_state_dict(state.optimiser)       # 3. restores the moments
        scheduler.load_state_dict(state.scheduler)       # 4. restores last_epoch

    ``build_scheduler`` REFUSES any optimiser whose param groups already carry
    ``initial_lr`` (``loop.py:578``), with a comment saying it exists so that "the
    day a resume path is added, it fails here instead of in the results table".
    ``optimiser.load_state_dict()`` restores ``initial_lr``, so steps 2 and 3
    swapped make the guard fire -- which is the *good* outcome. The bad outcome is
    the one that guard is aimed at and that a future refactor could still reach: a
    scheduler built on annealed rates peaks wherever the old schedule had got to,
    producing a silently under-trained run with a completely correct-looking
    config, and on a differential-LR run a silently different backbone/head ratio
    as well. ``tests/test_resume.py`` asserts both directions.

    ``weights_only=True`` for the same reason ``predict.load_checkpoint`` uses it:
    this payload is tensors and plain containers by construction, so the safe
    loader is sufficient and a file from an untrusted source cannot execute code.
    """
    path = checkpoint_of(out_root, expect.fold, expect.seed)
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ResumeMismatch(f"{path} does not hold a KAPPA resume mapping")

    version = payload.get("kappa_resume_version")
    if version != RESUME_VERSION:
        raise ResumeMismatch(
            f"{path} has resume version {version!r}, this build reads {RESUME_VERSION}. "
            "Refusing rather than guessing: a partially-understood payload is how an "
            "averager or an RNG stream goes quietly missing from a resumed run."
        )

    raw = payload.get("identity") or {}
    found = ResumeIdentity(
        config_hash=str(raw.get("config_hash", "")),
        git_sha=str(raw.get("git_sha", "")),
        select=str(raw.get("select", "")),
        epochs=int(raw.get("epochs", -1)),
        seed=int(raw.get("seed", -1)),
        fold=int(raw.get("fold", -1)),
        name=str(raw.get("name", "")),
    )
    bad = found.disagreements(expect)
    if bad:
        lines = "; ".join(f"{k}: on disk {v[0]!r}, this run {v[1]!r}" for k, v in bad.items())
        raise ResumeMismatch(
            f"{path} belongs to a different experiment -- {lines}. REFUSING to resume. "
            "`--select` is NOT in config_hash, so continuing a `last`-policy run under "
            "`best` would produce a policy-mixed fold that nothing downstream could tell "
            "from a good one. Start this fold from epoch 0 instead."
        )

    best = payload.get("best")
    last = payload.get("last")
    return ResumeState(
        identity=found,
        next_epoch=int(payload["next_epoch"]),
        global_step=int(payload.get("global_step", 0)),
        step_total=float(payload.get("step_total", 0.0)),
        model=dict(payload["state_dict"]),
        optimiser=dict(payload["optimiser"]),
        scheduler=dict(payload["scheduler"]),
        scaler=payload.get("scaler"),
        loader_generator=payload["loader_generator"],
        rng=dict(payload.get("rng") or {}),
        epochs=tuple(dict(row) for row in payload.get("epochs", ())),
        best=(
            None
            if best is None
            else (float(best["macro"]), int(best["epoch"]), _load_prediction(best["prediction"]))
        ),
        last=(None if last is None else (int(last["epoch"]), _load_prediction(last["prediction"]))),
        averager=payload.get("averager"),
        wandb_run_id=payload.get("wandb_run_id"),
    )


def install_averager(averager: WeightAverager | None, state: Mapping[str, Any] | None) -> None:
    """Public name for the averager restore, so ``loop.py`` does not reach into a private."""
    _load_averager(averager, state)
