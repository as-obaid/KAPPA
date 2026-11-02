"""Weights & Biases run tracking, and the one convention every run follows.

Why this exists: **a run that cannot be read a week later is not evidence.** Two
failures motivate it: runs that exist only as directories on scratch because
nothing synced them, and a nested config logged to W&B as a single object, which
the UI can display but cannot filter, group or sort on. Both produce work that
was done and cannot be used.

So the standard is encoded here rather than assembled at each call site:

1. **Config is entities, not a blob.** :func:`flatten_config` turns the
   *resolved* configuration into dotted keys -- ``data.n_slices``,
   ``model.backbone``, ``loss.gamma_neg`` -- so every parameter is its own
   filterable W&B config field. The resolved config, never the file: a YAML file
   plus a pile of ``--set`` overrides is not the configuration that ran, and the
   difference is exactly what makes an old run uninterpretable.
2. **Identity travels with every run.** :func:`identity_entities` attaches the
   git SHA, the config hash, the Slurm job and array ids, the node, the
   partition, the venv and the torch/CUDA versions. A number that cannot be
   traced to a commit, a node and an environment is not reproducible.
3. **Grouping reassembles an experiment.** Folds run as *parallel one-GPU Slurm
   jobs, never as ranks*, so the W&B group is the only thing that puts five
   folds and two seeds back together. The convention, enforced in
   :func:`run_name` and :func:`run_group` and nowhere else:

       name  = ``{config_name}-f{fold:02d}-s{seed:02d}``   (fold ``all`` when a
               single job trains several folds)
       group = ``{config_name}-{config_hash[:8]}``
       job_type = ``train`` | ``noise-floor`` | ``predict`` | ...

   The fold and seed are zero-padded so the runs table sorts lexically into
   numeric order. The group is keyed on
   :func:`kappa.config.config_hash`, which deliberately excludes ``seed``, so a
   seed sweep lands in one group differing only in the ``seed`` field -- which is
   what makes a noise floor readable in the UI at all. The gap between "works"
   and top-10 on this task is about 0.006 macro and a competitor measured a
   0.0020 macro noise floor at three folds, so until a same-config
   different-seed spread is visible, no comparison of two runs means anything.
4. **Metrics say what they are over.** :func:`label_metrics` logs all twelve
   per-label AUCs, the macro, the fold standard deviation for all twelve, *and*
   a per-label skipped flag. A label that could not be scored is named, never
   silently omitted -- eleven labels averaged and called macro-12 is a different
   quantity, not a rounding error.
5. **Artifacts make a run reproducible from the UI.** :func:`log_artifact` and
   :func:`log_standard_artifacts` attach the resolved config, the OOF parquet,
   the submission csv, the fold table and the environment lockfile. Model
   weights are the exception and are **off by default**: trained weights live in
   ``$KAPPA_WEIGHTS_DIR``, and uploading multi-gigabyte checkpoints on every
   fold would spend the network and the quota for something already on disk. Set
   ``KAPPA_WANDB_LOG_WEIGHTS=1`` to opt in; otherwise the checkpoint path is
   recorded in the summary and the file stays where it is.
6. **Offline is the normal case and it must actually sync.** Cluster compute
   nodes often have no outbound internet, so a run inside a Slurm allocation logs
   offline into ``WANDB_DIR`` and is pushed later from a login node.
   :func:`finish_run` prints the offline directory and the exact command that
   syncs it, because a run the operator cannot find is a run that will not be
   synced.

Tracking is optional by construction: if ``wandb`` is not installed, no API key
is configured, or ``KAPPA_NO_TRACKING=1`` is set, :func:`init_run` returns a
:class:`NoOpRun` and the caller carries on. **An experiment must never fail
because its logger is unavailable.** Every helper here accepts a
:class:`NoOpRun` and does nothing with it.

    from kappa.tracking import finish_run, label_metrics, start_training_run, summarise

    run = start_training_run(cfg, fold=0, seed=0, n_train=3500, n_val=880)
    summarise(run, label_metrics(auc, prefix="val"))
    finish_run(run)

Scratch is purged on a timer, so run directories go somewhere deliberate via
``WANDB_DIR`` (set by ``slurm/env_common.sh``) rather than next to whatever the
cwd happens to be when a job starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from kappa.constants import LABELS

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from kappa.config import TrainConfig
    from kappa.evaluation.metrics import AucResult, DiagnosticResult, FoldSummary

__all__ = [
    "DEFAULT_ENTITY",
    "DEFAULT_PROJECT",
    "FOLD_TABLE_COLUMNS",
    "LABEL_SUMMARY_COLUMNS",
    "LABEL_TABLE_COLUMNS",
    "LOCKFILE",
    "WEIGHTS_OPT_IN",
    "NoOpRun",
    "config_digest",
    "diagnostic_label_metrics",
    "environment_entities",
    "epoch_metrics",
    "file_digest",
    "fold_prefix",
    "finish_run",
    "install_walltime_guard",
    "flatten_config",
    "git_sha",
    "identity_entities",
    "init_run",
    "label_metrics",
    "log_artifact",
    "log_standard_artifacts",
    "offline_directory",
    "require_run_directory",
    "per_fold_table",
    "per_label_summary_table",
    "per_label_table",
    "step_metrics",
    "pr_curve_panel",
    "roc_curve_panel",
    "run_group",
    "run_name",
    "start_training_run",
    "summarise",
    "summary_metrics",
    "sync_hint",
    "training_entities",
]

DEFAULT_ENTITY = "muhammadobaidullah98-northeastern-university"
DEFAULT_PROJECT = "KAPPA"

#: Environment variable that opts a run into uploading its checkpoints. Off by
#: default on purpose -- see the module docstring's point 5.
WEIGHTS_OPT_IN = "KAPPA_WANDB_LOG_WEIGHTS"

#: The environment lockfile logged with every training run, relative to the repo
#: root. Logged as an artifact rather than described in prose because "the env
#: that produced this number" is not something anyone reconstructs by hand.
LOCKFILE = Path("slurm/locks/kappa-train.lock")

#: Columns of the per-label results table. Fixed here so every run's table has
#: the same shape and twelve rows, always: an earlier W&B table in this project
#: was truncated, and the fix was two tables with every row present and **no
#: cap**. Do not add one.
LABEL_TABLE_COLUMNS = (
    "label",
    "auc",
    "scored",
    "n_used",
    "n_pos",
    "prevalence",
    "fold_std",
    "folds_skipped",
)

#: Columns of the across-fold per-label table. Same twelve rows, aggregated: the
#: mean is unreadable without the spread on this task, so they share a row.
LABEL_SUMMARY_COLUMNS = (
    "label",
    "auc_mean",
    "auc_std",
    "folds_scored",
    "folds_skipped",
    "n_folds",
)

#: Columns of the per-fold table. The second of the two tables: per-label rows
#: answer "which label is weak", per-fold rows answer "is this spread or signal".
FOLD_TABLE_COLUMNS = (
    "fold",
    "seed",
    "macro",
    "n_scored",
    "n_skipped",
    "skipped",
    "n_train",
    "n_val",
    "wall_seconds",
    "step_seconds",
    "seconds_per_study",
    "checkpoint",
)


class NoOpRun:
    """Stand-in used when tracking is unavailable. Records nothing, breaks nothing.

    Carries the attributes callers actually touch -- ``summary``, ``config``,
    ``dir`` -- so a call site written against a real run needs no ``if run is
    not None`` around every line. ``offline`` is True because "not logged
    anywhere reachable" is the honest reading.
    """

    name = "untracked"
    id = "untracked"
    url = None
    offline = True
    group = None
    job_type = None

    def __init__(self) -> None:
        # Instance attributes, not class attributes: a shared mutable dict would
        # leak one no-op run's summary into the next one's.
        self.summary: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.dir: str | None = None
        self.tags: tuple[str, ...] = ()

    def log(self, *_args: Any, **_kwargs: Any) -> None: ...
    def summary_update(self, *_args: Any, **_kwargs: Any) -> None: ...
    def log_artifact(self, *_args: Any, **_kwargs: Any) -> None: ...
    def use_artifact(self, *_args: Any, **_kwargs: Any) -> None: ...
    def finish(self, *_args: Any, **_kwargs: Any) -> None: ...


def git_sha(repo: Path | None = None) -> str:
    """Current commit, with ``-dirty`` appended when the tree has changes.

    Returns ``"unknown"`` rather than raising: on the Kaggle runtime there is no
    git checkout at all, and losing a run over a missing SHA would be absurd.
    """
    try:
        from kappa.data.paths import repo_root

        cwd = repo or repo_root()
    except Exception:
        cwd = repo or Path.cwd()
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def file_digest(path: Path | str | None, *, length: int = 12) -> str:
    """Short sha256 of a file's bytes; ``"unset"`` or ``"missing"`` when there is none.

    Used for the label table a run trained against. Two runs quoting different
    macros with the same config hash and *different* ``label_source_hash`` are
    not a contradiction -- they trained on different labels -- and that is only
    visible if the hash is recorded. Never raises: a run must not die because a
    path it is merely describing has moved.
    """
    if path is None:
        return "unset"
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:length]
    except OSError:
        return "missing"


def _is_scalar(value: Any) -> bool:
    """True for the value types W&B can filter, group and sort on."""
    return value is None or isinstance(value, (bool, int, float, str))


def flatten_config(config: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config into dotted keys whose values are all scalars.

    W&B filters, groups and sorts on config **keys**, so a nested mapping logged
    as one object is unqueryable -- visible in the UI and useless in the runs
    table. ``{"model": {"backbone": "convnext_small"}}`` becomes
    ``{"model.backbone": "convnext_small"}``, which is a field you can filter a
    hundred runs by.

    Sequences become comma-joined strings rather than lists, for the same
    reason: ``data.slots == "Axial_fs1,Coronal_fs1"`` is a filterable value and
    ``["Axial_fs1", "Coronal_fs1"]`` is not. Pydantic models are dumped in JSON
    mode first, so a ``TrainConfig`` can be passed straight in and its ``Path``
    fields arrive as strings.

        flatten_config({"optim": {"lr": 5e-5}, "data": {"slots": ("Axial_fs1",)}})
        {'optim.lr': 5e-05, 'data.slots': 'Axial_fs1'}
    """
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        config = dump(mode="json")
    if isinstance(config, Mapping):
        flat: dict[str, Any] = {}
        for key, value in config.items():
            flat.update(flatten_config(value, prefix=f"{prefix}{key}."))
        return flat
    key = prefix.rstrip(".")
    if not key:
        raise ValueError(f"flatten_config needs a mapping at the top level, got {type(config)}")
    if isinstance(config, Path):
        return {key: str(config)}
    if isinstance(config, (list, tuple, set, frozenset)):
        items = sorted(config) if isinstance(config, (set, frozenset)) else list(config)
        if all(_is_scalar(item) for item in items):
            return {key: ",".join("" if item is None else str(item) for item in items)}
        # A list of mappings has no dotted spelling that stays filterable, so it
        # is recorded as JSON and named as such. Better an honest opaque field
        # than a key that looks queryable and is not.
        return {key: json.dumps(_jsonable(items), sort_keys=True)}
    if _is_scalar(config):
        return {key: config}
    return {key: str(config)}


def _jsonable(value: Any) -> Any:
    """Recursively convert to JSON-serialisable containers, stringifying the rest."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if _is_scalar(value):
        return value
    return str(value)


def config_digest(config: Any, *, length: int = 12) -> str:
    """Stable short digest of a flattened config, independent of key order.

    For the generic runs -- extraction, discovery -- that have no
    :class:`kappa.config.TrainConfig`. Training runs use
    :func:`kappa.config.config_hash` instead, which excludes ``seed`` so that a
    seed sweep shares one group; this one covers **everything it is given**,
    seed included, so two runs with the same digest ran the same thing.
    """
    flat = flatten_config(config)
    payload = json.dumps(flat, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _on_compute_node() -> bool:
    """True inside a Slurm allocation, where there is no outbound internet."""
    return bool(os.environ.get("SLURM_JOB_ID"))


def _resolve_mode(mode: str | None) -> str:
    """Explicit argument, then Slurm (forced offline), then ``WANDB_MODE``, then online.

    Slurm outranks ``WANDB_MODE`` deliberately. A compute node has no outbound
    internet, so an inherited ``WANDB_MODE=online`` there does not produce an
    online run -- it produces a job that stalls on a network call and then logs
    nothing at all. ``WANDB_MODE`` still decides on the login node and on a
    laptop, which is what makes ``WANDB_MODE=offline`` usable for a rehearsal.
    """
    if mode:
        return str(mode)
    if _on_compute_node():
        return "offline"
    return os.environ.get("WANDB_MODE") or "online"


def environment_entities() -> dict[str, Any]:
    """Where the run physically happened: node, partition, Slurm ids, venv, torch.

    ``torch`` is imported lazily and its absence recorded rather than raised.
    This module is imported by ``scripts/train.py`` before the ``--dry-run``
    branch that exists precisely so validating a config does not require torch.

    The CUDA line is not decoration. Under an older GPU driver (570.86.15 was
    observed), the cu130 wheel imports fine and reports
    ``cuda.is_available() == False`` -- a run that trains on CPU for eight hours
    and still reports a macro. ``cuda_available``
    in the config is how that shows up in the runs table instead of in a log
    nobody reopened.
    """
    entities: dict[str, Any] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": socket.gethostname(),
        "env": sys.prefix,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
    }
    try:
        import torch

        entities["torch_version"] = torch.__version__
        entities["cuda_version"] = torch.version.cuda
        entities["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        entities["torch_version"] = "absent"
        entities["cuda_version"] = None
        entities["cuda_available"] = False
    return entities


def identity_entities(*, config_hash: str | None = None) -> dict[str, Any]:
    """The fields that make a run identifiable, independent of what it trained.

    Always present, on every run type, so that a filter written once works
    across extraction, discovery and training runs alike.
    """
    entities = {"git_sha": git_sha()}
    if config_hash is not None:
        entities["config_hash"] = config_hash
    entities.update(environment_entities())
    return entities


def run_name(config_name: str, *, fold: int | None, seed: int) -> str:
    """``{config_name}-f{fold:02d}-s{seed:02d}``; ``fall`` when folds are not split out.

    Deterministic and sortable, which is the whole requirement: the runs table
    sorts lexically, so ``f10`` must not land between ``f01`` and ``f02``. A
    ``fold`` of ``None`` means one job trained several folds and is spelled
    ``fall`` rather than omitted, so a name is never ambiguous about it.
    """
    fold_part = "fall" if fold is None else f"f{int(fold):02d}"
    return f"{config_name}-{fold_part}-s{int(seed):02d}"


def run_group(config_name: str, config_hash: str) -> str:
    """``{config_name}-{config_hash[:8]}``: the experiment every fold and seed joins.

    Keyed on the hash rather than the name alone because a config file gets
    edited in place. Two runs of ``baseline_axial_fs1`` a week apart with a
    different ``n_slices`` are two experiments, and grouping them by name would
    average them into one meaningless line.
    """
    return f"{config_name}-{config_hash[:8]}"


def training_entities(
    cfg: "TrainConfig",
    *,
    fold: int | None,
    seed: int,
    config_hash: str | None = None,
    n_train: int | None = None,
    n_val: int | None = None,
) -> dict[str, Any]:
    """Every filterable field for a training run: flattened config plus identity.

    The *resolved* config, dumped from the validated object rather than read
    back from the YAML file, so ``--set optim.epochs=2`` is in the logged
    configuration instead of being invisible.

    ``label_source`` and ``label_source_hash`` state which ``derived.parquet``
    the run trained on. Only 58 of 4,407 train studies carry the twelve
    competition labels, so every other row's target came from a derivation whose
    output changes when its rules change; a macro quoted without saying which
    label table produced it cannot be compared to anything.
    """
    from kappa.config import config_hash as hash_config

    digest = hash_config(cfg) if config_hash is None else config_hash
    entities: dict[str, Any] = flatten_config(cfg)
    entities.update(
        {
            "config_name": cfg.name,
            "fold": fold,
            "n_folds": cfg.folds.n_folds,
            "fold_column": cfg.folds.column,
            "fold_file": None if cfg.folds.file is None else str(cfg.folds.file),
            "seed": seed,
            "backbone": cfg.model.backbone,
            "label_source": None if cfg.data.labels is None else str(cfg.data.labels),
            "label_source_hash": file_digest(cfg.data.labels),
            "manifest_source": None if cfg.data.manifest is None else str(cfg.data.manifest),
            "n_train": n_train,
            "n_val": n_val,
        }
    )
    entities.update(identity_entities(config_hash=digest))
    return entities


def init_run(
    name: str,
    *,
    job_type: str,
    config: dict | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
    group: str | None = None,
    config_hash: str | None = None,
    notes: str | None = None,
):
    """Start a W&B run, or return a :class:`NoOpRun` if tracking is unavailable.

    ``config`` is flattened into dotted keys and joined with
    :func:`identity_entities`, so no caller has to remember to log the git SHA.
    When ``config_hash`` is not given it is computed with :func:`config_digest`
    over the flattened config.

    ``mode`` defaults to ``offline`` inside a Slurm job -- compute nodes have no
    outbound internet -- and otherwise to ``WANDB_MODE`` or ``online``. Set
    ``KAPPA_NO_TRACKING=1`` to disable entirely, which is what tests should do.
    """
    if os.environ.get("KAPPA_NO_TRACKING") == "1":
        return NoOpRun()
    try:
        import wandb
    except ImportError:
        print("note: wandb not installed; running untracked")
        return NoOpRun()
    if not os.environ.get("WANDB_API_KEY"):
        # Offline logging does not technically need the key, but syncing does,
        # and a directory on scratch that nobody can push is the failure this
        # module exists to prevent. Fail visibly and early instead.
        print("note: WANDB_API_KEY unset; running untracked (set it in .env)")
        return NoOpRun()

    resolved = _resolve_mode(mode)
    flat = flatten_config(config or {})
    full_config = dict(flat)
    full_config.update(identity_entities(config_hash=config_hash or config_digest(flat)))

    try:
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY),
            project=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
            name=name,
            job_type=job_type,
            group=group,
            config=full_config,
            tags=tags,
            notes=notes,
            mode=resolved,
        )
    except Exception as exc:
        print(f"note: wandb.init failed ({exc}); running untracked")
        return NoOpRun()
    print(f"wandb      : {name}  group={group}  job_type={job_type}  mode={resolved}")
    directory = offline_directory(run)
    if directory is not None:
        print(f"wandb dir  : {directory}")
    return run


def start_training_run(
    cfg: "TrainConfig",
    *,
    fold: int | None,
    seed: int,
    job_type: str = "train",
    n_train: int | None = None,
    n_val: int | None = None,
    tags: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
    mode: str | None = None,
):
    """The one entry point for a training run. Name, group and entities by convention.

    Everything the convention promises happens here and nowhere else, so a future
    fold, seed sweep or ablation inherits it by construction rather than by
    somebody remembering: the name from :func:`run_name`, the group from
    :func:`run_group`, the flattened resolved config from
    :func:`training_entities`, and a printed banner naming the backbone and the
    config that actually loaded. That last one is a rule, not a nicety -- a run
    that does not print which backbone it loaded is not evidence about that
    backbone -- and it goes to stdout *and* into the run, because the log gets
    rotated and the run does not.
    """
    from kappa.config import config_hash as hash_config

    digest = hash_config(cfg)
    entities = training_entities(
        cfg, fold=fold, seed=seed, config_hash=digest, n_train=n_train, n_val=n_val
    )
    if extra:
        entities.update(flatten_config(dict(extra)))
    name = run_name(cfg.name, fold=fold, seed=seed)
    group = run_group(cfg.name, digest)
    print(
        f"tracking   : run={name} group={group} job_type={job_type}\n"
        f"             backbone={cfg.model.backbone} config={cfg.name} hash={digest} "
        f"labels={entities['label_source_hash']}"
    )
    return init_run(
        name,
        job_type=job_type,
        config=entities,
        tags=[*dict.fromkeys([job_type, cfg.model.backbone, *tags])],
        mode=mode,
        group=group,
        config_hash=digest,
        notes=f"{cfg.name} fold={fold} seed={seed} hash={digest}",
    )


def summarise(run: Any, metrics: Mapping[str, Any]) -> None:
    """Write headline numbers into ``run.summary``, not only into the step history.

    The runs table sorts and filters on the **summary**, so a macro that exists
    only as the last point of a logged series is invisible in the one view
    anybody uses to compare a hundred runs. Both are useful; only one is
    sortable.
    """
    if run is None:
        return
    summary = getattr(run, "summary", None)
    if summary is None:
        return
    try:
        summary.update(dict(metrics))
    except Exception as exc:  # pragma: no cover - defensive, wandb-side failure
        print(f"note: could not update the run summary ({exc})")


def label_metrics(
    auc: "AucResult",
    *,
    prefix: str = "val",
    summary: "FoldSummary | None" = None,
) -> dict[str, Any]:
    """Flat metrics for all twelve labels, the macro, and what was not scored.

    A skipped label -- one class present in the split, so no defined AUC --
    appears as ``nan`` under ``{prefix}/auc/<label>`` **and** as ``True`` under
    ``{prefix}/auc_skipped/<label>``, plus by name in ``{prefix}/skipped``. It is
    never a silently missing column: averaging eleven labels and calling the
    result macro-12 is how a number becomes a lie, and the only defence is that
    the denominator is logged next to the mean.

    Label names are used verbatim, apostrophe and spaces included. ``Baker's``
    and four two-word labels come from :data:`kappa.constants.LABELS`, and a
    slugified metric key would be one more spelling of the twelve that has to be
    kept in step with the submission header.
    """
    metrics: dict[str, Any] = {
        f"{prefix}/macro": auc.macro,
        f"{prefix}/n_scored": auc.n_scored,
        f"{prefix}/n_skipped": len(auc.skipped),
        f"{prefix}/skipped": ",".join(auc.skipped) if auc.skipped else "",
        f"{prefix}/is_complete": bool(auc.is_complete),
    }
    for label, value in auc.per_label.items():
        used = int(auc.n_used.get(label, 0))
        positives = int(auc.n_pos.get(label, 0))
        metrics[f"{prefix}/auc/{label}"] = float(value)
        metrics[f"{prefix}/auc_skipped/{label}"] = label in auc.skipped
        metrics[f"{prefix}/n_used/{label}"] = used
        metrics[f"{prefix}/n_pos/{label}"] = positives
        metrics[f"{prefix}/prevalence/{label}"] = (
            positives / used if used else float("nan")
        )
    if summary is not None:
        metrics.update(summary_metrics(summary, prefix=prefix))
    return metrics


def summary_metrics(summary: "FoldSummary", *, prefix: str = "val") -> dict[str, Any]:
    """Across-fold mean and **standard deviation for all twelve labels**, plus the macro.

    The standard deviation is not optional decoration here. A single fold's macro
    is one draw from a distribution whose width is of the same order as the whole
    gap between a working solution and a top-10 place on this task, so a mean
    logged without its spread invites reading noise as progress. It is ``nan``
    for a single fold rather than 0.0, because one observation has no spread and
    0.0 would claim certainty.
    """
    metrics: dict[str, Any] = {
        f"{prefix}/macro_mean": summary.macro_mean,
        f"{prefix}/macro_std": summary.macro_std,
        f"{prefix}/n_folds": summary.n_folds,
    }
    for label in summary.per_label_mean:
        metrics[f"{prefix}/auc_mean/{label}"] = summary.per_label_mean[label]
        metrics[f"{prefix}/auc_std/{label}"] = summary.per_label_std[label]
        metrics[f"{prefix}/fold_skips/{label}"] = summary.fold_skips[label]
    return metrics


def diagnostic_label_metrics(result: "DiagnosticResult", *, prefix: str = "val") -> dict[str, Any]:
    """Flat average precision / precision / recall / F1 metrics, plus their macros.

    **Diagnostic only, and it says so in the numbers it logs, not just in
    prose.** Nothing this returns may choose a checkpoint, a fold or a model --
    see :mod:`kappa.evaluation.metrics` and :class:`DiagnosticResult`. The
    competition is macro ROC-AUC; these exist so a weak rare label like
    Fracture is visible during a run instead of after the leaderboard, because
    ROC-AUC does not read the positive rate and average precision does.

    Emits, per label: ``{prefix}/ap/<label>``; ``{prefix}/precision/<label>``,
    ``{prefix}/recall/<label>``, ``{prefix}/f1/<label>`` at
    :data:`kappa.evaluation.metrics.FIXED_THRESHOLD`; and
    ``{prefix}/f1_best/<label>`` with ``{prefix}/threshold_best/<label>`` --
    the best F1 found on this split and the threshold that found it, which is
    exactly the pair that must never be carried forward, because a threshold
    picked on a validation fold fits that fold. Then four macros:
    ``{prefix}/ap_macro``, ``{prefix}/precision_macro``, ``{prefix}/recall_macro``,
    ``{prefix}/f1_macro``, each the mean :func:`kappa.evaluation.metrics.diagnostic_metrics`
    already restricted to the labels ``result.skipped`` does not name. Label
    names are used verbatim, apostrophe and spaces included, straight from
    ``result.ap``'s keys -- never rebuilt or slugified.
    """
    metrics: dict[str, Any] = {}
    for label in result.ap:
        metrics[f"{prefix}/ap/{label}"] = float(result.ap[label])
        metrics[f"{prefix}/precision/{label}"] = float(result.precision[label])
        metrics[f"{prefix}/recall/{label}"] = float(result.recall[label])
        metrics[f"{prefix}/f1/{label}"] = float(result.f1[label])
        metrics[f"{prefix}/f1_best/{label}"] = float(result.f1_best[label])
        metrics[f"{prefix}/threshold_best/{label}"] = float(result.threshold_best[label])
    metrics[f"{prefix}/ap_macro"] = float(result.ap_macro)
    metrics[f"{prefix}/precision_macro"] = float(result.precision_macro)
    metrics[f"{prefix}/recall_macro"] = float(result.recall_macro)
    metrics[f"{prefix}/f1_macro"] = float(result.f1_macro)
    return metrics


def fold_prefix(fold: int) -> str:
    """``fold00``, ``fold01``: the metric namespace for one fold, zero-padded to sort."""
    return f"fold{int(fold):02d}"


def epoch_metrics(
    auc: "AucResult",
    *,
    fold: int,
    epoch: int,
    train_loss: float,
    lr: float,
    train_seconds: float,
    val_seconds: float,
    step_seconds: float,
) -> dict[str, Any]:
    """Per-epoch step metrics for one fold, under the ``fold{NN}/`` namespace.

    The per-label AUC is here as well as in the end-of-fold summary, because a
    label whose AUC is a flat 0.5 for ten epochs and a label that peaks at epoch
    three and decays are the same final number and different problems. What is
    *not* repeated per epoch is the support -- ``n_used`` and ``n_pos`` are
    properties of the split, not of the epoch, and logging them ten times says
    nothing new.

    ``epoch`` is also logged unprefixed so it can serve as the x-axis for every
    fold's series in one chart.
    """
    prefix = fold_prefix(fold)
    metrics: dict[str, Any] = {
        "epoch": epoch,
        f"{prefix}/epoch": epoch,
        f"{prefix}/train_loss": float(train_loss),
        f"{prefix}/macro": float(auc.macro),
        f"{prefix}/n_scored": auc.n_scored,
        f"{prefix}/n_skipped": len(auc.skipped),
        f"{prefix}/lr": float(lr),
        f"{prefix}/train_seconds": float(train_seconds),
        f"{prefix}/val_seconds": float(val_seconds),
        f"{prefix}/step_seconds": float(step_seconds),
    }
    for label, value in auc.per_label.items():
        metrics[f"{prefix}/auc/{label}"] = float(value)
    return metrics


def step_metrics(
    *, fold: int, step: int, loss: float, lr: float, parts: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Per-optimiser-step loss and learning rate, under the ``fold{NN}/`` namespace.

    Why this is separate from :func:`epoch_metrics`: that one fires once per epoch,
    and the baseline's epoch budget is ten. **A ten-point loss curve cannot show a
    diverging run, a bad warmup, or a loss that spiked and recovered** -- all three
    look like one monotone line, and the first two are the ones worth catching in
    the first minutes rather than at the end of the walltime.

    ``step`` is the optimiser step, not the batch index, so the x-axis matches what
    the scheduler saw under gradient accumulation. ``parts`` carries the
    distillation split when there is one and is empty otherwise.
    """
    prefix = fold_prefix(fold)
    metrics: dict[str, Any] = {
        f"{prefix}/step": int(step),
        f"{prefix}/step_loss": float(loss),
        f"{prefix}/step_lr": float(lr),
    }
    for key, value in (parts or {}).items():
        metrics[f"{prefix}/step_{key}"] = float(value)
    return metrics


def per_label_table(auc: "AucResult", *, summary: "FoldSummary | None" = None) -> Any | None:
    """Twelve rows, always, as a ``wandb.Table``. Returns ``None`` without wandb.

    This is the artifact a human actually reads, so every label gets a row
    whether or not it could be scored, with its own denominator on the same line
    -- a positive rate without its support is a claim about a sample size it does
    not have. There is no row cap and there must never be one: an earlier table
    in this project was truncated, and the fix was two tables with every row
    present.
    """
    try:
        import wandb
    except ImportError:
        return None
    rows = []
    for label, value in auc.per_label.items():
        used = int(auc.n_used.get(label, 0))
        positives = int(auc.n_pos.get(label, 0))
        rows.append(
            [
                label,
                float(value),
                label not in auc.skipped,
                used,
                positives,
                positives / used if used else float("nan"),
                float("nan") if summary is None else summary.per_label_std.get(label, float("nan")),
                0 if summary is None else int(summary.fold_skips.get(label, 0)),
            ]
        )
    return wandb.Table(columns=list(LABEL_TABLE_COLUMNS), data=rows)


def per_label_summary_table(summary: "FoldSummary") -> Any | None:
    """Twelve rows of across-fold mean, spread and skip count. ``None`` without wandb.

    ``folds_skipped`` is the load-bearing column: a label whose mean rests on one
    of five folds and one that rests on all five are two different claims, and
    without the count they render as the same number.
    """
    try:
        import wandb
    except ImportError:
        return None
    rows = []
    for label, mean in summary.per_label_mean.items():
        skips = int(summary.fold_skips.get(label, 0))
        rows.append(
            [
                label,
                float(mean),
                float(summary.per_label_std.get(label, float("nan"))),
                summary.n_folds - skips,
                skips,
                summary.n_folds,
            ]
        )
    return wandb.Table(columns=list(LABEL_SUMMARY_COLUMNS), data=rows)


def per_fold_table(rows: Sequence[Mapping[str, Any]]) -> Any | None:
    """One row per fold, as a ``wandb.Table``. Returns ``None`` without wandb.

    The second table. Per-label rows answer "which label is weak"; per-fold rows
    answer "is this a spread or a signal", which on this task is the prior
    question: the entire gap between working and top-10 is about 0.006 macro and
    a single fold's draw is of the same order.

    Missing keys are filled with ``None`` rather than raising, so a caller that
    has no checkpoint or no timing still produces a table with the pinned
    :data:`FOLD_TABLE_COLUMNS` shape.
    """
    try:
        import wandb
    except ImportError:
        return None
    data = [[row.get(column) for column in FOLD_TABLE_COLUMNS] for row in rows]
    return wandb.Table(columns=list(FOLD_TABLE_COLUMNS), data=data)


def _validated_usable(
    y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray | None, labels: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared shape check and usable-cell mask for the two curve panels below.

    Same contract as :func:`kappa.evaluation.metrics.diagnostic_metrics`: NaN in
    ``y_true`` is *unknown* and dropped, ``mask`` narrows further. Kept minimal
    on purpose -- these two functions build a picture, not a score, so they
    validate shape and nothing else.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 2 or y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true and y_score must both be (n_studies, n_labels); "
            f"got {y_true.shape} and {y_score.shape}"
        )
    if y_true.shape[1] != len(labels):
        raise ValueError(f"y_true has {y_true.shape[1]} columns, expected {len(labels)}")
    usable = ~np.isnan(y_true)
    if mask is not None:
        usable &= np.asarray(mask).astype(bool)
    return y_true, y_score, usable


def roc_curve_panel(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    labels: Sequence[str] = LABELS,
) -> Any | None:
    """One ROC-curve panel overlaying every scoreable label. ``None`` without wandb.

    **Diagnostic and visual only**, like everything else this module logs from
    :mod:`kappa.evaluation.metrics` -- a picture invites eyeballing a threshold
    exactly as a number does, and neither may choose a checkpoint, a fold or a
    model. A label with one class present in the cells used has no ROC curve
    and is left out of the panel rather than plotted as a single point, the
    same skip :func:`kappa.evaluation.metrics.macro_roc_auc` already applies.
    """
    try:
        import wandb
    except ImportError:
        return None
    from kappa.evaluation.metrics import roc_curve_points

    y_true, y_score, usable = _validated_usable(y_true, y_score, mask, labels)
    xs, ys, keys = [], [], []
    for column, name in enumerate(labels):
        keep = usable[:, column]
        fpr, tpr = roc_curve_points(y_true[keep, column], y_score[keep, column])
        if fpr.size == 0:
            continue
        xs.append(fpr.tolist())
        ys.append(tpr.tolist())
        keys.append(name)
    if not keys:
        return None
    return wandb.plot.line_series(
        xs, ys, keys=keys, title="ROC curve (diagnostic only)", xname="false positive rate"
    )


def pr_curve_panel(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    labels: Sequence[str] = LABELS,
) -> Any | None:
    """One precision-recall panel overlaying every scoreable label. ``None`` without wandb.

    **Diagnostic and visual only** -- see :func:`roc_curve_panel`. A label with
    zero positives in the cells used has no precision-recall curve and is left
    out, the same skip :func:`kappa.evaluation.metrics.diagnostic_metrics`
    already applies; unlike the ROC panel, a label with zero negatives still
    has a curve (a trivial one) and is kept.
    """
    try:
        import wandb
    except ImportError:
        return None
    from kappa.evaluation.metrics import pr_curve_points

    y_true, y_score, usable = _validated_usable(y_true, y_score, mask, labels)
    xs, ys, keys = [], [], []
    for column, name in enumerate(labels):
        keep = usable[:, column]
        recall, precision = pr_curve_points(y_true[keep, column], y_score[keep, column])
        if recall.size == 0:
            continue
        xs.append(recall.tolist())
        ys.append(precision.tolist())
        keys.append(name)
    if not keys:
        return None
    return wandb.plot.line_series(
        xs, ys, keys=keys, title="Precision-recall curve (diagnostic only)", xname="recall"
    )


_ARTIFACT_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _safe_name(text: str) -> str:
    """W&B artifact names accept only ``[A-Za-z0-9_.-]``; everything else joins with ``-``."""
    cleaned = _ARTIFACT_SAFE.sub("-", str(text)).strip("-")
    return cleaned or "artifact"


def log_artifact(
    run: Any,
    path: Path | str,
    *,
    artifact_type: str,
    name: str | None = None,
    aliases: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Attach a file or directory to the run as a typed, aliased artifact.

    Returns whether it was logged. Never raises: an artifact is a convenience
    and an experiment must not die because one could not be attached. A missing
    path is reported and skipped rather than logged as an empty artifact, which
    would be worse -- an artifact that exists and contains nothing looks like a
    successful upload.

    Artifacts logged offline are staged in the run directory and travel with
    ``wandb sync``. ``run.use_artifact`` of a *remote* artifact does not, since
    resolving a name against the server needs the network, which is why an input
    table is recorded as ``label_source_hash`` in the config instead.
    """
    if run is None or isinstance(run, NoOpRun):
        return False
    target = Path(path)
    if not target.exists():
        print(f"note: no artifact at {target}; not logged")
        return False
    try:
        import wandb
    except ImportError:  # pragma: no cover - unreachable once a real run exists
        return False
    label = _safe_name(name or f"{artifact_type}-{target.stem}")
    try:
        artifact = wandb.Artifact(label, type=artifact_type, metadata=dict(metadata or {}))
        if target.is_dir():
            artifact.add_dir(str(target))
        else:
            artifact.add_file(str(target))
        run.log_artifact(artifact, aliases=list(aliases) or None)
    except Exception as exc:
        print(f"note: could not log artifact {label} ({exc})")
        return False
    return True


def log_standard_artifacts(
    run: Any,
    *,
    cfg: "TrainConfig | None" = None,
    entities: Mapping[str, Any] | None = None,
    oof: Path | str | None = None,
    submission: Path | str | None = None,
    fold_file: Path | str | None = None,
    lockfile: Path | str | None = None,
    weights: Sequence[Path | str] = (),
    aliases: Sequence[str] = (),
) -> tuple[str, ...]:
    """Log the set of artifacts that make a run reproducible from the UI.

    Returns the artifact types actually logged, so a caller can print what it
    attached rather than assuming.

    The resolved config is written into the run directory as JSON in **both**
    shapes: nested, which is what you would re-run from, and flat, which is what
    the W&B table filters on. Weights are skipped unless
    ``KAPPA_WANDB_LOG_WEIGHTS=1`` -- see the module docstring -- and their paths
    are recorded in the summary either way, because the policy is that
    checkpoints live in ``$KAPPA_WEIGHTS_DIR`` and a run should say where, not
    duplicate gigabytes into an artifact store.
    """
    if run is None or isinstance(run, NoOpRun):
        return ()
    logged: list[str] = []
    directory = getattr(run, "dir", None)

    if cfg is not None and directory:
        payload = {
            "resolved": _jsonable(cfg),
            "flat": dict(entities) if entities is not None else flatten_config(cfg),
        }
        config_path = Path(directory) / "resolved_config.json"
        try:
            config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            print(f"note: could not write {config_path} ({exc})")
        else:
            if log_artifact(
                run, config_path, artifact_type="config",
                name=f"config-{cfg.name}", aliases=aliases,
            ):
                logged.append("config")

    for value, kind in ((oof, "oof"), (submission, "submission"), (fold_file, "folds")):
        if value is not None and log_artifact(run, value, artifact_type=kind, aliases=aliases):
            logged.append(kind)

    lock = Path(lockfile) if lockfile is not None else LOCKFILE
    if not lock.is_absolute():
        try:
            from kappa.data.paths import repo_root

            lock = repo_root() / lock
        except Exception:
            lock = Path.cwd() / lock
    if lock.exists() and log_artifact(run, lock, artifact_type="lockfile", aliases=aliases):
        logged.append("lockfile")

    paths = [str(item) for item in weights]
    if paths:
        summarise(run, {"weights_paths": ",".join(paths)})
    if paths and os.environ.get(WEIGHTS_OPT_IN) == "1":
        for item in weights:
            if log_artifact(run, item, artifact_type="weights", aliases=aliases):
                logged.append("weights")
    elif paths:
        print(
            f"note: {len(paths)} checkpoint(s) not uploaded; paths recorded in the summary. "
            f"Set {WEIGHTS_OPT_IN}=1 to upload them."
        )
    return tuple(logged)


def offline_directory(run: Any) -> Path | None:
    """The run's own directory on disk, or ``None`` when there is nothing to sync.

    ``run.dir`` points at the ``files/`` subdirectory; ``wandb sync`` wants its
    parent, which is the ``offline-run-*`` directory itself.
    """
    if run is None or isinstance(run, NoOpRun):
        return None
    directory = getattr(run, "dir", None)
    if not directory:
        return None
    return Path(directory).parent


def require_run_directory(run: Any, *, tracked: bool) -> Path:
    """The run's directory, or a raise naming the one flag that explains its absence.

    ``tracked`` is the caller's INTENT -- it passed ``--track`` and therefore expects
    a syncable run. When intent and reality disagree this raises, because the
    alternative is silent: ``--track`` is ``action="store_true"`` in
    ``scripts/train.py``, so a job submitted without it builds a :class:`NoOpRun`,
    writes **no** ``offline-run-*`` directory, and trains perfectly. On a compute
    node with no outbound internet there is then nothing to sync afterwards and no
    way to backfill from the run side -- and ``sync_wandb.sh`` reports success,
    because an empty set syncs cleanly.

    The check is deliberately about the DIRECTORY and not about ``run.offline`` or
    ``run.id``: a :class:`NoOpRun` carries a plausible ``id`` and an honest
    ``offline = True``, and neither distinguishes it from a real offline run. Only the
    directory does.
    """
    directory = offline_directory(run)
    if not tracked:
        # Untracked on purpose. Nothing to assert; nothing will be syncable.
        return directory  # type: ignore[return-value]
    if directory is None:
        raise RuntimeError(
            "tracking was requested but this run has no directory on disk, so there "
            "is nothing for `wandb sync` to push and the run cannot be recovered "
            "later -- a compute node has no outbound internet. Cause: "
            f"{type(run).__name__} was built instead of a real run. Check that "
            "--track reached scripts/train.py (it is action=\"store_true\", so any "
            "wrapper script must forward its arguments), and that WANDB_DIR is "
            "writable."
        )
    return directory


def sync_hint(run: Any) -> str:
    """The exact command that pushes this run, or a line saying none is needed.

    Printed at finish because a run logged offline and never synced exists only as
    a directory on scratch. A run the operator cannot find is a run that will not be
    synced.
    """
    directory = offline_directory(run)
    if directory is None:
        return "wandb sync: nothing to sync (untracked run)"
    if not directory.name.startswith("offline-run"):
        return f"wandb sync: not needed, this run logged online ({directory})"
    return (
        f"wandb offline dir: {directory}\n"
        f"sync it from the LOGIN node (compute nodes have no internet):\n"
        f"  slurm/sync_wandb.sh          # syncs every offline run under $WANDB_DIR\n"
        f"  wandb sync {directory}"
    )


def install_walltime_guard(run: Any, *, signals: Sequence[int] | None = None) -> None:
    """Commit the run to disk when Slurm signals the walltime, instead of losing it.

    **This is not a wandb setting.** The offline transaction log ``run-*.wandb`` is
    block-buffered at 32 KiB with no time-based flush, and the only thing that
    commits the buffer is ``run.finish()``. Measured against the real metric shape
    (12 labels through ``epoch_metrics``): the first flush lands after **22 epochs**. The baseline config
    trains 10. So a job killed at the 8-hour walltime leaves a **7-byte** file --
    the ``:W&B`` magic header and nothing else -- having logged everything and kept
    none of it.

    What makes it dangerous rather than merely annoying is that the empty directory
    still matches ``sync_wandb.sh``'s ``offline-run-*`` glob and still reports as
    syncable, so the loss uploads as a run that looks like it happened.

    Pair this with ``#SBATCH --signal=B:USR1@300`` in the sbatch, which asks Slurm for
    SIGUSR1 five minutes before the kill. SIGTERM is handled too because that is what
    ``scancel`` sends. The handler re-raises with the default disposition afterwards,
    so the exit status still records that the job was signalled.
    """
    if run is None or isinstance(run, NoOpRun):
        return
    for number in signals or (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(number, _commit_and_die)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass
    _GUARDED_RUNS.append(run)


#: Runs to commit if a signal arrives. A list rather than a single slot because
#: --noise-floor trains the same config twice in one process.
_GUARDED_RUNS: list[Any] = []


def _commit_and_die(number: int, _frame: Any) -> None:  # pragma: no cover - signal path
    name = signal.Signals(number).name
    print(f"\n*** {name} received -- committing the wandb run before exit ***", flush=True)
    while _GUARDED_RUNS:
        run = _GUARDED_RUNS.pop()
        try:
            summarise(run, {"walltime_signal": name, "finished_cleanly": False})
            finish_run(run)
        except Exception as exc:
            print(f"note: could not commit run ({exc})", flush=True)
    signal.signal(number, signal.SIG_DFL)
    os.kill(os.getpid(), number)


def finish_run(run: Any, *, metrics: Mapping[str, Any] | None = None) -> None:
    """Close the run, after writing any last summary numbers and printing the sync line.

    Always call this rather than ``run.finish()``: the sync hint is the half of
    finishing that a human needs, and it is the half that gets forgotten.
    """
    if run is None:
        return
    if metrics:
        summarise(run, metrics)
    print(sync_hint(run))
    try:
        run.finish()
    except Exception as exc:  # pragma: no cover - defensive, wandb-side failure
        print(f"note: wandb run.finish failed ({exc})")
