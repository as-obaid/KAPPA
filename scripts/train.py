#!/usr/bin/env python3
"""Train one config: all folds, one fold, or the same config under several seeds.

Argparse plus one call into kappa.training, per the repo contract: the loop lives in
the package so it is testable and importable, not here.

    python scripts/train.py --config configs/baseline_axial_fs1.yaml
    python scripts/train.py --config configs/baseline_axial_fs1.yaml --fold 0
    python scripts/train.py --config configs/baseline_axial_fs1.yaml --noise-floor
    python scripts/train.py --config configs/baseline_axial_fs1.yaml --dry-run
    python scripts/train.py --config configs/baseline_axial_fs1.yaml --synthetic 12 \
        --set optim.epochs=2 --set model.pretrained=false

With no ``--fold`` it reads ``SLURM_ARRAY_TASK_ID``, so the same command line works
standalone and as an array element -- and folds run as **parallel one-GPU jobs**,
never as ranks of one job. On a queue that caps a job at one GPU, a request over the
cap can stay pending forever rather than run slowly.

``--noise-floor`` is the mode to run first. The gap between a working solution and a
top-10 place on this task is about 0.006 macro; until the seed-to-seed spread of this
exact config has been measured locally, no comparison between two runs is readable.

``--synthetic N`` trains on a generated corpus from ``kappa.data.synthetic``. **The
labels mean nothing** -- it exists to prove the wiring on CPU before the real cache
and the derived label table are on disk, and it prints that warning at the top of the
run so no output of it can be mistaken for a result.

``--resume`` restarts an interrupted fold at its last epoch boundary instead of at
epoch 0, reading the ``resume_fold<k>_seed<s>.pt`` that ``--resume`` itself wrote on a
previous attempt. It is **off by default** and nothing is read or written without it,
because a queued task must train the fold it was submitted for and not whatever is on
disk beside it. A checkpoint belonging to a different ``config_hash``, git SHA,
``--select`` or epoch budget is refused out loud and the fold starts at epoch 0 --
``--select`` in particular is not in ``config_hash``, so continuing a ``last``-policy
run under ``best`` would produce a policy-mixed fold nothing downstream could detect.
See ``kappa.training.resume``.

``--track`` logs to W&B through ``kappa.tracking``, which owns the whole convention:
the run is named ``{config}-f{fold}-s{seed}`` inside the group
``{config}-{config_hash[:8]}``, every config value becomes its own dotted, filterable
field, and all twelve per-label AUCs plus the fold spread land in the run summary.
Inside a Slurm job it logs **offline** -- compute nodes have no outbound internet --
and prints the offline directory and the ``wandb sync`` command at finish. Checkpoints
are not uploaded unless ``--log-weights`` says so.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.config import ConfigError, config_hash, describe, load_config  # noqa: E402
from kappa.constants import ID_COLUMN  # noqa: E402
from kappa.tracking import WEIGHTS_OPT_IN, git_sha  # noqa: E402


def _slurm_fold() -> int | None:
    """This element's fold from the Slurm array environment, or ``None`` standalone.

    ``SLURM_ARRAY_TASK_MIN`` is subtracted rather than assumed to be zero, because
    ``--array=1-5`` is a common spelling and an unshifted id would train fold 5 (which
    does not exist in a 5-fold table) and never train fold 0.
    """
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        return None
    return int(task) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))


def _synthetic_bundle(root: Path, n_studies: int, cfg, seed: int):
    """Generate a corpus and a fold table for it, and return a ``DataBundle``.

    ``mask_rate=0.25`` rather than the realistic 0.987: at the measured default only
    1.32% of label cells are usable, so 28.1% of batches of 8 contain no usable cell at
    all and the loss has nothing to move. That is a real property of the dataset and
    the loss is built to survive it, but a wiring run is trying to prove the plumbing,
    not rediscover the sparsity.
    """
    from kappa.data.folds import build_folds
    from kappa.data.synthetic import make_synthetic_dataset
    from kappa.training.loop import DataBundle, slot_indices

    slots = slot_indices(cfg.data.slots)
    corpus = make_synthetic_dataset(
        root,
        n_studies=n_studies,
        seed=seed,
        patterns=(slots,),
        cache_size=64,
        slice_counts=(12, 20),
        mask_rate=0.25,
    )
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
    # strata_key=None because a synthetic corpus has no report and therefore no
    # language column. The real fold build stratifies on it -- 9 language buckets,
    # 39.4% English -- so this differs from the production call by exactly that,
    # and the difference is stated rather than hidden behind a default.
    folds = build_folds(
        study_table, n_folds=cfg.folds.n_folds, strata_key=None, seed=seed
    )
    return DataBundle(
        manifest=corpus.manifest,
        labels=corpus.labels,
        folds=folds,
        cache_dir=corpus.cache_dir,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="YAML config; unknown keys fail")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override applied before validation, e.g. optim.epochs=2")
    ap.add_argument("--fold", type=int, action="append", default=None,
                    help="fold(s) to train; defaults to SLURM_ARRAY_TASK_ID, then all")
    ap.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    ap.add_argument("--fold-column", default=None, choices=["fold", "fold_scanner"],
                    help="defaults to folds.column in the config")
    ap.add_argument("--select", default="last", choices=["last", "best"],
                    help="which epoch's predictions become the OOF rows. 'last' by "
                         "default: picking the best epoch by validation macro fits the "
                         "split the number is then quoted from")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <out_dir>/<name> from the config")
    ap.add_argument("--device", default=None, help="cuda, cpu; defaults to cuda when usable")
    ap.add_argument("--noise-floor", action="store_true",
                    help="train the identical config under every seed in `seeds` and "
                         "report the macro spread. Run this before reading any delta")
    ap.add_argument("--noise-floor-seeds", type=int, nargs="+", default=None,
                    help="seeds for --noise-floor; defaults to the config's `seeds`")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute backbone activations in the backward pass: trades "
                         "~35%% more compute for a large cut in activation memory. "
                         "Needed by the 24-slice config, which OOMs an H200 without it. "
                         "A CLI flag and not a config field, so it moves no config_hash")
    # Off by default, and spelled as two explicit flags rather than one
    # BooleanOptionalAction, so that an sbatch line reads `--resume` or `--no-resume`
    # and never a bare `--resume=false` that argparse would accept as truthy. The
    # default matters more than the spelling: a queued task must train the fold it was
    # submitted for. Without the flag `train_fold` reads no sidecar, writes no
    # checkpoint and does not so much as construct a ResumeIdentity. `--no-resume`
    # exists because the parity recipe in kappa/training/resume.py passes it to the
    # reference half explicitly, so that a stale file cannot be picked up by the run
    # the comparison is against.
    ap.add_argument("--resume", dest="resume", action="store_true", default=False,
                    help="restart an interrupted fold at its last epoch boundary, "
                         "reading <out-dir>/resume_fold<k>_seed<s>.pt. A CLI flag and "
                         "not a config field, so it moves no config_hash. A checkpoint "
                         "from a different config_hash, git SHA, --select or epoch "
                         "budget is refused and the fold starts at epoch 0")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="the default, stated explicitly: ignore any resume checkpoint "
                         "on disk and train the fold from epoch 0")
    ap.add_argument("--no-checkpoint", action="store_true", help="do not write weights")
    ap.add_argument("--no-oof", action="store_true", help="do not write the OOF parquet")
    ap.add_argument("--track", action="store_true",
                    help="log to W&B; offline inside a Slurm job, see kappa.tracking")
    ap.add_argument("--log-weights", action="store_true",
                    help="upload checkpoints as W&B artifacts. Off by default: weights "
                         "live in $KAPPA_WEIGHTS_DIR and a run records their path, so "
                         "the default keeps multi-GB files out of the artifact store")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config, its hash and the git SHA, then exit")
    ap.add_argument("--synthetic", type=int, default=None, metavar="N",
                    help="train on N generated studies instead of the real cache. The "
                         "labels are meaningless; this proves wiring, not accuracy")
    ap.add_argument("--synthetic-root", type=Path, default=None,
                    help="where to write the synthetic corpus (default: <out-dir>/synthetic)")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config, overrides=args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(describe(cfg))
    print(f"git sha    : {git_sha()}")
    if args.log_weights:
        # An environment variable rather than a threaded-through argument, so the
        # same opt-in works from an sbatch file without a second flag to forget.
        os.environ[WEIGHTS_OPT_IN] = "1"
        print(f"weights    : will be uploaded as W&B artifacts ({WEIGHTS_OPT_IN}=1)")
    if args.dry_run:
        print("dry run    : config validated, nothing trained")
        return 0

    folds = args.fold
    if folds is None:
        from_slurm = _slurm_fold()
        folds = [from_slurm] if from_slurm is not None else None
    if folds is not None:
        print(f"folds      : training {folds} (this job only; folds run as parallel jobs)")

    # Imported after --dry-run so that validating a config does not require torch.
    from kappa.data.dataset import Normalisation
    from kappa.training.loop import DataBundle, measure_noise_floor, run_experiment

    out_dir = args.out_dir or (Path(cfg.out_dir) / cfg.name)
    normalisation = None
    if args.synthetic:
        root = args.synthetic_root or (out_dir / "synthetic")
        print(
            f"\n*** SYNTHETIC: {args.synthetic} generated studies under {root}. The labels "
            "are random and mean nothing. This run proves wiring, not accuracy, and its "
            "macro must never be quoted as a result. ***\n"
        )
        bundle = _synthetic_bundle(root, int(args.synthetic), cfg, args.seed or cfg.seed)
        # A generated disc is not anatomy, so the backbone's ImageNet statistics have
        # nothing to be right about. Identity normalisation keeps the pixels in [0, 1]
        # and makes the wiring claim exact.
        normalisation = Normalisation.identity()
    else:
        try:
            bundle = DataBundle.from_config(cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: could not assemble the data bundle: {exc}", file=sys.stderr)
            print(
                "hint: build the cache, manifest and folds first, or pass --synthetic N "
                "to prove the wiring without them.",
                file=sys.stderr,
            )
            return 2

    options = dict(
        device=args.device,
        out_dir=out_dir,
        fold_column=args.fold_column,
        select=args.select,
        save_checkpoint=not args.no_checkpoint,
        write_oof=not args.no_oof,
        normalisation=normalisation,
        grad_checkpoint=args.grad_checkpoint,
        resume=args.resume,
    )

    if args.noise_floor:
        # save_checkpoint and write_oof are dropped rather than forwarded: the output
        # of this mode is one number, and writing a set of weights per seed to produce
        # it spends the disk the real runs need.
        floor_options = {
            key: value
            for key, value in options.items()
            if key not in ("save_checkpoint", "write_oof")
        }
        result = measure_noise_floor(
            cfg, bundle, seeds=args.noise_floor_seeds, folds=folds,
            track=args.track, **floor_options
        )
        print(f"\nconfig hash: {result.config_hash} (identical across every seed above)")
        return 0

    result = run_experiment(
        cfg, bundle, seed=args.seed, folds=folds, track=args.track, **options
    )
    print(f"\nconfig hash: {config_hash(cfg)}  git sha: {result.git_sha}  seed: {result.seed}")
    if len(result.folds) < 2:
        print(
            "note: one fold gives no spread. Run --noise-floor before reading this "
            "number against another run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
