#!/usr/bin/env python3
"""Score studies from one or more checkpoints and write ``submission.csv``.

Argparse plus one call into kappa.inference, per the repo contract.

    python scripts/predict.py --checkpoint data/runs/base/fold0_seed0.pt \
        --manifest data/interim/manifest_test.parquet \
        --cache-dir data/processed/cache_test --out submission.csv

    # five folds, blended by per-column rank normalisation
    python scripts/predict.py --checkpoint data/runs/base/fold*_seed0.pt ...

Everything here runs **offline and image-only**. Submissions carry no report text,
no weight is fetched -- the checkpoint holds its own state dict and
its own normalisation -- and every path is an argument, because the Kaggle notebook
imports ``src/kappa/`` and runs where none of this machine's paths exist.

Several ``--checkpoint`` arguments are combined with per-column rank normalisation.
Never by averaging probabilities: macro ROC-AUC reads only the order within a column,
so a probability average is decided by whichever member has the widest spread -- a
property this competition does not score.

The measured seconds per study is printed and projected against the 32,400 s notebook
limit, because the efficiency track's score carries a ``T / 32400`` term and a
full-length run contributes 1.0 against accuracy gaps two orders of magnitude smaller.

``--track`` files the scoring pass as its own W&B run (``job_type=predict``), with the
submission csv as an artifact and the **measured** seconds per study in the summary.
That number is the one the efficiency term consumes, so it belongs in the runs table
next to the accuracy, not only in a log. Tracking is optional by construction: without
``wandb`` or an API key the flag is a no-op and the submission is written regardless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.constants import ID_COLUMN  # noqa: E402


def _expected_uids(path: Path) -> list[str]:
    """Study UIDs in the order a sample submission or ``test.csv`` states them."""
    import pandas as pd

    table = pd.read_csv(path, dtype=str)
    for column in (ID_COLUMN, "StudyInstanceUID", "study_uid"):
        if column in table.columns:
            return [str(value) for value in table[column]]
    raise ValueError(f"{path} has no study identifier column; saw {list(table.columns)}")


def _track(args, results, uids, out: Path, check, total: float, per_study: float) -> None:
    """File this scoring pass as a W&B run with the submission attached.

    Kept out of ``main`` so the submission is written and validated before anything
    touches the network: a submission that exists is worth more than a logged run,
    and this is called after ``validate_submission`` for exactly that reason.

    The checkpoints are named by path and hashed rather than uploaded. Which weights
    produced a submission has to be recorded -- otherwise a leaderboard score is
    attached to nothing -- but a submission run is not the place to move gigabytes.
    """
    from kappa.tracking import file_digest, finish_run, init_run, log_artifact, summarise

    entities = {
        "n_members": len(results),
        "n_studies": len(uids),
        "transform": args.transform,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": str(args.amp),
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "checkpoints": ",".join(str(path) for path in args.checkpoint),
        "checkpoint_hashes": ",".join(file_digest(path) for path in args.checkpoint),
    }
    name = args.run_name or f"predict-{len(results)}x{len(uids)}"
    run = init_run(name, job_type="predict", config=entities, tags=["predict"])
    summarise(
        run,
        {
            "predict/seconds_total": total,
            "predict/seconds_per_study": per_study,
            "predict/n_studies": len(uids),
            "predict/runtime_term": (per_study * (args.project_to or len(uids))) / 32_400,
            "predict/submission_ok": bool(check.ok),
            "predict/submission_path": str(out),
        },
    )
    log_artifact(run, out, artifact_type="submission", aliases=["latest"])
    finish_run(run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, action="append", required=True,
                    help="repeat to ensemble; members are blended by per-column rank")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="one row per series, the pinned 16-column manifest")
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="<cache>/<study_uid>/<series_uid>.npy")
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--expected-uids", type=Path, default=None,
                    help="test.csv or sample_submission.csv; reorders rows to match it")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda, cpu; defaults to cuda when usable")
    ap.add_argument("--amp", default=None, choices=["bf16", "fp16", "off"],
                    help="autocast dtype; default is bf16 on cuda and off elsewhere")
    ap.add_argument("--transform", default="rank", choices=["rank", "none"],
                    help="'rank' writes per-column rank-normalised scores in [0, 1]; "
                         "'none' writes raw logits. Both give a bit-identical AUC")
    # The legal names are not spelled out here: they live in kappa.config's
    # TTA_TRANSFORMS, importing that would make --help need pydantic, and a second
    # copy in a help string is a list that drifts. An unknown name is refused below
    # with the real set named.
    ap.add_argument("--tta", default="", metavar="A,B",
                    help="comma-separated test-time augmentations to average over. "
                         "Empty (the default) is a single plain pass. There is no "
                         "flip: this is a knee, and a horizontal flip swaps medial "
                         "and lateral. Each name costs one more forward pass over "
                         "the whole set, which the efficiency track's T/32400 term "
                         "charges for -- read the projection this prints.")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N studies")
    ap.add_argument("--project-to", type=int, default=None, metavar="N",
                    help="project the measured rate to an N-study test set")
    ap.add_argument("--track", action="store_true",
                    help="log this scoring pass to W&B: the submission as an artifact "
                         "and the measured s/study in the summary. Offline inside a "
                         "Slurm job; a no-op without wandb or an API key")
    ap.add_argument("--run-name", default=None,
                    help="W&B run name; defaults to predict-<n checkpoints>x<n studies>")
    args = ap.parse_args()

    import pandas as pd

    from kappa.inference.predict import (
        TTA_TRANSFORMS,
        load_checkpoint,
        predict_studies,
        write_predictions,
    )
    from kappa.inference.submission import validate_submission

    # Validated before any weights are loaded, so a misspelled name costs a second
    # rather than one scored checkpoint. The names come from the module that
    # implements them, never from a copy in the help text.
    tta = tuple(name.strip() for name in args.tta.split(",") if name.strip())
    unknown = [name for name in tta if name not in TTA_TRANSFORMS]
    if unknown:
        ap.error(f"unknown --tta {unknown}; legal names are {list(TTA_TRANSFORMS)}")
    if len(set(tta)) != len(tta):
        ap.error(f"--tta repeats a transform: {list(tta)}")

    manifest = pd.read_parquet(args.manifest)
    uids = sorted(str(uid) for uid in manifest["study_uid"].unique())
    if args.limit:
        uids = uids[: args.limit]
    print(f"manifest   : {args.manifest} -- {len(manifest)} series, {len(uids)} studies")
    print(f"cache      : {args.cache_dir}")

    results = []
    for path in args.checkpoint:
        loaded = load_checkpoint(path, device=args.device)
        print(loaded.describe())
        dataset = loaded.dataset(manifest, args.cache_dir, study_uids=uids)
        result = predict_studies(
            loaded.model,
            dataset,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=False if args.amp == "off" else args.amp,
            tta=tta,
        )
        print(result.timing_report(n_studies=args.project_to))
        results.append(result)

    if len(results) > 1:
        print(
            f"\nblending {len(results)} member(s) by per-column rank normalisation "
            "(never by averaging probabilities)"
        )

    out = write_predictions(
        args.out,
        results,
        transform=args.transform,
        expected_uids=_expected_uids(args.expected_uids) if args.expected_uids else None,
    )
    check = validate_submission(
        out, expected_uids=_expected_uids(args.expected_uids) if args.expected_uids else None
    )
    print()
    print(check.summary())

    total = sum(result.seconds for result in results)
    per_study = total / max(len(uids), 1)
    print(
        f"\ntotal      : {total:.2f} s for {len(results)} member(s) over {len(uids)} "
        f"studies = {per_study:.4f} s/study (measured)"
    )
    if args.project_to:
        projected = per_study * args.project_to
        print(
            f"projected  : {projected:.0f} s for {args.project_to} studies, runtime term "
            f"{projected / 32_400:.4f} against the 32,400 s notebook limit "
            "(projected from the measured rate, not measured at that size)"
        )

    if args.track:
        _track(args, results, uids, out, check, total, per_study)

    if not check.ok:
        print("FAILED: the submission did not pass its contract checks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
