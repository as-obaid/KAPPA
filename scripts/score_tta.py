#!/usr/bin/env python3
"""Re-score trained folds under several TTA view-sets. Inference only, no training.

Argparse plus calls into ``kappa.inference`` and ``kappa.evaluation``, per the repo
contract. Nothing here reimplements a model, a dataset, a blend or an AUC.

    python scripts/score_tta.py --run-dir data/runs/v2_fs_three_planes_convnext
    python scripts/score_tta.py --run-dir data/runs/v2_fs_three_planes_convnext \
        --run-dir data/runs/v3_all_slots_convnext --folds 0 --limit 32 --device cpu

**What this measures and why it is free.** Every TTA view-set is a different way of
reading weights that already exist, so the whole comparison costs inference passes and
no GPU training at all. The output is one table: per (run, fold, view-set), the macro
ROC-AUC, all twelve per-label AUCs, the
forward passes per study and the measured wall clock.

**Every input is an argument and a missing one is a named error, never a hang.** The
checkpoints usually live on cluster storage rather than a laptop, and compute nodes
often have no outbound internet. So the preflight below stats every weight file and every OOF parquet for
every requested cell *before* torch is imported, and raises :class:`MissingArtifact`
naming all of them at once. A GPU task that dies on a missing parquet has already
burned an allocation to print a stack trace.

**The held-out fold comes from the OOF parquet, not from the fold table.** A run's
``oof_seed<seed>_fold<k>.parquet`` already holds exactly the studies that fold
validated on, together with the truth and the mask they were scored under -- which is
the reason ``kappa.training`` writes all three into one file. Re-joining a label table
here would let a score column be re-scored against a *different* mask, and neither
number would say which mask produced it.

**The identity self-check is the thing that makes the other three numbers readable.**
``predict_studies`` documents, and ``tests/test_tta.py`` asserts, that a single-view
``identity`` TTA is *bit-identical* to no TTA at all -- the mean over one view divides
by 1.0, which is exact. This script re-asserts it on the real weights and the real
cache, per fold, and says so loudly if it ever fails. Until that equality holds on the
machine doing the measuring, a rotate-set delta is not evidence of anything; it could
equally be a loader, a dtype or an alignment bug. It costs one extra plain pass per
fold and buys the right to read the rest of the table.

**The first timed pass of each fold is cold and is flagged as such.** Epoch 0 of a
training run pays page-cache warming over the 182 GB slice cache and measured 3.2x the
steady rate; the same warm-up lands on whichever inference pass runs first. The plain
reference pass is deliberately run first and its row carries ``cold=True``. Read
seconds off the warm rows only.

**Scores are raw logits, and that is fine here.** ROC-AUC reads only the order within a
column, so no sigmoid, no calibration and no threshold enters this file. The measured
range of a stored OOF column is about [-2.53, 6.16]; nothing here may threshold one at
0.5.

**This script writes nothing that could be mistaken for a training artifact.** It emits
one summary table at ``--out`` and refuses to place it inside any ``--run-dir``. No
checkpoint, no ``oof_seed*_fold*.parquet``, no weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.constants import LABELS  # noqa: E402

#: The four candidate view-sets this script chooses between, in the order they are
#: measured. ``identity`` is first and is the baseline: it must reproduce the run's own
#: OOF number, and it is what the ``--no-self-check`` assertion is taken against. The
#: rotate pair is drawn from the train-time augment distribution (10 degrees, the same
#: operator), the zoom pair is not -- there is no train-time scale jitter -- so the two
#: are not interchangeable candidates and are measured separately rather than as one
#: "all four extras" blob.
#:
#: There is no flip in any of these and there must never be one. This is a knee: a
#: horizontal flip swaps medial and lateral, and four of the twelve labels (Medial and
#: Lateral Meniscus, Medial and Lateral OA) are defined on exactly that axis, while MCL
#: has no LCL column to be swapped with. ``kappa.config.TTA_TRANSFORMS`` contains no
#: flip by construction and this list is checked against it below.
DEFAULT_TTA_SETS = (
    "identity",
    "identity,rotate_cw,rotate_ccw",
    "identity,zoom_in,zoom_out",
    "identity,rotate_cw,rotate_ccw,zoom_in,zoom_out",
)

#: How far a recomputed identity macro may sit from the stored OOF macro before this
#: script calls it a mismatch rather than a rounding difference. 0.0020 is the
#: carried-in reference at ``kappa.evaluation.metrics.NOISE_FLOOR_MACRO`` -- a
#: competitor's three-fold figure, not a number measured in this repository -- and it is
#: used here only as an order-of-magnitude tripwire, never as evidence.
OOF_AGREEMENT_TOLERANCE = 0.0020


class MissingArtifact(FileNotFoundError):
    """A requested cell has no weights, or no OOF table, at the path the contract names.

    Its own class rather than a bare ``FileNotFoundError`` so a batch wrapper can tell
    "this fold was never trained" apart from "the cache mount is not there". Both are
    fatal; only one of them means training is incomplete.
    """


def _cell_paths(run_dir: Path, fold: int, seed: int) -> tuple[Path, Path]:
    """``(weights, oof)`` for one cell, by the layout ``kappa.training`` writes.

    ``out_root = cfg.out_dir / cfg.name``, then ``fold<k>_seed<s>.pt`` and
    ``oof_seed<s>_fold<k>.parquet``. Note the two spell the pair in opposite orders;
    that is the contract, not a typo here.
    """
    return (
        run_dir / f"fold{fold}_seed{seed}.pt",
        run_dir / f"oof_seed{seed}_fold{fold}.parquet",
    )


def _parse_sets(raw: list[str], legal: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Turn ``--tta-set`` strings into view tuples, refusing anything illegal up front.

    Validated before a single weight is read: a misspelled view should cost a second,
    not one loaded checkpoint per run directory. The legal names come from the module
    that implements them, never from a copy in a help string.
    """
    parsed: list[tuple[str, ...]] = []
    for item in raw:
        views = tuple(name.strip() for name in str(item).split(",") if name.strip())
        if not views:
            raise ValueError(f"--tta-set {item!r} names no views")
        unknown = [name for name in views if name not in legal]
        if unknown:
            raise ValueError(
                f"--tta-set {item!r} names unknown view(s) {unknown}; legal names are "
                f"{list(legal)}. There is no flip in that set and adding one would "
                "mislabel every lateralised column."
            )
        if len(set(views)) != len(views):
            raise ValueError(f"--tta-set {item!r} repeats a view: {list(views)}")
        if views in parsed:
            raise ValueError(f"--tta-set {item!r} is a duplicate of an earlier set")
        parsed.append(views)
    return tuple(parsed)


def _aligned(uids, wanted, matrix):
    """Reorder ``matrix`` (rows keyed by ``uids``) onto ``wanted``, by identifier.

    By identifier and never by position, for the reason ``OofTable.aligned_truth``
    exists: a prediction lined up positionally against the wrong label looks exactly
    like a mediocre model and raises nothing.
    """
    position = {str(uid): index for index, uid in enumerate(uids)}
    missing = [uid for uid in wanted if uid not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} scored study/studies are absent from the reference rows, "
            f"e.g. {missing[:3]}"
        )
    return matrix[[position[uid] for uid in wanted]]


def _row(
    *,
    run: str,
    fold,
    seed: int,
    views: tuple[str, ...],
    auc,
    n_studies: int,
    n_slots: int,
    n_slices: int,
    seconds: float,
    cold: bool,
) -> dict:
    """One line of the summary table: what was scored, how well, and what it cost.

    ``image_passes_per_study`` is the number the Kaggle budget actually consumes --
    views times slots times slices -- and is carried beside ``forward_passes_per_study``
    because the second is the TTA multiplier alone and quoting it as the cost
    understates a six-slot model by 96x.
    """
    n_views = max(len(views), 1)
    row = {
        "run": run,
        # A string, and "pooled" is one of its values. Not an int with -1 for the
        # pooled row: -1 already means "in no fold at all" in this project's fold
        # tables (folds_v2 marks its frozen test split that way), and reusing it for
        # "every fold at once" would put the two opposite meanings in one column.
        "scope": "pooled" if fold == "pooled" else "fold",
        "fold": str(fold),
        "seed": seed,
        "tta_set": ",".join(views) if views else "(none)",
        "n_views": n_views,
        "forward_passes_per_study": n_views,
        "image_passes_per_study": n_views * n_slots * n_slices,
        "n_studies": n_studies,
        "macro": float(auc.macro),
        "n_labels_scored": int(auc.n_scored),
        "skipped_labels": ",".join(auc.skipped),
        "seconds": float(seconds),
        "seconds_per_study": float(seconds) / max(n_studies, 1),
        "cold": bool(cold),
    }
    for label in LABELS:
        row[f"auc::{label}"] = float(auc.per_label[label])
    return row


def _print_table(rows: list[dict], title: str) -> None:
    """A fixed-width block. Deltas are printed against the identity row of the same key.

    The identity row is the reference and not zero, because "better than nothing" is not
    the question this script asks -- the question is whether a view-set beats a plain
    pass by more than run-to-run noise. Every delta below is a difference, not
    evidence.
    """
    if not rows:
        return
    reference = {}
    for row in rows:
        if row["n_views"] == 1 and row["tta_set"] == "identity":
            reference[(row["run"], row["fold"])] = row["macro"]
    print(f"\n{title}")
    print(
        f"{'run':<34}  {'fold':>6}  {'tta set':<46}{'views':>6}{'macro':>9}"
        f"{'d vs id':>10}{'sec':>9}{'s/study':>9}  cold"
    )
    for row in rows:
        base = reference.get((row["run"], row["fold"]))
        delta = "" if base is None else f"{row['macro'] - base:+.4f}"
        print(
            f"{row['run'][:33]:<34}  {str(row['fold']):>6}  {row['tta_set'][:45]:<46}"
            f"{row['n_views']:>6}{row['macro']:>9.4f}{delta:>10}"
            f"{row['seconds']:>9.1f}{row['seconds_per_study']:>9.4f}"
            f"  {'COLD' if row['cold'] else ''}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", type=Path, action="append", required=True,
                    metavar="DIR",
                    help="repeat per run, e.g. data/runs/v2_fs_three_planes_convnext. "
                         "Each must hold fold<k>_seed<s>.pt and oof_seed<s>_fold<k>.parquet")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="folds to re-score (default: all five)")
    ap.add_argument("--seed", type=int, default=0,
                    help="the seed in the artifact filenames (default: 0)")
    ap.add_argument("--tta-set", action="append", default=None, metavar="A,B,C",
                    help="comma-separated view names, repeatable. Defaults to the four "
                         "candidates: identity; identity,rotate_cw,rotate_ccw; "
                         "identity,zoom_in,zoom_out; all five. There is no flip and "
                         "there must never be one -- a knee's medial/lateral axis")
    ap.add_argument("--manifest", type=Path, default=Path("data/interim/manifest.parquet"),
                    help="one row per series, the pinned manifest")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/processed/cache"),
                    help="<cache>/<study_uid>/<series_uid>.npy")
    ap.add_argument("--out", type=Path, default=Path("data/tta/tta_scores.parquet"),
                    help=".parquet or .csv summary. Refused if it lands inside a --run-dir")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="cuda, cpu; defaults to cuda when it is actually usable")
    ap.add_argument("--amp", default=None, choices=["bf16", "fp16", "off"],
                    help="autocast dtype; default is bf16 on cuda and off elsewhere")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="score only the first N studies of each fold. A smoke run: the "
                         "macro it prints is over a subset and is not the fold's number")
    ap.add_argument("--no-verify-cache", action="store_true",
                    help="skip the dataset's per-series cache check (faster, less safe)")
    ap.add_argument("--no-self-check", action="store_true",
                    help="do not assert that a single identity view is bit-identical to "
                         "no TTA. Leave the check on: the assertion is what proves the harness "
                         "before any other number in the table is worth reading")
    ap.add_argument("--allow-self-check-failure", action="store_true",
                    help="report a failed self-check and keep going instead of exiting 1")
    args = ap.parse_args()

    # ---- validation and preflight, all of it before torch is imported ----------------
    from kappa.config import TTA_TRANSFORMS

    try:
        sets = _parse_sets(list(args.tta_set or DEFAULT_TTA_SETS), TTA_TRANSFORMS)
    except ValueError as exc:
        ap.error(str(exc))

    # The self-check needs a single-view identity set to compare the plain pass against,
    # so one is put back if the caller's --tta-set list has none. Prepended rather than
    # appended: it is also the baseline row that every delta is read against, and
    # it is what the reference column of the printed table looks for.
    if not args.no_self_check and ("identity",) not in sets:
        sets = (("identity",),) + sets
        print(
            "note       : added the single-view 'identity' set -- it is both the baseline "
            "row and what the harness self-check is taken against. --no-self-check drops it."
        )

    run_dirs = [Path(d) for d in args.run_dir]
    out = Path(args.out)
    for run_dir in run_dirs:
        try:
            out.resolve().relative_to(run_dir.resolve())
        except ValueError:
            continue
        ap.error(
            f"--out {out} is inside the run directory {run_dir}. This script is "
            "inference only and must not put a file where a training artifact is read "
            "from; write it somewhere clearly separate."
        )

    # Rows are keyed by the run directory's basename, which is `cfg.name` -- so two
    # --run-dir arguments with the same basename would pool two runs' logits into one
    # macro and print it under one name. It is the same collision as two configs
    # sharing a `name:` (excluded from config_hash, but used as the output path), one
    # layer up, so it is refused here the same way.
    by_name: dict[str, list[Path]] = {}
    for run_dir in run_dirs:
        by_name.setdefault(run_dir.name, []).append(run_dir)
    clashes = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if clashes:
        ap.error(
            "two or more --run-dir arguments share a basename, and the basename is what "
            "every row is keyed by: "
            + "; ".join(f"{name} -> {[str(p) for p in paths]}" for name, paths in clashes.items())
        )

    folds = sorted({int(k) for k in args.folds})
    absent: list[str] = []
    cells: list[tuple[Path, int, Path, Path]] = []
    for run_dir in run_dirs:
        for fold in folds:
            weights, oof = _cell_paths(run_dir, fold, args.seed)
            if not weights.is_file():
                absent.append(str(weights))
            if not oof.is_file():
                absent.append(str(oof))
            cells.append((run_dir, fold, weights, oof))
    for required in (args.manifest, args.cache_dir):
        if not Path(required).exists():
            absent.append(str(required))
    if absent:
        raise MissingArtifact(
            f"{len(absent)} required input(s) do not exist:\n  "
            + "\n  ".join(absent)
            + "\n\nTrained checkpoints are never downloaded by this script, and compute "
            "nodes often have no outbound internet -- nothing here will "
            "fetch a missing file. Pass --run-dir/--folds that name cells which actually "
            "ran, or --manifest/--cache-dir that point at the built inputs."
        )
    print(f"preflight  : {len(cells)} cell(s), all weights and OOF tables present")

    # ---- the heavy half --------------------------------------------------------------
    import numpy as np
    import pandas as pd

    from kappa.ensemble import read_oof
    from kappa.evaluation.metrics import macro_roc_auc
    from kappa.inference.predict import load_checkpoint, predict_studies

    manifest = pd.read_parquet(args.manifest)
    amp = False if args.amp == "off" else args.amp
    print(f"manifest   : {args.manifest} -- {len(manifest)} series")
    print(f"cache      : {args.cache_dir}")
    print(f"tta sets   : {[','.join(s) for s in sets]}")
    if args.limit:
        print(
            f"*** SMOKE RUN: --limit {args.limit}. Every macro below is over the first "
            f"{args.limit} studies of a fold and is NOT that fold's number. ***"
        )

    out.parent.mkdir(parents=True, exist_ok=True)

    def write_table(frame_rows: list[dict]) -> None:
        """Rewrite the summary from scratch. Called after EVERY fold, not only at the end.

        A batch job can be killed at its walltime with little warning, and a buffer
        committed only on a clean exit is then lost. Rewriting a small table five times
        costs nothing and means a job that dies on fold 4 still leaves folds 0-3 on
        disk, readable, instead of nothing at all.
        """
        frame = pd.DataFrame(frame_rows)
        if out.suffix == ".csv":
            frame.to_csv(out, index=False)
        else:
            frame.to_parquet(out, index=False)

    rows: list[dict] = []
    # Keyed by (run, views); each entry accumulates the per-fold scores, truth and mask
    # so a pooled macro can be taken over the concatenation -- which is what an OOF
    # number is. Logits are pooled, never ranks: a per-fold rank is only comparable
    # inside its own fold, so concatenating five of them would be five incomparable
    # scales under one plausible-looking mean.
    pooled: dict[tuple[str, tuple[str, ...]], list] = {}
    pooled_baseline: dict[str, list] = {}
    geometry: dict[str, tuple[int, int]] = {}
    self_check_failures: list[str] = []

    for run_dir, fold, weights, oof_path in cells:
        run = run_dir.name
        print(f"\n{'=' * 90}\n{run}  fold {fold}\n{'=' * 90}")

        table = read_oof(oof_path, name=f"{run}_fold{fold}")
        uids = [str(uid) for uid in table.study_uids]
        if args.limit:
            uids = sorted(uids)[: args.limit]
        truth, mask = table.aligned_truth(uids)
        stored = _aligned(table.study_uids, uids, table.scores)
        stored_auc = macro_roc_auc(truth, stored, mask)
        print(f"oof stored : {oof_path.name} -- {len(uids)} studies, "
              f"macro {stored_auc.macro:.4f}")

        loaded = load_checkpoint(weights, device=args.device)
        print(loaded.describe())
        n_slots, n_slices = len(loaded.slots), int(loaded.n_slices)
        geometry[run] = (n_slots, n_slices)

        # Built once per fold and reused for every view-set. The dataset is stateless
        # between passes -- predict_studies builds its own DataLoader each time -- so a
        # second construction would only re-run the per-series cache verification and
        # charge the fold for it five times over.
        dataset = loaded.dataset(
            manifest,
            args.cache_dir,
            study_uids=uids,
            verify_cache=not args.no_verify_cache,
        )

        def score(views: tuple[str, ...]):
            result = predict_studies(
                loaded.model,
                dataset,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                amp=amp,
                tta=views,
            )
            return result, _aligned(result.study_uids, uids, result.scores)

        # The plain pass runs first, on purpose: it warms the page cache over the slice
        # cache so every TTA row after it is a warm measurement, and it is the reference
        # the identity self-check is taken against. Its own seconds are flagged cold.
        plain, plain_scores = score(())
        plain_auc = macro_roc_auc(truth, plain_scores, mask)
        plain_row = _row(run=run, fold=fold, seed=args.seed, views=(), auc=plain_auc,
                         n_studies=len(uids), n_slots=n_slots, n_slices=n_slices,
                         seconds=plain.seconds, cold=True)
        plain_row["delta_vs_stored_oof"] = float(plain_auc.macro - stored_auc.macro)
        rows.append(plain_row)
        pooled_baseline.setdefault(run, []).append((stored, truth, mask))
        drift = plain_auc.macro - stored_auc.macro
        print(
            f"recomputed : macro {plain_auc.macro:.4f} with no TTA, "
            f"{drift:+.4f} against the stored OOF row"
        )
        if abs(drift) > OOF_AGREEMENT_TOLERANCE:
            print(
                f"!! WARNING: the recomputed plain pass differs from the stored OOF by "
                f"{drift:+.4f}, more than the {OOF_AGREEMENT_TOLERANCE} carried-in "
                "reference (a competitor's three-fold figure, not measured here). These "
                "weights and that parquet may not be the same epoch, or the cache has "
                "moved under them. Do not read a TTA delta off this cell until it is "
                "explained.",
                file=sys.stderr,
            )

        for views in sets:
            result, scores = score(views)
            auc = macro_roc_auc(truth, scores, mask)
            row = _row(run=run, fold=fold, seed=args.seed, views=views, auc=auc,
                       n_studies=len(uids), n_slots=n_slots, n_slices=n_slices,
                       seconds=result.seconds, cold=False)
            row["delta_vs_stored_oof"] = float(auc.macro - stored_auc.macro)
            rows.append(row)
            pooled.setdefault((run, views), []).append((scores, truth, mask))
            print(f"  {','.join(views):<44} macro {auc.macro:.4f}  "
                  f"{result.seconds:7.1f}s  {result.n_passes} pass/study")
            print("  " + result.timing_report().replace("\n", "\n  "))

            # The self-check. predict_studies divides the view sum by float(len(views)),
            # so a single view divides by 1.0, which is exact -- the two branches must
            # agree to the bit, not to a tolerance. tests/test_tta.py asserts this on a
            # toy model; this asserts it on the real weights, the real cache and the
            # real device, which is where a dtype or an ordering bug would actually
            # live. If it fails, nothing else in this table is readable.
            if views == ("identity",) and not args.no_self_check:
                if np.array_equal(plain_scores, scores):
                    print("  self-check : identity == no-TTA, BIT-IDENTICAL (harness sound)")
                else:
                    worst = float(np.max(np.abs(plain_scores - scores)))
                    message = (
                        f"{run} fold {fold}: a single identity view is NOT bit-identical "
                        f"to no TTA (max |diff| {worst:.3e})"
                    )
                    self_check_failures.append(message)
                    print(
                        "!! SELF-CHECK FAILED: " + message + "\n"
                        "!! predict_studies documents this equality and tests/test_tta.py "
                        "asserts it. Until it holds here, every other row in this table "
                        "is a measurement of an unknown harness, not of TTA.",
                        file=sys.stderr,
                    )

        del loaded, dataset
        write_table(rows)
        print(f"  wrote      : {out} ({len(rows)} rows so far)")

    _print_table(rows, "PER FOLD  (identity is the reference; the 'cold' row is the "
                       "page-cache warm-up, do not quote its seconds)")

    # ---- pooled over folds -----------------------------------------------------------
    def _pooled_auc(parts):
        """Macro over the concatenation of several folds' scores, truth and mask.

        Concatenating is what makes this an out-of-fold number rather than a mean of
        five fold numbers: each study appears exactly once, so the ranking the metric
        reads is global. It is only legitimate because the arrays are raw logits from
        one set of weights per fold and are never rank-normalised per fold first.
        """
        return macro_roc_auc(
            np.concatenate([t for _, t, _ in parts]),
            np.concatenate([s for s, _, _ in parts]),
            np.concatenate([m for _, _, m in parts]),
        )

    pooled_rows: list[dict] = []
    for run in dict.fromkeys(run_dir.name for run_dir, *_ in cells):
        base_parts = pooled_baseline.get(run) or []
        base_macro = _pooled_auc(base_parts).macro if base_parts else float("nan")
        if base_parts:
            print(f"\n{run}: pooled STORED OOF macro over folds {folds} = "
                  f"{base_macro:.4f}")
        n_slots, n_slices = geometry.get(run, (0, 0))
        for views in sets:
            parts = pooled.get((run, views))
            if not parts:
                continue
            auc = _pooled_auc(parts)
            row = _row(
                run=run, fold="pooled", seed=args.seed, views=views, auc=auc,
                n_studies=sum(len(s) for s, _, _ in parts),
                n_slots=n_slots, n_slices=n_slices,
                # Summed over the folds that were actually scored, so this is the cost
                # of the whole run under this view-set rather than one fold's.
                seconds=sum(
                    r["seconds"] for r in rows
                    if r["run"] == run and r["tta_set"] == ",".join(views)
                ),
                cold=False,
            )
            row["delta_vs_stored_oof"] = float(auc.macro - base_macro)
            pooled_rows.append(row)

    _print_table(pooled_rows, "POOLED OVER FOLDS  (concatenated logits, one macro)")
    if pooled_rows:
        print(
            "\nEvery delta above is a difference, not evidence: read it against "
            "run-to-run noise. 0.0020 is a carried-in reference, a competitor's "
            "three-fold figure. The measured fold-to-fold sd in this repo is 0.0087."
        )

    write_table(rows + pooled_rows)
    print(
        f"\nwrote      : {out}  ({len(rows) + len(pooled_rows)} rows) -- summary only, "
        "no weights, no OOF rows, nothing that could be mistaken for a trained fold"
    )

    if self_check_failures and not args.allow_self_check_failure:
        print(
            f"\nFAILED: {len(self_check_failures)} identity self-check(s) did not hold:\n  "
            + "\n  ".join(self_check_failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
