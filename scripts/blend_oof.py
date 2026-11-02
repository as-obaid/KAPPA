#!/usr/bin/env python3
"""Blend the finished 5-fold runs on their out-of-fold predictions, and cost it honestly.

    python scripts/blend_oof.py                      # the default seven-member recipe
    python scripts/blend_oof.py --members a,b,c      # an explicit recipe
    python scripts/blend_oof.py --all                # every complete config
    python scripts/blend_oof.py --greedy             # forward selection, LOFO-honest

**This is the cheapest measured gain the project has.** Seven models already on disk,
combined by per-fold rank average at equal weights, score **0.8612** against the best
single model's 0.8484 -- **+0.012 to +0.013**, one of the largest gains in the project,
for zero GPU. ``docs/results.md`` reports those numbers; this script is what produced
them and what must reproduce them.

**Four things here are load-bearing.**

1. **The fold assignment is verified before anything is blended.** Two configs that
   disagree on which studies sit in which fold cannot be averaged: a study held out by
   one model was *trained on* by the other, so the blend reads its own training data.
   That is a leak, not a nuisance, and it does not announce itself -- the blend simply
   scores higher. All 22 complete configs were checked and agree: 3,722 studies,
   identical ``StudyInstanceUID`` order, identical ``fold`` column at sizes
   744/744/747/742/745, bit-identical truth and mask. A future config need not, so the
   check runs every time and refuses rather than warns.

2. **Rank-normalise within each fold, never globally.** Each fold is a different model
   and writes logits on its own scale; rank-normalising the pooled table mixes five
   scales into one ordering. Measured: it drops even a *single* model from 0.8484 to
   0.8457 -- a 0.0027 loss conjured out of arithmetic. The script prints the global
   figure alongside as a diagnostic, so the trap stays visible.

3. **Equal weights.** The best fitted scheme tried (per-label logistic, C=0.05) reaches
   0.8627 against equal weights' 0.8614 -- **+0.0013, about one sixth of the smallest
   difference a 5-fold mean can resolve**. Fitted weighting's real use is rescuing a
   badly chosen pool; curate the pool by hand and equal weights arrive at the same
   place with nothing to overfit. No weighting is offered here on purpose.

4. **The 685 held-out test studies are never touched.** ``folds_v2.csv`` carries them at
   ``fold == -1`` / ``split == "test"``. They are the one uncontaminated read left on a
   final recipe, they are spent the moment they are scored, and overlap with the OOF
   tables is asserted to be exactly zero on every run.

``--greedy`` adds forward selection with **leave-one-fold-out honesty**: the membership
is chosen on four folds and the macro is read on the fifth, so a gain cannot come from
the selection having already seen the rows it is scored on. This is the path that proves
the seven-member number is not greedy overfitting -- LOFO landed on the same figure as the
in-sample fit -- and it is the reason the saturation curve flattening at seven members
can be believed.

CPU-only and seconds long; it needs no GPU and no batch job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.constants import LABELS  # noqa: E402
from kappa.evaluation.metrics import (  # noqa: E402
    AucResult,
    macro_roc_auc,
    rank_average,
    rank_normalise,
)

UID = "StudyInstanceUID"

#: The five cross-validation folds. A run directory is only usable when it has all
#: five: four folds of a config are not a 5-fold model, they are a partial table over
#: a different set of studies, and blending it against complete members would score
#: the blend on a subset one member never predicted.
FOLDS = tuple(range(5))

#: The measured recipe, ``docs/results.md`` (out-of-fold blend). Five ConvNeXt configs, one
#: volumetric and one MaxViT. It is not "the seven best" -- ``3d_ctg_d32`` standalone is
#: ~0.05 behind the leaders and still earns its place, because the blend pays for
#: decorrelation rather than for standalone score. Membership was chosen on this same
#: OOF table, so quote the result as +0.012 to +0.013 and not to four decimals.
CORE_7 = (
    "b2_aug_rot",
    "b2_aug_rotbc",
    "b3_v2_convnext_query_h1",
    "b3_v2_convnext_perlabel",
    "b2_v2_convnext_lr100",
    "3d_ctg_d32",
    "v3_all_slots_maxvit",
)

#: Members 8-12 measured between -0.0004 and +0.0004, and all 22 together score 0.8499 --
#: *worse* than seven, because the three DINOv2 configs each subtract 0.0027-0.0032. The
#: greedy search is capped a little past the knee so the flattening is visible in the
#: printed curve rather than assumed.
GREEDY_STEPS = 12


class FoldDisagreement(RuntimeError):
    """Two run directories do not describe the same out-of-fold split.

    Raised rather than warned. The blend of two disagreeing configs is not a
    degraded measurement, it is a leaked one, and it reads *better* than the honest
    number -- so nothing downstream would catch it.
    """


class HeldOutSplitTouched(RuntimeError):
    """An OOF table contains a study from the frozen held-out test split."""


# --------------------------------------------------------------------------- loading


def discover_complete(runs: Path) -> list[str]:
    """Run directory names holding a complete 5/5 ``oof_seed0_fold<k>.parquet`` set.

    Completeness is read off the filesystem rather than from any manifest: a submission
    log records what was *submitted*, and a cancelled task leaves the record intact and
    the artifact absent.
    """
    if not runs.is_dir():
        raise FileNotFoundError(f"no run directory at {runs}")
    return sorted(
        d.name
        for d in runs.iterdir()
        if d.is_dir() and all((d / f"oof_seed0_fold{k}.parquet").exists() for k in FOLDS)
    )


def load_oof(runs: Path, name: str) -> pd.DataFrame:
    """The five fold parquets of one config, concatenated and sorted by study UID.

    Sorting is what makes two configs row-comparable at all: the folds are written by
    five independent jobs in whatever order they finished, so concatenation order is
    not a property of the data. ``kind="stable"`` so that the order is a function of the
    UIDs alone and not of which job happened to land first.
    """
    parts = [pd.read_parquet(runs / name / f"oof_seed0_fold{k}.parquet") for k in FOLDS]
    frame = pd.concat(parts, ignore_index=True)
    return frame.sort_values(UID, kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------- guardrails


def verify_agreement(tables: dict[str, pd.DataFrame], reference: str) -> None:
    """Refuse unless every table describes the same studies, folds, truth and mask.

    Four separate things are compared, because they fail separately and only one of
    them is visible in a row count:

    * the **study set and its order** -- a differing set means the tables are not
      row-aligned and averaging them pairs unrelated studies;
    * the **fold column** -- equal sets with a different partition is the leak case,
      and it is invisible in every summary statistic;
    * **truth**, which would mean the label table was rebuilt between runs, so the two
      configs are not scored against the same target;
    * **mask**, which sets each label's denominator, so a difference silently changes
      what the macro is a mean over.
    """
    ref = tables[reference]
    ref_ids = ref[UID].to_numpy()
    ref_fold = ref["fold"].to_numpy()
    ref_true = ref[[f"{label}__true" for label in LABELS]].to_numpy(dtype=np.float64)
    ref_mask = ref[[f"{label}__mask" for label in LABELS]].to_numpy(dtype=bool)
    # NaN is the "unknown" encoding in 98.7% of raw label cells and NaN != NaN, so
    # compare on a sentinel rather than letting every unknown cell read as a mismatch.
    ref_true_cmp = np.nan_to_num(ref_true, nan=-9.0)

    problems: list[str] = []
    for name, frame in tables.items():
        if name == reference:
            continue
        ids = frame[UID].to_numpy()
        if ids.shape != ref_ids.shape or not (ids == ref_ids).all():
            problems.append(
                f"{name}: study set or order differs from {reference} "
                f"({len(ids)} rows vs {len(ref_ids)})"
            )
            continue  # nothing below is meaningful once the rows do not line up
        if not (frame["fold"].to_numpy() == ref_fold).all():
            differing = int((frame["fold"].to_numpy() != ref_fold).sum())
            problems.append(
                f"{name}: FOLD ASSIGNMENT differs from {reference} on {differing} studies "
                "-- blending these would score a model on studies it trained on"
            )
        truth = frame[[f"{label}__true" for label in LABELS]].to_numpy(dtype=np.float64)
        if not np.array_equal(np.nan_to_num(truth, nan=-9.0), ref_true_cmp):
            problems.append(f"{name}: truth columns differ from {reference}")
        frame_mask = frame[[f"{label}__mask" for label in LABELS]].to_numpy(dtype=bool)
        if not (frame_mask == ref_mask).all():
            problems.append(f"{name}: mask columns differ from {reference}")

    if problems:
        raise FoldDisagreement(
            "run directories do not share one out-of-fold split, so they cannot be "
            "blended:\n  " + "\n  ".join(problems)
        )


def assert_no_test_overlap(uids: np.ndarray, folds_csv: Path) -> int:
    """Refuse if any OOF row is one of the 685 frozen test studies. Returns that count.

    The test split is spent the first time it is scored. Nothing in the blend pipeline
    has any business reading it, and the failure it guards against is silent: a test
    study leaking into the OOF table would raise the number and look like a result.
    """
    if not folds_csv.is_file():
        raise FileNotFoundError(
            f"fold table {folds_csv} not found. This check is not optional -- without it "
            "there is no evidence the blend stayed off the held-out test studies."
        )
    folds = pd.read_csv(folds_csv, usecols=[UID, "split", "fold"])
    held_out = folds.loc[(folds["fold"] == -1) | (folds["split"] == "test"), UID]
    overlap = np.intersect1d(np.asarray(uids), held_out.to_numpy())
    if overlap.size:
        raise HeldOutSplitTouched(
            f"{overlap.size} of the {len(held_out)} held-out test studies appear in the "
            f"out-of-fold tables, e.g. {overlap[0]}. The test split is the only "
            "uncontaminated read left; do not spend it here."
        )
    return int(len(held_out))


# ------------------------------------------------------------------------- blending


def per_fold_rank_normalise(scores: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Rank-normalise each fold's rows independently, per label.

    The fold is the unit because the fold *is* the model: study 7's score comes from
    the fold-0 checkpoint and study 900's from the fold-3 checkpoint, and those two
    networks never agreed on a scale. Normalising the pooled table would rank a fold-0
    logit against a fold-3 logit and call the comparison meaningful.
    """
    out = np.empty_like(np.asarray(scores, dtype=np.float64))
    for fold in np.unique(fold_ids):
        rows = fold_ids == fold
        out[rows] = rank_normalise(scores[rows])
    return out


def per_fold_rank_average(members: list[np.ndarray], fold_ids: np.ndarray) -> np.ndarray:
    """Equal-weight rank average computed within each fold, then stitched back together.

    Delegates to :func:`kappa.evaluation.metrics.rank_average` one fold at a time, so
    the blend arithmetic has exactly one implementation in the repository.
    """
    if not members:
        raise ValueError("per_fold_rank_average needs at least one member")
    out = np.empty_like(np.asarray(members[0], dtype=np.float64))
    for fold in np.unique(fold_ids):
        rows = fold_ids == fold
        out[rows] = rank_average([member[rows] for member in members])
    return out


def score(truth: np.ndarray, mask: np.ndarray, blended: np.ndarray) -> AucResult:
    """Macro ROC-AUC over the cells the mask keeps."""
    return macro_roc_auc(truth, blended, mask=mask)


def per_fold_macro(
    truth: np.ndarray, mask: np.ndarray, fold_ids: np.ndarray, blended: np.ndarray
) -> np.ndarray:
    """Macro per fold, in fold order. The pooled macro is a different estimator.

    A pooled ranking compares studies across folds, i.e. across models; the per-fold
    mean is the quantity every other 5-fold number in this project reports, so it is
    the one that can be read against them.
    """
    return np.array(
        [
            score(truth[fold_ids == k], mask[fold_ids == k], blended[fold_ids == k]).macro
            for k in sorted(np.unique(fold_ids))
        ]
    )


def greedy_forward(
    pool: list[str],
    ranks: dict[str, np.ndarray],
    truth: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
    steps: int = GREEDY_STEPS,
) -> tuple[list[str], list[float]]:
    """Forward selection with replacement on ``rows``. Returns the picks and their curve.

    With replacement, so a strong member can be selected twice and thereby take double
    weight -- the cheapest form of weighting there is, and the one that needs no fitted
    coefficients. The returned curve is *in-sample over ``rows``* and is therefore
    optimistic by construction; only the caller's held-out score is honest.
    """
    chosen: list[str] = []
    curve: list[float] = []
    running: np.ndarray | None = None
    for _ in range(steps):
        best_name, best_macro, best_running = None, -np.inf, None
        for name in pool:
            candidate = ranks[name][rows] if running is None else running + ranks[name][rows]
            macro = score(truth[rows], mask[rows], candidate / (len(chosen) + 1)).macro
            if macro > best_macro:
                best_name, best_macro, best_running = name, macro, candidate
        assert best_name is not None  # pool is non-empty, so a best always exists
        chosen.append(best_name)
        curve.append(float(best_macro))
        running = best_running
    return chosen, curve


# -------------------------------------------------------------------------- reporting


def print_per_label(single_name: str, single: AucResult, blend: AucResult) -> None:
    """Per-label table beside the macro.

    A macro number with no per-label table is not a result. The table is also
    where the blend's mechanism shows -- the gain concentrating on the weakest columns
    is the signature of real decorrelation rather than of a rescaled winner.
    """
    print(f"\n=== PER-LABEL: blend vs best single ({single_name}) ===")
    print(f"{'label':<20}{'single':>9}{'blend':>9}{'delta':>9}{'n':>7}{'pos':>6}")
    for label in LABELS:
        one, many = single.per_label[label], blend.per_label[label]
        print(
            f"{label:<20}{one:>9.4f}{many:>9.4f}{many - one:>+9.4f}"
            f"{blend.n_used[label]:>7}{blend.n_pos[label]:>6}"
        )
    print(
        f"{'MACRO (pooled)':<20}{single.macro:>9.4f}{blend.macro:>9.4f}"
        f"{blend.macro - single.macro:>+9.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--folds",
        type=Path,
        default=Path("data/folds/folds_v2.csv"),
        help="frozen split, read only to prove the test studies were not touched",
    )
    parser.add_argument(
        "--members",
        default=None,
        help="comma-separated recipe (default: the measured seven-member recipe)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="blend every complete config (measured at 0.8499, worse than the default recipe)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="forward selection, scored leave-one-fold-out",
    )
    parser.add_argument("--greedy-steps", type=int, default=GREEDY_STEPS)
    args = parser.parse_args(argv)

    complete = discover_complete(args.runs)
    print(f"complete 5/5 configs under {args.runs}: {len(complete)}")
    if not complete:
        print("nothing to blend.")
        return 1
    for name in complete:
        print(f"  {name}")

    tables = {name: load_oof(args.runs, name) for name in complete}
    reference = complete[0]
    verify_agreement(tables, reference)
    ref = tables[reference]
    uids = ref[UID].to_numpy()
    fold_ids = ref["fold"].to_numpy()
    n_test = assert_no_test_overlap(uids, args.folds)
    sizes = [int((fold_ids == k).sum()) for k in sorted(np.unique(fold_ids))]
    print(
        f"\nverified: {len(complete)} configs agree on {len(uids)} studies, fold sizes "
        f"{sizes}, identical truth and mask; zero overlap with the {n_test} held-out "
        "test studies."
    )

    truth = ref[[f"{label}__true" for label in LABELS]].to_numpy(dtype=np.float64)
    mask = ref[[f"{label}__mask" for label in LABELS]].to_numpy(dtype=bool)
    preds = {name: tables[name][list(LABELS)].to_numpy(dtype=np.float64) for name in complete}
    ranks = {name: per_fold_rank_normalise(preds[name], fold_ids) for name in complete}

    # Standalone score of every member, by the 5-fold mean -- the estimator every other
    # number in docs/results.md uses, so these are directly comparable.
    standalone = {
        name: per_fold_macro(truth, mask, fold_ids, preds[name]) for name in complete
    }
    order = sorted(complete, key=lambda name: -float(standalone[name].mean()))
    # Population sd (ddof=0) for the standalone column, because that is the convention
    # every 5-fold cell in docs/results.md is quoted under and a column
    # that silently switched estimator would read as a disagreement with the doc.
    print("\n=== STANDALONE (5-fold mean, seed 0, population sd) ===")
    print(f"{'config':<34}{'mean':>9}{'sd':>9}")
    for name in order:
        folds = standalone[name]
        print(f"{name:<34}{folds.mean():>9.4f}{folds.std():>9.4f}")
    best_single = order[0]

    if args.members:
        recipe = [name.strip() for name in args.members.split(",") if name.strip()]
    elif args.all:
        recipe = list(order)
    else:
        recipe = list(CORE_7)
    missing = [name for name in recipe if name not in tables]
    if missing:
        print(f"\nERROR: recipe names no complete 5/5 run: {', '.join(missing)}", file=sys.stderr)
        return 2

    print(f"\n=== RECIPE ({len(recipe)} members, equal weights, per-fold rank average) ===")
    for name in recipe:
        print(f"  {name:<34}standalone {standalone[name].mean():.4f}")

    blended = per_fold_rank_average([ranks[name] for name in recipe], fold_ids)
    blend_folds = per_fold_macro(truth, mask, fold_ids, blended)
    single_folds = standalone[best_single]
    delta = blend_folds - single_folds
    print(f"\nper-fold blend  {np.round(blend_folds, 4).tolist()}  mean {blend_folds.mean():.4f}")
    print(f"per-fold single {np.round(single_folds, 4).tolist()}  mean {single_folds.mean():.4f}")
    print(
        f"per-fold delta  {np.round(delta, 4).tolist()}  mean {delta.mean():+.4f} "
        f"sd {delta.std(ddof=1):.4f} sem {delta.std(ddof=1) / np.sqrt(len(delta)):.4f}"
    )

    # The wrong arithmetic, printed on purpose, on the estimator where it does damage.
    # Rank-normalising within a fold is monotone within that fold, so it cannot move a
    # per-fold macro at all -- which is why the trap survives so long. It shows on the
    # pooled table, where a global normalisation ranks a fold-0 logit against a fold-3
    # logit: measured at 0.8484 -> 0.8457 for a single model.
    global_single = rank_average([preds[best_single]])
    global_blend = rank_average([preds[name] for name in recipe])
    print("\n=== DIAGNOSTIC: per-fold vs GLOBAL rank normalisation, pooled macro ===")
    print(f"{'':<34}{'per-fold':>10}{'global':>10}{'cost':>10}")
    for tag, per_fold_score, global_score in (
        (f"single ({best_single})", per_fold_rank_normalise(preds[best_single], fold_ids),
         global_single),
        (f"recipe ({len(recipe)} members)", blended, global_blend),
    ):
        one = score(truth, mask, per_fold_score).macro
        other = score(truth, mask, global_score).macro
        print(f"{tag:<34}{one:>10.4f}{other:>10.4f}{other - one:>+10.4f}")
    print("the per-fold MEAN is blind to this; only the pooled table shows it")

    print_per_label(
        best_single,
        score(truth, mask, per_fold_rank_normalise(preds[best_single], fold_ids)),
        score(truth, mask, blended),
    )

    if args.greedy:
        print("\n=== GREEDY FORWARD SELECTION, LEAVE-ONE-FOLD-OUT ===")
        print("picked on four folds, read on the fifth -- the selection never sees the score")
        curves, picked = [], []
        for k in sorted(np.unique(fold_ids)):
            train, held = fold_ids != k, fold_ids == k
            chosen, curve = greedy_forward(
                order, ranks, truth, mask, train, steps=args.greedy_steps
            )
            # Held-out curve: the same pick order, scored on rows the search never saw.
            running = np.zeros_like(ranks[chosen[0]][held])
            held_curve = []
            for step, name in enumerate(chosen, start=1):
                running = running + ranks[name][held]
                held_curve.append(score(truth[held], mask[held], running / step).macro)
            curves.append(held_curve)
            # Size the blend by the in-sample knee, then read the held-out fold there.
            knee = int(np.argmax(curve)) + 1
            picked.append((k, knee, held_curve[knee - 1], chosen[:knee]))
        curve_array = np.array(curves)

        print(f"\n{'k':>3}{'held-out macro':>16}{'sd':>9}{'gain vs k-1':>13}")
        for index in range(curve_array.shape[1]):
            column = curve_array[:, index]
            gain = column.mean() - curve_array[:, index - 1].mean() if index else 0.0
            print(f"{index + 1:>3}{column.mean():>16.4f}{column.std(ddof=1):>9.4f}{gain:>+13.4f}")

        print()
        held_scores = []
        for k, knee, held_macro, chosen in picked:
            single = score(
                truth[fold_ids == k],
                mask[fold_ids == k],
                ranks[best_single][fold_ids == k],
            ).macro
            held_scores.append((held_macro, single))
            print(f"  fold {k}: greedy({knee} picks) {held_macro:.4f} vs single {single:.4f} "
                  f"({held_macro - single:+.4f})")
            print(f"          {chosen}")
        held_array = np.array(held_scores)
        gap = held_array[:, 0] - held_array[:, 1]
        print(
            f"  LOFO mean: greedy {held_array[:, 0].mean():.4f}  "
            f"single {held_array[:, 1].mean():.4f}  delta {gap.mean():+.4f} "
            f"(sd {gap.std(ddof=1):.4f}, sem {gap.std(ddof=1) / np.sqrt(len(gap)):.4f})"
        )

    print(
        f"\nBLEND {blend_folds.mean():.4f} vs BEST SINGLE {single_folds.mean():.4f} "
        f"= {delta.mean():+.4f}. Quote a blend gain to three decimals, not four: the "
        "membership was chosen on this same table."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
