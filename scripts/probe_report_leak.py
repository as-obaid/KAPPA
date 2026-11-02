#!/usr/bin/env python3
"""Text-only probe: can the report features alone reproduce the twelve labels?

    python scripts/probe_report_leak.py \
        --features data/interim/labels/report_features.parquet \
        --labels   data/interim/labels/derived.parquet \
        --folds    data/folds/folds_v2.csv

**This is a gate, not an experiment.** The grounding pipeline maps report text
-> 12 labels for 4,349 of 4,407 studies,
so a teacher reading that same text can minimise its loss by learning that map
without reading a pixel. Every consequence of the leak *looks like success* --
the teacher's score inflates, its scores collapse onto the labels, and the
student gains nothing. A near-zero transfer gain beside a large teacher gain is
the leak's signature, not evidence that transfer fails.

So: fit the cheapest possible text-only model, out of fold, and read its macro
AUC. **No GPU, no training job.** One logistic regression per label on 156 float
features, on the same folds every image model in this repository used, so the
number is directly comparable to theirs.

**How to read the number.**

* ``~0.99`` -- the features are the labels in another basis. The teacher would be
  a lookup table, and neither it nor a student distilled from it means anything;
  the text channel has to be cut back to a residue first.
* ``~0.765`` -- the features reproduce the label pipeline about as well as the
  pipeline reproduces the anchor ground truth. Still a leak: the teacher has a
  shortcut to the labels and will take it.
* clearly below ``0.765`` -- the text carries less than the labels already hold,
  which is the only case in which distilling from it can transfer anything.

Nothing here reads pixels, so a high score is not a result about the images. It
is a statement about how much of the label table is recoverable from the text it
was made from.

Exit status is the verdict: 1 for a leak, 0 for clear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.constants import LABELS  # noqa: E402
from kappa.evaluation.metrics import macro_roc_auc  # noqa: E402
from kappa.labels.features import load_features  # noqa: E402

UID = "StudyInstanceUID"

#: The label pipeline's own measured accuracy against the 58 anchor studies. A
#: text-only probe that reaches this has recovered the pipeline, not the knee.
PIPELINE_MACRO = 0.765

#: The best five-fold image-only mean of a single ConvNeXt config when the probe
#: was written. Here only to put the probe's number in scale.
IMAGE_MACRO = 0.8444


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--fold-column", default="fold")
    ap.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="inverse regularisation strength (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    uids, features, present = load_features(args.features)
    feature_frame = pd.DataFrame({UID: uids})
    feature_frame["_row"] = np.arange(len(uids))

    labels = pd.read_parquet(args.labels)
    folds = pd.read_csv(args.folds, usecols=[UID, args.fold_column])

    # Inner joins throughout: a study missing from any of the three cannot be
    # scored, and carrying it as NaN would quietly shrink a label's support
    # without saying so.
    joined = feature_frame.merge(labels, on=UID, how="inner").merge(
        folds, on=UID, how="inner"
    )
    print(f"features : {len(uids)} studies, {features.shape[1]} channels")
    print(f"present  : {int(present.sum())} of {len(present)}")
    print(f"labels   : {len(labels)} rows")
    print(f"joined   : {len(joined)} studies usable")
    if joined.empty:
        print("nothing joined; check that the uid columns agree", file=sys.stderr)
        return 2

    rows = joined["_row"].to_numpy()
    x_all = features[rows]
    fold_of = joined[args.fold_column].to_numpy()

    y_true = np.full((len(joined), len(LABELS)), np.nan, dtype=np.float64)
    mask = np.zeros((len(joined), len(LABELS)), dtype=bool)
    for column, label in enumerate(LABELS):
        y_true[:, column] = joined[label].to_numpy(dtype=np.float64)
        mask_column = f"{label}__mask"
        if mask_column in joined.columns:
            mask[:, column] = joined[mask_column].to_numpy().astype(bool)
        else:
            mask[:, column] = ~np.isnan(y_true[:, column])

    # Out of fold, on the same split the image models used. In fold would report a
    # fit, not a generalisation, and the whole question is whether the mapping
    # transfers to a study the fitter never saw.
    scores = np.full_like(y_true, np.nan, dtype=np.float64)
    for fold in sorted(np.unique(fold_of)):
        is_val = fold_of == fold
        is_train = ~is_val
        for column, label in enumerate(LABELS):
            usable = is_train & mask[:, column] & ~np.isnan(y_true[:, column])
            y = y_true[usable, column]
            if len(np.unique(y)) < 2:
                # One class in the training half: nothing to fit, and a constant
                # prediction has no ROC. Leave NaN; the label drops out below and
                # appears in `skipped`, which is visible rather than silent.
                continue
            model = LogisticRegression(max_iter=2000, C=args.C)
            model.fit(x_all[usable], y)
            scores[is_val, column] = model.predict_proba(x_all[is_val])[:, 1]

    # macro_roc_auc refuses NaN in y_score -- a NaN prediction is a broken model,
    # not a missing observation -- so narrow the mask to the cells the probe
    # actually produced and fill the rest with a value that is never scored.
    scored = mask & ~np.isnan(scores)
    filled = np.where(np.isnan(scores), 0.5, scores)
    result = macro_roc_auc(y_true, filled, scored)

    print()
    print("TEXT-ONLY, OUT OF FOLD")
    for label in LABELS:
        auc = result.per_label[label]
        shown = "  n/a " if np.isnan(auc) else f"{auc:6.4f}"
        print(
            f"  {label:18s} {shown}   "
            f"n={result.n_used.get(label, 0):5d}  pos={result.n_pos.get(label, 0):5d}"
        )
    print(f"  {'MACRO':18s} {result.macro:6.4f}   complete={result.is_complete}")
    if result.skipped:
        print(f"  skipped: {list(result.skipped)}")

    print()
    print(f"label pipeline vs anchors : {PIPELINE_MACRO:.4f}")
    print(f"best image-only 5-fold    : {IMAGE_MACRO:.4f}")
    print()
    if result.macro >= PIPELINE_MACRO:
        print("VERDICT: LEAK. The text features recover the labels at or above the")
        print("         pipeline's own accuracy. A teacher reading them has a")
        print("         shortcut to the label and will take it, so its gain would")
        print("         not transfer. Do not build a teacher on this feature set as it")
        print("         stands -- cut the channel back to a residue first.")
        return 1
    print("VERDICT: no leak at the pipeline's level. The text carries less than the")
    print("         label pipeline does, so a teacher must use the images to beat")
    print("         it. A teacher is worth building.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
