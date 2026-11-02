#!/usr/bin/env python3
"""Zero the label-determining channels of the report feature table.

    python scripts/build_residue_features.py \
        --features data/interim/labels/report_features.parquet \
        --out      data/interim/labels/report_features_residue.parquet

**Why this exists.** The full 156-channel table leaks the labels, and the leak
is measured, not suspected: a plain logistic regression on all 156 channels
reproduces the twelve labels at **0.9538 macro out of fold**, with ACL, MCL,
Medial Meniscus and Lateral Meniscus at **1.0000**. The label pipeline itself
only reaches 0.765 against the anchors and the best image model on disk reaches
0.8444, so a teacher handed that table is a lookup table for the labels and
would transfer nothing -- while looking exactly like a success.

**The rule this script implements, and the four that failed.** Keep a cluster's
two channels only when :data:`kappa.labels.label_map.CLUSTER_LABELS` maps it to
**no label at all**. That is 29 of the 78 clusters, so 58 of the 156 channels
survive and 98 are zeroed. Measured text-only macro under each candidate rule:

    keep unmapped clusters only (this one)            0.6910   <- below 0.765
    drop the four perfectly-separated labels' inputs  0.8928
    drop every DIRECT mapping                         0.7809
    no direct, plus ambiguous with >= 3 labels        0.7643
    unmapped, plus ambiguous negatives                0.7497

Only the first clears the bar `scripts/probe_report_leak.py` sets, and it clears
it by 0.074 rather than by a rounding error.

**It is not clean, and the residue that remains is named.** Even under this
rule, ACL still scores **0.784** and MCL **0.752** text-only. That is a real
shortcut on two of the twelve columns, and the decision to proceed anyway comes
with an obligation: **every distillation result must report ACL and MCL
separately from the other ten.** A gain concentrated on those two is the
shortcut resurfacing, not transfer, and a macro mean hides it perfectly.

**Why the output is 156 columns wide and not 58.** This is the design decision
in the file and it is not a convenience.
:func:`kappa.labels.features.load_features` refuses any table whose feature
columns are not exactly ``REPORT_FEATURE_NAMES``, in order, and
:func:`kappa.training.loop.load_frozen_teacher` re-checks the same vocabulary
when a student reloads a teacher checkpoint. A narrow 58-column table would be
refused at load, and widening either check to accept a variable vocabulary would
remove the guard that stops a permuted feature table from training a teacher on
scrambled inputs -- a failure that converges, prints a loss curve and reports a
number. So the residue table keeps the full width and the full name order, and
the 98 label-determining channels are set to **0.0**. ``report_dim`` stays 156
and the 98 dead inputs are dead units into the report MLP's first Linear, which
is harmless: they contribute a constant zero to every study.

**The selection is derived at runtime, never hardcoded.** The kept set is
computed from ``CLUSTER_LABELS`` on every run, so if the label map gains or
loses a mapping the residue moves with it instead of silently drifting into a
table whose name no longer describes it. The count is printed, and
``--expect-kept`` fails the run if it has moved from the 58 the 0.6910 was
measured on.

**Run the probe afterwards; this script does not decide anything.**

    python scripts/probe_report_leak.py \
        --features data/interim/labels/report_features_residue.parquet \
        --labels   data/interim/labels/derived.parquet \
        --folds    data/folds/folds_v2.csv

Expect macro **0.6910**. Use it as a preflight gate: do not train a teacher on
the table if the probe comes back at or above 0.765.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.labels.features import (  # noqa: E402
    FEATURE_TABLE_COLUMNS,
    PRESENT_COLUMN,
    REPORT_CLUSTERS,
    REPORT_FEATURE_DIM,
    REPORT_FEATURE_NAMES,
    UID_COLUMN,
    load_features,
)
from kappa.labels.label_map import CLUSTER_LABELS  # noqa: E402

#: The number of channels the measured 0.6910 was produced on. Not the rule --
#: the rule is `residue_clusters()` below -- but a tripwire on it. If the label
#: map changes so that this moves, every quoted 0.6910 describes a different
#: table and has to be re-measured before it is quoted again.
EXPECTED_KEPT_CHANNELS = 58

#: What `probe_report_leak.py` calls a leak. Repeated here only so this script's
#: own output can say where the table it just wrote is expected to land.
PIPELINE_MACRO = 0.765

#: The measured text-only macro of the table this script writes.
RESIDUE_MACRO = 0.6910


def residue_clusters() -> tuple[str, ...]:
    """The clusters that map to NO label, in ``REPORT_CLUSTERS`` order.

    ``CLUSTER_LABELS.get(cluster)`` is falsy in exactly two ways and both mean
    the same thing to :func:`kappa.labels.label_map.derive_study`: the cluster is
    absent from the table (it is in ``UNMAPPED_CLUSTERS``), or it is present with
    an empty mapping (PCL, LCL, the tendons, the retinaculum -- real structures
    that no competition label covers). Both contribute nothing to any label, so
    both are residue, and collapsing them with one ``not`` is the same test
    ``derive_study`` performs rather than a second opinion about it.
    """
    return tuple(name for name in REPORT_CLUSTERS if not CLUSTER_LABELS.get(name))


def residue_channels() -> tuple[str, ...]:
    """The feature column names that survive, in ``REPORT_FEATURE_NAMES`` order."""
    kept = set(residue_clusters())
    # Split on the LAST colon, not the first: every channel name is
    # `<cluster>:<pos|neg>` and no cluster name contains a colon today, but
    # rsplit costs nothing and does not quietly mis-parse the day one does.
    return tuple(
        name for name in REPORT_FEATURE_NAMES if name.rsplit(":", 1)[0] in kept
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--features",
        type=Path,
        default=Path("data/interim/labels/report_features.parquet"),
        help="the full 156-channel table written by build_report_features.py "
        "(default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/labels/report_features_residue.parquet"),
        help="destination parquet (default: %(default)s)",
    )
    ap.add_argument(
        "--expect-kept",
        type=int,
        default=EXPECTED_KEPT_CHANNELS,
        help="refuse to write if the derived rule keeps a different number of "
        "channels than the measured 0.6910 was produced on. Pass 0 to disable, "
        "and re-run the probe if you do (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    if args.out.resolve() == args.features.resolve():
        # Overwriting the full table in place would destroy the input every other
        # rule is measured against, and the leaked-teacher negative control reads
        # it directly.
        print("REFUSED: --out is --features; the full table is an input, not scratch",
              file=sys.stderr)
        return 2

    import pandas as pd

    # Through `load_features`, deliberately. It is the function the teacher's
    # loader uses, so reading the source through it proves the input carries the
    # exact vocabulary in the exact order before anything is derived from it.
    uids, features, present = load_features(args.features)
    print(f"source   : {args.features}")
    print(f"shape    : {features.shape} (expect (n, {REPORT_FEATURE_DIM}))")

    kept_names = residue_channels()
    kept_index = np.array(
        [REPORT_FEATURE_NAMES.index(name) for name in kept_names], dtype=np.int64
    )
    zeroed = REPORT_FEATURE_DIM - len(kept_names)
    print(f"clusters : {len(residue_clusters())} of {len(REPORT_CLUSTERS)} map to no label")
    print(f"kept     : {len(kept_names)} channels; {zeroed} zeroed")
    if args.expect_kept and len(kept_names) != args.expect_kept:
        print(
            f"REFUSED  : the rule keeps {len(kept_names)} channels, not "
            f"{args.expect_kept}. label_map.CLUSTER_LABELS has moved, so the "
            f"measured text-only {RESIDUE_MACRO} describes a different table than "
            "this one. Re-run scripts/probe_report_leak.py and update every header "
            "that quotes it before passing --expect-kept 0.",
            file=sys.stderr,
        )
        return 3

    # Zero by construction rather than by subtraction: start from all-zero and
    # copy the kept columns in. Masking the full matrix in place would leave a
    # NaN, if one ever appeared, sitting in a "zeroed" column.
    residue = np.zeros_like(features)
    residue[:, kept_index] = features[:, kept_index]

    frame = pd.DataFrame(residue, columns=list(REPORT_FEATURE_NAMES))
    frame.insert(0, UID_COLUMN, pd.Series(uids, dtype=object))
    frame[PRESENT_COLUMN] = np.asarray(present, dtype=bool)
    frame = frame.loc[:, list(FEATURE_TABLE_COLUMNS)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    # Read it straight back through the loader the teacher will use, for the same
    # reason build_report_features.py does: the only useful moment to catch a
    # permuted table is now, not at the top of a five-fold run.
    back_uids, back, back_present = load_features(args.out)
    live = int((back != 0.0).any(axis=0).sum())
    dead = int((back[:, kept_index] != 0.0).any(axis=0).sum())
    print(f"written  : {args.out}")
    print(f"reread   : {back.shape}, {len(set(back_uids))} unique uids")
    print(f"present  : {int(back_present.sum())} of {len(back_present)} studies")
    print(f"non-zero : {live} channels fire at least once ({dead} of them kept)")
    print(f"density  : {float((back != 0.0).sum(axis=1).mean()):.2f} channels/study")
    print()
    print(f"measured text-only macro on this table : {RESIDUE_MACRO:.4f}")
    print(f"the leak bar (label pipeline)          : {PIPELINE_MACRO:.4f}")
    print("Run scripts/probe_report_leak.py against it before training a teacher.")
    print("ACL and MCL still score 0.784 and 0.752 text-only under this rule --")
    print("report them separately from the other ten in every distillation result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
