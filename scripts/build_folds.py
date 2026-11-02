#!/usr/bin/env python3
"""Cut the folds and prove they do not leak, then write data/folds/folds_vN.csv.

Argparse plus one call into kappa.data.folds, per the repo contract.

    python scripts/build_folds.py                     # v1: five folds, no holdout
    python scripts/build_folds.py --version v2        # v2: 15% frozen test + 5 folds
    python scripts/build_folds.py --version v2 --dry-run       # measure only
    python scripts/build_folds.py --group-key scanner_fingerprint

``--version v2`` is the one to use for anything new. It holds out 15% of the
corpus -- 661 of 4,407 studies -- as ``split == "test"`` with ``fold == -1``, and
cuts five folds over the remaining 3,746. **That test set is frozen and is not
looked at until the pipeline is finished**; the 58 anchor studies were spent on
exactly that mistake and are pinned into ``dev`` here so the test set is clean of
anything already tuned against. The cost is stated rather than hidden: a fold
trains on ~2,997 studies where v1 gave it ~3,526, and the submission models
should be retrained on dev+test once the pipeline is frozen.

``folds_v1.csv`` stays exactly as it is. The default version is ``v1`` so that
the command which produced the frozen file still produces that file, byte for
byte.

Three inputs are joined, and each contributes one thing the split needs:

* the **manifest** gives ``patient_id`` and ``scanner_fingerprint`` per study,
* the **derived label table** gives the twelve labels and their usability masks,
* **train.csv** gives the report text, from which the language bucket is derived
  with ``kappa.labels.normalise.detect_script`` -- offline, no model, four
  buckets measured over all 4,407 reports as latin 3,320, turkish 546, greek
  321, cyrillic 220.

The gate is asserted here, not asserted in prose: at least one usable positive
for every one of the twelve labels in every fold -- and, under ``--version v2``,
in the test set too -- plus zero group overlap between every pair of folds and
between dev and test. The script exits non-zero if any of them fails, so a bad
split cannot be written and then trusted.

Two fold columns are written and neither is preferred. ``fold`` groups by report
content; ``fold_scanner`` also keeps each scanner inside one fold. See
``kappa.data.folds`` for why that choice is left open. Under v2 the fold arms are
measured over the **dev pool alone**: the test set is scanner-shared on purpose,
so a scanner spanning dev and test is the design and not a leak, and gating the
``fold_scanner`` arm over the whole table would fail the very split it asks for.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

import kappa.labels  # noqa: E402
from kappa.constants import ID_COLUMN, LABELS  # noqa: E402
from kappa.data.folds import (  # noqa: E402
    DEFAULT_GROUP_KEYS,
    MASK_SUFFIX,
    N_FOLDS,
    SCANNER_GROUP_KEYS,
    TEST_FRACTION,
    build_folds,
    build_folds_v2,
    dev_mask,
    fold_report,
    holdout_report,
    write_folds,
)
from kappa.data.manifest import read_manifest, study_table  # noqa: E402
from kappa.data.paths import data_root, folds_dir, interim_dir, raw_dir  # noqa: E402
from kappa.labels.normalise import content_hash, detect_script, normalise_report  # noqa: E402


#: Default home of anchors.csv, alongside the original label-mapping scripts. Not
#: published, because it contains report text; nothing in shipped code imports from
#: it -- anchors.csv is read as *data*, with csv.
RECOVERED = Path(kappa.labels.__path__[0]) / "_recovered"


#: Substring identifying the balance warning raised by :func:`kappa.data.folds.assign_folds`.
#: Matched on text because the warning carries no dedicated category; if that message
#: is reworded, ``test_build_folds.py`` fails rather than the gate going quiet.
_BALANCE_WARNING = "cannot be balanced"


def _build_gated(build):
    """Run a fold builder and count an unbalanceable split as a gate failure.

    ``assign_folds`` *warns* rather than raises when a group is larger than one
    fold's share, and a warning does not change an exit code. Without this, a
    split the library had just called unbalanceable was still written to disk
    under "GATE PASSED" -- the failure mode this whole script exists to prevent,
    reproduced by the gate itself.

    The warnings are re-emitted to stderr rather than swallowed:
    ``catch_warnings(record=True)`` suppresses display, so capturing without
    printing would trade a silent pass for a silent gate.

    Measured on this corpus, it does not fire: the largest scanner group is 327
    studies against a fold share of 881, and the largest patient group is 37. So
    the gate costs nothing on the current table and exists for one that changes
    that.

    Returns the builder's result and the number of balance warnings seen.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build()
    unbalanced = 0
    for w in caught:
        print(f"warning : {w.message}", file=sys.stderr)
        if _BALANCE_WARNING in str(w.message):
            unbalanced += 1
    if unbalanced:
        print(
            f"GATE FAILED: {unbalanced} grouping(s) cannot be balanced across folds. "
            f"Writing an unbalanceable split would produce folds that look valid and "
            f"are not.",
            file=sys.stderr,
        )
    return result, unbalanced


def _anchor_ids(path: Path) -> tuple[str, ...]:
    """StudyInstanceUIDs of the 58 annotated anchor studies, read as data.

    These are the studies that must stay in ``dev``. They are **spent**: they were
    read to measure the label derivation at macro balanced accuracy 0.765, which turned them from an unseen test set into a regression check. A
    test set containing them is not clean, and nothing may be tuned against them.

    Same leading-``#``-comment skip as ``scripts/derive_labels.py:read_truth``,
    against the same file. Read with ``csv``, never imported.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        if reader.fieldnames is None or ID_COLUMN not in reader.fieldnames:
            raise ValueError(f"{path} has no {ID_COLUMN!r} column: {reader.fieldnames}")
        return tuple(row[ID_COLUMN] for row in reader)


def _find(*candidates: Path) -> Path | None:
    """First candidate that exists, or None."""
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _report_facts(train_csv: Path) -> tuple[dict[str, str], dict[str, str]]:
    """One pass over train.csv for the two things folds need from report text.

    Returns ``(language, report_group)``. ``language`` is a script bucket, used as
    the stratification key. ``report_group`` is the content hash of the normalised
    report, used as a **grouping** key -- two studies carrying the same report text
    must not land in different folds.

    ``normalise_report`` then ``content_hash``, matching ``patient_linkage`` exactly
    so the grouping reproduces the measurement that motivated it (52 groups over 197
    studies). Deliberately **not** ``fold_lookalikes``: folding U+2206 onto delta is
    right when matching Greek and Turkish, and wrong here, because it would merge
    genuinely distinct reports and quietly bind unrelated studies into one fold.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    language: dict[str, str] = {}
    report_group: dict[str, str] = {}
    with open(train_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row["StudyInstanceUID"]
            normalised = normalise_report(row["Report"])
            language[uid] = detect_script(normalised)
            report_group[uid] = content_hash(normalised)
    return language, report_group


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="defaults to <interim>/manifest.parquet")
    ap.add_argument("--labels", type=Path, default=None,
                    help="defaults to <interim>/labels/derived.parquet")
    ap.add_argument("--train-csv", type=Path, default=None,
                    help="train.csv, for the language bucket")
    ap.add_argument("--version", choices=("v1", "v2"), default="v1",
                    help="v1: five folds over everything. v2: a frozen 15%% test set "
                         "plus five folds over the rest. Default v1 so the command "
                         "that produced the frozen folds_v1.csv still produces it")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to data/folds/folds_<version>.csv. Must be .csv")
    ap.add_argument("--test-frac", type=float, default=None,
                    help=f"v2 only: share of the corpus frozen as the test set "
                         f"(default {TEST_FRACTION}, i.e. 661 of 4,407 studies)")
    ap.add_argument("--anchors-csv", type=Path, default=RECOVERED / "anchors.csv",
                    help="v2 only: the 58 spent anchor studies, pinned into dev. "
                         "They were tuned against, so a test set "
                         "holding them is not clean")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=0,
                    help="decides the group order, so it decides the split. Recorded.")
    ap.add_argument("--group-key", action="append", default=None,
                    help="grouping column for the `fold` arm; repeat for several. "
                         "Groups are the connected components over all of them. "
                         "Defaults to patient_id -- change this if patients turn "
                         "out to recur, no rewrite needed")
    ap.add_argument("--no-language-strata", action="store_true",
                    help="turn off language stratification. Only to measure what it buys")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    # Refused rather than ignored. A silently dropped --test-frac hands back five
    # folds over the whole corpus while the operator believes 15% is held out.
    if args.version == "v1" and args.test_frac is not None:
        print("--test-frac only means something with --version v2; v1 holds nothing out",
              file=sys.stderr)
        return 2
    test_frac = TEST_FRACTION if args.test_frac is None else args.test_frac

    manifest_path = args.manifest or (interim_dir() / "manifest.parquet")
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}. Run: python scripts/build_manifest.py",
              file=sys.stderr)
        return 1
    studies = study_table(read_manifest(manifest_path))
    studies = studies.rename(columns={"study_uid": ID_COLUMN})
    print(f"manifest: {manifest_path} -> {len(studies)} studies")

    labels_path = args.labels or (interim_dir() / "labels" / "derived.parquet")
    if not labels_path.is_file():
        print(f"no label table at {labels_path}. Run: python scripts/derive_labels.py",
              file=sys.stderr)
        return 1
    labels = pd.read_parquet(labels_path)
    missing = [label for label in LABELS if label not in labels.columns]
    if missing:
        print(f"label table is missing {missing}", file=sys.stderr)
        return 1
    print(f"labels  : {labels_path} -> {len(labels)} studies, "
          f"{sum(1 for c in labels.columns if c.endswith(MASK_SUFFIX))} mask columns")

    train_csv = args.train_csv or _find(
        data_root() / "meta" / "train.csv", raw_dir() / "train.csv",
        data_root() / "subset" / "train.csv",
    )
    if train_csv is None:
        print("no train.csv found, so language stratification is impossible. Pass "
              "--train-csv, or --no-language-strata to split without it.", file=sys.stderr)
        if not args.no_language_strata:
            return 1
        language, report_group = {}, {}
    else:
        language, report_group = _report_facts(train_csv)
        print(f"language: {train_csv} -> {len(language)} reports")

    table = studies.merge(labels, on=ID_COLUMN, how="inner")
    if table.empty:
        print("the manifest and the label table share no studies", file=sys.stderr)
        return 1
    table["language"] = table[ID_COLUMN].map(language).fillna("unknown")
    # patient_id links nothing -- the IDs are re-keyed per study, so they group
    # nothing. Identical report text is the only remaining link between two studies;
    # see kappa.data.folds.DEFAULT_GROUP_KEYS. A study with no report falls back to
    # its own uid, which makes it a singleton rather than merging every such study
    # into one giant group.
    table["report_group"] = (
        table[ID_COLUMN].map(report_group).fillna(table[ID_COLUMN])
    )
    _linked = int(table["report_group"].duplicated(keep=False).sum())
    print(f"report groups: {table['report_group'].nunique()} distinct, "
          f"{_linked} studies share their text with another study")
    print(f"joined  : {len(table)} studies with both a manifest row and a label row")
    print("language buckets: "
          + ", ".join(f"{k} {v}" for k, v in table['language'].value_counts().items()))

    group_keys = tuple(args.group_key) if args.group_key else None
    common = dict(
        n_folds=args.n_folds,
        strata_key=None if args.no_language_strata else "language",
        seed=args.seed,
        **({"group_keys": group_keys} if group_keys else {}),
    )

    failures = 0
    # Three gates, not two: the two fold arms plus the balance check below.
    n_gates = 3
    if args.version == "v1":
        folds, unbalanced = _build_gated(lambda: build_folds(table, **common))
        failures += unbalanced
        arms = (
            ("fold", "patient_group", ("scanner_group",)),
            ("fold_scanner", "scanner_group", ("patient_group",)),
        )
        cv_rows = folds
    else:
        if not args.anchors_csv.is_file():
            print(f"no anchor list at {args.anchors_csv}. The 58 spent anchors must be "
                  f"pinned into dev; building without them would put studies that have "
                  f"already been tuned against into the frozen test set.\n"
                  f"  The file is the 58 labelled training studies: StudyInstanceUID "
                  f"plus the 12 label columns, taken from the labelled rows of "
                  f"train.csv. The default under src/kappa/labels/_recovered/ is not "
                  f"published, because it contains report text. Build the file and "
                  f"pass --anchors-csv PATH.", file=sys.stderr)
            return 1
        anchors = _anchor_ids(args.anchors_csv)
        print(f"anchors : {args.anchors_csv} -> {len(anchors)} spent studies pinned to dev")
        if group_keys:
            # The holdout is grouped on the same keys as the `fold` arm, so
            # --group-key scanner_fingerprint here does not just change the folds:
            # it makes the test set scanner-DISJOINT, which is the opposite of the
            # design and would be invisible in the gate output.
            print(f"NOTE    : --group-key {list(group_keys)} also groups the dev/test "
                  f"boundary. A scanner key here makes the test set scanner-disjoint, "
                  f"which is pessimistic: the hidden test set comes from the same sites.")
        folds, unbalanced = _build_gated(
            lambda: build_folds_v2(
                table, test_frac=test_frac, keep_ids=anchors, **common
            )
        )
        failures += unbalanced

        n_gates = 4
        holdout = holdout_report(folds, table, group_column="report_group",
                                 extra_group_columns=("scanner_group",),
                                 gate_languages=not args.no_language_strata)
        print("\n=== dev vs test === (the test set is FROZEN: do not look at it again "
              "until the pipeline is finished)")
        print(holdout.table())
        if not holdout.ok:
            failures += 1

        # The fold arms are measured over the dev pool alone. Over the whole table
        # the fold_scanner arm would fail on a scanner spanning dev and test --
        # which is the design, not a leak: the hidden Kaggle test set comes from
        # the same sites, so a scanner-disjoint local test measures a harder
        # problem than the one being scored.
        cv_rows = folds.loc[dev_mask(folds)].reset_index(drop=True)
        arms = (
            ("fold", "report_group", ("scanner_group",)),
            ("fold_scanner", "scanner_group", ("report_group",)),
        )
        n_test = len(folds) - len(cv_rows)
        print(f"\nholdout : {n_test} of {len(folds)} studies frozen as test "
              f"({n_test / len(folds):.1%}), {len(cv_rows)} left in the dev pool. "
              f"A fold therefore trains on ~{len(cv_rows) - len(cv_rows) // args.n_folds} "
              f"studies instead of ~{len(folds) - len(folds) // args.n_folds}. Retrain "
              f"the submission models on dev+test once the pipeline is frozen.")

    for column, group_column, extras in arms:
        # Gated on the `fold` arm only. Language collapses across scanner groups
        # by construction here -- cyrillic 164/13/5/2/3 as measured -- which is a
        # fact about where the scanners are, not a defect in the split, and gating
        # it would reject the pessimistic arm the leakage read depends on.
        report = fold_report(
            cv_rows, table, fold_column=column, group_column=group_column,
            extra_group_columns=extras,
            gate_languages=column == "fold" and not args.no_language_strata,
        )
        print(f"\n=== {column} ===")
        print(report.table())
        if not report.ok:
            failures += 1

    if failures:
        print(f"\nGATE FAILED for {failures} of {n_gates} checks; writing nothing",
              file=sys.stderr)
        return 1

    print("\nGATE PASSED: every label has a positive in every fold, and no group "
          "spans two folds in the arm that groups on it.")
    if args.version == "v2":
        print("           : every label also has a positive in the test set, and no "
              "report group spans dev and test.")
        print("NOTE       : scanner_group deliberately DOES span dev and test -- the "
              "test set is scanner-shared, because the hidden test set is drawn from "
              "the same sites. The fold -> fold_scanner CV gap is the pessimistic "
              "bound, and it is an UPPER bound on site memorisation, not a "
              "measurement: fold_scanner is degenerate here (337 groups, largest 327 "
              "studies, language collapses across it).")
    if args.dry_run:
        print("--dry-run: nothing written")
        return 0

    out = args.out or (folds_dir() / f"folds_{args.version}.csv")
    write_folds(folds, out)
    print(f"wrote {len(folds)} rows -> {out}")
    print("verify it is tracked, not silently gitignored:")
    print(f"    git check-ignore -v {out}     # must exit 1")
    # DEFAULT_GROUP_KEYS, not a literal. This line was hardcoded to ['patient_id']
    # and kept printing it after report_group became a default key, so the run log
    # misreported the provenance of the very file it had just written.
    print(f"{args.version}, seed {args.seed}, group keys "
          f"{list(group_keys) if group_keys else list(DEFAULT_GROUP_KEYS)} / "
          f"{list(SCANNER_GROUP_KEYS)}"
          + (f", test_frac {test_frac}" if args.version == "v2" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
