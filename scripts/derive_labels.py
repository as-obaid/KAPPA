#!/usr/bin/env python3
"""Derive the 12 competition labels from extraction records and write the table.

Argparse plus one call into kappa.labels, per the repo contract: the derivation
and the scoring arithmetic live in the package so they are testable and
importable, and only file I/O and formatting live here.

    python scripts/derive_labels.py --records data/interim/discovery/records.jsonl
    python scripts/derive_labels.py --records ... --report-features
    python scripts/derive_labels.py --anchors

``--report-features`` additionally writes the teacher's privileged input table
from the same records in the same pass. Two passes over separately resolved shard
files is how a feature table ends up describing a different corpus than the
labels it will be trained beside.

``--anchors`` scores the derivation against the 58 annotated studies. Those
anchors are **spent** -- they were used once to measure the derivation and are
no longer an unseen test set -- so this mode is a regression check, not a
tuning loop. Do not adjust a gate or a cluster because a row here moved.

The printed table gives an explicit per-label denominator (``n_cov``) and a
coverage rate. That is not decoration: a derived label is *unknown* wherever
every finding bearing on it was masked, so ten of the twelve rows cover fewer
than 58 studies. Printing prevalence over all 58 regardless -- which the
original driver did -- credits a row with evidence it never had.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import kappa.labels  # noqa: E402
from kappa.constants import ID_COLUMN, LABELS  # noqa: E402
from kappa.data.paths import interim_dir  # noqa: E402
from kappa.labels.label_map import (  # noqa: E402
    TABLE_COLUMNS, UNMENTIONED_BY_LABEL, StudyLabels, derive_studies, score_derivation,
)

#: Default home of the anchor ground truth and the anchor extractions, alongside the
#: original label-mapping scripts. Not published, because it contains report text;
#: read as data only -- nothing in ``src/kappa/`` imports code from there.
RECOVERED = Path(kappa.labels.__path__[0]) / "_recovered"

#: What each ``--anchors`` input is, for the error printed when one is missing.
ANCHOR_INPUTS = {
    "--anchors-csv": "the 58 labelled training studies: StudyInstanceUID plus the 12 "
                     "label columns, taken from the labelled rows of train.csv",
    "--anchor-records": "the LLM extraction output (records.jsonl, as written by "
                        "scripts/extract_reports.py) for those 58 studies",
}

#: What ``unmentioned`` a label nothing mentions gets, by policy name. ``zero``
#: is the default because the measured macro balanced accuracy on the anchors
#: was produced with it, and that number has to stay reproducible; ``per-label``
#: is the measured-better policy for four labels and changes the number.
UNMENTIONED_POLICIES = ("zero", "unknown", "per-label")


def read_records(path: Path) -> list[dict]:
    """Load a records.jsonl, one extraction record per line."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve_record_paths(given: Sequence[Path] | None, default_dir: Path) -> list[Path]:
    """Expand what the caller asked for into an explicit, sorted list of files.

    The extraction pass writes ONE FILE PER SHARD -- ``records.shard03of08.jsonl``
    and so on -- because a single output file would mean eight array tasks
    appending to the same path. So the common case here is eight inputs, not
    one, and requiring the caller to concatenate them by hand invites doing it
    wrong: a missed shard produces a label table that is quietly short by ~550
    studies and looks perfectly healthy.

    Accepts explicit paths, directories, and glob patterns. Directories are
    searched for shard files first and fall back to a plain ``records.jsonl``.
    """
    if given:
        out: list[Path] = []
        for item in given:
            if item.is_dir():
                out.extend(sorted(item.glob("records.shard*of*.jsonl")) or
                           sorted(item.glob("records*.jsonl")))
            elif item.is_file():
                out.append(item)
            else:
                # An unmatched glob must not silently contribute nothing.
                matches = sorted(item.parent.glob(item.name))
                if not matches:
                    raise FileNotFoundError(f"no records matched {item}")
                out.extend(matches)
        # De-duplicate while keeping order, so `--records dir dir/f.jsonl` is safe.
        seen, uniq = set(), []
        for f in out:
            r = f.resolve()
            if r not in seen:
                seen.add(r)
                uniq.append(f)
        return uniq

    shards = sorted(default_dir.glob("records.shard*of*.jsonl"))
    if shards:
        return shards
    single = default_dir / "records.jsonl"
    return [single] if single.is_file() else []


def load_all_records(paths: Sequence[Path]) -> tuple[list[dict], list[str]]:
    """Concatenate several shard files, refusing to hide a duplicated study.

    Shards are disjoint by construction -- membership is a hash of the report
    text modulo the shard count -- so a ``study_uid`` appearing twice means the
    inputs are wrong: a stale file from a different shard count, or the same
    shard passed twice. Keeping the first and moving on would produce a
    plausible table from mismatched inputs, so the duplicates are returned and
    the caller decides.
    """
    records: list[dict] = []
    first_seen: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        for record in read_records(path):
            uid = record.get("_study_uid") or record.get("study_uid")
            if uid is not None and uid in first_seen:
                duplicates.append(f"{uid} in {path.name} and {first_seen[uid].name}")
                continue
            if uid is not None:
                first_seen[uid] = path
            records.append(record)
    return records, duplicates


def read_truth(path: Path) -> dict[str, dict[str, int]]:
    """Load the anchor ground truth, skipping the leading ``#`` comment line."""
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        return {row[ID_COLUMN]: {label: int(row[label]) for label in LABELS} for row in reader}


def unmentioned_for(policy: str) -> float | None | dict[str, float | None]:
    """Translate a policy name into the ``unmentioned`` argument."""
    if policy == "zero":
        return 0.0
    if policy == "unknown":
        return None
    return dict(UNMENTIONED_BY_LABEL)


def print_anchor_table(derived: dict[str, StudyLabels], truth: dict[str, dict[str, int]]) -> None:
    """Print per-label agreement with every denominator stated."""
    score = score_derivation(derived, truth)
    print(f"scored studies: {score.n_compared} "
          f"(derived {len(derived)}, ground truth {len(truth)})")
    print(f"{'label':<18}{'n_cov':>6}{'cov%':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
          f"{'sens':>7}{'spec':>7}{'bal-acc':>9}{'gold+':>7}{'der+':>7}")
    for row in score.per_label:
        # Both prevalences are per label, over `n_covered`. The row's own
        # denominator is printed next to them so the two can never drift.
        gold = 100 * row.n_gold_positive / row.n_covered if row.n_covered else math.nan
        derived_positive = (
            100 * row.n_derived_positive / row.n_covered if row.n_covered else math.nan
        )
        print(f"{row.label:<18}{row.n_covered:6d}{100 * row.coverage:5.0f}%"
              f"{row.tp:5d}{row.fp:5d}{row.fn:5d}{row.tn:5d}"
              f"{row.sensitivity:7.2f}{row.specificity:7.2f}{row.balanced_accuracy:9.2f}"
              f"{gold:6.0f}%{derived_positive:6.0f}%")
    print(f"\nmacro balanced accuracy: {score.macro_balanced_accuracy:.3f}  (0.50 = coin flip)")
    print(f"label coverage: {100 * score.coverage:.1f}% of "
          f"{score.n_compared} x {len(LABELS)} label cells were usable")
    print("gold+ and der+ are percentages of n_cov, NOT of the studies compared: "
          "a masked label is absent evidence, not a derived negative.")
    if score.skipped:
        print(f"balanced accuracy undefined (one class only, or no coverage), "
              f"excluded from the macro: {', '.join(score.skipped)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, nargs="+", default=None, metavar="PATH",
                    help="one or more records.jsonl files, directories, or globs. "
                         "The extraction array writes one file per shard, so this "
                         "normally takes all eight. Defaults to the shard files "
                         "under <data_root>/interim/extraction, then "
                         "<data_root>/interim/discovery/records.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <data_root>/interim/labels/derived.parquet")
    ap.add_argument("--anchors", action="store_true",
                    help="score against the 58 annotated studies instead of the "
                         "extraction run. THE ANCHORS ARE SPENT: this is a "
                         "regression check, never a tuning signal.")
    ap.add_argument("--anchors-csv", type=Path, default=RECOVERED / "anchors.csv",
                    help="ground truth, 58 studies x 12 labels")
    ap.add_argument("--anchor-records", type=Path, default=RECOVERED / "anchors_records.jsonl",
                    help="extraction output for the annotated studies")
    ap.add_argument("--unmentioned", choices=UNMENTIONED_POLICIES, default="zero",
                    help="value for a label no finding mentions (default: zero, "
                         "which is what the quoted anchor score was measured with)")
    ap.add_argument("--no-fan-out", action="store_true",
                    help="ablation: let an ambiguous cluster-label pair contribute "
                         "nothing instead of negating on a negative statement")
    ap.add_argument("--no-write", action="store_true",
                    help="derive and summarise without writing the parquet")
    ap.add_argument("--report-features", type=Path, nargs="?", const=Path("-"),
                    default=None, metavar="PATH",
                    help="ALSO write the teacher's privileged feature table: one row "
                         "per study, 78 anatomy clusters x 2 polarity channels. "
                         "Derived from the SAME records as the labels in the same "
                         "pass, so the two artifacts cannot drift. Defaults to "
                         "<data_root>/interim/labels/report_features.parquet. "
                         "TRAINING ONLY -- test studies have no report")
    ap.add_argument("--by-language", action="store_true",
                    help="also print per-label prevalence broken out by language "
                         "bucket. A label whose "
                         "prevalence tracks language is a scanner-signature "
                         "shortcut the image model can take instead of learning "
                         "the pathology, and no robust loss repairs it")
    args = ap.parse_args()

    if args.anchors:
        for flag, path in (("--anchors-csv", args.anchors_csv),
                           ("--anchor-records", args.anchor_records)):
            if not path.is_file():
                print(f"error: {flag} not found: {path}\n"
                      f"  It is {ANCHOR_INPUTS[flag]}.\n"
                      f"  The default under src/kappa/labels/_recovered/ is not "
                      f"published, because it contains report text. Build the file "
                      f"and pass {flag} PATH.", file=sys.stderr)
                return 2
        record_paths = [args.anchor_records]
    else:
        try:
            record_paths = resolve_record_paths(args.records, interim_dir() / "extraction")
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not record_paths:
            record_paths = resolve_record_paths(args.records, interim_dir() / "discovery")
    if not record_paths:
        print("error: no records found; pass --records", file=sys.stderr)
        return 2
    missing = [p for p in record_paths if not p.is_file()]
    if missing:
        print(f"error: no records at {missing[0]}; pass --records", file=sys.stderr)
        return 2

    records, duplicates = load_all_records(record_paths)
    if duplicates:
        # Not a warning. Shards are disjoint by construction, so this means the
        # inputs are mismatched and every number below would be wrong.
        print(f"error: {len(duplicates)} study_uid(s) appear in more than one input, "
              f"so the shard files do not agree. First few:", file=sys.stderr)
        for line in duplicates[:5]:
            print(f"  {line}", file=sys.stderr)
        return 2
    labels = list(derive_studies(
        records,
        fan_out_negatives=not args.no_fan_out,
        unmentioned=unmentioned_for(args.unmentioned),
    ))
    if len(record_paths) == 1:
        print(f"records   : {record_paths[0]}")
    else:
        print(f"records   : {len(record_paths)} files, {len(records)} records, "
              f"{record_paths[0].parent}")
        for path in record_paths:
            print(f"            {path.name}")
    print(f"derived   : {len(labels)} studies, unmentioned={args.unmentioned}, "
          f"fan_out_negatives={not args.no_fan_out}")

    if args.by_language:
        from kappa.evaluation.label_audit import audit_by_language
        pairs = [
            (str(record.get("language") or "unknown"), study)
            for record, study in zip(records, labels)
        ]
        print()
        print(audit_by_language((), unmentioned=unmentioned_for(args.unmentioned),
                               derived=pairs).render())
        print()

    reasons = collections.Counter(r for study in labels for r in study.mask_reasons)
    usable = sum(study.n_usable for study in labels)
    cells = len(labels) * len(LABELS)
    print(f"usable    : {usable}/{cells} label cells "
          f"({100 * usable / cells if cells else 0:.1f}%)")
    for reason, count in reasons.most_common():
        print(f"  flagged : {reason} on {count} studies")

    if args.anchors:
        print()
        truth = read_truth(args.anchors_csv)
        print_anchor_table({s.study_uid: s for s in labels if s.study_uid}, truth)

    if args.report_features is not None:
        # Written from `records`, the same list `derive_studies` consumed above. A
        # separate script over a separately resolved set of shards is how the
        # feature table ends up describing a different corpus than the labels.
        from kappa.labels.features import REPORT_FEATURE_DIM, write_features
        features_out = (
            interim_dir() / "labels" / "report_features.parquet"
            if str(args.report_features) == "-"
            else args.report_features
        )
        write_features(records, features_out)
        print(f"features  : {len(records)} studies x {REPORT_FEATURE_DIM} -> {features_out}")

    if args.no_write:
        return 0
    if args.anchors and args.out is None:
        # Never let the scoring mode overwrite the real table by default. The
        # anchor run covers 58 studies out of 4,407, and a 58-row file sitting
        # at the pinned path looks exactly like a finished derivation.
        print("note      : --anchors scored only; pass --out to write its 58-row table")
        return 0

    # pandas is imported here rather than at module scope so that --anchors,
    # which writes nothing, does not depend on it.
    import pandas as pd

    out_path = args.out or (interim_dir() / "labels" / "derived.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([study.as_row() for study in labels], columns=list(TABLE_COLUMNS))
    frame.to_parquet(out_path, index=False)
    print(f"wrote     : {out_path} ({len(frame)} rows x {len(frame.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
