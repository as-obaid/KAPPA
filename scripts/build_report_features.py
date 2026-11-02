#!/usr/bin/env python3
"""Write the teacher's report-feature table: one row per study, 156 channels.

    python scripts/build_report_features.py \
        --records /path/to/extraction \
        --out data/interim/labels/report_features.parquet

This is the input ``DistillConfig.report_features`` names, and without it a
teacher cannot be built at all. This script is the only caller of
``kappa.labels.features.write_features``.

**The records are not under ``data/``.** The extraction shards live in the
extraction job's output directory, not in ``data/interim/discovery/``, which
holds only a 300-study pilot and a 58-study anchor run. Pointing this at
``discovery`` produces a table quietly short by 4,107 studies that looks
perfectly healthy -- the same failure ``resolve_record_paths`` was written to
prevent one level up. So ``--records`` is required with no default: there is no
directory this script could guess that is right more often than it is wrong.

**What a row is.** ``study_features`` emits 78 anatomy clusters x 2 polarity
channels, named ``ACL:pos``, ``ACL:neg`` and so on. The ``report_present`` flag
is False only for ``extraction_failed``; an empty findings section is True,
because a report that exists and says nothing is evidence and must not be gated
like a report that failed to parse.

**This does not decide whether the teacher is worth building.** 4,349 of the
4,407 label rows were themselves derived from these same reports, so a teacher
reading them can minimise its loss by relearning the label pipeline without ever
reading a pixel -- and that failure looks exactly like success. The text-only
probe is what answers that, and it reads the table this script writes. Build the
table, run the probe, then decide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from derive_labels import load_all_records, resolve_record_paths  # noqa: E402

from kappa.labels.features import (  # noqa: E402
    REPORT_FEATURE_DIM,
    load_features,
    write_features,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--records",
        type=Path,
        nargs="+",
        required=True,
        help="extraction shard files, directories or globs. Required and "
        "deliberately without a default -- see the module docstring.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/interim/labels/report_features.parquet"),
        help="destination parquet (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    paths = resolve_record_paths(args.records, default_dir=Path("."))
    if not paths:
        print(f"no records matched {args.records}", file=sys.stderr)
        return 2
    print(f"records  : {len(paths)} file(s)")
    for path in paths:
        print(f"           {path}")

    records, duplicates = load_all_records(paths)
    if duplicates:
        # Shards are disjoint by construction -- membership is a hash of the report
        # text modulo the shard count -- so a duplicate means the inputs are wrong:
        # a stale file from a different shard count, or a shard passed twice.
        # Refuse, rather than build a plausible table from mismatched inputs.
        print(f"REFUSED  : {len(duplicates)} duplicated study_uid(s)", file=sys.stderr)
        for line in duplicates[:10]:
            print(f"           {line}", file=sys.stderr)
        return 3
    print(f"studies  : {len(records)}")

    out = write_features(records, args.out)

    # Read it straight back through the loader the teacher will use. A permuted or
    # truncated feature table trains on permuted inputs and fails silently, so the
    # only useful moment to catch it is now, not at the top of a five-fold run.
    uids, features, present = load_features(out)
    print(f"written  : {out}")
    print(f"shape    : {features.shape} (expect (n, {REPORT_FEATURE_DIM}))")
    print(f"present  : {int(present.sum())} of {len(present)} studies")
    print(f"uids     : {len(set(uids))} unique")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
