#!/usr/bin/env python3
"""Build the per-series manifest, and with it answer: do patients recur?

Argparse plus one call into kappa.data.manifest, per the repo contract.

    # local 558-study subset, no cluster needed
    python scripts/build_manifest.py

    # full pass, one shard of a Slurm array
    python scripts/build_manifest.py --series-root "$KAPPA_DATA_ROOT/raw/train_series" \\
        --shard "$SLURM_ARRAY_TASK_ID" --n-shards 16
    python scripts/build_manifest.py --merge          # then stitch the shards

Reading a header per series is the whole cost. Measured on the local subset over
exFAT: **160 s for 3,095 series of 24,371 index rows**, most of it filesystem
stat calls rather than parsing, which is why the full pass is sharded.

Patient recurrence is the reason this script exists. Uniqueness is only provisional
until every one of the 4,407 studies has contributed a ``PatientID``, and a
negative result there is **not** proof that patients are distinct -- see
``kappa.data.manifest.PatientLinkage``. Pass ``--train-csv`` (the default finds
it) so the duplicate-report cross-check runs; without it the script cannot tell
"distinct patients" from "re-keyed per study" and says so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from kappa.data.manifest import (  # noqa: E402
    MANIFEST_COLUMNS,
    availability_patterns,
    build_manifest,
    patient_linkage,
    read_series_index,
    slot_coverage,
    study_scanner_fingerprint,
    write_manifest,
)
from kappa.data.paths import data_root, interim_dir, raw_dir  # noqa: E402

# Where the series tree and its index CSV might be, most specific first. The
# local machine holds only the 558-study subset while the cluster holds the full
# 4,407, and the same command has to work on both without an edit.
_ROOT_CANDIDATES = ("subset", "raw", "")


def _resolve_root() -> Path:
    """Directory holding train_series.csv and the train_series/ tree."""
    try:  # tolerate a kappa.data.paths without subset_dir()
        from kappa.data.paths import subset_dir  # type: ignore[attr-defined]

        candidate = Path(subset_dir())
        if (candidate / "train_series.csv").is_file():
            return candidate
    except (ImportError, AttributeError):
        pass
    for name in _ROOT_CANDIDATES:
        candidate = (data_root() / name) if name else raw_dir()
        if (candidate / "train_series.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"no train_series.csv under {data_root()}. Pass --series-csv and "
        f"--series-root explicitly, or stage the data with: make data"
    )


def _report_texts(train_csv: Path) -> dict[str, str]:
    """study_uid -> raw report text, for the duplicate-text cross-check."""
    import csv

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(train_csv, encoding="utf-8", newline="") as handle:
        return {row["StudyInstanceUID"]: row["Report"] for row in csv.DictReader(handle)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series-csv", type=Path, default=None,
                    help="train_series.csv; found under the data root by default")
    ap.add_argument("--series-root", type=Path, default=None,
                    help="tree of <study_uid>/<series_uid>/*.dcm")
    ap.add_argument("--train-csv", type=Path, default=None,
                    help="train.csv, for the duplicate-report cross-check on patient recurrence")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <interim>/manifest.parquet")
    ap.add_argument("--shard", type=int, default=None,
                    help="0-based shard index; shards over sorted study UIDs")
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--merge", action="store_true",
                    help="stitch <out>.part*.parquet into <out> and report, reading no DICOMs")
    ap.add_argument("--require-all", action="store_true",
                    help="fail instead of skipping series with no directory on disk")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of studies; use for a smoke test")
    args = ap.parse_args()

    out = args.out or (interim_dir() / "manifest.parquet")

    if args.merge:
        parts = sorted(out.parent.glob(f"{out.stem}.part*{out.suffix}"))
        if not parts:
            print(f"no shards matching {out.stem}.part*{out.suffix} in {out.parent}",
                  file=sys.stderr)
            return 1
        manifest = pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
        manifest = manifest.drop_duplicates(subset=["series_uid"])
        manifest = manifest.sort_values(["study_uid", "slot", "series_uid"]).reset_index(drop=True)
        manifest = manifest[list(MANIFEST_COLUMNS)]
        print(f"merged {len(parts)} shards -> {len(manifest)} series")
    else:
        series_csv = args.series_csv
        series_root = args.series_root
        if series_csv is None or series_root is None:
            root = _resolve_root()
            series_csv = series_csv or root / "train_series.csv"
            series_root = series_root or root / "train_series"

        studies = None
        index = read_series_index(series_csv)
        all_studies = sorted(index["study_uid"].unique())
        if args.limit is not None:
            all_studies = all_studies[: args.limit]
            studies = all_studies
        if args.shard is not None or args.n_shards is not None:
            if args.shard is None or args.n_shards is None:
                ap.error("--shard and --n-shards must be given together")
            # Strided, not contiguous. Study directories are UUID-named, so
            # there is no disk locality to preserve, and striding spreads the
            # ragged per-study series counts evenly over the array tasks.
            studies = all_studies[args.shard :: args.n_shards]
            out = out.with_name(f"{out.stem}.part{args.shard:03d}{out.suffix}")
            print(f"shard {args.shard}/{args.n_shards}: {len(studies)} studies")

        print(f"index   : {series_csv} ({len(index)} series, "
              f"{index['study_uid'].nunique()} studies)")
        print(f"series  : {series_root}")
        print(f"out     : {out}")
        manifest = build_manifest(
            series_csv, series_root, studies=studies, require_all=args.require_all
        )

    write_manifest(manifest, out)
    print(f"wrote {len(manifest)} series rows for "
          f"{manifest['study_uid'].nunique()} studies -> {out}")

    if manifest.empty:
        print("manifest is empty: nothing to report", file=sys.stderr)
        return 1

    print("\n=== slot coverage ===")
    print(slot_coverage(manifest).to_string(index=False))
    patterns = availability_patterns(manifest)
    print(f"\n=== availability patterns: {len(patterns)} distinct ===")
    print(patterns.head(15).to_string())

    fingerprints = study_scanner_fingerprint(manifest)
    print("\n=== scanner ===")
    print(f"per-series fingerprints : {manifest['scanner_fingerprint'].nunique()}")
    print(f"per-study fingerprints  : {fingerprints.nunique()}")
    print(f"make/model/field triples: "
          f"{len(manifest[['manufacturer', 'model', 'field_strength']].drop_duplicates())}")
    print(f"largest single scanner  : {int(fingerprints.value_counts().iloc[0])} studies "
          f"({fingerprints.value_counts().iloc[0] / len(fingerprints):.1%})")

    train_csv = args.train_csv
    if train_csv is None:
        for candidate in (data_root() / "meta" / "train.csv", raw_dir() / "train.csv",
                          _resolve_root() / "train.csv"):
            if candidate.is_file():
                train_csv = candidate
                break
    reports = _report_texts(train_csv) if train_csv and train_csv.is_file() else None
    print("\n=== patient linkage ===")
    if reports is None:
        print("no train.csv found, so the duplicate-report cross-check is OFF. A clean "
              "'all IDs distinct' result below cannot be distinguished from per-study "
              "re-keying; pass --train-csv.")
    print(patient_linkage(manifest, reports).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
