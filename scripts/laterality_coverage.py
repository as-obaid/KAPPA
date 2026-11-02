#!/usr/bin/env python3
"""How much of the corpus carries a DICOM side tag, and where it is missing.

Why this is a script rather than a grep: the available figures do not agree, and the
gap between them decides whether laterality canonicalisation is a config change or a
research project.

  * A competition-forum claim that ``ImageLaterality`` is present on **100%** of series.
  * This repo's discovery survey (``src/kappa/labels/discover.py``): ``Laterality`` on
    **47.3%** of series, ``ImageLaterality`` on **0%**.
  * ``src/kappa/data/manifest.py``, in the ``study_table`` docstring: the tag is blank on
    **1,539 of 3,095** series -- 50.27% present -- explicitly scoped to "the local
    subset".

A 50.3% figure that sometimes circulates as an independent check is the third one
rounded: 1,539/3,095 is 49.73% blank. So there are only two measurements, both from
this repo, both below 51%, and they differ by three points because they were taken over
different row sets.

That matters because the downstream design forks on it. If the tag is near-universal,
canonicalisation is a lookup. If it is near half, every untagged series needs its side
derived from the direction cosines walked to the image centre -- reported on the forum to
drop agreement with the populated tag from 96.9% to 58.8% when that walk is skipped --
and that is a different piece of work with a different risk profile.

What this measures: the manifest's own ``laterality`` column, over whatever rows the
manifest holds, sliced by the axes that could explain the disagreement: by slot, by plane,
by scanner fingerprint and by manufacturer. It reports at both the series level and the
study level, because they are different questions and the published figures do not say
which one they answered.

This reads one parquet. It opens no DICOM, needs no GPU, and runs in seconds, so it is
safe to run on a login node.

    python3 scripts/laterality_coverage.py
    python3 scripts/laterality_coverage.py --manifest data/interim/manifest.parquet --json out.json

What a result means, decided before the number is seen so the reading is not fitted to it:

    >= 95% present   canonicalisation is a lookup; no side derivation is needed.
    50-95%           derive the remainder from direction cosines, and validate the
                     derivation against the tagged majority before trusting it.
    < 50%            the repo's figure stands and the forum's does not. Side derivation
                     is mandatory, and its agreement rate against the tagged half gates
                     any training run that relies on it.

The forum's 100% claim is testable here only for ``Laterality``. ``ImageLaterality`` is a
separate DICOM tag that this manifest does not carry at all, and the discovery survey
measured it at 0%. If that survey is right, no amount of re-reading this manifest will
find it, and the forum is describing a different corpus or a different tag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# The manifest folds the tag onto exactly these, or empty. See
# kappa.data.manifest.normalise_laterality and _LATERALITY_MAP: the corpus writes both
# `R` and `RIGHT`, so a raw value count would split one side across two bins.
SIDES = ("L", "R")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalised_side(manifest: pd.DataFrame) -> pd.Series:
    """The ``laterality`` column as upper-case text, NaN folded to empty."""
    return manifest["laterality"].fillna("").astype(str).str.strip().str.upper()


def series_coverage(manifest: pd.DataFrame) -> dict[str, object]:
    """Per-series tag coverage. This is the unit both published figures used."""
    total = len(manifest)
    side = _normalised_side(manifest)
    present = side.isin(SIDES)
    return {
        "unit": "series",
        "total": int(total),
        "present": int(present.sum()),
        "missing": int((~present).sum()),
        "fraction_present": float(present.mean()) if total else 0.0,
        "value_counts": {str(k): int(v) for k, v in side.value_counts().items()},
        # An unexpected value is the failure mode that would make every percentage above
        # wrong in the same direction, so it is surfaced rather than folded into "missing".
        "unexpected_values": sorted({v for v in side.unique() if v and v not in SIDES}),
    }


def study_coverage(manifest: pd.DataFrame) -> dict[str, object]:
    """Per-study coverage, and the studies whose series disagree about the side.

    A study is what the model scores and what the submission has one row of, so this is
    the number that prices the work. ``LR`` is called out separately because a study whose
    series claim both sides cannot be canonicalised by a lookup at all -- it is either a
    bilateral acquisition or a mis-tag, and both need a human decision, not a default.
    """
    side = _normalised_side(manifest)
    frame = pd.DataFrame({"study_uid": manifest["study_uid"].astype(str), "side": side})
    grouped = frame.groupby("study_uid")["side"].apply(
        lambda s: "".join(sorted({v for v in s if v in SIDES}))
    )
    total = len(grouped)
    single = grouped.isin(SIDES)
    return {
        "unit": "study",
        "total": int(total),
        "single_sided": int(single.sum()),
        "conflicted_LR": int((grouped == "LR").sum()),
        "no_tag_at_all": int((grouped == "").sum()),
        "fraction_single_sided": float(single.mean()) if total else 0.0,
    }


def coverage_by(manifest: pd.DataFrame, column: str) -> pd.DataFrame:
    """Tag coverage sliced by one manifest column, commonest stratum first.

    The slice is the point: a corpus at 50% overall could be 100% on three scanners and 0%
    on three others, which is a tractable problem, or uniformly 50%, which is not. The two
    cases need different code and the aggregate cannot tell them apart.
    """
    if column not in manifest.columns:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "key": manifest[column].fillna("").astype(str),
            "present": _normalised_side(manifest).isin(SIDES),
        }
    )
    out = frame.groupby("key")["present"].agg(["size", "sum", "mean"])
    out.columns = ["series", "tagged", "fraction_tagged"]
    return out.sort_values("series", ascending=False)


_SLICE_COLUMNS = ("slot", "plane", "manufacturer", "scanner_fingerprint")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure DICOM side-tag coverage on the manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=_repo_root() / "data/interim/manifest.parquet",
        help="manifest parquet (default: data/interim/manifest.parquet)",
    )
    ap.add_argument("--json", type=Path, default=None, help="also write the summary as JSON")
    ap.add_argument("--top", type=int, default=12, help="strata to print per slice (default 12)")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        # Named, immediate, and it says how to make the file -- the alternative is a
        # pandas traceback that does not.
        print(
            f"manifest not found: {args.manifest}\n"
            "Build it first: sbatch slurm/manifest.sbatch (or scripts/build_manifest.py).",
            file=sys.stderr,
        )
        return 2

    manifest = pd.read_parquet(args.manifest)
    for required in ("laterality", "study_uid"):
        if required not in manifest.columns:
            print(
                f"{args.manifest} has no {required!r} column; it predates the current "
                f"MANIFEST_COLUMNS contract. Rebuild the manifest before reading this.\n"
                f"columns: {list(manifest.columns)}",
                file=sys.stderr,
            )
            return 2

    series = series_coverage(manifest)
    study = study_coverage(manifest)

    print("=" * 78)
    print("DICOM side-tag coverage")
    print("=" * 78)
    print(f"manifest : {args.manifest}\n")
    print(
        f"SERIES   : {series['present']:,} of {series['total']:,} tagged "
        f"= {series['fraction_present']:.4%}"
    )
    print(f"           value counts: {series['value_counts']}")
    if series["unexpected_values"]:
        print(f"           !! UNEXPECTED VALUES (not L/R/empty): {series['unexpected_values']}")
    print()
    print(f"STUDIES  : {study['total']:,} total")
    print(
        f"           {study['single_sided']:,} single-sided "
        f"({study['fraction_single_sided']:.4%})  <- canonicalisable by lookup"
    )
    print(f"           {study['conflicted_LR']:,} claim BOTH sides  <- need a human decision")
    print(f"           {study['no_tag_at_all']:,} carry no tag on any series\n")

    for column in _SLICE_COLUMNS:
        table = coverage_by(manifest, column)
        if table.empty:
            continue
        print(f"--- by {column} (top {args.top} by series count) ---")
        for key, row in table.head(args.top).iterrows():
            label = (key or "<blank>")[:46]
            print(
                f"  {label:<46} {int(row['tagged']):>6,}/{int(row['series']):>6,}"
                f"  {row['fraction_tagged']:>8.2%}"
            )
        if len(table) > args.top:
            print(f"  ... {len(table) - args.top} more strata")
        # A stratum at 0% or 100% is the whole question: it says the tag is missing by
        # source rather than at random, which is what makes a derivation checkable.
        allornone = table[table["fraction_tagged"].isin((0.0, 1.0))]
        if not allornone.empty:
            print(
                f"  [{len(allornone)} of {len(table)} strata are all-or-nothing "
                "-- the tag is missing by source, not at random]"
            )
        print()

    frac = series["fraction_present"]
    print("=" * 78)
    print("VERDICT, against the thresholds set in this file's docstring before the run")
    print("=" * 78)
    if frac >= 0.95:
        print(f"  {frac:.2%} >= 95%: the forum's claim survives for `Laterality`.")
        print("  Canonicalisation is a lookup; no side derivation is needed.")
    elif frac >= 0.50:
        print(f"  {frac:.2%} in [50%, 95%): a real majority is tagged, a real minority is not.")
        print("  Derive the remainder from direction cosines, and validate the derivation")
        print("  against the tagged majority before any training run relies on it.")
    else:
        print(f"  {frac:.2%} < 50%: the repo's figure stands; the forum's does not.")
        print("  Side derivation is mandatory; its agreement rate against the tagged half")
        print("  gates any training run that relies on it.")
    print()
    print("  Note: `ImageLaterality` is a different tag and is not in this manifest.")
    print("  The discovery survey measured it at 0%. This script cannot refute or confirm that.")
    print("  Record the number with the other measured results before quoting it anywhere.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        sliced = {c: coverage_by(manifest, c) for c in _SLICE_COLUMNS}
        payload = {
            "manifest": str(args.manifest),
            "series": series,
            "study": study,
            "by": {
                column: table.reset_index().to_dict("records")
                for column, table in sliced.items()
                if not table.empty
            },
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
