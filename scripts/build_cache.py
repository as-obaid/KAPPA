#!/usr/bin/env python3
"""Decode DICOM series into the uint8 cache the dataloader reads.

Argparse plus one call into kappa.data.dicom, per the repo contract: the
decoding, ordering, windowing and letterboxing all live in the package so they
are testable and importable, not here.

    python scripts/build_cache.py --limit 4                 # smoke test
    python scripts/build_cache.py --shard-index 3 --shard-count 64

With no shard flags it reads ``SLURM_ARRAY_TASK_ID``, so the same command line
works standalone and as an array element. Resumable: a series whose ``.npy``
already exists is skipped, so a cancelled element costs one element.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.data.dicom import CACHE_SIZE, MAX_SLICES, WINDOW_PERCENTILES, build_cache  # noqa: E402
from kappa.data.paths import cache_dir, require_series_root, subset_dir  # noqa: E402


def _slurm_shard() -> tuple[int, int]:
    """This element's ``(index, count)`` from the Slurm array environment.

    ``SLURM_ARRAY_TASK_MIN`` is subtracted rather than assumed to be zero.
    ``--array=1-64`` is the more common spelling, and using ``TASK_ID``
    unshifted against a 64-way stride would then never process shard 0 and would
    ask for shard 64 -- so 1/64th of the corpus would be silently missing from
    the cache and the job would report success.
    """
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        return 0, 1
    first = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    return int(task) - first, count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--series-root", type=Path, default=None,
                    help="defaults to <raw>/<split>_series, falling back to the "
                         f"local subset at {subset_dir()}")
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="defaults to <processed>/cache; layout is "
                         "<cache>/<study_uid>/<series_uid>.npy")
    ap.add_argument("--studies-file", type=Path, default=None,
                    help="newline-separated study UIDs to restrict the pass to")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="defaults to SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN")
    ap.add_argument("--shard-count", type=int, default=None,
                    help="defaults to SLURM_ARRAY_TASK_COUNT")
    ap.add_argument("--size", type=int, default=CACHE_SIZE,
                    help="cached edge length. 512 is the pinned cache contract: the "
                         "median native matrix is 512, so it is near-lossless, and "
                         "the dataloader can downsample but cannot invent detail")
    ap.add_argument("--max-slices", type=int, default=MAX_SLICES)
    ap.add_argument("--window", type=float, nargs=2, default=list(WINDOW_PERCENTILES),
                    metavar=("LOW", "HIGH"),
                    help="per-series percentile window, not per-slice and not global")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap studies in this shard; use for a smoke test")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-decode series that are already cached, which disables "
                         "the resume and makes a killed job cost the whole shard")
    ap.add_argument("--max-failures", type=float, default=0.01,
                    help="fraction of attempted series allowed to fail before this "
                         "exits nonzero; 0 means any failure fails the job")
    args = ap.parse_args()

    series_root = args.series_root or require_series_root(args.split)
    cache_root = args.cache_root or cache_dir()
    shard_index, shard_count = _slurm_shard()
    if args.shard_index is not None:
        shard_index = args.shard_index
    if args.shard_count is not None:
        shard_count = args.shard_count

    studies = None
    if args.studies_file is not None:
        studies = [
            line.strip()
            for line in args.studies_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    print(f"series root : {series_root}")
    print(f"cache root  : {cache_root}")
    print(f"shard       : {shard_index} of {shard_count}")
    print(f"geometry    : {args.size}x{args.size}, <= {args.max_slices} slices, "
          f"window {tuple(args.window)}")
    if studies is not None:
        print(f"studies     : {len(studies)} from {args.studies_file}")

    report = build_cache(
        series_root,
        cache_root,
        studies=studies,
        shard_index=shard_index,
        shard_count=shard_count,
        size=args.size,
        max_slices=args.max_slices,
        window=(float(args.window[0]), float(args.window[1])),
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(report.summary())

    for name, message in report.failures[:20]:
        print(f"  FAILED {name}: {message}", file=sys.stderr)
    if len(report.failures) > 20:
        print(f"  ... and {len(report.failures) - 20} more", file=sys.stderr)

    # A shard whose studies were all cached by an earlier run is a success with
    # nothing to do, so "nothing written" is only a failure when something was
    # also attempted and failed.
    if report.n_attempted and report.n_failed > args.max_failures * report.n_attempted:
        print(
            f"FAILED: {report.n_failed} of {report.n_attempted} attempted series "
            f"failed to decode, over the {args.max_failures:.1%} allowance. Rerun "
            f"is safe -- the successes are already cached.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
