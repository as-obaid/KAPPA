#!/usr/bin/env python3
"""Extract the competition archive in parallel, resumably.

`unzip` is single-threaded and the archive holds 819,640 small DICOM files. On a
shared parallel filesystem the cost is per-file metadata, not bandwidth, so a
single writer measured ~81 files/s (~2.7 h). Concurrent writers to distinct
paths scale that most of the way to linear.

Resumable by design: an entry whose destination already exists at the right size
is skipped, so a cancelled or pre-empted run continues instead of restarting.
Each worker opens its own ZipFile handle -- a handle cannot be shared across
processes, and reads never contend because the archive is opened read-only.

    python scripts/extract_archive.py --workers 32

CRC is verified per entry as it is written, which is what `unzip` does during
extraction anyway, so no separate `unzip -t` pass is needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kappa.data.paths import STAMP_NAME, data_root  # noqa: E402

_ARCHIVE: str | None = None


def _init(archive: str) -> None:
    global _ARCHIVE
    _ARCHIVE = archive


def _extract_shard(names: list[str], dest: str) -> tuple[int, int, list[str]]:
    """Extract one shard. Returns (written, skipped, errors)."""
    written = skipped = 0
    errors: list[str] = []
    dest_path = Path(dest)
    with zipfile.ZipFile(_ARCHIVE) as zf:
        for name in names:
            info = zf.getinfo(name)
            target = dest_path / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            try:
                if target.exists() and target.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Write to a temp name then rename, so an interrupted write never
                # leaves a truncated file that a later resume would trust.
                tmp = target.with_suffix(target.suffix + f".part{os.getpid()}")
                with zf.open(info) as src, open(tmp, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                os.replace(tmp, target)
                written += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
    return written, skipped, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=2000)
    args = ap.parse_args()

    root = data_root()
    archive = args.archive
    if archive is None:
        candidates = sorted((root / ".stage").glob("*.zip"))
        if len(candidates) != 1:
            print(f"error: expected one .zip in {root/'.stage'}, found {len(candidates)}",
                  file=sys.stderr)
            return 2
        archive = candidates[0]

    dest = args.dest
    if dest is None:
        existing = sorted(root.glob(".raw.incoming.*"))
        dest = existing[0] if existing else root / ".raw.incoming.parallel"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"archive : {archive} ({archive.stat().st_size / 2**30:.1f} GiB)")
    print(f"dest    : {dest}")
    print(f"workers : {args.workers}")

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    print(f"entries : {len(names)}")

    shards = [names[i:i + args.shard_size] for i in range(0, len(names), args.shard_size)]
    written = skipped = 0
    errors: list[str] = []
    started = time.time()

    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(str(archive),)) as pool:
        futures = [pool.submit(_extract_shard, shard, str(dest)) for shard in shards]
        for n, future in enumerate(as_completed(futures), 1):
            w, s, errs = future.result()
            written += w
            skipped += s
            errors.extend(errs)
            if n % 25 == 0 or n == len(futures):
                done = written + skipped
                rate = done / max(time.time() - started, 1e-9)
                eta = (len(names) - done) / rate / 60 if rate else 0
                print(f"[{n}/{len(futures)} shards] {done}/{len(names)} files "
                      f"| {rate:.0f}/s | eta {eta:.0f} min", flush=True)

    print(f"written : {written}  skipped : {skipped}  errors : {len(errors)}")
    for err in errors[:20]:
        print(f"  ! {err}", file=sys.stderr)
    if errors:
        print("extraction incomplete; not publishing", file=sys.stderr)
        return 1

    raw = root / "raw"
    if raw.exists():
        print(f"note: {raw} already exists; leaving {dest} in place", file=sys.stderr)
        return 1
    os.replace(dest, raw)
    stamp = raw / STAMP_NAME
    stamp.write_text(
        f"competition={os.environ.get('KAPPA_COMPETITION', 'unknown')}\n"
        f"completed_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"entries={len(names)}\n"
        f"method=extract_archive.py\n",
        encoding="utf-8",
    )
    print(f"published: {raw}")
    print(f"stamped  : {stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
