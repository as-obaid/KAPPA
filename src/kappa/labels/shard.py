"""Plan the extraction pass so a walltime kill costs one batch, not the run.

The full corpus is 4,407 reports at a measured 0.81 s/report, i.e. roughly 59
minutes of generation, and the only node the vLLM environment is measured
working on sits in a partition capped at **1 hour**. One run is therefore
exactly at the wall. Worse, the reconnaissance script accumulated every record
in memory and wrote ``records.jsonl`` once at the end, so a kill at minute 59
produced an empty file. This module is the fix, and it is deliberately pure --
no vLLM, no GPU, no model -- so the resume and shard arithmetic is testable on
a laptop.

The answer is to shard, not to lengthen: the 8-hour partition's nodes are
measured to fail on this environment's cu130 wheels, so more walltime is not
available at any price.

Six silent failures this module exists to prevent:

* **A buffered write that is never flushed.** ``RecordWriter`` flushes and
  ``fsync``s every record, so an interrupted job loses at most the batch still
  inside ``generate()``.
* **A truncated final line taken as a completed study.** A half-written JSON
  object on the last line is exactly what a walltime ``SIGKILL`` produces.
  ``read_done`` discards it rather than crashing or counting it as done.
* **Shard membership that moves between runs.** ``shard_index`` hashes with
  ``sha256``, never with ``hash()``: Python randomises ``hash()`` of a ``str``
  per process unless ``PYTHONHASHSEED`` is pinned, so a resumed shard would
  cover a different set of studies than the run it resumed -- leaving some
  studies extracted twice and others never.
* **The same report text extracted twice, or split across two shards.**
  Measured on the 4,407-row corpus (``normalise_report`` then ``content_hash``):
  **4,262 distinct texts in 52 duplicate groups spanning 197 rows**, so
  **145 of 4,407 calls (3.3%) are redundant**. De-duplication happens *before*
  sharding, because two studies sharing a text must not land in different
  shards and be extracted independently.
* **A study counted twice in the aggregates.** A resume can legitimately append
  a second record for a study whose first record was unreadable, so
  ``load_records`` keeps the last record per study instead of both.
* **The first record of a resume glued onto the truncated last line of the
  previous run.** Append mode writes at the end of the file, not at the end of
  the last *line*, so resuming a file that a kill left un-terminated concatenated
  the new record onto the broken one and destroyed both -- silently, and always
  for the one study that was in flight when the job died. ``RecordWriter``
  repairs the tail before it appends.

**Hashing order matters.** The key is ``content_hash(normalise_report(text))``.
``normalise.fold_lookalikes`` is called before *matching* and never before
hashing: it is lossy by design, and folding it in here would merge texts that
are genuinely distinct. On this corpus folding changes nothing (4,262 groups
either way, measured), which is exactly why the mistake would go unnoticed
until a corpus where it does.

For the record, the two duplicate counts in circulation measure different
things and both are right: on **raw** ``Report`` bytes there are 4,276 distinct
texts and **131** redundant calls; after ``normalise_report`` there are 4,262
distinct texts and **197** rows sitting in duplicate groups. This module uses
the normalised key, so it saves 145 calls.

    from kappa.labels.shard import plan_units, pending, read_done, select_shard

    units = plan_units(load_reports(train_csv))
    mine = select_shard(units, 3, 8)
    todo = pending(mine, read_done(out_path).study_uids)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kappa.labels.normalise import content_hash, normalise_report

__all__ = [
    "WorkUnit",
    "ResumeState",
    "RecordWriter",
    "plan_units",
    "describe_units",
    "shard_index",
    "select_shard",
    "parse_shard",
    "shard_path",
    "shard_paths",
    "read_done",
    "pending",
    "records_for_unit",
    "load_records",
    "missing_studies",
    "STUDY_KEY",
]

#: Field every record carries the study it belongs to in. Resume keys on it.
STUDY_KEY = "_study_uid"


@dataclass(frozen=True)
class WorkUnit:
    """One model call, plus every study whose record that call will produce.

    ``study_uids`` is the set of studies still to be written and shrinks under
    ``pending``; ``primary_uid`` and ``group_size`` are fixed at plan time so
    that a record can say which study the extraction was actually run for and
    how many studies shared the text. Without those two fields a propagated
    record is indistinguishable from an independently extracted one, and the
    de-duplication becomes unauditable after the fact.
    """

    content_hash: str
    report: str
    script: str
    primary_uid: str
    study_uids: tuple[str, ...]
    group_size: int
    is_anchor: bool = False


def plan_units(rows: Sequence[Mapping[str, Any]]) -> list[WorkUnit]:
    """Collapse rows that share a report text into one work unit each.

    ``rows`` is what ``sample.load_reports`` returns: ``study_uid``, normalised
    ``report``, ``script``, ``hash`` and ``is_anchor``. The ``hash`` field is
    reused when present so this module and ``sample.stratified_sample`` cannot
    drift into two different notions of "same report".

    The result is sorted by content hash, so the plan is byte-identical between
    a run and its resume regardless of CSV row order.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = row.get("hash") or content_hash(normalise_report(row["report"]))
        groups.setdefault(str(key), []).append(row)

    units: list[WorkUnit] = []
    for key, group in groups.items():
        uids = tuple(sorted(str(row["study_uid"]) for row in group))
        first = group[0]
        units.append(
            WorkUnit(
                content_hash=key,
                report=first["report"],
                script=str(first.get("script", "unknown")),
                primary_uid=uids[0],
                study_uids=uids,
                group_size=len(uids),
                # An anchor's gold row is the project's only ground truth, so a
                # group containing one is flagged as a whole: the flag exists to
                # make the propagation visible, not to select on it.
                is_anchor=any(bool(row.get("is_anchor")) for row in group),
            )
        )
    return sorted(units, key=lambda unit: unit.content_hash)


def describe_units(units: Sequence[WorkUnit]) -> str:
    """One line stating calls, studies and how many calls the dedup saved."""
    studies = sum(unit.group_size for unit in units)
    shared = sum(unit.group_size for unit in units if unit.group_size > 1)
    groups = sum(1 for unit in units if unit.group_size > 1)
    return (
        f"{len(units)} calls covering {studies} studies "
        f"({groups} duplicate groups over {shared} studies; "
        f"{studies - len(units)} calls saved)"
    )


def shard_index(key: str, n: int) -> int:
    """Which of ``n`` shards ``key`` belongs to, stably across processes.

    ``sha256`` rather than ``hash()``: Python salts ``hash()`` of a ``str`` per
    interpreter unless ``PYTHONHASHSEED`` is set, so ``hash()`` here would put a
    study in shard 3 on one run and shard 6 on its resume. Re-hashing an
    already-hashed key is cheap and makes the function total -- it accepts a
    study UID as happily as a content hash, and neither is required to be hex.
    """
    if n < 1:
        raise ValueError(f"shard count must be >= 1, got {n}")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def select_shard(units: Sequence[WorkUnit], index: int, n: int) -> list[WorkUnit]:
    """The ``index``-th of ``n`` disjoint shards, keyed on the content hash.

    Keyed on the content hash and not the study UID, because de-duplication has
    already happened: a text shared by 37 studies is one call and must be one
    shard's responsibility. Measured balance over the real 4,262-unit corpus at
    ``n=8``: **483 to 576 units per shard, i.e. 495 to 615 studies**. The spread
    is 19% wide, so size the walltime against the *largest* shard and not
    against 4,407/8 -- an even split is what a round-robin would give, and a
    round-robin is exactly what cannot survive a resume.
    """
    if not 0 <= index < n:
        raise ValueError(f"shard index {index} out of range for {n} shards")
    return [unit for unit in units if shard_index(unit.content_hash, n) == index]


_SHARD_SPEC = re.compile(r"\s*(\d+)\s*/\s*(\d+)\s*")


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse ``"3/8"`` into ``(3, 8)``, rejecting anything out of range.

    Rejecting loudly matters more than it looks: ``--shard 8/8`` is the natural
    off-by-one for a ``--array=0-7`` job, and silently selecting an empty shard
    would report "0 pending, nothing to do" and exit 0.
    """
    match = _SHARD_SPEC.fullmatch(spec)
    if match is None:
        raise ValueError(f"expected --shard as i/n, e.g. 3/8; got {spec!r}")
    index, n = int(match.group(1)), int(match.group(2))
    if n < 1:
        raise ValueError(f"shard count must be >= 1, got {n}")
    if not 0 <= index < n:
        raise ValueError(
            f"shard index {index} out of range for {n} shards; "
            f"valid indices are 0..{n - 1}"
        )
    return index, n


def shard_path(base: Path | str, index: int | None = None, n: int | None = None) -> Path:
    """Per-shard output file beside ``base``, or ``base`` itself when unsharded.

    Zero-padded so that a shell glob lists shards in shard order; an unpadded
    ``shard10of12`` sorts before ``shard2of12`` and a merge that trusts glob
    order then silently reorders the corpus.
    """
    base = Path(base)
    if index is None or n is None:
        return base
    if not 0 <= index < n:
        raise ValueError(f"shard index {index} out of range for {n} shards")
    width = max(2, len(str(n - 1)))
    stem = f"{base.stem}.shard{index:0{width}d}of{n:0{width}d}"
    return base.with_name(stem + base.suffix)


def shard_paths(base: Path | str, n: int) -> tuple[Path, ...]:
    """Every shard file for an ``n``-way split, in shard order."""
    if n < 1:
        raise ValueError(f"shard count must be >= 1, got {n}")
    return tuple(shard_path(base, index, n) for index in range(n))


@dataclass(frozen=True)
class ResumeState:
    """What an existing output file says has already been done."""

    study_uids: frozenset[str]
    n_lines: int = 0
    n_unreadable: int = 0
    n_missing_key: int = 0
    truncated_tail: bool = False

    @property
    def warnings(self) -> tuple[str, ...]:
        """Human-readable notes, empty when the file was clean.

        A truncated tail is reported but not alarming -- it is the expected
        signature of a walltime kill. An unreadable line in the *middle* of the
        file is a different animal and is called out separately, because
        nothing normal produces one.
        """
        notes: list[str] = []
        if self.truncated_tail:
            notes.append(
                "discarded a truncated final line; that study will be re-extracted"
            )
        mid_file = self.n_unreadable - (1 if self.truncated_tail else 0)
        if mid_file > 0:
            notes.append(
                f"{mid_file} unreadable line(s) away from the end of the file, "
                "which no clean kill produces -- inspect before trusting this file"
            )
        if self.n_missing_key:
            notes.append(
                f"{self.n_missing_key} record(s) carried no {STUDY_KEY}; "
                "not counted as done"
            )
        return tuple(notes)


def read_done(path: Path | str, key: str = STUDY_KEY) -> ResumeState:
    """Collect the studies an existing output file already holds.

    Robust to the two things a walltime kill actually leaves behind:

    * a final line that is a half-written JSON object, which is discarded
      rather than crashing the resume and rather than being counted as done;
    * a final line cut mid-codepoint. The corpus is multilingual, so a kill can
      land inside a multi-byte UTF-8 sequence, and a strict decode would raise
      ``UnicodeDecodeError`` over the *whole file* -- losing 500 good records
      to one bad byte. Decoding is therefore lossy-tolerant, and the damaged
      line then fails the JSON parse and is dropped on the first rule.
    """
    p = Path(path)
    if not p.is_file():
        return ResumeState(frozenset())

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    # A trailing newline yields a final empty element, which is not a line.
    if lines and lines[-1] == "":
        lines.pop()

    done: set[str] = set()
    n_unreadable = 0
    n_missing_key = 0
    truncated_tail = False
    last = len(lines) - 1
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            n_unreadable += 1
            if position == last:
                truncated_tail = True
            continue
        uid = record.get(key) if isinstance(record, dict) else None
        if isinstance(uid, str) and uid:
            done.add(uid)
        else:
            n_missing_key += 1
    return ResumeState(
        study_uids=frozenset(done),
        n_lines=len(lines),
        n_unreadable=n_unreadable,
        n_missing_key=n_missing_key,
        truncated_tail=truncated_tail,
    )


def pending(units: Sequence[WorkUnit], done: Iterable[str]) -> list[WorkUnit]:
    """Units with at least one study still unwritten, narrowed to those studies.

    A unit is re-extracted when *any* of its studies is missing, which costs one
    extra model call if a kill landed between writing the first and last record
    of a duplicate group. Paying for at most one redundant call per group beats
    the alternative, which is deciding a group is finished on the strength of
    its first record and leaving the rest permanently unwritten.
    """
    already = frozenset(done)
    out: list[WorkUnit] = []
    for unit in units:
        todo = tuple(uid for uid in unit.study_uids if uid not in already)
        if todo:
            out.append(replace(unit, study_uids=todo))
    return out


def records_for_unit(
    unit: WorkUnit,
    payload: Mapping[str, Any],
    *,
    shard: str | None = None,
) -> list[dict]:
    """Fan one extraction out to every study that shared the report text.

    This is the write side of the de-duplication: the model is called once per
    distinct text and this function turns that single answer into one record per
    study, because everything downstream -- the resume key, the label table, the
    fold assignment -- is per study. A caller that instead wrote one record per
    *call* would leave 145 of the 4,407 studies with no row at all (measured on
    this corpus), and the shortfall would show up much later as a silently
    smaller training set.

    ``_propagated`` is set against ``unit.primary_uid``, which is fixed at plan
    time and therefore still meaningful after ``pending`` has narrowed
    ``study_uids``. On a resume where the primary study was already written,
    every record this call emits is flagged propagated -- correct, since they all
    come from one extraction of one text.

    ``payload`` is copied, never mutated, so the caller can reuse the same
    validated extraction for all 37 members of the largest measured group.
    """
    records: list[dict] = []
    for uid in unit.study_uids:
        record = dict(payload)
        record[STUDY_KEY] = uid
        record["_content_hash"] = unit.content_hash
        record["_script"] = unit.script
        record["_primary_uid"] = unit.primary_uid
        record["_group_size"] = unit.group_size
        record["_propagated"] = uid != unit.primary_uid
        record["_is_anchor"] = unit.is_anchor
        if shard is not None:
            record["_shard"] = shard
        records.append(record)
    return records


def load_records(
    paths: Iterable[Path | str], key: str = STUDY_KEY
) -> tuple[list[dict], list[str]]:
    """Merge shard files into one record list, last record per study winning.

    Last-wins rather than first-wins, and never both: a resume legitimately
    appends a second record for a study whose earlier line was unreadable, and
    the good record is the later one. Keeping both would double-count that study
    in every aggregate the records feed, which is invisible in a rate and wrong
    in a count.

    Returns the records sorted by study UID, plus warnings naming every file
    that was missing or damaged, so a partial merge is never mistaken for a
    complete one.
    """
    merged: dict[str, dict] = {}
    warnings: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            warnings.append(f"missing: {p}")
            continue
        state = read_done(p, key=key)
        for note in state.warnings:
            warnings.append(f"{p.name}: {note}")
        text = p.read_text(encoding="utf-8", errors="replace")
        for line in text.split("\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            uid = record.get(key) if isinstance(record, dict) else None
            if isinstance(uid, str) and uid:
                merged[uid] = record
    return [merged[uid] for uid in sorted(merged)], warnings


def missing_studies(
    records: Sequence[Mapping[str, Any]], expected: Iterable[str], key: str = STUDY_KEY
) -> list[str]:
    """Studies in ``expected`` that no record covers, sorted.

    The merge step needs this to fail loudly. Eight array tasks can leave seven
    shard files, and a merge that reports "wrote 3,730 records" without saying
    which 677 studies are absent reads as success.
    """
    have = {
        record.get(key)
        for record in records
        if isinstance(record.get(key), str)
    }
    return sorted(uid for uid in expected if uid not in have)


class RecordWriter:
    """Append-only JSONL writer that flushes and ``fsync``s every record.

    ``flush()`` alone only moves bytes out of Python's buffer into the kernel's
    page cache; ``os.fsync`` is what puts them on the device. A Slurm walltime
    kill delivers ``SIGKILL``, so ``__exit__`` never runs and anything still in
    either buffer is gone -- a buffered write that is never flushed is the same
    bug wearing a disguise. One ``fsync`` per record is ~4,262 round trips for
    the whole corpus, which is nothing against a 59-minute run and is the entire
    difference between losing one batch and losing everything.

    Opened in append mode on purpose, so ``--resume`` adds to the existing file
    instead of truncating the work it just read -- but append mode writes at the
    end of the *file*, not the end of the last *line*, and a killed run leaves a
    file whose last line has no newline. Appending to that concatenates the new
    record onto the broken one, and the result parses as neither: the resume
    then loses the very study it restarted to write, every time, with nothing in
    the log to say so. ``_repair_tail`` closes that, and ``repaired_tail``
    records what it did so the caller can print it.

        with RecordWriter(path) as writer:
            writer.write(record)
    """

    def __init__(self, path: Path | str, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.n_written = 0
        self._fsync = fsync
        self._fh: Any = None
        #: What ``_repair_tail`` had to do at open time, or None if nothing.
        self.repaired_tail: str | None = None

    def __enter__(self) -> RecordWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.repaired_tail = self._repair_tail()
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def _repair_tail(self) -> str | None:
        """Leave the file newline-terminated, so the next append starts a line.

        The trailing partial line is dropped rather than terminated, because it
        is unparseable by definition and ``read_done`` would otherwise count it
        as an unreadable line in the *middle* of the file -- a warning reserved
        for damage that no clean kill produces, which is worth keeping rare.

        The parse is what decides. A kernel write can stop exactly at the end of
        a record and lose only the newline, and that record is *good*: it was
        already counted as done, so the study will not be re-extracted and
        deleting the line here would lose it permanently. So a tail that parses
        is terminated and kept; only a tail that does not parse is discarded.
        """
        if not self.path.is_file():
            return None
        chunk = 1 << 16
        with open(self.path, "r+b") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size == 0:
                return None
            tail = b""
            start = 0
            position = size
            while position > 0:
                step = min(chunk, position)
                position -= step
                fh.seek(position)
                block = fh.read(step)
                cut = block.rfind(b"\n")
                if cut != -1:
                    tail = block[cut + 1 :] + tail
                    start = position + cut + 1
                    break
                tail = block + tail
            if not tail:
                return None
            try:
                json.loads(tail.decode("utf-8", errors="replace"))
            except ValueError:
                fh.truncate(start)
                if self._fsync:
                    os.fsync(fh.fileno())
                return (
                    f"dropped {len(tail)} trailing bytes of an unparseable final "
                    "line before appending"
                )
            fh.seek(0, os.SEEK_END)
            fh.write(b"\n")
            if self._fsync:
                os.fsync(fh.fileno())
            return "terminated a final line that was missing its newline"

    def write(self, record: Mapping[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("RecordWriter.write called outside its context manager")
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())
        self.n_written += 1

    def __exit__(self, *_exc: Any) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None
