"""Script-stratified sampling of report text.

Two rules this module exists to enforce, both of which are easy to get wrong by
hand and expensive to discover later:

1. **Only unlabelled studies are sampled.** The 58 annotated studies are the
   only ground truth in the project. Any report the model has been shown during
   design is no longer available for honest validation, so the anchors are
   excluded here and the exclusion is asserted, not assumed.

   ``anchors_only=True`` inverts this rule, and it is a **deliberate, one-time
   spend of the project's only ground truth**. The reason: the 12-label derivation had *no per-study validation whatsoever*.
   Every rule in `polarity.py`, `masking.py` and the anatomy map was built from
   the 300-report unlabelled survey, so the anchors are a genuine held-out test
   set for it -- but only if measured once and not tuned against. The flag is
   explicit, and named for what it costs, so that no future run reaches for it
   casually. Reports exist for train studies only, so this cannot touch the
   image model or the submission.
2. **Stratify, do not sample proportionally.** Greek is 321 of 4,407 reports and
   Cyrillic 220, so a proportional 300-report sample yields ~22 and ~15 -- too
   few to characterise a reporting convention, and precisely the languages where
   extraction is most likely to fail.

    from kappa.labels.sample import load_reports, stratified_sample

    rows = load_reports(raw_dir() / "train.csv")
    sample = stratified_sample(rows, per_script={"greek": 50}, seed=0)
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kappa.constants import ID_COLUMN, LABELS
from kappa.labels.normalise import content_hash, detect_script, normalise_report

__all__ = ["load_reports", "stratified_sample", "DEFAULT_PER_SCRIPT"]

#: Oversamples the non-Latin scripts on purpose -- see the module docstring.
DEFAULT_PER_SCRIPT = {"latin": 150, "turkish": 50, "greek": 50, "cyrillic": 50}


def load_reports(train_csv: Path) -> list[dict]:
    """Read train.csv into rows carrying normalised text, script and label state.

    ``train.csv`` holds reports up to 4,743 characters inside quoted CSV fields,
    which exceeds Python's default field limit on some builds, so the limit is
    raised explicitly rather than left to chance.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows = []
    with open(train_csv, encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            text = normalise_report(raw["Report"])
            rows.append(
                {
                    "study_uid": raw[ID_COLUMN],
                    "report": text,
                    "script": detect_script(text),
                    "hash": content_hash(text),
                    # A study is an anchor only if every label is populated; the
                    # audit found 0 partially-labelled studies, so `any` and
                    # `all` agree today. `all` is the safer reading of "has
                    # ground truth" if that ever stops being true.
                    "is_anchor": all(raw[label] not in ("", None) for label in LABELS),
                }
            )
    return rows


def stratified_sample(
    rows: list[dict],
    per_script: dict[str, int] | None = None,
    seed: int = 0,
    anchors_only: bool = False,
) -> list[dict]:
    """Sample unlabelled reports per script, deduplicated by content.

    Deduplication is by report text, not study: 49 texts repeat across 183 rows
    (the largest group is one Turkish all-normal template shared by 37 studies).
    Sampling the same text twice would spend budget learning nothing.

    ``anchors_only`` returns **every** annotated study instead, ignoring
    ``per_script`` and the dedup -- see rule 1 in the module docstring. All 58
    are returned because a per-label comparison against 58 studies is already
    thin; dropping duplicates would make several of the 12 labels unreadable.
    """
    if anchors_only:
        picked = [r for r in rows if r["is_anchor"]]
        if not picked:
            raise AssertionError("anchors_only=True but no annotated studies found")
        return sorted(picked, key=lambda r: r["study_uid"])

    per_script = per_script or DEFAULT_PER_SCRIPT
    rng = random.Random(seed)

    pools: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["is_anchor"]:
            continue
        pools[row["script"]].setdefault(row["hash"], row)

    picked: list[dict] = []
    for script, want in per_script.items():
        candidates = sorted(pools.get(script, {}).values(), key=lambda r: r["study_uid"])
        if len(candidates) < want:
            print(
                f"warning: script {script!r} has {len(candidates)} distinct "
                f"unlabelled reports, wanted {want}; taking all",
                file=sys.stderr,
            )
        picked.extend(rng.sample(candidates, min(want, len(candidates))))

    anchors = {r["study_uid"] for r in rows if r["is_anchor"]}
    overlap = anchors & {r["study_uid"] for r in picked}
    if overlap:
        raise AssertionError(f"sample touched {len(overlap)} anchor studies: {overlap}")

    return picked


def describe(rows: list[dict]) -> str:
    """One-line-per-script summary, for logging what a sample actually contains."""
    counts = Counter(r["script"] for r in rows)
    return " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
