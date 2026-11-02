"""Is a derived label tracking the report's language rather than the pathology?

This module exists because of a measured effect that is invisible in the
headline number. Stage 3 drove polarity's language bias to 0pp, but it returns
at stages 5-6: on the first shard of the full extraction run (524 hash-random
studies) Turkish reports derive ``Synovitis`` at 2% against English's 16%,
``Medial OA`` at 6% against 25%, and ``PF OA`` at 11% against 31%, while
deriving ``Effusion`` HIGHER than English, 77% against 64%.

Why that is worse for an image model than it looks. Language implies
institution, institution implies scanner, and scanner implies image appearance
-- so a convolutional model can satisfy a language-correlated label from the
scanner signature instead of from the pathology. That inflates cross-validation
and is invisible at test time. And it cannot be papered over downstream:
robust-loss methods all assume i.i.d. label noise, and a systematic
per-language shift violates that assumption, so the fix has to happen at the
label source.

The 58 anchors structurally cannot show any of this -- they are 79%
Latin-script with two to six studies per non-English language and none in
French -- which is why this audit runs over derived output on the full corpus
rather than over the anchor set.

    from kappa.evaluation.label_audit import audit_by_language

    report = audit_by_language(records)
    print(report.render())
    for label, lo, hi, spread in report.worst_spreads(3):
        ...

Prevalence is reported per label over that label's OWN usable denominator, not
over all studies in the bucket. A masked label is absent evidence, not a
derived negative, and averaging the two together is the mistake that made an
earlier validation table overstate its coverage.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from kappa.constants import LABELS
from kappa.labels.label_map import StudyLabels, derive_study

__all__ = [
    "LanguageAudit",
    "LabelPrevalence",
    "audit_by_language",
    "MIN_BUCKET_FOR_INFERENCE",
]

#: Below this many studies, a bucket's extremes are one or two studies away from
#: being something else, so they are marked rather than read as evidence. Chosen
#: because the measured Dutch (15), French (17) and Bulgarian (23) buckets on a
#: single shard produced 0% and 75% cells that say nothing on their own.
MIN_BUCKET_FOR_INFERENCE = 30


@dataclass(frozen=True)
class LabelPrevalence:
    """One label in one language bucket, with the denominator that produced it."""

    positive: int
    usable: int

    @property
    def rate(self) -> float | None:
        """Positive fraction over usable cells, or None when nothing is usable.

        None rather than 0.0: "no study in this bucket could supervise this
        label" and "no study in this bucket was positive" are different facts,
        and collapsing them into 0.0 invents a negative.
        """
        return self.positive / self.usable if self.usable else None


@dataclass(frozen=True)
class LanguageAudit:
    """Per-label prevalence for every language bucket, plus bucket sizes."""

    #: language -> number of studies, descending by size.
    bucket_sizes: Mapping[str, int]
    #: label -> language -> prevalence.
    table: Mapping[str, Mapping[str, LabelPrevalence]]

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(self.bucket_sizes)

    @property
    def n_studies(self) -> int:
        return sum(self.bucket_sizes.values())

    def large_buckets(self) -> tuple[str, ...]:
        """Buckets big enough for their extremes to carry weight."""
        return tuple(
            lang for lang, n in self.bucket_sizes.items()
            if n >= MIN_BUCKET_FOR_INFERENCE
        )

    def worst_spreads(
        self, limit: int = 5, *, large_only: bool = True
    ) -> list[tuple[str, float, float, float]]:
        """Labels whose prevalence varies most across buckets, widest first.

        ``large_only`` restricts to buckets over MIN_BUCKET_FOR_INFERENCE, which
        is the honest default: the widest raw spreads on a single shard all came
        from buckets of 15 to 23 studies.
        """
        langs = self.large_buckets() if large_only else self.languages
        out = []
        for label in LABELS:
            rates = [
                r for lang in langs
                if (r := self.table[label][lang].rate) is not None
            ]
            if len(rates) > 1:
                out.append((label, min(rates), max(rates), max(rates) - min(rates)))
        return sorted(out, key=lambda row: -row[3])[:limit]

    def render(self) -> str:
        """A fixed-width table, bucket sizes in the header, small buckets marked."""
        langs = self.languages
        head = "label".ljust(17) + "".join(
            f"{lang[:4]}{'' if self.bucket_sizes[lang] >= MIN_BUCKET_FOR_INFERENCE else '*'}".rjust(7)
            for lang in langs
        )
        lines = [
            f"derived-label prevalence by language, n = {self.n_studies}",
            head,
            "n studies".ljust(17) + "".join(str(self.bucket_sizes[l]).rjust(7) for l in langs),
            "-" * len(head),
        ]
        for label in LABELS:
            row = ""
            for lang in langs:
                rate = self.table[label][lang].rate
                row += ("--".rjust(7) if rate is None else f"{100 * rate:5.0f}%".rjust(7))
            lines.append(label.ljust(17) + row)
        lines.append("")
        lines.append(
            f"* bucket under {MIN_BUCKET_FOR_INFERENCE} studies: its extremes are "
            "one or two studies from being different, so do not read them as evidence"
        )
        lines.append("")
        lines.append("widest spreads across buckets of adequate size:")
        for label, lo, hi, spread in self.worst_spreads():
            lines.append(
                f"  {label:<17} {100 * lo:5.0f}% -> {100 * hi:5.0f}%"
                f"   spread {100 * spread:5.0f} pp"
            )
        return "\n".join(lines)


def audit_by_language(
    records: Iterable[Mapping[str, object]],
    *,
    unmentioned: float | None | Mapping[str, float | None] = 0.0,
    derived: Sequence[tuple[str, StudyLabels]] | None = None,
) -> LanguageAudit:
    """Group derived labels by the report's language and tabulate prevalence.

    ``derived`` lets a caller pass ``(language, StudyLabels)`` pairs it has
    already computed, so a full-corpus audit does not derive twice.
    """
    if derived is None:
        derived = [
            (str(record.get("language") or "unknown"),
             derive_study(dict(record), unmentioned=unmentioned))
            for record in records
        ]

    buckets: dict[str, list[StudyLabels]] = collections.defaultdict(list)
    for language, study in derived:
        buckets[language].append(study)

    sizes = dict(sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])))
    table: dict[str, dict[str, LabelPrevalence]] = {}
    for label in LABELS:
        table[label] = {}
        for language, studies in sizes.items():
            usable = [s for s in buckets[language] if s.usable[label]]
            table[label][language] = LabelPrevalence(
                positive=sum(1 for s in usable if s.values[label] == 1.0),
                usable=len(usable),
            )
    return LanguageAudit(
        bucket_sizes={k: len(v) for k, v in sizes.items()}, table=table
    )
