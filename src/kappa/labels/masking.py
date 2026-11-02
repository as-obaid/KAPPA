"""Report-level hazards that make a row *unknown* rather than negative.

Why this exists: **an empty findings section is not twelve zeros.** The obvious
aggregation -- "no finding mentions a tear, so Tear = 0" -- turns every kind of
missing information into a confident negative. Under macro ROC-AUC, where each
of the 12 labels contributes 1/12 of the score, a fabricated negative is not a
neutral default; it is a wrong rank.

Three hazards were found in the 300-report survey, and they need different
treatment. Counts are indicative: vLLM is not reproducible across batch
composition, so these are the shape of the problem, not fixed figures.

**1. Empty findings section -> mask (1 of 300).** A Dutch report carried only
clinical indications and diagnostic questions; its ``Bevindingen`` section was
literally empty, so ``findings`` came back ``[]``. Nothing about the knee was
observed. Twelve zeros would assert twelve normal structures on no evidence.

**2. Bilateral report -> mask (5 of 300, ~1.7%).** Some reports describe both
knees in one text while the study images *one*. In the worst case the model
coped by prefixing ``right`` onto 15 of 32 anatomy strings while ``knee_side``
stayed ``unstated``. Findings from the unimaged knee would be attributed to the
imaged one -- a positive label on a structure that was never scanned. Note that
``knee_side`` is *not* a detector here: 4 of the 5 had it ``unstated``.

**3. Restated findings -> deduplicate, do NOT mask (~30 of 300, ~10%).** Many
reports repeat their findings in a summary or Impression section
(``The findings are repeated identically in both the Results and Impression
sections``). This inflates finding counts, not polarity, so masking would throw
away good evidence. It matters for any count- or confidence-weighted
aggregation and is harmless for plain any-positive. Recorded here because it
was found alongside the other two and is easy to mistake for a prior-study
comparison; only 2 of the ~32 notes matches were genuinely a second study.

    from kappa.labels.masking import report_mask, resolve_label

    mask = report_mask(extraction, report_text=text)
    if mask.masked:
        ...  # all 12 labels are unknown for this study

Detection deliberately does not rest on ``notes``. ``notes`` is free-text
model self-report -- non-null in only 39% of records, English regardless of the
report's language, and specific to whichever model produced it. It is used as a
*supporting* cue and never as the sole one, so this module keeps working when
the model changes.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, NamedTuple

from kappa.labels.normalise import fold_lookalikes
from kappa.labels.polarity import MASK, NEGATIVE, POSITIVE, Polarity

__all__ = [
    "MaskReason",
    "ReportMask",
    "report_mask",
    "resolve_label",
    "is_bilateral_text",
]

MaskReason = Literal[
    "extraction_failed", "empty_findings", "bilateral_report", "multiple_studies"
]


class ReportMask(NamedTuple):
    """Whether a study's 12-label row is unknowable, and why.

    ``reasons`` is kept as a tuple rather than collapsed to a bool so a run can
    be counted by cause. "2% of studies were masked" is not reviewable; "1.7%
    were masked for bilateral text and 0.3% for an empty findings section" is.
    """

    masked: bool
    reasons: tuple[MaskReason, ...]


# How a two-knee report is actually detected. **Not by keyword.**
#
# The corpus marks them itself: a bracketed note beginning `[BILATERAL NOTE`
# separates two complete reports concatenated under one StudyInstanceUID. That
# marker appears in exactly **27 of 4,407** reports and is unambiguous.
#
# A keyword scan for "bilateral"/"both knees" and their translations was tried
# and **measured to be wrong**: it fires on **131 of 4,407** reports, only 27 of
# which are genuinely two-knee. The other 104 use the word for two structures
# inside *one* knee, or for clinical history about the other leg. Paraphrased
# examples of what it caught:
#
#   Spanish clinical history of bilateral knee pain         (indication)
#   English indication of long-standing pain in both knees  (indication)
#   "bilateral menisci" described as intact                 (both menisci, one knee)
#   Bulgarian "двустранно" applied to the collateral ligaments
#   Spanish bone oedema in "ambos platillos tibiales"       (both tibial plateaus)
#   Croatian "obostrano" applied to femoral condyle chondromalacia
#
# So masking on the keyword would have discarded 104 usable studies -- 2.4% of
# the corpus -- to catch 27. The marker and the structural tests below are used
# instead, and the keyword scan is deliberately absent.
_BILATERAL_MARKER = "[BILATERAL NOTE"

# Reports that carry no marker but do address both knees give themselves away
# structurally, with a left-side and a right-side section heading in the same
# text. Matched after `fold_lookalikes`, because the corpus writes Greek capital
# delta as U+2206 INCREMENT: `∆ΕΞΙΟΥ ΓΟΝΑΤΟΣ` does not lowercase to `δεξιου`,
# so an unfolded Greek pattern silently matches nothing at all.
_SIDE_SECTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("дясна колянна става", "лява колянна става"),   # Bulgarian
    ("дясното колянна", "лявото колянна"),           # Bulgarian, neuter
    ("δεξιου γονατος", "αριστερου γονατος"),         # Greek
    ("δεξιού γόνατος", "αριστερού γόνατος"),         # Greek, accented
)

# An English `right knee` / `left knee` pair was tried here and **removed**. It
# fired on 10 reports, and every one was a single-knee study naming the other
# knee in its indication, comparison or a dictation slip. Paraphrased:
#
#   a left-knee study whose indication is months of right knee pain
#   a right-knee study noting it had been ordered for the left knee
#   a left-knee study whose indication names the right knee (and a kidney)
#
# All three genuinely two-knee English reports carry the marker instead, so the
# pair added false positives and nothing else. Those 10 are a *different*
# problem -- laterality-discordant text -- and the fix for them is to strip the
# history/indication span before extraction, not to mask the study.

#: Greek plural genitive "of the knees" -- one report covering both, with no
#: left/right sections to pair up.
_GREEK_BOTH_KNEES = "γονατων"

# The model's own note, kept as a *supporting* detector only: it is English
# whatever the report says, non-null in 39% of records, and specific to
# whichever model produced it. Narrow on purpose -- it must describe two knees,
# not merely contain the word "bilateral".
_NOTES_BILATERAL_RE = re.compile(
    r"describes both the (?:left and right|right and left)"
    r"|for two different knee"
    r"|two separate mri findings and conclusions for two"
    r"|separated by a bilateral note",
    re.IGNORECASE,
)

# A prior or follow-up study concatenated into the same text. Distinct from a
# report that merely *restates* its findings in a summary section, which is
# ~10% of the corpus and must not be masked.
_NOTES_TWO_STUDIES_RE = re.compile(
    r"two separate (?:technical|mri) "
    r"|current study and a (?:comparison|prior|previous)"
    r"|comparison with (?:a |the )?(?:prior|previous|earlier)"
    r"|contradicts or updates the initial",
    re.IGNORECASE,
)

_SIDE_RE = re.compile(r"\b(right|left)\b", re.IGNORECASE)


def is_bilateral_text(report_text: str) -> bool:
    """Does this report text cover both knees?

    Marker first, because it is the corpus's own statement and needs no
    interpretation. Then the structural tests, which need none of the keyword
    scan's guesswork about what "bilateral" modifies.
    """
    if _BILATERAL_MARKER in report_text:
        return True
    folded = fold_lookalikes(report_text).lower()
    for right, left in _SIDE_SECTION_PAIRS:
        if right in folded and left in folded:
            return True
    return _GREEK_BOTH_KNEES in folded


def _anatomy_sides(findings: Iterable) -> set[str]:
    """Sides named inside the anatomy strings themselves.

    The model resolves laterality into ``anatomy`` ("right medial meniscus")
    even when it leaves the report-level ``knee_side`` unstated, so this is the
    signal that actually fires on a bilateral report.
    """
    sides: set[str] = set()
    for finding in findings:
        anatomy = getattr(finding, "anatomy", None)
        if anatomy is None and isinstance(finding, dict):
            anatomy = finding.get("anatomy")
        if not isinstance(anatomy, str):
            continue
        for match in _SIDE_RE.finditer(anatomy):
            sides.add(match.group(1).lower())
    return sides


def _get(extraction, name: str, default=None):
    """Read a field from either a ReportExtraction or a plain dict."""
    if isinstance(extraction, dict):
        return extraction.get(name, default)
    return getattr(extraction, name, default)


def report_mask(extraction, report_text: str | None = None) -> ReportMask:
    """Decide whether a study's whole 12-label row is unknown, and why.

    ``report_text`` is optional but strongly preferred: the multilingual cue
    scan over the source text is the only detector that does not depend on the
    model having noticed the problem. Reports exist for train studies only, so
    passing it costs nothing at label-derivation time.
    """
    reasons: list[MaskReason] = []

    findings = _get(extraction, "findings") or []
    # An extraction that never parsed is not a report with nothing in it. Both
    # arrive here as `findings == []`, and conflating them hides a broken model
    # behind a property of the corpus -- the 122B-FP8 smoke test returned 0 of 5
    # parseable responses and every record looked like an empty findings
    # section. Same masking decision, different thing to go fix.
    if _get(extraction, "_parsed") is False or _get(extraction, "_valid") is False:
        reasons.append("extraction_failed")
    elif not findings:
        reasons.append("empty_findings")

    notes = _get(extraction, "notes") or ""

    # Bilateral: any one of three independent signals. The anatomy-side test is
    # the strongest, the text scan is model-independent, and notes is a cue of
    # last resort.
    sides = _anatomy_sides(findings)
    bilateral = len(sides) > 1
    if not bilateral and report_text:
        bilateral = is_bilateral_text(report_text)
    if not bilateral and notes:
        bilateral = _NOTES_BILATERAL_RE.search(notes) is not None
    if bilateral:
        reasons.append("bilateral_report")

    if notes and _NOTES_TWO_STUDIES_RE.search(notes):
        reasons.append("multiple_studies")

    return ReportMask(masked=bool(reasons), reasons=tuple(reasons))


def resolve_label(
    polarities: Iterable[Polarity],
    *,
    unmentioned: float | None = None,
) -> float | None:
    """Collapse the polarities of findings assigned to one label into its value.

    ``1.0`` if any finding asserts the abnormality, else ``0.0`` if any finding
    rules it out, else ``unmentioned``. Positive wins over negative because a
    report that both rules a tear out in the body and reports one in the
    impression is describing a tear.

    ``unmentioned`` is the value for a label no finding mentions at all, and it
    defaults to ``None`` -- unknown -- **on measured grounds, not caution.**
    Defaulting it to 0.0 reads "not mentioned" as "normal", which is only true
    for reports that sweep through normal structures, and that habit is
    strongly language-dependent: normality statements are 35% of English
    findings and 4% of Greek ones. A Greek report that never mentions the ACL
    is silent about it, while an English one usually would have said "ACL
    intact". Setting ``unmentioned=0.0`` therefore buys coverage by inventing
    negatives predominantly in the terser languages -- a language-correlated
    rank shift of exactly the kind the 12-label macro AUC punishes.

    Which value to use is the open decision in task 18, so it is a parameter
    with an honest default rather than a constant baked into the aggregation.
    """
    seen_negative = False
    seen_any = False
    for polarity in polarities:
        seen_any = True
        if polarity == POSITIVE:
            return 1.0
        if polarity == NEGATIVE:
            seen_negative = True
        elif polarity != MASK:
            raise ValueError(f"not a polarity: {polarity!r}")
    if seen_negative:
        return 0.0
    if seen_any:
        return None  # every finding for this label was masked
    return unmentioned
