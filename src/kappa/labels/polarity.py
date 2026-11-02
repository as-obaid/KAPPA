"""Derive a finding's polarity from ``(assertion, description)`` jointly.

Why this exists: ``assertion`` alone means two different things, and which one
it means tracks the report's language. Statements of pure normality ("ACL:
Preserved", "Lateral menisküs normal") are labelled ``affirmed`` -- the model is
asserting *that the report says the structure is normal*, not asserting an
abnormality. Measured over the 300-report survey, on the findings whose
description names a normality word and no abnormality:

===========  ====  ==================
language        n  labelled affirmed
===========  ====  ==================
English       483                 93%
Croatian       85                 94%
Bulgarian     323                 78%
Turkish       195                 37%
Greek          69                 30%
===========  ====  ==================

So ``affirmed -> 1`` misreads a normality sweep as a lesion, and does so ~3x
more often in English than in Turkish or Greek. Under macro ROC-AUC that is a
language-correlated rank shift, and the 58 labelled anchors cannot detect it:
they give 2-6 studies per non-English language and none in French.

**The mirror trap is real too, so ``description`` alone is equally unsafe.** Of
the negated findings, 117 name an abnormality with the negation living only in
the span -- ``No MR evidence of ACL lesion`` yields description ``"lesion of
the anterior cruciate ligament"``. Trusting descriptions flips those positive.

Hence: polarity comes from both fields, or it is wrong.

The description field is the reason this is tractable. It is **English for all
3,962 findings** (99.9% pure ASCII) whatever the report's language, because the
model normalises it, so one English lexicon covers all nine languages. That is
a measured property of *this* model's output, not a guarantee -- re-run
``polarity_report`` on any new model's records before trusting it.

    from kappa.labels.polarity import polarity

    polarity("affirmed", "myxoid degeneration without tear")  # -> "positive"
    polarity("affirmed", "lateral meniscus is intact")        # -> "negative"
    polarity("affirmed", "TT-TG distance is 15 mm")           # -> "mask"

Validation, per language, on the 300-report gemma-4-31B survey:

* On normality statements, the naive rule's false-positive rate spans
  **30-94%** (64pp of language-correlated bias). This rule gives **0% in every
  language**, 0pp spread.
* Mean positive findings per report -- the quantity that becomes a label -- for
  the four languages with 50+ reports goes from **6.9-12.5 (spread 5.7)** under
  the naive rule to **6.1-6.7 (spread 0.6)** under this one.
* Finding-level totals: 1,867 positive / 1,936 negative / 159 masked.

``mask`` is not a failure state and must not be collapsed to ``negative``. The
masked residue is measurements (``TT-TG distance is 15 mm``, ``Insall-Salvati
index is 1.2``), normal variants (fabella, bipartite patella, Wiberg type,
meniscal flounce) and benign incidentals (enchondroma, non-ossifying fibroma)
that name none of the 12 targets. Vagueness correlates with negation in this
corpus, so defaulting the ambiguous to ``negative`` inflates the negative class
and defaulting it to ``positive`` inflates prevalence; only a real
no-information state is honest. Report-level aggregation decides what to do
with it -- see ``kappa.labels.masking``.
"""

from __future__ import annotations

import re
from typing import Literal, NamedTuple

__all__ = [
    "Polarity",
    "POSITIVE",
    "NEGATIVE",
    "MASK",
    "DescriptionSignals",
    "signals",
    "polarity",
]

Polarity = Literal["positive", "negative", "mask"]

POSITIVE: Polarity = "positive"
NEGATIVE: Polarity = "negative"
MASK: Polarity = "mask"

# Every stem below was harvested from the survey's own description vocabulary
# (595 tokens at n>=4), not from outside clinical knowledge, and each was
# checked against the findings it changes.

#: Pathology stems matched at a word start.
_ABNORMAL_PREFIX = r"""
tear|torn|tearing|ruptur|degenerat|edema|oedema|edematous|effusion|cyst|
osteophyt|chondromalac|fractur|contusion|bruise|sprain|strain|lesion|defect|
thinning|thinned|narrowing|extrusion|extruded|subluxation|dislocat|luxation|
discontinuity|atrophy|erosion|ulcer|impingement|injur|h[ae]matoma|alta|tilt|
malalign|myxoid|mucoid|ganglion|hypertroph|hypoplas|dysplas|irregular|fissur|
damage|abnormal|pathologic|spur|collapse|necrosis|fragment|displac|swelling|
inflamm|heterogene|hyperintens|hypointens|distend|distension|irritation|
thickening|thickened|blunting|convexity|translation|h[ae]morrhag|lipoma|tumo|
loss|reduced|decreased|decrease|increased|increase|elevation|deformit|
granulation|septat|varic|dilat|plica|grade|chang|sclerot|sharpen|fluid|flap|
suffering|foci|focus|island|nodular|sequelae|scar|effac|lys|worn|wear|wearing|
denudation|hyperpressure|syndrome|disrupt|collection|friction|compress|
protrus|loose|shallow
"""

# Matched anywhere in the word: Greek and Latin compounds put the pathology at
# the end, so a word-start anchor misses every one of them -- `gonarthrosis`,
# `meniscopathy`, `osteoarthritis`, `osteochondral`, `fibrocystic`.
_ABNORMAL_SUFFIX = r"""
arthros|arthrit|arthrop|opath|itis|malaci|chondrosis|chondral|sclerosis|
fibrosis|tendinosis|cystic
"""

#: Words asserting the structure is normal.
_NORMAL_PREFIX = r"""
normal|normally|intact|preserv|unremarkable|maintained|appropriate|natural|
physiolog|regular|homogen|smooth|anatomic|peculiarit|particularit|conserved|
unchanged|symmetric|correct|centered|continuous
"""

# Cues that negate what follows. Closed on *both* sides deliberately: an
# open-ended `no` matches `normal`, `nodular` and `non-displaced`, which
# silently negated a fracture and a patellar tilt. `free` is excluded because
# `free fluid` and `free edge` are findings, not negations -- only `free of`
# negates.
_NEGATION = r"""
no|not|without|absence|absent|negative|lack|lacks|denies|excluded|neither|nor
"""


def _alternation(block: str) -> str:
    return "|".join(part for part in "".join(block.split()).split("|") if part)


_AB_RE = re.compile(
    r"\b(?:" + _alternation(_ABNORMAL_PREFIX) + r")|(?:" + _alternation(_ABNORMAL_SUFFIX) + r")",
    re.IGNORECASE,
)
_NM_RE = re.compile(r"\b(?:" + _alternation(_NORMAL_PREFIX) + r")", re.IGNORECASE)
_NG_RE = re.compile(r"\b(?:" + _alternation(_NEGATION) + r")\b|\bfree of\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")


class DescriptionSignals(NamedTuple):
    """Character offsets of the first hit of each cue class, or None.

    Offsets rather than booleans because the *order* decides polarity:
    ``myxoid degeneration without tear`` and ``no evidence of tear`` both carry
    an abnormality and a negation, and only their relative position separates a
    positive finding from a negative one.
    """

    abnormal: int | None
    normal: int | None
    negation: int | None


def signals(description: str | None) -> DescriptionSignals:
    """Locate abnormality, normality and negation cues in a description."""
    text = _PUNCT_RE.sub(" ", (description or "").lower())
    text = _WS_RE.sub(" ", text).strip()
    ab = _AB_RE.search(text)
    nm = _NM_RE.search(text)
    ng = _NG_RE.search(text)
    return DescriptionSignals(
        abnormal=ab.start() if ab else None,
        normal=nm.start() if nm else None,
        negation=ng.start() if ng else None,
    )


def polarity(assertion: str, description: str | None) -> Polarity:
    """Return the polarity of one finding from both of its fields.

    ``negated`` and ``uncertain`` are taken at face value: the model only
    reaches for them when the report is explicitly ruling something out or
    hedging, and in those cases the description is unreliable on its own (the
    mirror trap). ``affirmed`` carries no information by itself and is resolved
    entirely from the description.
    """
    if assertion == "uncertain":
        return MASK
    if assertion == "negated":
        return NEGATIVE
    if not description or not description.strip():
        return MASK

    ab, nm, ng = signals(description)
    if ab is not None:
        # An abnormality is asserted unless a negation cue precedes it. A
        # co-occurring normality word does **not** veto it: reports routinely
        # say what is preserved alongside what is damaged ("intact, with
        # periligamentous edema", "loss of integrity secondary to complete
        # rupture"), and in the 39 such findings the abnormality was the point
        # every time.
        return NEGATIVE if (ng is not None and ng < ab) else POSITIVE
    if ng is not None or nm is not None:
        # Nothing abnormal named, and the description either negates or asserts
        # normality. This is the normality sweep, and it is the whole reason
        # this module exists.
        return NEGATIVE
    return MASK
