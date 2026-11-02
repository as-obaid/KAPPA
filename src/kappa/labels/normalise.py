"""Report text normalisation and script detection.

Normalisation is deliberately conservative: it collapses whitespace and nothing
else. De-identification placeholders such as ``[DATE]`` are *kept*, because a
report that says "[DATE]" shows that de-identification ran, and stripping
them would silently change token counts between a run and its re-run.

    from kappa.labels.normalise import normalise_report, content_hash

    text = normalise_report(row["Report"])
    key = content_hash(text)          # identical reports share a key

``detect_script`` is a script test, not language identification. It answers
"which alphabet is this written in", which is all that is needed to stratify a
sample so low-frequency writing systems are represented. Naming it ``language``
would overclaim: Cyrillic covers Bulgarian, Macedonian and Serbian at least.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

__all__ = ["normalise_report", "content_hash", "detect_script", "fold_lookalikes", "SCRIPTS"]

#: Script buckets, in match order. Latin is the fallback, never matched directly.
SCRIPTS = ("greek", "cyrillic", "turkish", "latin")

_CYRILLIC = re.compile(r"[Ѐ-ӿԀ-ԯ]")
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
# Turkish-specific letters. Excludes the ones shared with Spanish/French/German
# (o-umlaut, u-umlaut, c-cedilla) so that this does not fire on those languages.
_TURKISH = re.compile(r"[ığşİĞŞ]")

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def normalise_report(text: str) -> str:
    """Collapse whitespace and normalise Unicode, preserving all content.

    NFC normalisation matters here: the corpus mixes precomposed and combining
    forms of accented Greek and Latin characters, so two reports that render
    identically can hash differently without it.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


#: Characters this corpus uses in place of the letter they look like. NFC does
#: not touch these -- they are different codepoints with different meanings, not
#: composition variants -- so every one of them silently zeroes out a match.
_LOOKALIKES = {
    "\u2206": "\u0394",  # INCREMENT -> GREEK CAPITAL DELTA. The corpus writes
                          # `∆ΕΞΙΟΥ` ("right"), which does not lowercase to
                          # `δεξιου`, so a Greek laterality pattern finds nothing.
    "\u00b5": "\u03bc",  # MICRO SIGN -> GREEK SMALL MU, as in `µηνίσκων`.
    "\u0130": "I",       # LATIN CAPITAL I WITH DOT ABOVE. Turkish `DİZ` ("knee")
                          # lowercases to `i` + U+0307 in Python's default
                          # casing, so a plain `diz` pattern misses every
                          # capitalised Turkish header.
}

_LOOKALIKE_RE = re.compile("|".join(map(re.escape, _LOOKALIKES)))


def fold_lookalikes(text: str) -> str:
    """Map visually identical codepoints onto the letter they stand for.

    Use this before **matching**, never before hashing or storing: it is lossy
    by design, and `normalise_report` deliberately does not apply it so that
    stored text stays byte-faithful to the source.

    This is not hypothetical. Folding U+0130 alone moved measured Turkish
    laterality from 0.4% to 67.8% of Turkish reports -- the pattern had been
    missing every capitalised header in the language.
    """
    return _LOOKALIKE_RE.sub(lambda m: _LOOKALIKES[m.group(0)], text)


def content_hash(text: str) -> str:
    """Stable key for a normalised report, so duplicates are extracted once."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def detect_script(text: str) -> str:
    """Return the dominant non-Latin script, or ``"latin"``.

    Order matters. Greek and Cyrillic are tested first because they are
    unambiguous codepoint ranges; the Turkish test is a diacritic heuristic and
    would otherwise misfire on a Greek report containing a Latin-script header.
    """
    if _GREEK.search(text):
        return "greek"
    if _CYRILLIC.search(text):
        return "cyrillic"
    if _TURKISH.search(text):
        return "turkish"
    return "latin"
