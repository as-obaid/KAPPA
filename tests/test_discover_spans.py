"""Span verification must not mistake a faithful quote for a paraphrase.

``load_reports`` puts every report through ``normalise_report``, which applies
NFC and collapses whitespace. The model's decoded output goes through no such
step, so a span it quoted verbatim can still fail a raw substring test -- and
both ways that happens fall hardest on the non-Latin scripts this survey exists
to assess. Counting them as paraphrases would understate Greek and Cyrillic
span fidelity, which is the single number the reconnaissance pass is for.

The strict count is kept alongside the tolerant one so the size of the artifact
stays visible instead of being normalised away.
"""

from __future__ import annotations

import unicodedata

from kappa.labels.discover import verify_spans
from kappa.labels.normalise import normalise_report


def _rec(*spans):
    return {"findings": [{"span": s} for s in spans]}


def test_no_findings_is_zero_of_zero():
    assert verify_spans({}, "any report") == (0, 0, 0)
    assert verify_spans({"findings": []}, "any report") == (0, 0, 0)


def test_verbatim_quote_counts_both_ways():
    report = normalise_report("Medial meniscus tear is seen.")
    assert verify_spans(_rec("Medial meniscus tear"), report) == (1, 1, 1)


def test_paraphrase_counts_neither_way():
    report = normalise_report("Medial meniscus tear is seen.")
    assert verify_spans(_rec("meniscal injury"), report) == (0, 0, 1)


def test_decomposed_greek_quote_is_not_a_paraphrase():
    """NFC report vs NFD model output -- identical text, different codepoints."""
    report = normalise_report("οστεοαρθρίτιδα του έσω μηριαίου κονδύλου")
    span = unicodedata.normalize("NFD", "οστεοαρθρίτιδα")
    assert span not in report  # the artifact being guarded against
    verbatim, tolerant, total = verify_spans(_rec(span), report)
    assert verbatim == 0
    assert tolerant == 1  # a faithful quote, so it has to count
    assert total == 1


def test_quote_across_a_line_break_is_not_a_paraphrase():
    report = normalise_report("Grade 2 chondromalacia\nof the patella")
    span = "chondromalacia of the patella"  # the break arrives as a space
    assert span not in report
    assert verify_spans(_rec(span), report) == (0, 1, 1)


def test_non_string_span_is_a_finding_but_never_a_match():
    assert verify_spans({"findings": [{"span": None}, {}]}, "text") == (0, 0, 2)


def test_counts_are_per_finding_not_deduplicated():
    report = normalise_report("Effusion. Effusion.")
    assert verify_spans(_rec("Effusion", "Effusion"), report) == (2, 2, 2)
