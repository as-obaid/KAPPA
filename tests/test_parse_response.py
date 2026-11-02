"""One encoding artifact in the corpus must not discard a whole extraction.

Measured on the full extraction run: 1 of the first 2,168 reports produced a
response that was complete, correctly closed, and still unparseable. The source
report carries RTF encoding artifacts and the model reproduced one verbatim
inside a span -- `graad 2 \\'e0 3`, where `\\'e0` is RTF for a-grave -- and `\\'`
is not a legal JSON escape, so `json.loads` rejected the entire document. The
extraction itself was perfect: recovering it yielded 9 findings with zero
validation warnings.

The tests here pin two things that pull in opposite directions. The repair must
be tolerant enough to rescue that document, and conservative enough not to
touch a valid one or to "helpfully" decode the artifact -- because span
verification asks whether a span occurs in the report, and the report contains
the literal artifact, not the character it stands for. Decoding would raise
span fidelity's apparent quality while making it wrong.
"""

from __future__ import annotations

import json

from kappa.labels.discover import parse_response, repair_invalid_escapes

#: The exact byte sequence from the corpus record that failed, reduced.
RTF_ARTIFACT = r'{"span": "graad 2 \'e0 3", "n": 1}'


def _document(body: str) -> str:
    """A minimal well-formed response wrapping one string value."""
    return '{"language": "Dutch", "findings": [{"span": "%s"}]}' % body


class TestRepairIsNeeded:
    def test_the_corpus_artifact_defeats_the_standard_library(self):
        # If this ever stops raising, the repair below is dead code and the
        # test suite should say so rather than quietly passing.
        try:
            json.loads(RTF_ARTIFACT)
        except json.JSONDecodeError as exc:
            assert "escape" in str(exc).lower()
        else:  # pragma: no cover
            raise AssertionError("stdlib parsed an invalid escape")

    def test_parse_response_recovers_the_corpus_artifact(self):
        parsed = parse_response(RTF_ARTIFACT)
        assert parsed is not None
        assert parsed["n"] == 1


class TestFidelityToTheSource:
    def test_the_artifact_is_escaped_and_never_decoded(self):
        # a-grave would be the friendly reading. It is the wrong one: the report
        # contains the literal backslash sequence, so a decoded span would no
        # longer be found in the report it came from.
        parsed = parse_response(RTF_ARTIFACT)
        assert parsed["span"] == r"graad 2 \'e0 3"
        assert "à" not in parsed["span"]

    def test_a_repaired_span_still_matches_the_report_it_came_from(self):
        report = "Chondropathie graad 2 \\'e0 3 mediaal."
        parsed = parse_response(_document(r"graad 2 \'e0 3"))
        assert parsed["findings"][0]["span"] in report


class TestConservatism:
    def test_a_valid_document_is_not_altered(self):
        valid = '{"a": "an \\n escape, a \\" quote, a \\\\ backslash", "b": 2}'
        repaired, n = repair_invalid_escapes(valid)
        assert n == 0
        assert repaired == valid

    def test_a_unicode_escape_is_left_alone(self):
        repaired, n = repair_invalid_escapes(r'{"a": "à"}')
        assert n == 0
        assert parse_response(r'{"a": "à"}')["a"] == "à"

    def test_every_legal_json_escape_survives(self):
        for esc in ('\\"', "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t"):
            body = '{"a": "x%sy"}' % esc
            _, n = repair_invalid_escapes(body)
            assert n == 0, f"{esc!r} was wrongly treated as invalid"

    def test_the_repair_reports_how_many_escapes_it_fixed(self):
        # A silent repair is how a corpus fact stops being visible.
        _, n = repair_invalid_escapes(r'{"a": "\'x", "b": "\'y"}')
        assert n == 2


class TestExistingBehaviourIsPreserved:
    def test_a_markdown_fence_is_still_tolerated(self):
        assert parse_response('```json\n{"a": 1}\n```')["a"] == 1

    def test_prose_around_the_object_is_still_sliced_away(self):
        assert parse_response('Here you go:\n{"a": 1}\nHope that helps.')["a"] == 1

    def test_a_response_with_no_object_is_still_None(self):
        assert parse_response("I cannot answer that.") is None

    def test_an_unrecoverable_response_is_None_rather_than_a_raise(self):
        # Truncated mid-structure: no repair can close it, and losing one
        # response must not lose the rest of the run.
        assert parse_response('{"a": "unterminated') is None
