"""Extraction validation must salvage what it can and report what it fixed.

The design rule under test: a survey over 4,407 reports loses more by
discarding a report than by recording that one field needed repair. So these
assert two things together -- that the coercion happened, *and* that it was
reported. A silent fix would pass a naive test and defeat the point, since the
value of the warning list is being able to count how often the model's output
needed repair.
"""

from __future__ import annotations

import pytest

from kappa.labels.schema import Finding, ReportExtraction, validate_extraction


def _payload(**over):
    base = {
        "language": "Bulgarian",
        "language_confident": True,
        "knee_side": "left",
        "knee_side_evidence": "лявата колянна става",
        "sections": ["МР находка"],
        "findings": [
            {
                "span": "малък излив",
                "description": "small effusion",
                "anatomy": "suprapatellar bursa",
                "assertion": "affirmed",
                "severity_wording": "малък",
            }
        ],
        "stated_grades": ["2а степен"],
        "notes": None,
    }
    base.update(over)
    return base


def test_clean_payload_validates_without_warnings():
    model, warnings = validate_extraction(_payload())
    assert warnings == []
    assert model is not None
    assert model.knee_side == "left"
    assert len(model.findings) == 1
    assert model.findings[0].assertion == "affirmed"


def test_non_object_payload_is_rejected():
    for bad in ([], "text", 3, None):
        model, warnings = validate_extraction(bad)
        assert model is None
        assert warnings and "expected object" in warnings[0]


def test_zero_findings_is_legal():
    # One report in the measured 300 produced no findings at all.
    model, warnings = validate_extraction(_payload(findings=[]))
    assert model is not None and model.findings == [] and warnings == []


def test_missing_optional_fields_default_rather_than_fail():
    model, warnings = validate_extraction({"findings": []})
    assert model is not None
    assert model.language is None
    assert model.knee_side == "unstated"
    assert model.sections == [] and model.stated_grades == []


def test_null_like_strings_become_none():
    f = _payload()["findings"][0] | {"severity_wording": "null", "anatomy": "  "}
    model, _ = validate_extraction(_payload(findings=[f]))
    assert model is not None
    assert model.findings[0].severity_wording is None
    assert model.findings[0].anatomy is None


def test_unknown_assertion_becomes_uncertain_not_none():
    f = _payload()["findings"][0] | {"assertion": "Negative"}
    model, _ = validate_extraction(_payload(findings=[f]))
    assert model is not None
    assert model.findings[0].assertion == "negated"  # via alias

    f2 = _payload()["findings"][0] | {"assertion": "wat"}
    model2, _ = validate_extraction(_payload(findings=[f2]))
    assert model2 is not None
    assert model2.findings[0].assertion == "uncertain"


def test_missing_assertion_does_not_become_none():
    """Unguarded, a missing assertion would enter the distribution as None."""
    f = {k: v for k, v in _payload()["findings"][0].items() if k != "assertion"}
    model, _ = validate_extraction(_payload(findings=[f]))
    assert model is not None
    assert model.findings[0].assertion == "uncertain"


def test_finding_without_span_is_dropped_and_reported():
    good = _payload()["findings"][0]
    bad = {k: v for k, v in good.items() if k != "span"}
    model, warnings = validate_extraction(_payload(findings=[good, bad]))
    assert model is not None
    assert len(model.findings) == 1, "the good finding must survive"
    assert any("finding[1] invalid" in w for w in warnings)


def test_non_dict_finding_is_dropped_and_reported():
    model, warnings = validate_extraction(_payload(findings=["not a dict"]))
    assert model is not None and model.findings == []
    assert any("finding[0] is str, dropped" in w for w in warnings)


def test_findings_not_a_list_is_reported():
    model, warnings = validate_extraction(_payload(findings={"span": "x"}))
    assert model is not None and model.findings == []
    assert any("not a list" in w for w in warnings)


def test_bare_string_where_list_expected_is_absorbed():
    model, warnings = validate_extraction(_payload(sections="FINDINGS"))
    assert model is not None
    assert model.sections == ["FINDINGS"]
    assert warnings == []


def test_side_aliases_and_unknown_sides():
    assert validate_extraction(_payload(knee_side="R"))[0].knee_side == "right"
    assert (
        validate_extraction(_payload(knee_side="bilateral"))[0].knee_side == "unstated"
    )
    assert validate_extraction(_payload(knee_side=42))[0].knee_side == "unstated"


def test_side_evidence_consistency_is_flagged_both_ways():
    # measured: evidence was null in exactly the 140 unstated cases
    _, w1 = validate_extraction(_payload(knee_side="left", knee_side_evidence=None))
    assert any("no evidence span" in w for w in w1)

    _, w2 = validate_extraction(
        _payload(knee_side="unstated", knee_side_evidence="left knee")
    )
    assert any("unstated but an evidence span" in w for w in w2)


def test_extra_keys_are_preserved_not_discarded():
    """Unknown keys are forensic evidence about the model, so keep them."""
    model, _ = validate_extraction(_payload(surprise="value"))
    assert model is not None
    assert model.model_dump().get("surprise") == "value"


def test_language_confident_is_preserved_but_optional():
    model, _ = validate_extraction(_payload(language_confident=None))
    assert model is not None and model.language_confident is None


def test_finding_model_rejects_empty_span_directly():
    with pytest.raises(Exception):
        Finding.model_validate({"span": ""})


def test_report_model_accepts_the_measured_maximum_findings():
    # The measured run peaked at 37 findings in one report.
    many = [{"span": f"s{i}", "assertion": "affirmed"} for i in range(37)]
    model = ReportExtraction.model_validate({"findings": many})
    assert len(model.findings) == 37
