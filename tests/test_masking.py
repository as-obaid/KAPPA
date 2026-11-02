"""Missing information must stay missing, and the wrong hazard must not mask.

The asymmetry these guard: masking a study costs one row of training signal,
while mislabelling one asserts a normal structure that was never scanned. The
tests are written so that the cheap fix -- treating an empty findings list as
twelve zeros -- fails loudly.
"""

from __future__ import annotations

import pytest

from kappa.labels.masking import is_bilateral_text, report_mask, resolve_label
from kappa.labels.polarity import MASK, NEGATIVE, POSITIVE
from kappa.labels.schema import validate_extraction


def _extraction(**over):
    base = {
        "language": "English",
        "knee_side": "left",
        "knee_side_evidence": "left knee",
        "findings": [
            {"span": "ACL intact", "description": "ACL is intact", "anatomy": "ACL",
             "assertion": "affirmed"}
        ],
        "notes": None,
    }
    base.update(over)
    return base


class TestEmptyFindings:
    def test_empty_findings_is_masked_not_twelve_zeros(self):
        mask = report_mask(_extraction(findings=[]))
        assert mask.masked
        assert mask.reasons == ("empty_findings",)

    def test_a_report_with_findings_is_not_masked(self):
        assert report_mask(_extraction()) == (False, ())


class TestBilateral:
    def test_anatomy_naming_both_sides_is_bilateral(self):
        # The real case: knee_side stayed unstated while the model resolved
        # laterality into the anatomy strings.
        mask = report_mask(
            _extraction(
                knee_side="unstated",
                knee_side_evidence=None,
                findings=[
                    {"span": "a", "anatomy": "right medial meniscus", "assertion": "affirmed"},
                    {"span": "b", "anatomy": "left anterior cruciate ligament", "assertion": "affirmed"},
                ],
            )
        )
        assert mask.masked
        assert "bilateral_report" in mask.reasons

    def test_one_side_only_is_not_bilateral(self):
        mask = report_mask(
            _extraction(
                findings=[
                    {"span": "a", "anatomy": "right medial meniscus", "assertion": "affirmed"},
                    {"span": "b", "anatomy": "right anterior cruciate ligament", "assertion": "affirmed"},
                ]
            )
        )
        assert not mask.masked

    def test_the_corpus_marker_is_definitive(self):
        # The corpus inserts a `[BILATERAL NOTE` marker where two reports were
        # filed under one study (27 of 4,407 reports). The marker alone decides.
        text = ("MT ΔΕΞΙΟΥ ΓΟΝΑΤΟΣ ... "
                "[BILATERAL NOTE: a second knee report follows] ...")
        mask = report_mask(_extraction(), report_text=text)
        assert mask.masked
        assert "bilateral_report" in mask.reasons

    @pytest.mark.parametrize(
        "text",
        [
            "\u2206\u0395\u039e\u0399\u039f\u03a5 \u0393\u039f\u039d\u0391\u03a4\u039f\u03a3: ... "
            "\u0391\u03a1\u0399\u03a3\u03a4\u0395\u03a1\u039f\u03a5 \u0393\u039f\u039d\u0391\u03a4\u039f\u03a3: ...",
            "\u0434\u044f\u0441\u043d\u0430 \u043a\u043e\u043b\u044f\u043d\u043d\u0430 \u0441\u0442\u0430\u0432\u0430: ... "
            "\u043b\u044f\u0432\u0430 \u043a\u043e\u043b\u044f\u043d\u043d\u0430 \u0441\u0442\u0430\u0432\u0430: ...",
        ],
    )
    def test_paired_side_sections(self, text):
        mask = report_mask(_extraction(), report_text=text)
        assert mask.masked, text
        assert "bilateral_report" in mask.reasons

    def test_greek_delta_written_as_increment_still_matches(self):
        # U+2206 INCREMENT, not U+0394. NFC leaves it alone, so an unfolded
        # pattern matches nothing -- this is the encoding trap.
        assert "\u2206" in "\u2206\u0395\u039e\u0399\u039f\u03a5"
        text = ("\u2206\u0395\u039e\u0399\u039f\u03a5 \u0393\u039f\u039d\u0391\u03a4\u039f\u03a3: a. "
                "\u0391\u03a1\u0399\u03a3\u03a4\u0395\u03a1\u039f\u03a5 \u0393\u039f\u039d\u0391\u03a4\u039f\u03a3: b.")
        assert is_bilateral_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            # Each of these fired the keyword scan that was removed. All are
            # single-knee reports; masking them would discard usable studies.
            "Antecedentes Cl\u00ednicos: Gonalgia bilateral. Hallazgos: ...",
            "History: bilateral knee pain over 3 months. No prior studies.",
            "The bilateral menisci are intact.",
            "\u041a\u043e\u043b\u0430\u0442\u0435\u0440\u0430\u043b\u043d\u0438 \u043b\u0438\u0433\u0430\u043c\u0435\u043d\u0442\u0438 \u0434\u0432\u0443\u0441\u0442\u0440\u0430\u043d\u043d\u043e \u2013 \u0431.\u043e.",
            "edema \u00f3seo en ambos platillos tibiales, sobre todo posterior",
            "Hondromalacija III. obostrano, femoralni kondili",
            "COMPARISON: prior standing bilateral knee radiographs of [Date].",
            "MRT beide Knie",
        ],
    )
    def test_the_keyword_scan_false_positives_are_not_masked(self, text):
        mask = report_mask(_extraction(), report_text=text)
        assert not mask.masked, f"masked a single-knee report on: {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "LEFT KNEE MRI. History: two months of pain in the right knee.",
            "RIGHT KNEE MRI. History: [Age], long-standing pain in the left knee.",
            "History: right knee pain for 3 weeks, left kidney stone noted. "
            "TECHNIQUE: left knee MRI, standard protocol.",
            "Right knee: anterior subcutaneous soft tissue infiltration seen. "
            "Left knee MRI, sagittal planes only.",
        ],
    )
    def test_the_other_knee_in_the_indication_is_not_a_bilateral_report(self, text):
        # 10 single-knee reports name the other knee in their indication,
        # comparison or a dictation slip. Masking them loses usable studies;
        # the fix is to strip the history span before extraction.
        assert not report_mask(_extraction(), report_text=text).masked, text

    def test_a_normal_unilateral_report_text_does_not_fire(self):
        mask = report_mask(_extraction(), report_text="MRI of the left knee. ACL intact.")
        assert not mask.masked

    def test_knee_side_alone_never_decides(self):
        # 4 of the 5 bilateral reports had knee_side == "unstated"; an unstated
        # side is the corpus norm (47%) and must not mask on its own.
        mask = report_mask(_extraction(knee_side="unstated", knee_side_evidence=None))
        assert not mask.masked


class TestMultipleStudies:
    def test_prior_comparison_study_is_flagged(self):
        mask = report_mask(
            _extraction(
                notes="The report contains two separate technical descriptions and findings "
                "sections, likely representing a current study and a comparison with a prior."
            )
        )
        assert "multiple_studies" in mask.reasons

    def test_restated_findings_are_not_masked(self):
        # ~10% of reports repeat their findings in a summary section. That
        # inflates counts, not polarity -- masking would discard good evidence.
        mask = report_mask(
            _extraction(
                notes="The findings are repeated identically in both the Results and "
                "Impression sections."
            )
        )
        assert not mask.masked, mask.reasons


class TestResolveLabel:
    def test_any_positive_wins(self):
        assert resolve_label([NEGATIVE, MASK, POSITIVE]) == 1.0

    def test_negative_when_ruled_out_and_never_asserted(self):
        assert resolve_label([NEGATIVE, MASK]) == 0.0

    def test_all_masked_is_unknown(self):
        assert resolve_label([MASK, MASK]) is None

    def test_unmentioned_label_defaults_to_unknown_not_zero(self):
        # Defaulting to 0.0 reads "not mentioned" as "normal", which is only
        # true for reports that sweep normal structures -- 35% of English
        # findings, 4% of Greek.
        assert resolve_label([]) is None

    def test_unmentioned_policy_is_caller_supplied(self):
        assert resolve_label([], unmentioned=0.0) == 0.0

    def test_rejects_a_non_polarity(self):
        with pytest.raises(ValueError):
            resolve_label(["affirmed"])


class TestAgainstValidatedExtraction:
    def test_accepts_a_reportextraction_model_not_only_a_dict(self):
        model, warnings = validate_extraction(_extraction(findings=[]))
        assert model is not None and not warnings
        assert report_mask(model).reasons == ("empty_findings",)


class TestExtractionFailureIsNotAnEmptyReport:
    """Both arrive as `findings == []`; only one is the corpus's fault."""

    def test_unparsed_extraction_is_reported_as_such(self):
        mask = report_mask(
            {"_parsed": False, "_valid": False, "findings": [],
             "_schema_warnings": ["response was not parseable JSON"]}
        )
        assert mask.masked
        assert mask.reasons == ("extraction_failed",)

    def test_a_genuinely_empty_findings_section_still_reads_as_empty(self):
        mask = report_mask({"_parsed": True, "_valid": True, "findings": []})
        assert mask.reasons == ("empty_findings",)

    def test_schema_invalid_is_a_failure_not_an_empty_report(self):
        mask = report_mask({"_parsed": True, "_valid": False, "findings": []})
        assert mask.reasons == ("extraction_failed",)

    def test_absent_flags_do_not_invent_a_failure(self):
        # Records predating schema validation carry neither flag.
        assert report_mask({"findings": []}).reasons == ("empty_findings",)
