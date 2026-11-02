"""A language-correlated label is a shortcut the image model will happily take.

The effect this guards is measured, not hypothetical: on the first shard of the
full extraction run, Turkish reports derived Synovitis at 2% against English's
16% and Effusion at 77% against English's 64%. Language implies institution
implies scanner implies image appearance, so a model can satisfy such a label
from the scanner signature rather than the pathology -- which inflates CV and
is invisible at test time.

Two ways of reporting that effect are wrong in opposite directions, and both
are tested here. Dividing positives by the bucket size instead of by the
label's own usable count turns absent evidence into derived negatives and
understates prevalence. Ranking spreads without a minimum bucket size promotes
whichever label happened to be 0% in a bucket of fifteen.
"""

from __future__ import annotations

from kappa.evaluation.label_audit import (
    MIN_BUCKET_FOR_INFERENCE,
    audit_by_language,
)


def _record(language: str, *, anatomy: str = "medial meniscus",
            description: str = "medial meniscus tear",
            assertion: str = "affirmed") -> dict:
    return {
        "language": language,
        "knee_side": "left",
        "knee_side_evidence": "left knee",
        "_parsed": True,
        "_valid": True,
        "findings": [
            {"span": description, "description": description, "anatomy": anatomy,
             "assertion": assertion, "severity_wording": None}
        ],
    }


def _corpus(**counts: int) -> list[dict]:
    """`_corpus(English=3, Turkish=2)` -> five records in those two buckets."""
    out = []
    for language, n in counts.items():
        out.extend(_record(language) for _ in range(n))
    return out


class TestBucketing:
    def test_buckets_are_ordered_largest_first(self):
        audit = audit_by_language(_corpus(Dutch=1, English=5, Turkish=3))
        assert audit.languages == ("English", "Turkish", "Dutch")

    def test_a_missing_language_field_becomes_an_explicit_unknown_bucket(self):
        record = _record("English")
        del record["language"]
        audit = audit_by_language([record])
        # Not silently dropped: a record whose language failed to parse still
        # carries derived labels, and losing it would bias the very comparison
        # this module exists to make.
        assert audit.languages == ("unknown",)

    def test_total_studies_equals_the_records_given(self):
        audit = audit_by_language(_corpus(English=4, Greek=2))
        assert audit.n_studies == 6


class TestPrevalenceDenominator:
    def test_prevalence_divides_by_usable_cells_not_by_bucket_size(self):
        # One report names the medial meniscus; the other eleven labels are
        # unmentioned. With unmentioned=None those cells are unusable, so a
        # label nobody addressed must not report 0% over the whole bucket.
        audit = audit_by_language(_corpus(English=4), unmentioned=None)
        mm = audit.table["Medial Meniscus"]["English"]
        assert mm.usable == 4
        assert mm.rate == 1.0
        acl = audit.table["ACL"]["English"]
        assert acl.usable == 0
        # None, not 0.0 -- "nobody could supervise this" is not "all negative".
        assert acl.rate is None

    def test_unmentioned_zero_makes_every_cell_usable(self):
        audit = audit_by_language(_corpus(English=4), unmentioned=0.0)
        assert audit.table["ACL"]["English"].usable == 4
        assert audit.table["ACL"]["English"].rate == 0.0


class TestSpreadRanking:
    def test_a_real_split_across_large_buckets_is_detected(self):
        n = MIN_BUCKET_FOR_INFERENCE
        records = [_record("English") for _ in range(n)]
        records += [
            _record("Turkish", anatomy="anterior cruciate ligament",
                    description="anterior cruciate ligament is intact",
                    assertion="negated")
            for _ in range(n)
        ]
        audit = audit_by_language(records, unmentioned=0.0)
        worst = dict((row[0], row[3]) for row in audit.worst_spreads(limit=12))
        # English derives Medial Meniscus positive, Turkish does not.
        assert worst["Medial Meniscus"] == 1.0

    def test_small_buckets_are_excluded_from_the_ranking_by_default(self):
        # A 100%-vs-0% split that exists ONLY in a tiny bucket must not top the
        # table. This is the honest-reporting requirement: on one real shard the
        # widest raw spreads all came from buckets of 15 to 23 studies.
        big = [_record("English") for _ in range(MIN_BUCKET_FOR_INFERENCE)]
        tiny = [
            _record("Dutch", anatomy="anterior cruciate ligament",
                    description="anterior cruciate ligament is intact",
                    assertion="negated")
        ]
        audit = audit_by_language(big + tiny, unmentioned=0.0)
        assert "Dutch" not in audit.large_buckets()
        assert audit.worst_spreads() == []
        # Still reachable when explicitly asked for.
        assert audit.worst_spreads(large_only=False)

    def test_large_buckets_reports_only_adequately_sized_ones(self):
        audit = audit_by_language(
            _corpus(English=MIN_BUCKET_FOR_INFERENCE, Dutch=MIN_BUCKET_FOR_INFERENCE - 1)
        )
        assert audit.large_buckets() == ("English",)


class TestRender:
    def test_small_buckets_are_marked_in_the_rendered_header(self):
        audit = audit_by_language(
            _corpus(English=MIN_BUCKET_FOR_INFERENCE, Dutch=2)
        )
        rendered = audit.render()
        assert "Dutc*" in rendered
        assert "Engl*" not in rendered

    def test_render_names_every_label_and_the_study_count(self):
        audit = audit_by_language(_corpus(English=3))
        rendered = audit.render()
        assert "n = 3" in rendered
        for label in ("ACL", "Baker's", "PF OA", "Synovitis"):
            assert label in rendered
