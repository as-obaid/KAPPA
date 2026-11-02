"""The W&B tables had never run. These make the row shape checkable offline.

Every logged run to date predates the two-table split, so the replacement was
written, committed, and never exercised -- its only caller sits inside a
`try/except` in a script that runs on a GPU node, which means a shape bug would
surface as a printed note and a silently missing table.
"""

from __future__ import annotations

import pytest

from kappa.labels.discover import (
    FINDINGS_COLUMNS,
    RECORDS_COLUMNS,
    findings_table,
    records_table,
)


def _record(**over):
    base = {
        "_study_uid": "1.2.3",
        "_script": "latin",
        "_parsed": True,
        "_valid": True,
        "_spans_ok": 2,
        "_spans_exact": 1,
        "_spans_total": 2,
        "_schema_warnings": [],
        "language": "English",
        "knee_side": "left",
        "knee_side_evidence": "left knee",
        "sections": ["FINDINGS", "IMPRESSION"],
        "stated_grades": ["grade II"],
        "notes": None,
        "findings": [
            {"span": "partial tear", "description": "partial tear of the medial meniscus",
             "anatomy": "medial meniscus", "assertion": "affirmed",
             "severity_wording": "partial"},
            {"span": "ACL intact", "description": "ACL is intact", "anatomy": "ACL",
             "assertion": "affirmed", "severity_wording": None},
        ],
    }
    base.update(over)
    return base


SOURCES = {"1.2.3": "MRI left knee. partial tear. ACL intact."}


class TestRowShape:
    def test_every_record_row_matches_the_declared_columns(self):
        rows = records_table([_record()], SOURCES)
        assert len(rows) == 1
        assert len(rows[0]) == len(RECORDS_COLUMNS)

    def test_every_finding_row_matches_the_declared_columns(self):
        rows = findings_table([_record()], SOURCES)
        assert len(rows) == 2
        for row in rows:
            assert len(row) == len(FINDINGS_COLUMNS)

    def test_one_row_per_finding_not_one_per_report(self):
        rows = findings_table([_record(), _record()], SOURCES)
        assert len(rows) == 4


class TestColumnsThatWereMissingBefore:
    @pytest.mark.parametrize(
        "column",
        ["knee_side_evidence", "sections", "stated_grades", "notes", "warnings",
         "valid", "spans_exact", "masked", "mask_reasons"],
    )
    def test_records_table_carries_it(self, column):
        # None of these were reachable from W&B in any logged run.
        assert column in RECORDS_COLUMNS

    def test_findings_table_carries_polarity_beside_assertion(self):
        assert "polarity" in FINDINGS_COLUMNS
        assert "assertion" in FINDINGS_COLUMNS

    def test_records_table_has_no_findings_blob(self):
        # The blob is what truncated 63 of 300 rows into invalid JSON.
        assert "findings" not in RECORDS_COLUMNS


class TestValues:
    def _row(self, records=None, sources=None):
        rows = records_table(records or [_record()], sources or SOURCES)
        return dict(zip(RECORDS_COLUMNS, rows[0]))

    def test_polarity_counts_are_per_report(self):
        row = self._row()
        assert (row["n_positive"], row["n_negative"], row["n_masked"]) == (1, 1, 0)

    def test_report_text_is_not_truncated(self):
        long_report = "x" * 4_743  # the longest report in the corpus
        row = self._row(sources={"1.2.3": long_report})
        assert row["report"] == long_report

    def test_mask_reasons_render_as_text_not_a_tuple(self):
        row = self._row(records=[_record(findings=[])])
        assert row["masked"] is True
        assert row["mask_reasons"] == "empty_findings"

    def test_an_unparsed_record_still_produces_a_row(self):
        # A failed extraction is exactly what a reviewer needs to see.
        rows = records_table(
            [_record(_parsed=False, _valid=False, findings=[],
                     _schema_warnings=["response was not parseable JSON"])],
            SOURCES,
        )
        row = dict(zip(RECORDS_COLUMNS, rows[0]))
        assert row["parsed"] is False
        assert row["mask_reasons"] == "extraction_failed"
        assert row["warnings"] == "response was not parseable JSON"

    def test_polarity_column_disagrees_with_assertion_where_it_should(self):
        rows = findings_table([_record()], SOURCES)
        by_col = [dict(zip(FINDINGS_COLUMNS, r)) for r in rows]
        tear, acl = by_col[0], by_col[1]
        assert (tear["assertion"], tear["polarity"]) == ("affirmed", "positive")
        # Same assertion, opposite polarity -- the whole reason for the column.
        assert (acl["assertion"], acl["polarity"]) == ("affirmed", "negative")

    def test_non_dict_findings_are_skipped_not_crashed(self):
        rows = findings_table([_record(findings=[{"span": "a"}, "junk", None])], SOURCES)
        assert len(rows) == 1
