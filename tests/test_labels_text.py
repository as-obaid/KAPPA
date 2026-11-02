"""Normalisation and sampling invariants.

The sampling rule that matters most: the 58 annotated studies are the only
ground truth in the project, and a report shown to a model during design is no
longer available for honest validation. That exclusion is asserted in the code;
this checks the assertion actually fires.
"""

from __future__ import annotations

import pytest

from kappa.labels.normalise import content_hash, detect_script, normalise_report
from kappa.labels.sample import stratified_sample


def test_normalise_collapses_whitespace_but_keeps_content():
    assert normalise_report("a   b\t\tc") == "a b c"
    assert normalise_report("  lead\n\n\n\ntrail  ") == "lead\n\ntrail"


def test_deidentification_placeholders_are_preserved():
    """153 reports lead with [DATE]; stripping them would change token counts."""
    assert "[DATE]" in normalise_report("Exam: [DATE] knee")


def test_unicode_forms_that_render_alike_hash_alike():
    """The corpus mixes precomposed and combining accents; NFC unifies them."""
    precomposed, combining = "Técnica", "Técnica"
    assert precomposed != combining
    assert content_hash(normalise_report(precomposed)) == content_hash(
        normalise_report(combining)
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("МР находка: фрактура", "cyrillic"),
        ("ΤΕΧΝΙΚΗ Η εξέταση", "greek"),
        ("SAĞ DİZ MRG. Tetkik", "turkish"),
        ("Técnica: RMN de la rodilla", "latin"),
        ("Mediales Kompartiment", "latin"),
        ("CONSTATATIONS : Aucune", "latin"),
    ],
)
def test_script_detection(text, expected):
    assert detect_script(text) == expected


def test_turkish_test_does_not_fire_on_shared_diacritics():
    """o-umlaut, u-umlaut and c-cedilla are shared with Spanish/French/German,
    so keying on them would misclassify a large slice of the corpus."""
    for text in ("Sección de rodilla", "Über das Kniegelenk", "Le ménisque, ça va"):
        assert detect_script(text) == "latin", text


def test_sampling_never_touches_an_anchor():
    rows = [
        {"study_uid": f"anchor-{i}", "report": f"anchor text {i}",
         "script": "latin", "hash": f"a{i}", "is_anchor": True}
        for i in range(5)
    ] + [
        {"study_uid": f"plain-{i}", "report": f"plain text {i}",
         "script": "latin", "hash": f"p{i}", "is_anchor": False}
        for i in range(10)
    ]
    picked = stratified_sample(rows, per_script={"latin": 8}, seed=0)
    assert len(picked) == 8
    assert all(not r["is_anchor"] for r in picked)
    assert not {r["study_uid"] for r in picked} & {f"anchor-{i}" for i in range(5)}


def test_sampling_dedupes_by_report_text():
    """49 texts repeat across 183 rows; the largest group is 37 identical
    Turkish all-normal templates. Sampling one twice learns nothing."""
    rows = [
        {"study_uid": f"s{i}", "report": "identical", "script": "latin",
         "hash": "same", "is_anchor": False}
        for i in range(20)
    ]
    assert len(stratified_sample(rows, per_script={"latin": 10}, seed=0)) == 1


def test_sampling_is_deterministic():
    rows = [
        {"study_uid": f"s{i}", "report": f"t{i}", "script": "latin",
         "hash": f"h{i}", "is_anchor": False}
        for i in range(50)
    ]
    a = [r["study_uid"] for r in stratified_sample(rows, {"latin": 10}, seed=7)]
    b = [r["study_uid"] for r in stratified_sample(rows, {"latin": 10}, seed=7)]
    assert a == b


class TestAnchorsOnly:
    """Inverting rule 1 is deliberate, so it is tested as deliberate."""

    def _rows(self):
        return [
            {"study_uid": "a", "script": "latin", "hash": "h1", "is_anchor": False,
             "report": "x"},
            {"study_uid": "b", "script": "latin", "hash": "h2", "is_anchor": True,
             "report": "y"},
            {"study_uid": "c", "script": "greek", "hash": "h3", "is_anchor": True,
             "report": "z"},
        ]

    def test_default_still_excludes_every_anchor(self):
        from kappa.labels.sample import stratified_sample
        picked = stratified_sample(self._rows(), per_script={"latin": 1, "greek": 1})
        assert [r["study_uid"] for r in picked] == ["a"]

    def test_anchors_only_returns_exactly_the_anchors(self):
        from kappa.labels.sample import stratified_sample
        picked = stratified_sample(self._rows(), anchors_only=True)
        assert [r["study_uid"] for r in picked] == ["b", "c"]

    def test_anchors_only_ignores_per_script_caps(self):
        # All of them, because a per-label comparison on 58 studies is already
        # thin and a cap would make several of the 12 labels unreadable.
        from kappa.labels.sample import stratified_sample
        picked = stratified_sample(self._rows(), per_script={"latin": 0, "greek": 0},
                                   anchors_only=True)
        assert len(picked) == 2

    def test_anchors_only_fails_loudly_when_there_are_none(self):
        import pytest as _pytest
        from kappa.labels.sample import stratified_sample
        rows = [r for r in self._rows() if not r["is_anchor"]]
        with _pytest.raises(AssertionError):
            stratified_sample(rows, anchors_only=True)
