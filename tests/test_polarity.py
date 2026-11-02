"""Polarity must come from both fields, and the cheap shortcuts must stay dead.

Every case below is modelled on a description string from the 300-report survey,
reworded where needed but keeping the cue words and their order, which are all the
rule reads. The regressions matter more than the happy path: three of them
(`normal`, `nodular`, `non-displaced`) came from an open-ended `no` cue that
silently negated a fracture and a patellar tilt, and one (`free fluid`) from
treating `free` as a negation.
"""

from __future__ import annotations

import pytest

from kappa.labels.polarity import MASK, NEGATIVE, POSITIVE, polarity, signals


class TestAssertionBranches:
    def test_negated_is_negative_whatever_the_description_says(self):
        # The mirror trap: 117 negated findings name an abnormality because the
        # negation lives in the span, not the description.
        assert polarity("negated", "lesion of the anterior cruciate ligament") == NEGATIVE
        assert polarity("negated", "joint effusion") == NEGATIVE

    def test_uncertain_is_masked_never_guessed(self):
        assert polarity("uncertain", "partial tear of the medial meniscus") == MASK
        assert polarity("uncertain", "") == MASK

    def test_affirmed_alone_decides_nothing(self):
        # Same assertion, opposite polarity -- this is the whole point.
        assert polarity("affirmed", "partial tear of the medial meniscus") == POSITIVE
        assert polarity("affirmed", "lateral meniscus is intact") == NEGATIVE


class TestNormalitySweep:
    """`affirmed` on a normality statement is the language-correlated bug."""

    @pytest.mark.parametrize(
        "description",
        [
            "normal",
            "intact",
            "lateral meniscus is intact",
            "medial collateral ligament is normal",
            "articular cartilage is within normal limits",
            "patellar retinacula, collateral ligaments - without peculiarities",
            "menisci morphology and signal preserved",
            "retinacula are unremarkable",
            "bones forming the left knee joint have normal shape",
            "extensor mechanism is preserved",
            "Hoffa's fat pad has homogeneous structure and signal intensity",
            "cruciate and collateral ligaments are continuous",
            "alignment is anatomic",
            "patella is centered",
        ],
    )
    def test_normality_statement_is_negative(self, description):
        assert polarity("affirmed", description) == NEGATIVE


class TestNegationScope:
    """Order decides. A negation only governs what comes after it."""

    @pytest.mark.parametrize(
        "description",
        [
            "myxoid degeneration without tear",
            "simple degeneration without tear",
            "degenerative change without tear",
            "thinning of cartilage over the weightbearing condyle, without a focal defect",
            "focus of chondromalacia without erosions in the central and lateral part",
            "mild bursitis of Baker's bursa, without a clearly formed cyst",
            "Baker's cyst beside the medial gastrocnemius head, without signs of rupture",
            "sequelae of Osgood-Schlatter with partial and deep bone fragment, without bone edema",
        ],
    )
    def test_abnormality_before_negation_is_positive(self, description):
        assert polarity("affirmed", description) == POSITIVE

    @pytest.mark.parametrize(
        "description",
        [
            "visualized tendons are intact without tendinosis or tear",
            "menisci with preserved morphology and signal, no signs of tear",
            "bones have normal shape and signal intensity, no fracture",
            "menisci have normal signal characteristics and preserved integrity, no evidence of rupture",
            "lateral meniscus normally configured, without evidence of tearing",
            "preserved femorotibial cartilage without deep-reaching chondropathy",
        ],
    )
    def test_negation_before_abnormality_is_negative(self, description):
        assert polarity("affirmed", description) == NEGATIVE


class TestNegationCueBoundaries:
    """An open-ended `no` matches `normal`, `nodular` and `non-displaced`."""

    def test_non_displaced_fracture_is_still_a_fracture(self):
        assert polarity("affirmed", "non-displaced fracture of the medial tibial plateau") == POSITIVE
        assert (
            polarity("affirmed", "non-displaced incomplete fracture at the fibular head")
            == POSITIVE
        )

    def test_nodular_is_not_a_negation(self):
        assert polarity("affirmed", "ossified nodular lesion belonging to a loose body") == POSITIVE

    def test_nonspecific_is_not_a_negation(self):
        assert polarity("affirmed", "nonspecific scattered edema of prepatellar subcutis") == POSITIVE

    def test_normal_position_with_a_tilt_is_positive(self):
        assert polarity("affirmed", "patella in normal position with a slight lateral patellar tilt") == POSITIVE

    def test_free_fluid_is_a_finding_not_a_negation(self):
        assert polarity("affirmed", "small amount of free fluid in the suprapatellar bursa") == POSITIVE

    def test_free_of_still_negates(self):
        assert polarity("affirmed", "free of tear") == NEGATIVE


class TestCompoundPathologyStems:
    """Greek and Latin compounds put the pathology at the end of the word."""

    @pytest.mark.parametrize(
        "description",
        [
            "medial gonarthrosis",
            "advanced gonarthrosis",
            "grade I meniscopathy",
            "patellofemoral osteoarthritis",
            "small osteochondral loose body with a diameter of 8mm",
            "subchondral medullary fibrocystic and fibrovascular changes",
            "signs of external patellar hyperpressure syndrome",
        ],
    )
    def test_suffix_stem_is_found(self, description):
        assert polarity("affirmed", description) == POSITIVE


class TestAbnormalityWinsOverCooccurringNormality:
    """Reports say what is preserved alongside what is damaged."""

    @pytest.mark.parametrize(
        "description",
        [
            "loss of integrity secondary to complete rupture",
            "intact, but periligamentous edema, probably reactive to the medial meniscal pathology",
            "anterior cruciate ligament with increased signal and maintained continuity, suggesting degeneration",
            "posterior cruciate ligament is mildly degeneratively changed, preserved continuity",
            "femorotibial cartilage thickness preserved with slight signal alteration and partial ulcers",
        ],
    )
    def test_positive(self, description):
        assert polarity("affirmed", description) == POSITIVE


class TestMask:
    """`mask` is a real state and must not collapse to `negative`."""

    @pytest.mark.parametrize(
        "description",
        [
            "TT-TG distance is 15 mm",
            "Insall-Salvati index is 1.2",
            "patella morphology Wiberg II",
            "presence of a fabella",
            "meniscal flounce",
            "linear signal intensity",
        ],
    )
    def test_names_neither_abnormality_nor_normality(self, description):
        assert polarity("affirmed", description) == MASK

    @pytest.mark.parametrize("description", [None, "", "   "])
    def test_missing_description_is_masked_not_assumed(self, description):
        assert polarity("affirmed", description) == MASK


class TestSignals:
    def test_offsets_are_returned_not_booleans(self):
        # The offsets are the mechanism; asserting on them keeps the scope
        # rule testable independently of the polarity decision.
        s = signals("myxoid degeneration without tear")
        assert s.abnormal is not None and s.negation is not None
        assert s.abnormal < s.negation

    def test_no_cues_at_all(self):
        s = signals("TT-TG distance is 15 mm")
        assert s == (None, None, None)

    def test_case_and_punctuation_are_irrelevant(self):
        assert polarity("affirmed", "PARTIAL TEAR!!") == POSITIVE
        assert polarity("affirmed", "Normal.") == NEGATIVE
