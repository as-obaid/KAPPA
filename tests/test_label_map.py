"""A cluster is not a label, and an unknown label is not a zero.

Two silent failures live here and each one produces a plausible wrong number.

The first is the missing finding-type gate. Five of the twelve labels name a
*finding type*, not a site: a medial femoral condyle with an osteophyte is a
Medial OA and the same condyle with marrow oedema is a Contusion, and the
anatomy field reads "medial femoral condyle" in both cases. Mapping cluster to
label without a gate invents those five wholesale, so every gate is tested from
both sides -- fires when the description names the type, silent when it does
not.

The second is the confident negative. An *ambiguous* cluster-label pair is
useful in one direction only: "no synovial thickening or effusion" rules
Synovitis out, while "joint effusion" cannot confirm it, because the positive
statement is about fluid. Reading both as positive is what fabricates ranks
under macro ROC-AUC, where every label is worth exactly 1/12. So the fan-out is
tested for the asymmetry, not merely for firing.

``StudyLabels.values`` is asserted to be twelve floats with no NaN even where
``usable`` is False, because a sentinel float survives every dtype cast and
arrives at ``BCEWithLogitsLoss``. The field is called ``usable``, not ``mask``:
it is True where the label may enter the loss.
"""

from __future__ import annotations

import math

import pytest

from kappa.constants import ID_COLUMN, LABELS, N_LABELS
from kappa.labels.anatomy_map import CLUSTER_NAMES
from kappa.labels.label_map import (
    AMBIGUOUS,
    CLUSTER_LABELS,
    DIRECT,
    GATE_ALONE,
    GATES,
    MASK_SUFFIX,
    OUTSIDE_KNOWLEDGE,
    TABLE_COLUMNS,
    UNMAPPED_CLUSTERS,
    UNMENTIONED_BY_LABEL,
    Agreement,
    StudyLabels,
    derive_studies,
    derive_study,
    mask_column,
    score_derivation,
)


def _finding(**over):
    """One extracted finding. ``description`` doubles as the span by default."""
    base = {
        "anatomy": "medial meniscus",
        "description": "tear of the medial meniscus",
        "assertion": "affirmed",
        "severity_wording": None,
    }
    base.update(over)
    base.setdefault("span", base["description"])
    return base


def _record(*findings, **over):
    """One extraction record, with the flags `report_mask` reads set to sane values.

    Anatomy strings here must not name a side: ``report_mask`` reads a
    left-and-right pair inside the anatomy fields as a two-knee report, which
    would flag the record for a reason these tests are not about.
    """
    base = {
        "_study_uid": "1.2.826.0.1.3680043.8.498.0",
        "_parsed": True,
        "_valid": True,
        "language": "English",
        "knee_side": "left",
        "findings": list(findings),
        "notes": None,
    }
    base.update(over)
    return base


def _known(study: StudyLabels) -> dict[str, float]:
    """Only the labels this study can actually supervise, with their values."""
    return {label: study.values[label] for label in LABELS if study.usable[label]}


def _labels(uid: str, values: dict[str, float], usable: dict[str, bool]) -> StudyLabels:
    """A StudyLabels with every label filled, for the scoring tests."""
    return StudyLabels(
        study_uid=uid,
        values={label: values.get(label, 0.0) for label in LABELS},
        usable={label: usable.get(label, False) for label in LABELS},
        mask_reasons=(),
    )


class TestStudyLabelsShape:
    def test_values_is_always_twelve_floats(self):
        study = derive_study(_record(_finding()))
        assert list(study.values) == list(LABELS)
        assert len(study.values) == N_LABELS
        assert all(isinstance(v, float) for v in study.values.values())

    def test_no_nan_ever_reaches_values_even_where_the_label_is_unusable(self):
        # This is the assertion that stops NaN reaching the loss. 98.7% of raw
        # label cells are NaN, and a NaN target silently destroys training
        # rather than failing.
        study = derive_study(
            _record(_finding(description="tear of the cruciate ligaments",
                             anatomy="cruciate ligaments")),
            unmentioned=None,
        )
        assert study.n_usable < N_LABELS, "expected at least one unusable label here"
        assert not any(math.isnan(v) for v in study.values.values())
        assert all(v in (0.0, 1.0) for v in study.values.values())

    def test_the_field_is_named_usable_not_mask(self):
        # `usable=True` means "may enter the loss". A field named `mask` inverts
        # that reading, and the inversion is invisible until a run trains on
        # exactly the rows it should have skipped.
        study = derive_study(_record(_finding()))
        assert study.usable["Medial Meniscus"] is True
        assert study.n_usable == sum(study.usable.values())

    def test_as_row_matches_the_pinned_table_columns(self):
        row = derive_study(_record(_finding())).as_row()
        assert list(row) == list(TABLE_COLUMNS)
        assert row[ID_COLUMN] == "1.2.826.0.1.3680043.8.498.0"

    def test_the_table_carries_a_value_and_a_mask_column_per_label(self):
        # Never encode "unknown" as a magic float: value and mask are separate
        # columns so a sentinel cannot survive a dtype cast into the loss.
        assert TABLE_COLUMNS[0] == ID_COLUMN
        assert len(TABLE_COLUMNS) == 1 + 2 * N_LABELS
        for label in LABELS:
            assert label in TABLE_COLUMNS
            assert mask_column(label) == f"{label}{MASK_SUFFIX}"
            assert mask_column(label) in TABLE_COLUMNS

    def test_the_apostrophe_and_space_bearing_labels_survive_column_naming(self):
        # `Baker's` has an apostrophe and five labels contain a space, which is
        # why the header is never built by splitting on punctuation.
        assert mask_column("Baker's") == "Baker's__mask"
        assert mask_column("Medial Meniscus") == "Medial Meniscus__mask"
        assert "Baker's__mask" in TABLE_COLUMNS

    def test_a_record_without_a_study_uid_still_derives(self):
        assert derive_study({"findings": []}).study_uid is None

    def test_derive_studies_preserves_input_order(self):
        records = [_record(_finding(), _study_uid=f"u{i}") for i in range(3)]
        assert [s.study_uid for s in derive_studies(records)] == ["u0", "u1", "u2"]


class TestGates:
    def test_an_oa_gate_is_required_before_a_bone_site_sets_an_oa_label(self):
        osteophyte = derive_study(
            _record(_finding(anatomy="medial femoral condyle",
                             description="osteophytes and joint space narrowing")),
            unmentioned=None,
        )
        assert _known(osteophyte) == {"Medial OA": 1.0}

    def test_the_same_site_with_a_different_finding_type_sets_a_different_label(self):
        # Identical anatomy, different description. Without the gate both of
        # these become Medial OA *and* Contusion.
        oedema = derive_study(
            _record(_finding(anatomy="medial femoral condyle", description="marrow oedema")),
            unmentioned=None,
        )
        assert _known(oedema) == {"Contusion": 1.0}

    @pytest.mark.parametrize(
        ("label", "description"),
        [
            ("Effusion", "moderate joint effusion"),
            ("Synovitis", "synovial hypertrophy"),
            ("Baker's", "popliteal cyst"),
            ("Fracture", "undisplaced fracture"),
            ("Contusion", "bone marrow oedema"),
            ("Medial OA", "gonarthrosis"),
            ("Medial OA", "chondromalacia"),
            ("Medial OA", "cartilage thinning"),
            ("Medial OA", "joint space narrowing"),
        ],
    )
    def test_each_gate_matches_the_phrase_family_it_is_for(self, label, description):
        assert GATES[label].search(description)

    def test_the_three_oa_labels_share_one_gate(self):
        # They are the same judgement about which phrases count as
        # osteoarthritis; three copies would drift apart.
        assert GATES["Medial OA"].pattern == GATES["Lateral OA"].pattern
        assert GATES["Medial OA"].pattern == GATES["PF OA"].pattern

    @pytest.mark.parametrize("label", ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"])
    def test_the_four_structure_labels_take_no_gate(self, label):
        # They are structures, not finding types: any positive finding on the
        # structure is evidence for the label, and there is no separate "does
        # this count" question to ask.
        assert label not in GATES

    def test_a_structure_label_fires_on_any_positive_finding(self):
        study = derive_study(
            _record(_finding(anatomy="anterior cruciate ligament",
                            description="complete rupture")),
            unmentioned=None,
        )
        assert _known(study) == {"ACL": 1.0}

    def test_a_gated_label_stays_unknown_when_the_description_does_not_name_the_type(self):
        study = derive_study(
            _record(_finding(anatomy="medial femoral condyle",
                            description="subchondral sclerosis of the medial femoral condyle")),
            unmentioned=None,
        )
        assert "Medial OA" not in _known(study)


class TestGateAlone:
    def test_exactly_four_labels_are_decided_by_the_gate_alone(self):
        assert GATE_ALONE == frozenset({"Effusion", "Synovitis", "Baker's", "Fracture"})

    def test_a_gate_alone_label_ignores_the_anatomy_field(self):
        # 129 findings name a Baker's or popliteal cyst in `description`
        # against only 34 whose anatomy lands in the Baker's cluster, so
        # requiring both would discard three quarters of the evidence.
        study = derive_study(
            _record(_finding(anatomy="medial meniscus",
                            description="Baker's cyst in the popliteal fossa")),
            unmentioned=None,
        )
        assert study.values["Baker's"] == 1.0
        assert study.usable["Baker's"]

    def test_contusion_is_deliberately_not_gate_alone(self):
        # Its gate has to match "bone marrow oedema", which appears in ~76% of
        # reports and is mostly degenerative subchondral oedema rather than
        # trauma. Gate-only would over-call the label massively, so a bone site
        # is required as well.
        assert "Contusion" not in GATE_ALONE
        assert GATES["Contusion"].search("bone marrow oedema")
        study = derive_study(
            _record(_finding(anatomy="medial meniscus", description="bone marrow oedema")),
            unmentioned=None,
        )
        assert "Contusion" not in _known(study), (
            "marrow oedema on a non-bone site set Contusion: Contusion has been "
            "made gate-only, which over-calls it on ~76% of reports"
        )

    def test_contusion_still_fires_when_a_bone_site_agrees(self):
        study = derive_study(
            _record(_finding(anatomy="medial femoral condyle",
                            description="bone marrow oedema")),
            unmentioned=None,
        )
        assert _known(study) == {"Contusion": 1.0}

    def test_a_gate_alone_label_is_negated_by_a_negative_statement(self):
        study = derive_study(
            _record(_finding(anatomy="popliteal fossa",
                            description="no popliteal cyst")),
            unmentioned=None,
        )
        assert study.values["Baker's"] == 0.0
        assert study.usable["Baker's"]


class TestPolarityConditionedFanOut:
    def test_a_negative_bundle_statement_negates_every_member(self):
        # "the cruciate ligaments are intact" is one statement about two
        # structures, and it rules both out.
        study = derive_study(
            _record(_finding(anatomy="cruciate ligaments",
                            description="the cruciate ligaments are intact")),
            unmentioned=None,
        )
        assert _known(study) == {"ACL": 0.0}

    def test_a_positive_bundle_statement_is_masked_not_attributed(self):
        # "tear of the cruciate ligaments" cannot be attributed to the ACL, so
        # the honest answer is unknown. Reading it as positive is what
        # fabricates a rank.
        study = derive_study(
            _record(_finding(anatomy="cruciate ligaments",
                            description="tear of the cruciate ligaments")),
            unmentioned=None,
        )
        assert _known(study) == {}
        assert study.usable["ACL"] is False
        assert study.values["ACL"] == 0.0

    def test_the_asymmetry_is_the_point(self):
        negative = derive_study(
            _record(_finding(anatomy="joint space",
                            description="no joint effusion or synovial thickening")),
            unmentioned=None,
        )
        positive = derive_study(
            _record(_finding(anatomy="joint space", description="moderate joint effusion")),
            unmentioned=None,
        )
        # Effusion is DIRECT for this cluster, Synovitis only AMBIGUOUS.
        assert _known(negative) == {"Effusion": 0.0, "Synovitis": 0.0}
        assert _known(positive) == {"Effusion": 1.0}

    def test_the_fan_out_still_respects_the_members_own_gate(self):
        # An ambiguous pair is not a licence to ignore the finding type: the
        # OA labels are only reached when the description names an OA phrase.
        study = derive_study(
            _record(_finding(anatomy="joint space",
                            description="no joint space narrowing and no effusion")),
            unmentioned=None,
        )
        assert _known(study) == {
            "Medial OA": 0.0, "Lateral OA": 0.0, "PF OA": 0.0, "Effusion": 0.0
        }

    def test_switching_the_fan_out_off_contributes_nothing_at_all(self):
        record = _record(_finding(anatomy="cruciate ligaments",
                                 description="the cruciate ligaments are intact"))
        assert _known(derive_study(record, unmentioned=None)) == {"ACL": 0.0}
        assert _known(derive_study(record, fan_out_negatives=False, unmentioned=None)) == {}

    def test_direct_evidence_is_sticky_against_an_ambiguous_cluster(self):
        # One finding, two clusters: COMPARTMENT_PATELLOFEMORAL gives PF OA
        # directly and COMPARTMENT_UNSPECIFIED calls it ambiguous. The direct
        # reading must win, or the positive gets masked.
        study = derive_study(
            _record(_finding(anatomy="patellofemoral compartment, femorotibial compartment",
                            description="moderate arthrosis")),
            unmentioned=None,
        )
        assert _known(study) == {"PF OA": 1.0}

    def test_a_positive_anywhere_beats_a_negative_elsewhere(self):
        # A report that rules a tear out in the body and reports one in the
        # impression is describing a tear.
        study = derive_study(
            _record(
                _finding(anatomy="medial meniscus", description="medial meniscus is intact"),
                _finding(anatomy="medial meniscus", description="tear of the medial meniscus"),
            ),
            unmentioned=None,
        )
        assert study.values["Medial Meniscus"] == 1.0

    def test_an_uncertain_assertion_masks_rather_than_guessing(self):
        study = derive_study(
            _record(_finding(anatomy="medial meniscus",
                            description="tear of the medial meniscus",
                            assertion="uncertain")),
            unmentioned=None,
        )
        assert _known(study) == {}


class TestThePatellofemoralCompartmentFixReachesALabel:
    def test_a_positive_finding_on_it_now_sets_pf_oa_instead_of_being_masked(self):
        # This is what the task-31 fix bought: the cluster used to be
        # COMPARTMENT_UNSPECIFIED, which is ambiguous for all three OA labels,
        # so under polarity-conditioned fan-out the positive was masked.
        fixed = derive_study(
            _record(_finding(anatomy="patellofemoral compartment",
                            description="moderate arthrosis")),
            unmentioned=None,
        )
        assert _known(fixed) == {"PF OA": 1.0}

    def test_the_unspecified_compartment_is_still_masked_on_a_positive(self):
        # The contrast that shows the fix is about naming one compartment, not
        # about weakening the ambiguity rule.
        study = derive_study(
            _record(_finding(anatomy="femorotibial compartment",
                            description="moderate arthrosis")),
            unmentioned=None,
        )
        assert _known(study) == {}

    def test_the_new_cluster_is_direct_for_pf_oa_only(self):
        assert CLUSTER_LABELS["COMPARTMENT_PATELLOFEMORAL"] == {"PF OA": DIRECT}

    def test_it_buys_coverage_not_accuracy(self):
        # Stated because the temptation is to read the fix as an improvement.
        # On the anchors it lifted per-label coverage (PF OA 52->53, Medial OA
        # 50->51, Lateral OA 49->50) and the one newly covered PF OA study is a
        # false positive, so the macro balanced accuracy went 0.765 -> 0.764.
        # What the fix removes is a mask, which is what this asserts.
        masked_before = derive_study(
            _record(_finding(anatomy="femorotibial compartment", description="arthrosis")),
            unmentioned=None,
        )
        usable_now = derive_study(
            _record(_finding(anatomy="patellofemoral compartment", description="arthrosis")),
            unmentioned=None,
        )
        assert masked_before.usable["PF OA"] is False
        assert usable_now.usable["PF OA"] is True


class TestUnmentioned:
    def test_the_per_label_mapping_works_and_not_only_the_scalar(self):
        # Measured P(gold positive | report silent): Synovitis 0.34 (and silent
        # 83.7% of the time), PF OA 0.21, Baker's 0.03, Medial OA 0.00. A single
        # scalar treats "absent" and "nobody looked" alike.
        study = derive_study(
            _record(_finding(anatomy="medial meniscus",
                            description="tear of the medial meniscus")),
            unmentioned=UNMENTIONED_BY_LABEL,
        )
        assert study.usable["Synovitis"] is False
        assert study.usable["PF OA"] is False
        assert study.usable["Baker's"] is True and study.values["Baker's"] == 0.0
        assert study.usable["Medial OA"] is True and study.values["Medial OA"] == 0.0

    def test_the_measured_policy_masks_exactly_the_two_coin_flip_labels(self):
        assert [label for label, value in UNMENTIONED_BY_LABEL.items() if value is None] == [
            "Synovitis", "PF OA"
        ]

    def test_every_label_appears_in_the_policy_so_an_omission_is_visible(self):
        assert set(UNMENTIONED_BY_LABEL) == set(LABELS)

    def test_the_scalar_path_is_kept_because_the_quoted_score_used_it(self):
        # The measured macro balanced accuracy on the 58 anchors was produced
        # with unmentioned=0.0, and that number has to stay reproducible.
        study = derive_study(_record(_finding()))
        assert study.n_usable == N_LABELS
        assert study.values["Synovitis"] == 0.0

    def test_a_scalar_none_masks_every_unmentioned_label(self):
        study = derive_study(_record(_finding()), unmentioned=None)
        assert _known(study) == {"Medial Meniscus": 1.0}

    def test_a_partial_mapping_keeps_the_scalar_default_for_the_rest(self):
        study = derive_study(_record(_finding()), unmentioned={"Synovitis": None})
        assert study.usable["Synovitis"] is False
        assert study.usable["Baker's"] is True

    def test_a_misspelled_label_key_is_rejected_rather_than_ignored(self):
        # Silently ignoring an unknown key turns "Bakers" into a no-op, which
        # is indistinguishable from the override having been applied.
        with pytest.raises(ValueError, match="not competition labels"):
            derive_study(_record(_finding()), unmentioned={"Bakers": None})


class TestReportLevelMasking:
    def test_a_failed_extraction_voids_the_whole_row(self):
        # No parseable output means no evidence of any kind. Distinct from a
        # report that parsed and said little.
        study = derive_study(_record(_parsed=False, _valid=False))
        assert study.n_usable == 0
        assert study.mask_reasons == ("extraction_failed",)
        assert set(study.values.values()) == {0.0}
        assert not any(math.isnan(v) for v in study.values.values())

    def test_an_empty_findings_section_is_flagged_but_its_values_are_not_voided(self):
        # Deliberate and worth knowing: only `extraction_failed` voids the row,
        # because acting on the other reasons changes the derivation that was
        # measured. With unmentioned=0.0 an empty findings section therefore
        # becomes twelve usable zeros, and the reason is carried so a run can
        # count them rather than discover them.
        study = derive_study(_record())
        assert study.mask_reasons == ("empty_findings",)
        assert study.n_usable == N_LABELS
        assert set(study.values.values()) == {0.0}

    def test_the_same_record_under_the_honest_policy_is_all_unknown(self):
        study = derive_study(_record(), unmentioned=None)
        assert study.n_usable == 0
        assert study.mask_reasons == ("empty_findings",)

    def test_a_bilateral_record_is_flagged_and_carried(self):
        study = derive_study(
            _record(
                _finding(anatomy="right medial meniscus"),
                _finding(anatomy="left anterior cruciate ligament"),
            )
        )
        assert "bilateral_report" in study.mask_reasons

    def test_a_non_dict_finding_is_skipped_rather_than_crashing(self):
        study = derive_study(_record("not a finding", _finding()), unmentioned=None)
        assert _known(study) == {"Medial Meniscus": 1.0}


class TestTheTableIsComplete:
    def test_every_cluster_the_clusterer_can_emit_has_a_decision_or_is_listed(self):
        # `dict.get` makes an absent cluster and a decided-empty one identical,
        # so the gap has to be enumerated to be visible. This mirrors the
        # import-time guard, which fails the module rather than the run.
        assert CLUSTER_NAMES - set(CLUSTER_LABELS) - UNMAPPED_CLUSTERS == set()

    def test_the_unmapped_list_names_only_real_clusters(self):
        assert UNMAPPED_CLUSTERS <= CLUSTER_NAMES
        assert UNMAPPED_CLUSTERS.isdisjoint(set(CLUSTER_LABELS))

    def test_the_table_names_no_cluster_the_clusterer_cannot_emit(self):
        assert set(CLUSTER_LABELS) <= CLUSTER_NAMES

    def test_every_label_in_the_table_is_a_competition_label(self):
        # A misspelled label name would map a real cluster to a column that
        # does not exist, and contribute nothing, silently.
        for cluster, entries in CLUSTER_LABELS.items():
            assert set(entries) <= set(LABELS), cluster
            assert set(entries.values()) <= {DIRECT, AMBIGUOUS}, cluster

    def test_the_synthetic_cluster_is_decided_so_the_ablation_does_not_crash(self):
        # LIGAMENT_PATELLAR_UNFOLDED is reachable only with fold_patlig=False.
        assert CLUSTER_LABELS["LIGAMENT_PATELLAR_UNFOLDED"] == {}

    def test_the_residual_buckets_contribute_nothing(self):
        for cluster in CLUSTER_NAMES:
            if cluster.startswith("ZZ_"):
                assert CLUSTER_LABELS[cluster] == {}, cluster

    def test_every_outside_knowledge_pair_is_actually_in_the_table(self):
        # The set exists so an ablation can drop these pairs and measure what
        # the borrowed clinical knowledge is worth. A stale entry would drop
        # nothing and report a null result.
        for cluster, label in OUTSIDE_KNOWLEDGE:
            assert label in CLUSTER_LABELS[cluster], (cluster, label)

    def test_a_structure_no_competition_label_covers_maps_to_nothing(self):
        # PCL, LCL, the retinaculum and most tendons are real structures with
        # no label. A finding on them must contribute nothing rather than be
        # rounded to the nearest label.
        for cluster in ("PCL", "LCL", "PATELLAR_RETINACULUM", "TENDON_QUADRICEPS",
                        "MENISCOFEMORAL_LIGAMENT", "POSTEROLATERAL_CORNER"):
            assert CLUSTER_LABELS[cluster] == {}, cluster
        study = derive_study(
            _record(_finding(anatomy="posterior cruciate ligament",
                            description="complete rupture")),
            unmentioned=None,
        )
        assert _known(study) == {}


class TestScoreDerivation:
    """Every rate is per label, over the studies that label actually covers."""

    def _case(self):
        # Three studies. For ACL: one TP, one FN, and one study whose ACL was
        # masked. Ten of the twelve anchor rows look like this.
        derived = {
            "a": _labels("a", {"ACL": 1.0}, {"ACL": True}),
            "b": _labels("b", {"ACL": 0.0}, {"ACL": True}),
            "c": _labels("c", {"ACL": 0.0}, {"ACL": False}),
        }
        truth = {
            "a": dict.fromkeys(LABELS, 0) | {"ACL": 1},
            "b": dict.fromkeys(LABELS, 0) | {"ACL": 1},
            "c": dict.fromkeys(LABELS, 0) | {"ACL": 0},
        }
        return derived, truth

    def _row(self, score, label: str) -> Agreement:
        return next(row for row in score.per_label if row.label == label)

    def test_a_masked_label_is_not_counted_as_a_derived_negative(self):
        # Counting it would credit the derivation with a true negative it never
        # claimed, and inflate specificity on exactly the labels it covers worst.
        score = score_derivation(*self._case())
        acl = self._row(score, "ACL")
        assert (acl.tp, acl.fp, acl.fn, acl.tn) == (1, 0, 1, 0)
        assert acl.n_covered == 2
        assert acl.n_compared == 3

    def test_the_per_label_denominator_is_reported_alongside_the_rates(self):
        acl = self._row(score_derivation(*self._case()), "ACL")
        assert acl.coverage == pytest.approx(2 / 3)
        assert acl.sensitivity == pytest.approx(0.5)

    def test_prevalence_is_over_the_covered_studies_not_the_compared_ones(self):
        # The original driver printed gold prevalence over all 58 studies while
        # ten of the twelve rows covered fewer, which overstated the evidence
        # behind those rows. Both numbers exist here; the covered one is the
        # one the row is about.
        acl = self._row(score_derivation(*self._case()), "ACL")
        assert acl.n_gold_positive == 2
        assert acl.n_gold_positive / acl.n_covered == pytest.approx(1.0)
        assert acl.n_gold_positive / acl.n_compared == pytest.approx(2 / 3)

    def test_a_rate_with_no_denominator_is_nan_and_the_label_is_named(self):
        # NaN, never 0.0: a specificity of zero and a specificity that does not
        # exist are different claims, and averaging the second as zero drags the
        # macro down invisibly.
        score = score_derivation(*self._case())
        acl = self._row(score, "ACL")
        assert math.isnan(acl.specificity)
        assert math.isnan(acl.balanced_accuracy)
        assert "ACL" in score.skipped

    def test_the_macro_excludes_the_undefined_rows_it_names(self):
        derived = {
            "a": _labels("a", {"ACL": 1.0, "MCL": 1.0}, {"ACL": True, "MCL": True}),
            "b": _labels("b", {"ACL": 0.0, "MCL": 0.0}, {"ACL": True, "MCL": True}),
        }
        truth = {
            "a": dict.fromkeys(LABELS, 0) | {"ACL": 1, "MCL": 1},
            "b": dict.fromkeys(LABELS, 0) | {"ACL": 0, "MCL": 1},
        }
        score = score_derivation(derived, truth)
        assert self._row(score, "ACL").balanced_accuracy == pytest.approx(1.0)
        assert math.isnan(self._row(score, "MCL").balanced_accuracy)
        # The ten labels with no coverage at all are skipped too, so the macro
        # is over ACL alone here.
        assert score.macro_balanced_accuracy == pytest.approx(1.0)
        assert "MCL" in score.skipped and "ACL" not in score.skipped

    def test_only_studies_present_in_both_sides_are_compared(self):
        derived = {"a": _labels("a", {"ACL": 1.0}, {"ACL": True}),
                   "z": _labels("z", {"ACL": 1.0}, {"ACL": True})}
        truth = {"a": dict.fromkeys(LABELS, 0) | {"ACL": 1}}
        score = score_derivation(derived, truth)
        assert score.n_compared == 1
        assert self._row(score, "ACL").n_covered == 1

    def test_label_coverage_is_reported_over_all_twelve_columns(self):
        score = score_derivation(*self._case())
        # Only the ACL column is usable anywhere in this case: 2 usable cells
        # out of 3 studies x 12 labels.
        assert score.coverage == pytest.approx(2 / 36)

    def test_an_empty_comparison_does_not_divide_by_zero(self):
        score = score_derivation({}, {})
        assert score.n_compared == 0
        assert score.coverage == 0.0
        assert math.isnan(score.macro_balanced_accuracy)
        assert set(score.skipped) == set(LABELS)
