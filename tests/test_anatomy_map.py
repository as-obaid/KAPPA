"""One anatomy string must land in the cluster that decides the right label.

The failure this guards is not a crash, it is a plausible wrong side. ``medial
compartment`` and ``lateral compartment`` differ by one token and feed two
different competition labels, ``outer compartment`` is the lateral one written
as an English synonym, and the corpus misspells ``patelofemoral``. Every one of
those flattens silently under a keyword scan, so the tests below are written
against the distinctions rather than against the code's shape: they pin what
each stage of the pipeline must preserve, and the cascade order that decides
which structure wins when a bag names two.

The four fold flags get their own class. They are deliberate ablation switches
over judgement calls, not corpus facts, so each one has to keep working in both
positions -- ``LIGAMENT_PATELLAR_UNFOLDED`` exists for no other reason than to
serve ``fold_patlig=False``, and a cluster nothing can emit is dead weight the
next reader will delete.

The corpus differential lives in ``tests/test_anatomy_map_equivalence.py``.
This file is the unit-level companion to it.
"""

from __future__ import annotations

import pytest

from kappa.labels.anatomy_map import (
    ABBREVIATIONS,
    CLUSTER_NAMES,
    INDUCED_VOCABULARY,
    canonicalise,
    cluster_of_bag,
    clusters_of,
    preprocess,
    singularise,
    split_components,
    tokens,
)
from kappa.labels.normalise import content_hash, fold_lookalikes


def _bag(*words: str) -> frozenset[str]:
    """A canonicalised bag, built directly so a cascade test states its input."""
    return frozenset(words)


class TestTokens:
    def test_splits_on_punctuation_and_lowercases(self):
        assert tokens("Medial Meniscus, posterior horn") == [
            "medial", "meniscus", "posterior", "horn"
        ]

    def test_nfkc_folds_full_width_forms(self):
        # The corpus contains full-width and ligature forms. Without NFKC they
        # survive the non-word strip as separate, unmatchable tokens.
        assert tokens("ＡＣＬ") == ["acl"]

    def test_keeps_digits(self):
        assert tokens("grade 3 chondromalacia") == ["grade", "3", "chondromalacia"]


class TestPreprocess:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ACL", "anterior cruciate ligament"),
            ("PCL", "posterior cruciate ligament"),
            ("MCL", "medial collateral ligament"),
            ("LCL", "lateral collateral ligament"),
            ("FCL", "fibular collateral ligament"),
            ("MPFL", "medial patellofemoral ligament"),
            ("ITB", "iliotibial band"),
            ("IT band", "iliotibial band"),
            ("PF joint", "patellofemoral joint"),
            ("FT compartment", "femorotibial compartment"),
        ],
    )
    def test_expands_abbreviations(self, raw, expected):
        assert preprocess(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("VKB", "anterior cruciate ligament"),   # Turkish for the ACL
            ("HKB", "posterior cruciate ligament"),  # Turkish for the PCL
            ("LCA", "anterior cruciate ligament"),   # Spanish/French for the ACL
        ],
    )
    def test_expands_the_non_english_abbreviations(self, raw, expected):
        # Reports are multilingual, so an English-only abbreviation list drops
        # a whole cluster. These three were harvested from reports that glossed
        # them inline; a hand-written list would not have contained them.
        assert preprocess(raw) == expected

    def test_a_parenthetical_gloss_is_dropped_as_redundant(self):
        assert preprocess("anterior cruciate ligament (ACL)") == "anterior cruciate ligament"

    def test_a_wholly_parenthesised_field_keeps_its_content(self):
        # Dropping the parenthetical would empty the string and send a perfectly
        # identifiable structure to the unmappable bucket.
        assert preprocess("(ACL)") == "anterior cruciate ligament"

    def test_tt_tg_is_expanded_before_the_parentheses_are_stripped(self):
        # TT-TG is a measurement, and its expansion is what routes it to the
        # non-anatomical bucket. Expanding it after the strip would lose it.
        assert preprocess("TT-TG (measured)") == "tibial tubercle trochlear groove"
        assert clusters_of("TT-TG distance") == frozenset({"ZZ_NONANAT_measurement"})

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("patelofemoral compartment", "patellofemoral compartment"),
            ("femoro-tibial compartment", "femorotibial compartment"),
            ("femoro tibial compartment", "femorotibial compartment"),
            ("trochea", "trochlea"),
            ("semimenbranosus tendon", "semimembranosus tendon"),
            ("poplitiofibular ligament", "popliteofibular ligament"),
            ("quadricep tendon", "quadriceps tendon"),
            ("gastronomes", "gastrocnemius"),
            ("eminance", "eminence"),
        ],
    )
    def test_repairs_the_misspellings_the_corpus_actually_contains(self, raw, expected):
        assert preprocess(raw) == expected

    def test_hyphen_and_space_forms_of_a_compartment_agree(self):
        assert preprocess("patello-femoral") == preprocess("patello femoral")

    def test_drops_the_possessive_and_expands_the_ampersand(self):
        assert preprocess("Baker's cyst") == "baker cyst"
        assert preprocess("medial & lateral menisci") == "medial and lateral menisci"

    def test_strips_trailing_sentence_punctuation_and_collapses_whitespace(self):
        assert preprocess("  medial   meniscus.  ") == "medial meniscus"


class TestSingularise:
    @pytest.mark.parametrize(
        ("plural", "singular"),
        [
            ("menisci", "meniscus"),
            ("retinacula", "retinaculum"),
            ("retinaculums", "retinaculum"),
            ("plicae", "plica"),
            ("condyles", "condyle"),
            ("cartilages", "cartilage"),
            ("tendons", "tendon"),
            ("compartments", "compartment"),
        ],
    )
    def test_folds_plurals(self, plural, singular):
        assert singularise(plural) == singular

    @pytest.mark.parametrize("word", ["pes", "quadriceps", "biceps", "femoris", "osseous",
                                      "anserinus", "synchondroses", "hoffa", "baker"])
    def test_protects_singulars_that_merely_end_in_s(self, word):
        # Stripping the `s` silently renames the structure: `pes anserinus`
        # would become `pe anserinu` and match nothing at all.
        assert singularise(word) == word

    def test_only_strips_a_suffix_when_the_singular_is_a_corpus_word(self):
        # The membership test is the whole safety net. `bodies` has no singular
        # in the corpus vocabulary, so it stays put rather than becoming a
        # token the cascade has never seen.
        assert "bodie" not in INDUCED_VOCABULARY
        assert singularise("bodies") == "bodies"

    def test_leaves_a_short_word_alone(self):
        assert singularise("s") == "s"


class TestSplitComponents:
    def test_a_shared_head_noun_before_the_sites_is_distributed(self):
        # `cartilages of the lateral and the medial femorotibial compartments` is
        # one finding and two sites. Treating it as a unit loses a label.
        assert clusters_of("cartilages of the lateral and the medial femorotibial compartments") == (
            frozenset({"CARTILAGE_MEDIAL_TIBIOFEMORAL", "CARTILAGE_LATERAL_TIBIOFEMORAL"})
        )

    def test_a_shared_head_noun_after_the_sites_is_inherited_forward(self):
        # English puts the shared head last, so the donor search goes forward
        # first: in `medial and lateral meniscus` only the second fragment has
        # the head noun.
        assert split_components("medial and lateral meniscus") == [
            ["medial", "meniscus"], ["lateral", "meniscus"]
        ]

    def test_an_inherited_head_does_not_drag_the_donors_side_along(self):
        # The cheap implementation -- inherit every token the donor has --
        # collapses both fragments to MENISCUS_BOTH, which is the wrong label on
        # both sides.
        assert clusters_of("medial and lateral meniscus") == frozenset(
            {"MENISCUS_MEDIAL", "MENISCUS_LATERAL"}
        )

    def test_a_fragment_with_its_own_head_noun_borrows_nothing(self):
        assert split_components("medial meniscus, anterior cruciate ligament") == [
            ["medial", "meniscus"], ["anterior", "cruciate", "ligament"]
        ]

    def test_three_sites_in_one_string_become_three_clusters(self):
        assert clusters_of(
            "lateral compartment, lateral collateral ligament, cruciate ligaments"
        ) == frozenset({"COMPARTMENT_LATERAL", "LCL", "CRUCIATE_UNSPECIFIED"})

    def test_a_single_site_stays_one_bag(self):
        assert split_components("medial collateral ligament") == [
            ["medial", "collateral", "ligament"]
        ]

    def test_a_lone_shared_head_is_not_split_off_from_its_only_site(self):
        # One part after the `of`, so there is nothing to distribute the head
        # over and the phrase must stay intact.
        assert len(split_components(preprocess("cartilage of the medial compartment"))) == 1
        assert clusters_of("cartilage of the medial compartment") == frozenset(
            {"CARTILAGE_MEDIAL_TIBIOFEMORAL"}
        )


class TestCanonicalise:
    def test_returns_a_set_so_inherited_duplicates_collapse(self):
        # Head-noun inheritance can hand a bag the same token twice, and a
        # repeated token must not read as emphasis to the cascade.
        assert canonicalise(["medial", "medial", "meniscus"]) == _bag("medial", "meniscus")

    def test_drops_stopwords_and_generic_locators(self):
        assert canonicalise(["the", "medial", "aspect", "of", "the", "meniscus"]) == _bag(
            "medial", "meniscus"
        )

    def test_drops_laterality_because_the_study_images_one_knee(self):
        # "right" says nothing about *where* in the imaged knee.
        assert canonicalise(["right", "medial", "meniscus"]) == _bag("medial", "meniscus")

    def test_folds_inner_and_outer_onto_medial_and_lateral(self):
        assert canonicalise(["outer", "compartment"]) == _bag("lateral", "compartment")
        assert canonicalise(["internal", "meniscus"]) == _bag("medial", "meniscus")

    def test_the_inout_fold_can_be_switched_off(self):
        assert canonicalise(["outer", "compartment"], inout=False) == _bag(
            "outer", "compartment"
        )

    def test_hedge_words_can_be_kept(self):
        assert canonicalise(["other", "compartment"]) == _bag("compartment")
        assert canonicalise(["other", "compartment"], hedge=False) == _bag(
            "other", "compartment"
        )


class TestCascadeOrder:
    """The order of the tests in ``cluster_of_bag`` *is* the algorithm."""

    def test_ligament_is_decided_before_cartilage(self):
        # `patellofemoral ligament` is the MPFL, not patellofemoral cartilage.
        # Swapping the two branches changes both of these.
        assert cluster_of_bag(_bag("patellofemoral", "ligament")) == (
            "PATELLOFEMORAL_LIGAMENT_MPFL"
        )
        assert cluster_of_bag(_bag("patellofemoral", "cartilage")) == "CARTILAGE_PATELLOFEMORAL"

    def test_a_named_ligament_beats_the_unspecified_bucket(self):
        assert cluster_of_bag(_bag("meniscofemoral", "ligament")) == "MENISCOFEMORAL_LIGAMENT"
        assert cluster_of_bag(_bag("oblique", "ligament")) == "POSTEROMEDIAL_OBLIQUE_LIGAMENT"
        assert cluster_of_bag(_bag("ligament")) == "LIGAMENT_UNSPECIFIED"

    def test_non_anatomical_content_is_removed_before_any_structure_matches(self):
        # `TT-TG distance` names the trochlea and is a measurement. Matching
        # the structure first invents a finding on it.
        assert cluster_of_bag(_bag("tubercle", "groove")) == "ZZ_NONANAT_measurement"
        assert cluster_of_bag(_bag("alignment")) == "ZZ_NONANAT_measurement"

    @pytest.mark.parametrize(
        ("bag", "expected"),
        [
            (("meniscus", "medial"), "MENISCUS_MEDIAL"),
            (("meniscus", "lateral"), "MENISCUS_LATERAL"),
            (("meniscus", "medial", "lateral"), "MENISCUS_BOTH"),
            (("meniscus",), "MENISCUS_SIDE_UNSPECIFIED"),
            (("cruciate", "anterior"), "ACL"),
            (("cruciate", "posterior"), "PCL"),
            (("cruciate", "anterior", "posterior"), "CRUCIATE_BOTH"),
            (("cruciate",), "CRUCIATE_UNSPECIFIED"),
            (("collateral", "medial"), "MCL"),
            (("collateral", "lateral"), "LCL"),
            (("collateral", "fibular"), "LCL"),
            (("collateral", "medial", "lateral"), "COLLATERAL_BOTH"),
            (("collateral",), "COLLATERAL_SIDE_UNSPECIFIED"),
            (("marrow", "medial"), "BONE_MARROW_MEDIAL"),
            (("marrow", "lateral"), "BONE_MARROW_LATERAL"),
            (("marrow",), "BONE_MARROW_UNSPECIFIED"),
            (("condyle", "femoral", "medial"), "BONE_MEDIAL_FEMORAL_CONDYLE"),
            (("condyle", "femoral", "lateral"), "BONE_LATERAL_FEMORAL_CONDYLE"),
            (("plateau", "tibial", "medial"), "BONE_MEDIAL_TIBIAL_PLATEAU"),
            (("plateau", "tibial", "lateral"), "BONE_LATERAL_TIBIAL_PLATEAU"),
            (("condyle",), "BONE_CONDYLE_PLATEAU_UNSPEC"),
        ],
    )
    def test_the_side_bearing_clusters(self, bag, expected):
        # These are the pairs that differ by one token and feed different
        # competition labels, so each one is pinned rather than sampled.
        assert cluster_of_bag(_bag(*bag)) == expected

    def test_a_bag_naming_both_sides_is_not_silently_one_of_them(self):
        assert cluster_of_bag(_bag("marrow", "medial", "lateral")) == "BONE_MARROW_UNSPECIFIED"

    def test_every_returned_name_is_in_the_declared_vocabulary(self):
        assert cluster_of_bag(_bag("meniscus", "medial")) in CLUSTER_NAMES


class TestPatellofemoralCompartmentFix:
    """Task 31. The one deliberate difference from the original clusterer."""

    def test_a_patellofemoral_compartment_is_its_own_cluster(self):
        # It used to fall to COMPARTMENT_UNSPECIFIED, which the label map calls
        # ambiguous for all three OA labels -- so a positive finding on it was
        # masked instead of setting PF OA.
        assert clusters_of("patellofemoral compartment") == frozenset(
            {"COMPARTMENT_PATELLOFEMORAL"}
        )

    @pytest.mark.parametrize(
        "raw", ["patellofemoral compartment", "patelofemoral compartment",
                "femoropatellar compartment", "retropatellar compartment",
                "PF compartment"],
    )
    def test_every_surface_form_lands_in_it(self, raw):
        assert clusters_of(raw) == frozenset({"COMPARTMENT_PATELLOFEMORAL"})

    def test_it_is_guarded_exactly_as_the_patellofemoral_joint_test_is(self):
        # The two branches have to agree, or `patellofemoral joint` and
        # `patellofemoral compartment` disagree about the same site.
        assert cluster_of_bag(_bag("patellofemoral", "joint")) == "JOINT_PATELLOFEMORAL"
        assert cluster_of_bag(_bag("patellofemoral", "compartment")) == (
            "COMPARTMENT_PATELLOFEMORAL"
        )

    @pytest.mark.parametrize(
        ("bag", "expected"),
        [
            (("patellofemoral", "compartment", "medial"), "COMPARTMENT_MEDIAL"),
            (("patellofemoral", "compartment", "lateral"), "COMPARTMENT_LATERAL"),
            (("patellofemoral", "compartment", "femorotibial"), "COMPARTMENT_UNSPECIFIED"),
        ],
    )
    def test_a_compound_the_splitter_missed_keeps_its_previous_cluster(self, bag, expected):
        # A bag carrying both a patellofemoral token and a side is a compound
        # the splitter failed to separate. Inventing a combined cluster for it
        # would be a second, unmeasured change.
        assert cluster_of_bag(_bag(*bag)) == expected

    def test_a_tibiofemoral_compartment_is_untouched_by_the_fix(self):
        assert clusters_of("femorotibial compartment") == frozenset(
            {"COMPARTMENT_UNSPECIFIED"}
        )
        assert clusters_of("medial compartment") == frozenset({"COMPARTMENT_MEDIAL"})
        assert clusters_of("outer compartment") == frozenset({"COMPARTMENT_LATERAL"})


class TestResidualBuckets:
    """A string the vocabulary cannot place must be *named* unplaceable."""

    @pytest.mark.parametrize(
        ("raw", "bucket"),
        [
            ("TT-TG distance", "ZZ_NONANAT_measurement"),
            ("patellar tracking", "ZZ_NONANAT_measurement"),
            ("joint effusion", "ZZ_NONANAT_finding_in_anatomy_field"),
            ("fracture", "ZZ_NONANAT_finding_in_anatomy_field"),
            ("loose bodies", "ZZ_NONANAT_finding_in_anatomy_field"),
            ("perilesional", "ZZ_UNMAPPABLE_vague"),
            ("complex", "ZZ_UNMAPPABLE_vague"),
            ("anterior", "ZZ_UNMAPPABLE_vague"),
            ("right thigh", "ZZ_OUTSIDE_KNEE"),
            ("lower leg", "ZZ_OUTSIDE_KNEE"),
            ("knee", "ZZ_WHOLE_KNEE_nonlocalising"),
        ],
    )
    def test_each_bucket_is_reachable(self, raw, bucket):
        # An unreachable residual bucket is dead weight; a missing one means a
        # non-anatomical string is being rounded to the nearest structure.
        assert clusters_of(raw) == frozenset({bucket})

    def test_an_empty_or_blank_anatomy_field_is_unmappable_not_a_crash(self):
        assert clusters_of("") == frozenset({"ZZ_UNMAPPABLE_vague"})
        assert clusters_of("   ") == frozenset({"ZZ_UNMAPPABLE_vague"})

    def test_a_fluid_word_inside_a_container_is_still_the_container(self):
        # `effusion` alone is a finding in the anatomy field, but `bursa` names
        # a real site and must survive the residual test above it.
        assert cluster_of_bag(_bag("effusion", "bursa")) == "BURSA"

    def test_a_structure_named_alongside_the_thigh_stays_inside_the_knee(self):
        assert cluster_of_bag(_bag("thigh", "meniscus")) == "MENISCUS_SIDE_UNSPECIFIED"

    def test_every_bucket_name_is_declared(self):
        assert {n for n in CLUSTER_NAMES if n.startswith("ZZ_")} == {
            "ZZ_NONANAT_finding_in_anatomy_field",
            "ZZ_NONANAT_measurement",
            "ZZ_OUTSIDE_KNEE",
            "ZZ_UNMAPPABLE_vague",
            "ZZ_WHOLE_KNEE_nonlocalising",
        }


class TestFoldFlags:
    """Four ablation switches over judgement calls. Both positions must work."""

    def test_inout_on_gives_the_synonym_its_side(self):
        assert clusters_of("outer compartment") == frozenset({"COMPARTMENT_LATERAL"})

    def test_inout_off_loses_the_side_as_documented(self):
        assert clusters_of("outer compartment", inout=False) == frozenset(
            {"COMPARTMENT_UNSPECIFIED"}
        )
        assert clusters_of("inner meniscus", inout=False) == frozenset(
            {"MENISCUS_SIDE_UNSPECIFIED"}
        )

    def test_fold_fibular_on_reads_the_fibular_collateral_as_the_lcl(self):
        assert clusters_of("fibular collateral ligament") == frozenset({"LCL"})

    def test_fold_fibular_off_declines_to_name_the_side(self):
        assert clusters_of("fibular collateral ligament", fold_fibular=False) == frozenset(
            {"COLLATERAL_SIDE_UNSPECIFIED"}
        )

    def test_fold_fibular_off_still_names_an_explicitly_lateral_one(self):
        # The flag questions the fibular synonym, not the word `lateral`.
        assert clusters_of("lateral collateral ligament", fold_fibular=False) == frozenset(
            {"LCL"}
        )
        assert clusters_of(
            "lateral (fibular) collateral ligament", fold_fibular=False
        ) == frozenset({"LCL"})

    def test_fold_patlig_on_reads_the_patellar_ligament_as_the_tendon(self):
        assert clusters_of("patellar ligament") == frozenset({"TENDON_PATELLAR"})

    def test_fold_patlig_off_is_the_only_producer_of_the_unfolded_cluster(self):
        # LIGAMENT_PATELLAR_UNFOLDED exists to serve this flag and nothing else.
        # If this stops firing, the cluster is unreachable and the label map's
        # entry for it is dead.
        assert "LIGAMENT_PATELLAR_UNFOLDED" in CLUSTER_NAMES
        assert clusters_of("patellar ligament", fold_patlig=False) == frozenset(
            {"LIGAMENT_PATELLAR_UNFOLDED"}
        )

    def test_fold_patlig_off_leaves_the_patellar_tendon_alone(self):
        assert clusters_of("patellar tendon", fold_patlig=False) == frozenset(
            {"TENDON_PATELLAR"}
        )

    def test_expand_extensor_off_keeps_the_mechanism_as_one_site(self):
        assert clusters_of("extensor mechanism") == frozenset({"EXTENSOR_MECHANISM"})

    def test_expand_extensor_on_replaces_it_with_its_three_members(self):
        assert clusters_of("extensor mechanism", expand_extensor=True) == frozenset(
            {"TENDON_QUADRICEPS", "TENDON_PATELLAR", "PATELLA"}
        )
        assert "EXTENSOR_MECHANISM" not in clusters_of(
            "extensor apparatus", expand_extensor=True
        )

    def test_the_flags_are_independent(self):
        # All four at once, on a string that exercises three of them.
        assert clusters_of(
            "fibular collateral ligament, patellar ligament, extensor mechanism",
            inout=False, fold_fibular=False, fold_patlig=False, expand_extensor=True,
        ) == frozenset(
            {"COLLATERAL_SIDE_UNSPECIFIED", "LIGAMENT_PATELLAR_UNFOLDED",
             "TENDON_QUADRICEPS", "TENDON_PATELLAR", "PATELLA"}
        )


class TestLookalikeFolding:
    """``normalise.fold_lookalikes`` before matching, never before hashing.

    Measured: **0 of the 1,010** anatomy strings this model produced carry a
    non-ASCII codepoint, so the ASCII tokeniser is sound for *this* model's
    output and the clusterer does not fold. That is a property of the model, not
    of the corpus, and these tests state what the caller must do if a new model
    starts writing the anatomy field in the report's own language.
    """

    def test_the_turkish_dotted_capital_i_is_mangled_unless_folded_first(self):
        # U+0130 lowercases to `i` + U+0307 COMBINING DOT ABOVE, and the
        # non-word strip then splits the word in two. Folding U+0130 alone moved
        # measured Turkish laterality from 0.4% to 67.8% of Turkish reports.
        assert tokens("DİZ") == ["di", "z"]
        assert tokens(fold_lookalikes("DİZ")) == ["diz"]

    def test_nfc_does_not_repair_these_codepoints(self):
        import unicodedata

        assert unicodedata.normalize("NFC", "∆") == "∆"
        assert fold_lookalikes("∆") == "Δ"

    def test_folding_before_hashing_would_change_the_key(self):
        # Hence "never before hashing": the stored text has to stay
        # byte-faithful to the source, so folding is a matching step only.
        raw = "DİZ"
        assert content_hash(raw) != content_hash(fold_lookalikes(raw))

    def test_a_non_latin_anatomy_string_contributes_no_label(self):
        # Folding is necessary but not sufficient: the induced vocabulary is
        # Latin-script, so no amount of folding makes a Greek anatomy field
        # match. It falls to OTHER, which the label map leaves undecided, so the
        # failure mode is *no label* rather than a wrong one -- asserted here
        # rather than assumed, because OTHER also collects Latin-script strings
        # the vocabulary simply misses, and the two look identical in a count.
        from kappa.labels.label_map import CLUSTER_LABELS, UNMAPPED_CLUSTERS

        assert tokens("∆ΕΞΙΟΥ ΓΟΝΑΤΟΣ") == []
        assert clusters_of("μηνίσκου") == frozenset({"OTHER"})
        assert "OTHER" in UNMAPPED_CLUSTERS
        assert CLUSTER_LABELS.get("OTHER", {}) == {}


class TestKnownLimitations:
    """Pinned, not fixed: changing these changes the measured derivation."""

    @pytest.mark.parametrize(
        ("raw", "clusters"),
        [
            ("articular cartilage of the femur and tibia (medial side)",
             {"CARTILAGE_TIBIOFEMORAL_SIDE_UNSPEC"}),
            ("knee joint (patellofemoral and medial femoral tibial compartments)",
             {"JOINT_SPACE_EFFUSION"}),
            ("distal femur (intercondylar and medial epicondyle)", {"BONE_UNSPECIFIED"}),
        ],
    )
    def test_a_parenthetical_that_specifies_rather_than_glosses_is_still_dropped(
        self, raw, clusters
    ):
        # Measured over the corpus: 25 strings carry a parenthetical, and for
        # **6 strings / 6 findings** keeping its content would change the
        # cluster. It is not a clear win -- for `anterior cruciate ligament
        # (posterolateral bundle)` keeping it yields CRUCIATE_BOTH, which is
        # worse. So the gloss rule stands, and this test exists so the loss is
        # visible instead of being rediscovered as a mystery.
        assert clusters_of(raw) == frozenset(clusters)

    def test_the_bundle_parenthetical_is_why_the_gloss_rule_stays(self):
        assert clusters_of("anterior cruciate ligament (posterolateral bundle)") == frozenset(
            {"ACL"}
        )

    def test_a_sub_part_with_no_structure_is_not_a_site(self):
        # `posterior horn` names a location within a meniscus, not a site, and
        # falls to OTHER. `label_map` deliberately leaves OTHER undecided.
        assert clusters_of("posterior horn") == frozenset({"OTHER"})

    def test_the_frozen_constants_are_frozen(self):
        # Recomputing them from whatever findings are in the batch would make
        # `singularise` batch-dependent: the same string would cluster
        # differently in a smoke test and in a full run, invisibly.
        assert isinstance(INDUCED_VOCABULARY, frozenset)
        assert len(INDUCED_VOCABULARY) == 312
        assert len(ABBREVIATIONS) == 14
