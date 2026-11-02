"""Turn one study's extracted findings into the 12 competition labels.

Why this exists: **anatomy alone cannot decide five of the twelve labels.**
Effusion, Synovitis, Baker's, Contusion and Fracture name a *finding type*, not
a site, so a cluster on its own is not evidence for them -- a medial femoral
condyle with an osteophyte is a Medial OA, not a Contusion, and the anatomy
field says only "medial femoral condyle" in both cases. Mapping cluster to label
without a finding-type gate invents those five labels wholesale. Every gate in
:data:`GATES` exists to stop exactly that.

The second silent failure this module prevents is the confident negative. A
cluster can be *ambiguous* for a label: a finding on ``JOINT_SPACE_EFFUSION``
rules Synovitis out when it is negative ("no synovial thickening or effusion")
but cannot confirm it when positive, because the positive statement is about
fluid. So ambiguity is resolved **conditioned on polarity**: a negative
statement fans out to every member of the bundle, a positive one is masked
rather than attributed. Reading both as positive is what fabricates ranks under
macro ROC-AUC, where each of the twelve labels is worth exactly 1/12.

    from kappa.labels.label_map import UNMENTIONED_BY_LABEL, derive_study

    labels = derive_study(record)                                   # scalar path
    labels = derive_study(record, unmentioned=UNMENTIONED_BY_LABEL)  # per-label

``unmentioned`` -- the value for a label no finding mentions -- takes a
**per-label mapping**, not only a scalar, and that is a measured requirement.
Silence does not mean the same thing for every label: see
:data:`UNMENTIONED_BY_LABEL`. The scalar path is kept because the measured
macro balanced accuracy of 0.765 on the 58 anchors was produced with
``unmentioned=0.0``, and that number has to stay reproducible.

Polarity comes from ``kappa.labels.polarity.polarity(assertion, description)``,
which needs both fields: ``assertion`` alone is language-dependent (normality
statements are ``affirmed`` 93% of the time in English and 30% in Greek), so it
must never set polarity by itself.

:func:`score_derivation` lives here rather than in ``kappa.evaluation`` because
what it scores is this module's output, and the thing it exists to get right is
a property of masking: **every rate is per label, over the studies that label
actually covers.** Ten of the twelve labels cover fewer than all 58 anchors, so
a prevalence printed over the full comparison overstates the evidence behind
the row. :class:`Agreement` therefore carries ``n_covered`` and ``coverage``
alongside the confusion counts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from kappa.constants import ID_COLUMN, LABELS
from kappa.labels.anatomy_map import CLUSTER_NAMES, clusters_of
from kappa.labels.masking import MaskReason, report_mask, resolve_label
from kappa.labels.polarity import polarity

__all__ = [
    "AMBIGUOUS",
    "CLUSTER_LABELS",
    "DIRECT",
    "Directness",
    "GATES",
    "GATE_ALONE",
    "MASK_SUFFIX",
    "OUTSIDE_KNOWLEDGE",
    "Agreement",
    "DerivationScore",
    "StudyLabels",
    "TABLE_COLUMNS",
    "UNMAPPED_CLUSTERS",
    "UNMENTIONED_BY_LABEL",
    "derive_studies",
    "derive_study",
    "mask_column",
    "score_derivation",
]

Directness = Literal["D", "A"]

#: The cluster is evidence for the label on its own.
DIRECT: Directness = "D"
#: The cluster names a site that *could* carry the label but shares it with
#: others. Only useful in the negative direction -- see the fan-out in
#: :func:`derive_study`.
AMBIGUOUS: Directness = "A"

_SHORT_TO_LABEL = {
    "ACL": "ACL",
    "MCL": "MCL",
    "MM": "Medial Meniscus",
    "LM": "Lateral Meniscus",
    "MOA": "Medial OA",
    "LOA": "Lateral OA",
    "PFOA": "PF OA",
    "EFF": "Effusion",
    "SYN": "Synovitis",
    "BAK": "Baker's",
    "CON": "Contusion",
    "FRA": "Fracture",
}

# Authored with the short names so that one cluster is one readable line. An
# empty mapping is a deliberate decision, not an omission: PCL, LCL, the
# retinaculum and most tendons are real structures that no competition label
# covers, and a finding on them must contribute to nothing rather than be
# rounded to the nearest label.
_CLUSTER_LABELS_SHORT: dict[str, dict[str, Directness]] = {
    "JOINT_SPACE_EFFUSION":        {"EFF": "D", "SYN": "A", "MOA": "A", "LOA": "A", "PFOA": "A"},
    "MENISCUS_MEDIAL":             {"MM": "D"},
    "MENISCUS_LATERAL":            {"LM": "D"},
    "MENISCUS_SIDE_UNSPECIFIED":   {"MM": "A", "LM": "A"},
    "BONE_UNSPECIFIED":            {"FRA": "D", "CON": "D", "MOA": "A", "LOA": "A", "PFOA": "A"},
    "PATELLA":                     {"PFOA": "D", "FRA": "D", "CON": "D"},
    "ACL":                         {"ACL": "D"},
    "PCL":                         {},
    "TENDON_PATELLAR":             {},
    "MCL":                         {"MCL": "D"},
    "TENDON_QUADRICEPS":           {},
    "COLLATERAL_SIDE_UNSPECIFIED": {"MCL": "A"},
    "CARTILAGE_PATELLOFEMORAL":    {"PFOA": "D"},
    "LCL":                         {},
    "POPLITEAL_FOSSA":             {"BAK": "A"},
    "PATELLAR_RETINACULUM":        {},
    "CRUCIATE_UNSPECIFIED":        {"ACL": "A"},
    "CARTILAGE_FULLY_UNSPECIFIED": {"MOA": "A", "LOA": "A", "PFOA": "A"},
    "SOFT_TISSUE":                 {},
    "BONE_MARROW_UNSPECIFIED":     {"CON": "D", "FRA": "D", "MOA": "A", "LOA": "A", "PFOA": "A"},
    "CARTILAGE_TIBIOFEMORAL_SIDE_UNSPEC": {"MOA": "A", "LOA": "A"},
    "BONE_MEDIAL_FEMORAL_CONDYLE": {"MOA": "D", "FRA": "D", "CON": "D"},
    "EXTENSOR_MECHANISM":          {"PFOA": "A", "FRA": "A", "CON": "A"},
    "CARTILAGE_MEDIAL_TIBIOFEMORAL": {"MOA": "D"},
    "BONE_LATERAL_FEMORAL_CONDYLE": {"LOA": "D", "FRA": "D", "CON": "D"},
    "BURSA":                       {"EFF": "A", "SYN": "A"},
    "BONE_TROCHLEA":               {"PFOA": "D", "FRA": "D", "CON": "D"},
    "FAT_PAD_HOFFA_INFRAPATELLAR": {"SYN": "A"},
    "CARTILAGE_LATERAL_TIBIOFEMORAL": {"LOA": "D"},
    "JOINT_PATELLOFEMORAL":        {"PFOA": "D"},
    "BONE_MEDIAL_TIBIAL_PLATEAU":  {"MOA": "D", "FRA": "D", "CON": "D"},
    "ILIOTIBIAL_BAND":             {},
    "TENDON_PES_ANSERINUS_HAMSTRING": {"BAK": "A"},
    "MUSCLE":                      {},
    "POSTEROLATERAL_CORNER":       {},
    "SYNOVIUM_PLICA":              {"SYN": "D"},
    "TENDON_POPLITEUS":            {},
    "CYST_POPLITEAL_BAKER":        {"BAK": "D"},
    "TENDON_UNSPECIFIED":          {},
    "TENDON_BICEPS_FEMORIS":       {},
    "GASTROCNEMIUS_SEMIMEMBRANOSUS": {"BAK": "A"},
    "BONE_CONDYLE_PLATEAU_UNSPEC": {"MOA": "A", "LOA": "A", "FRA": "D", "CON": "D"},
    "PATELLOFEMORAL_LIGAMENT_MPFL": {},
    "NEUROVASCULAR":               {},
    "BONE_LATERAL_TIBIAL_PLATEAU": {"LOA": "D", "FRA": "D", "CON": "D"},
    "COMPARTMENT_MEDIAL":          {"MOA": "D"},
    "BONE_INTERCONDYLAR":          {"FRA": "D", "CON": "D"},
    "COMPARTMENT_LATERAL":         {"LOA": "D"},
    # Added with the fix to `cluster_of_bag`'s compartment branch, and direct
    # for the same reason CARTILAGE_PATELLOFEMORAL and JOINT_PATELLOFEMORAL
    # are: the string names one compartment, and only one label covers it.
    "COMPARTMENT_PATELLOFEMORAL":  {"PFOA": "D"},
    "COMPARTMENT_UNSPECIFIED":     {"MOA": "A", "LOA": "A", "PFOA": "A"},
    "LIGAMENT_UNSPECIFIED":        {"ACL": "A", "MCL": "A"},
    "BONE_SUBCHONDRAL":            {"CON": "D", "FRA": "D", "MOA": "A", "LOA": "A", "PFOA": "A"},
    "CARTILAGE_MULTICOMPARTMENT":  {"PFOA": "D", "MOA": "A", "LOA": "A"},
    "FAT_PAD_OTHER":               {},
    "JOINT_MARGIN_ambiguous":      {"MOA": "A", "LOA": "A", "PFOA": "A"},
    "JOINT_CAPSULE":               {"SYN": "A"},
    "BONE_FIBULA_FABELLA":         {"FRA": "D", "CON": "D"},
    "BONE_METAPHYSIS_EPIPHYSIS":   {"FRA": "D", "CON": "D"},
    "JOINT_TIBIOFIBULAR":          {},
    "JOINT_MEDIAL_TIBIOFEMORAL":   {"MOA": "D"},
    "POSTEROMEDIAL_OBLIQUE_LIGAMENT": {"MCL": "A"},
    "MENISCOFEMORAL_LIGAMENT":     {},
    "BONE_TIBIAL_TUBEROSITY":      {"FRA": "D", "CON": "D"},
    "PARAMENISCAL_REGION":         {"MM": "A", "LM": "A"},
    "JOINT_LATERAL_TIBIOFEMORAL":  {"LOA": "D"},
    "BONE_MARROW_MEDIAL":          {"CON": "D", "FRA": "D", "MOA": "A"},
    # Residual buckets. Empty by construction -- they exist so that a string
    # the vocabulary cannot place is *named* as unplaceable instead of being
    # silently dropped into the nearest structure.
    "ZZ_NONANAT_measurement":            {},
    "ZZ_WHOLE_KNEE_nonlocalising":       {},
    "ZZ_UNMAPPABLE_vague":               {},
    "ZZ_NONANAT_finding_in_anatomy_field": {},
    "ZZ_OUTSIDE_KNEE":                   {},
    # Synthetic, reachable only when a fold in `clusters_of` is switched off.
    "LIGAMENT_PATELLAR_UNFOLDED":        {},
}

#: cluster -> {full label name: "D" | "A"}. The table used at derivation time.
CLUSTER_LABELS: dict[str, dict[str, Directness]] = {
    cluster: {_SHORT_TO_LABEL[short]: kind for short, kind in entries.items()}
    for cluster, entries in _CLUSTER_LABELS_SHORT.items()
}

# Clusters `clusters_of` can emit that this table does not decide. Enumerated
# rather than left implicit, because the lookup in `derive_study` treats an
# absent cluster and an empty mapping identically: both contribute no labels.
# Written down so `tests/test_label_map.py` fails the moment a *new* cluster
# appears without a decision, instead of it going unnoticed.
#
# Measured over the survey's anatomy table (3,962 findings): none of these six
# is reachable, so leaving them undecided costs nothing there. Over the 58
# anchors' 859 findings it is not free -- `OTHER` fires 3 times and
# `CRUCIATE_BOTH` once -- so filling them in is a real change to the 0.765 and
# belongs in its own task, not in this port.
UNMAPPED_CLUSTERS: frozenset[str] = frozenset(
    {
        "BONE_MARROW_LATERAL",
        "COLLATERAL_BOTH",
        "CRUCIATE_BOTH",
        "MENISCUS_BOTH",
        "OTHER",
        "TENDON_QUAD_AND_PATELLAR",
    }
)

#: Cluster-label pairs that rest on outside clinical knowledge rather than on
#: anything the corpus states. Kept as data so an ablation can drop them and
#: measure what the borrowed knowledge is worth.
OUTSIDE_KNOWLEDGE: frozenset[tuple[str, str]] = frozenset(
    {
        ("BURSA", "Effusion"),
        ("BURSA", "Synovitis"),
        ("FAT_PAD_HOFFA_INFRAPATELLAR", "Synovitis"),
        ("TENDON_PES_ANSERINUS_HAMSTRING", "Baker's"),
        ("POSTEROMEDIAL_OBLIQUE_LIGAMENT", "MCL"),
        ("BONE_MARROW_UNSPECIFIED", "Medial OA"),
        ("BONE_MARROW_UNSPECIFIED", "Lateral OA"),
        ("BONE_MARROW_UNSPECIFIED", "PF OA"),
        ("BONE_SUBCHONDRAL", "Medial OA"),
        ("BONE_SUBCHONDRAL", "Lateral OA"),
        ("BONE_SUBCHONDRAL", "PF OA"),
        ("BONE_INTERCONDYLAR", "Fracture"),
        ("BONE_INTERCONDYLAR", "Contusion"),
        ("BONE_TIBIAL_TUBEROSITY", "Fracture"),
        ("BONE_TIBIAL_TUBEROSITY", "Contusion"),
        ("BONE_FIBULA_FABELLA", "Fracture"),
        ("BONE_FIBULA_FABELLA", "Contusion"),
        ("EXTENSOR_MECHANISM", "PF OA"),
        ("EXTENSOR_MECHANISM", "Fracture"),
        ("EXTENSOR_MECHANISM", "Contusion"),
        ("JOINT_MARGIN_ambiguous", "Medial OA"),
        ("JOINT_MARGIN_ambiguous", "Lateral OA"),
        ("JOINT_MARGIN_ambiguous", "PF OA"),
    }
)

# Finding-type gates, matched against `description` -- which is English for
# every finding measured, whatever the report's language, so one English
# lexicon covers all nine. A label with a gate is set only by a finding whose
# description names the finding type, however well the anatomy matches.
#
# ACL, MCL and the two menisci take **no gate** on purpose: they are
# structures, so any positive finding on the structure is evidence for its
# label and there is no separate "does this count" question to ask.
_GATE_PATTERNS = {
    "Effusion": r"effusion|joint fluid|intra-?articular fluid|fluid (?:in|within)|hydrops",
    "Synovitis": r"synovit|synovial (?:hypertroph|thicken|proliferat|enhance)|pannus",
    "Baker's": r"baker|popliteal cyst|gastrocnemio-?semimembranosus",
    "Fracture": r"fractur|avulsion|osteochondral (?:lesion|defect|fragment)",
    "Contusion": r"contusion|bruise|bone (?:marrow )?(?:o?edema|bruis)|marrow o?edema",
    # JUDGEMENT, and marked as one: which phrase families count as
    # osteoarthritis. The three OA labels share this pattern exactly.
    "Medial OA": r"arthros|arthrit|osteophyt|chondromalac|chondrosis|chondropath|"
                 r"cartilage (?:loss|thinning|defect|denud|wear)|joint space narrow|"
                 r"chondral (?:loss|thinning|defect)|degenerat",
}
_GATE_PATTERNS["Lateral OA"] = _GATE_PATTERNS["Medial OA"]
_GATE_PATTERNS["PF OA"] = _GATE_PATTERNS["Medial OA"]

#: label -> compiled gate. Labels absent from this mapping take no gate.
GATES: dict[str, re.Pattern[str]] = {
    label: re.compile(pattern, re.I) for label, pattern in _GATE_PATTERNS.items()
}

# Labels where the gate ALONE decides, with no anatomy agreement required.
# These name a finding type rather than a site, so the question is "does this
# study show a Baker's cyst", not "does the anatomy field say popliteal fossa".
# The corpus makes the cost of getting that wrong concrete: 129 findings name a
# Baker's or popliteal cyst in `description` against only 34 whose anatomy
# lands in the Baker's cluster, so requiring both would discard three quarters
# of the evidence.
#
# `Contusion` is deliberately NOT in this set. Its gate has to match "bone
# marrow oedema", and bone marrow oedema appears in ~76% of reports -- mostly
# degenerative subchondral oedema, not trauma. Requiring a bone site keeps that
# in check; gate-only would over-call the label massively.
GATE_ALONE: frozenset[str] = frozenset({"Effusion", "Synovitis", "Baker's", "Fracture"})

# What silence means, per label. `resolve_label` needs a value for a label that
# no finding mentioned at all, and a single scalar is measurably wrong: on the
# 58 anchors, P(gold positive | report silent) is 0.34 for Synovitis (silent
# 83.7% of the time) and 0.21 for PF OA, against 0.03 for Baker's and 0.00 for
# Medial OA. Silence means *absent* for the last two and *nobody looked* for the
# first two, and 0.0 everywhere treats those two states alike -- inventing a
# confident negative on the exact labels where it is a coin flip.
#
# `None` masks the label, which is the honest answer for "nobody looked". Only
# these four labels have a measurement; the other eight keep the scalar default
# of 0.0 and are listed here so the omission is visible rather than implied.
UNMENTIONED_BY_LABEL: Mapping[str, float | None] = {
    "Synovitis": None,       # P(positive | silent) = 0.34, silent in 83.7% of reports
    "PF OA": None,           # P(positive | silent) = 0.21
    "Baker's": 0.0,          # P(positive | silent) = 0.03
    "Medial OA": 0.0,        # P(positive | silent) = 0.00
    "ACL": 0.0,              # unmeasured
    "MCL": 0.0,              # unmeasured
    "Medial Meniscus": 0.0,  # unmeasured
    "Lateral Meniscus": 0.0, # unmeasured
    "Lateral OA": 0.0,       # unmeasured
    "Effusion": 0.0,         # unmeasured
    "Contusion": 0.0,        # unmeasured
    "Fracture": 0.0,         # unmeasured
}

#: Suffix of the per-label usability column in the derived label table.
MASK_SUFFIX = "__mask"


def mask_column(label: str) -> str:
    """Name of the usability column paired with ``label``."""
    return f"{label}{MASK_SUFFIX}"


#: Column order of ``data/interim/labels/derived.parquet``. Value and mask are
#: separate columns on purpose: encoding "unknown" as a magic float is how a
#: NaN or a -1 reaches ``BCEWithLogitsLoss`` and silently destroys training.
TABLE_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    *(column for label in LABELS for column in (label, mask_column(label))),
)


@dataclass(frozen=True)
class StudyLabels:
    """One study's twelve labels, each with whether it may enter the loss.

    ``values`` is always a full set of twelve floats and never carries NaN, even
    where ``usable`` is False. The mask is the only thing that says "unknown";
    a sentinel float would survive every dtype cast and arrive at the loss.
    """

    study_uid: str | None
    values: Mapping[str, float]
    usable: Mapping[str, bool]
    mask_reasons: tuple[MaskReason, ...]

    @property
    def n_usable(self) -> int:
        """How many of the twelve labels this study can actually supervise."""
        return sum(1 for label in LABELS if self.usable[label])

    def as_row(self) -> dict[str, object]:
        """One row of the derived label table, keyed by :data:`TABLE_COLUMNS`."""
        row: dict[str, object] = {ID_COLUMN: self.study_uid}
        for label in LABELS:
            row[label] = float(self.values[label])
            row[mask_column(label)] = bool(self.usable[label])
        return row


def _unmentioned_per_label(
    unmentioned: float | None | Mapping[str, float | None],
) -> dict[str, float | None]:
    """Expand the ``unmentioned`` argument into a value for all twelve labels.

    A mapping may be partial -- the labels it omits keep the scalar default of
    0.0 -- but its keys are validated against ``LABELS``. Silently ignoring an
    unknown key would turn a misspelling such as ``"Bakers"`` into a no-op,
    which is indistinguishable from the override having been applied.
    """
    if isinstance(unmentioned, Mapping):
        unknown = sorted(set(unmentioned) - set(LABELS))
        if unknown:
            raise ValueError(f"not competition labels: {unknown}")
        return {label: unmentioned.get(label, 0.0) for label in LABELS}
    return {label: unmentioned for label in LABELS}


def _finding_labels(anatomy: str, description: str) -> dict[str, Directness]:
    """Labels this one finding bears on, and how directly.

    ``DIRECT`` is sticky: once one of a finding's clusters gives a label
    directly, another cluster calling it ambiguous must not downgrade it.
    """
    found: dict[str, Directness] = {}
    for cluster in clusters_of(anatomy):
        for label, kind in CLUSTER_LABELS.get(cluster, {}).items():
            if found.get(label) != DIRECT:
                found[label] = kind
    # A finding whose description names one of these labels counts for it
    # directly, whatever its anatomy field says.
    for label in GATE_ALONE:
        if GATES[label].search(description):
            found[label] = DIRECT
    return found


def derive_study(
    record: dict,
    *,
    fan_out_negatives: bool = True,
    unmentioned: float | None | Mapping[str, float | None] = 0.0,
) -> StudyLabels:
    """Derive one study's twelve labels from its extraction record.

    ``fan_out_negatives`` controls what happens to an *ambiguous* cluster-label
    pair. On, a negative bundle statement ("the cruciate ligaments are intact")
    negates every member, while a positive one is masked because it cannot be
    attributed to a member. Off, ambiguous pairs contribute nothing at all,
    which is the ablation that measures what the fan-out is worth.
    """
    uid = record.get("_study_uid")
    silence = _unmentioned_per_label(unmentioned)
    mask = report_mask(record)

    # Only `extraction_failed` voids the whole row here. It means the model
    # never produced parseable output, so there is no evidence of any kind --
    # distinct from a report that parsed and simply said little. The other
    # reasons `report_mask` can return are carried on the result instead of
    # applied, because acting on them changes the derivation that measured
    # 0.765 and that is a separate, measurable decision.
    if "extraction_failed" in mask.reasons:
        return StudyLabels(
            study_uid=uid,
            values=dict.fromkeys(LABELS, 0.0),
            usable=dict.fromkeys(LABELS, False),
            mask_reasons=mask.reasons,
        )

    per_label: dict[str, list[str]] = {label: [] for label in LABELS}
    for finding in record.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        description = finding.get("description") or ""
        found_polarity = polarity(finding.get("assertion"), description)
        for label, kind in _finding_labels(finding.get("anatomy") or "", description).items():
            gate = GATES.get(label)
            if gate is not None and not gate.search(description):
                continue
            if kind == DIRECT:
                per_label[label].append(found_polarity)
            elif fan_out_negatives:
                per_label[label].append(
                    "negative" if found_polarity == "negative" else "mask"
                )

    values: dict[str, float] = {}
    usable: dict[str, bool] = {}
    for label in LABELS:
        resolved = resolve_label(per_label[label], unmentioned=silence[label])
        values[label] = 0.0 if resolved is None else float(resolved)
        usable[label] = resolved is not None
    return StudyLabels(study_uid=uid, values=values, usable=usable, mask_reasons=mask.reasons)


def derive_studies(
    records: Iterable[dict],
    *,
    fan_out_negatives: bool = True,
    unmentioned: float | None | Mapping[str, float | None] = 0.0,
) -> Iterator[StudyLabels]:
    """Derive labels for a stream of extraction records, in input order."""
    for record in records:
        yield derive_study(
            record, fan_out_negatives=fan_out_negatives, unmentioned=unmentioned
        )


@dataclass(frozen=True)
class Agreement:
    """One label's agreement with ground truth, over the studies it covers.

    ``n_covered`` is the load-bearing field. A derived label is unknown for some
    studies -- every finding that bore on it was masked -- and those studies are
    not scored, so the confusion counts and every rate below rest on
    ``n_covered``, not on the number of studies compared. Reporting a prevalence
    over the full comparison instead is how a table claims 58 studies' worth of
    evidence for a row that saw 48.
    """

    label: str
    tp: int
    fp: int
    fn: int
    tn: int
    n_compared: int

    @property
    def n_covered(self) -> int:
        """Studies where the derivation produced a usable value for this label."""
        return self.tp + self.fp + self.fn + self.tn

    @property
    def coverage(self) -> float:
        """``n_covered / n_compared``, or 0.0 when nothing was compared."""
        return self.n_covered / self.n_compared if self.n_compared else 0.0

    @property
    def n_gold_positive(self) -> int:
        """Gold positives **among the covered studies only**."""
        return self.tp + self.fn

    @property
    def n_derived_positive(self) -> int:
        """Derived positives among the covered studies."""
        return self.tp + self.fp

    @property
    def sensitivity(self) -> float:
        """NaN, not 0.0, when no covered study is a gold positive."""
        return self.tp / self.n_gold_positive if self.n_gold_positive else math.nan

    @property
    def specificity(self) -> float:
        """NaN, not 0.0, when no covered study is a gold negative."""
        negatives = self.tn + self.fp
        return self.tn / negatives if negatives else math.nan

    @property
    def balanced_accuracy(self) -> float:
        """Mean of sensitivity and specificity, or NaN if either is undefined."""
        sens, spec = self.sensitivity, self.specificity
        if math.isnan(sens) or math.isnan(spec):
            return math.nan
        return (sens + spec) / 2


@dataclass(frozen=True)
class DerivationScore:
    """Per-label agreement plus the macro figure the project quotes.

    ``skipped`` names the labels whose balanced accuracy is undefined rather
    than letting them fall silently out of the mean, for the same reason
    ``kappa.evaluation.metrics`` does: a macro over an unstated subset of the
    twelve is not comparable with a macro over all twelve.
    """

    per_label: tuple[Agreement, ...]
    n_compared: int
    skipped: tuple[str, ...]

    @property
    def macro_balanced_accuracy(self) -> float:
        """Mean balanced accuracy over the labels that have one."""
        scored = [
            row.balanced_accuracy
            for row in self.per_label
            if not math.isnan(row.balanced_accuracy)
        ]
        return sum(scored) / len(scored) if scored else math.nan

    @property
    def coverage(self) -> float:
        """Fraction of the ``n_compared x 12`` label cells that were usable."""
        cells = self.n_compared * len(LABELS)
        return sum(a.n_covered for a in self.per_label) / cells if cells else 0.0


def score_derivation(
    derived: Mapping[str, StudyLabels],
    truth: Mapping[str, Mapping[str, int]],
) -> DerivationScore:
    """Compare derived labels against ground truth on the studies both cover.

    Only studies present in both mappings are compared, and within a study only
    the labels the derivation marked usable are scored. A masked label is
    absent evidence, and counting it as a derived 0.0 would credit the
    derivation with a true negative it never claimed.
    """
    shared = [uid for uid in derived if uid in truth]
    rows: list[Agreement] = []
    for label in LABELS:
        tp = fp = fn = tn = 0
        for uid in shared:
            study = derived[uid]
            if not study.usable[label]:
                continue
            gold = int(truth[uid][label])
            positive = study.values[label] == 1.0
            if gold == 1 and positive:
                tp += 1
            elif gold == 0 and positive:
                fp += 1
            elif gold == 1:
                fn += 1
            else:
                tn += 1
        rows.append(Agreement(label=label, tp=tp, fp=fp, fn=fn, tn=tn, n_compared=len(shared)))
    skipped = tuple(a.label for a in rows if math.isnan(a.balanced_accuracy))
    return DerivationScore(per_label=tuple(rows), n_compared=len(shared), skipped=skipped)


# Guard the one gap the lookup in `_finding_labels` cannot report: a cluster
# that `clusters_of` can emit but this table does not decide contributes no
# labels, and `dict.get` makes that indistinguishable from a decided-empty
# cluster. Checked at import so the mismatch surfaces the moment the module
# loads, not in a run whose output merely looks thin.
_UNDECIDED = CLUSTER_NAMES - set(CLUSTER_LABELS) - UNMAPPED_CLUSTERS
if _UNDECIDED:  # pragma: no cover - a guard, not a branch
    raise RuntimeError(
        f"clusters with no label decision and not listed in UNMAPPED_CLUSTERS: "
        f"{sorted(_UNDECIDED)}"
    )
