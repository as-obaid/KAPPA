"""Cluster a finding's free-text ``anatomy`` string into a controlled vocabulary.

Why this exists: the extraction model writes ``anatomy`` as prose, and matching
label keywords against that prose directly gets the *side* wrong. The 300-report
survey produced **1,010 distinct anatomy strings over 3,962 findings**, and the
distinctions that decide a label are exactly the ones a keyword scan flattens:

* ``medial compartment`` and ``lateral compartment`` differ by one token and map
  to two different competition labels. So do ``medial meniscus`` and ``outer
  compartment`` -- the second is *lateral*, written as an English synonym.
* One string routinely names two structures. ``lateral compartment, lateral
  collateral ligament, cruciate ligaments`` is one finding and three sites, and
  ``medial and lateral femoral condyles`` shares its head noun across two sites. Treating either as a single unit loses a label.
* The corpus misspells. ``patelofemoral``, ``trochea``, ``semimenbranosus`` and
  ``poplitiofibular`` all occur, and each one is a whole cluster missed.

So the mapping is a small pipeline: :func:`preprocess` (expand abbreviations,
repair known typos), :func:`split_components` (one string to one bag of tokens
per site, inheriting the shared head noun), :func:`canonicalise` (drop
stopwords, fold inner/outer to medial/lateral) and finally the ordered cascade
in :func:`cluster_of_bag`.

    from kappa.labels.anatomy_map import clusters_of

    clusters_of("medial and lateral meniscus")
    # frozenset({'MENISCUS_MEDIAL', 'MENISCUS_LATERAL'})
    clusters_of("patellofemoral compartment")
    # frozenset({'COMPARTMENT_PATELLOFEMORAL'})

**The induced vocabulary is frozen, deliberately.** :data:`INDUCED_VOCABULARY`
and :data:`ABBREVIATIONS` were induced from the survey's own anatomy strings,
and they are checked in as constants rather than recomputed from whatever input
is being clustered. Recomputing would make the depluralisation in
:func:`singularise` depend on which findings happen to be in the batch: the same
string would cluster differently in a smoke test and in a full run, and the
difference would be invisible. ``tests/test_anatomy_map_equivalence.py``
re-derives both constants from the survey's anatomy table and fails if they
drift, and runs the whole corpus through this module and the original clusterer
to pin the difference between them to five enumerated strings. That table and
the original label-mapping scripts are not published, because they contain
report text, so the test skips in a public checkout.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "ABBREVIATIONS",
    "CLUSTER_NAMES",
    "INDUCED_VOCABULARY",
    "TYPO_FIXES",
    "canonicalise",
    "cluster_of_bag",
    "clusters_of",
    "preprocess",
    "singularise",
    "split_components",
    "tokens",
]

# Every token that appears in any anatomy string of the 300-report survey (312
# of them). Used only for membership: `singularise` strips a plural suffix only
# when the singular is a word the corpus actually uses, so `menisci` folds but
# `pes` does not become `pe`.
INDUCED_VOCABULARY: frozenset[str] = frozenset(
    "acl acromial adipose adjacent alignment and anserine anserinus anterior anteromedial "
    "anteroposterior apex aponeurosis apophysis apparatus area areas around artery articular "
    "articulating articulation articulations aspect at attachment attachments baker band base "
    "bearing between biceps bilateral bodies body bone bones bony both bundle bundles bursa "
    "bursas capsule cartilage cartilages cartilaginous cavity cellular central chondral "
    "collateral compartment compartments complex complexes component components condyle condyles "
    "connective corner cranial cruciate crural cyst deep diaphysis distal distance dorsal edge "
    "effusion elements eminance eminence eminences ends epicondyle epiphysis extensor external "
    "fabella facet facets fascicles fat fatty fcl femoral femoris femoro femoropatellar "
    "femorotibial femur fibula fibular fluid formations fossa fracture fragments ft gastrocnemius "
    "gastronomes general groove groups half head hkb hoffa horn horns humphrey hyaline iliotibial "
    "in inferior infrapatellar inner insertion intercondylar interface intermediate intermuscular "
    "internal intra it joint joints knee lateral lca lcl left leg legs level ligament ligamentous "
    "ligaments loose lower m margin margins marrow mcl mechanism medial medialis median "
    "mediopatellar membrane meniscal menisci meniscocapsular meniscofemoral meniscopopliteal "
    "meniscus mesocondyle metaphysis mid middle mpfl multiple muscle muscles musculature nerve "
    "neurovascular nodular notch oblique of or osseous osteochondral other outer pad parameniscal "
    "parapatellar part patella patellar patellofemoral patelofemoral pcl periarticular perigonal "
    "perilesional peritrochanteric peroneal pes pf planes plateau plateaus plica plicae pole "
    "popliteal popliteofibular popliteus poplitiofibular portion posterior posterolateral "
    "posteromedial posterosuperior pre prepatellar proximal quadricep quadriceps recess recesses "
    "region regional remaining retinacula retinacular retinaculum retinaculums retrofemoral "
    "retropatellar ridge right rim root s section segments semimembranosus semimenbranosus "
    "semitendinosus shaft sheaths side site slope soft soleus space spaces spine structure "
    "structures subarticular subchondral subcortical subcutaneous subcutis subpatellar sulcus "
    "superficial superior superolateral supra suprapatellar surface surfaces surrounding "
    "synchondroses synovial synovium system tendon tendons tg the thigh thighs tibia tibial "
    "tibialis tibiofemoral tibiofibular tissue tissues to tracking tract tricompartmental trochea "
    "trochear trochlea trochlear trochleas tt tubercle tuberosity under upper various vascular "
    "vastus vein veins vertex vessels vicinity visualized vkb vsm weight weightbaring wrisberg "
    "zone".split()
)

# Abbreviation expansions. The first ten were *harvested from the corpus*, which
# glosses its own abbreviations inline ("anterior cruciate ligament (ACL)") --
# including the non-English ones a hand-written list would have missed: `VKB` is
# Turkish for the ACL, `HKB` Turkish for the PCL, `LCA` Spanish/French for the
# ACL. The rest are the bare forms that occur with no gloss anywhere.
ABBREVIATIONS: dict[str, str] = {
    "acl": "anterior cruciate ligament",
    "fcl": "fibular collateral ligament",
    "hkb": "posterior cruciate ligament",
    "lca": "anterior cruciate ligament",
    "lcl": "lateral collateral ligament",
    "mcl": "medial collateral ligament",
    "mpfl": "medial patellofemoral ligament",
    "pcl": "posterior cruciate ligament",
    "vkb": "anterior cruciate ligament",
    "vsm": "vascular system",
    "ft": "femorotibial",
    "it": "iliotibial",
    "itb": "iliotibial band",
    "pf": "patellofemoral",
}

# Misspellings observed in the corpus, and the word-space repairs that keep a
# compartment name one token. Applied as plain substring replacement after
# hyphens have become spaces, so `femoro-tibial` and `femoro tibial` both land
# on `femorotibial`.
TYPO_FIXES: tuple[tuple[str, str], ...] = (
    ("femoro tibial", "femorotibial"),
    ("patello femoral", "patellofemoral"),
    ("tibio femoral", "tibiofemoral"),
    ("semimenbranosus", "semimembranosus"),
    ("trochea", "trochlea"),
    ("trochear", "trochlear"),
    ("patelofemoral", "patellofemoral"),
    ("poplitiofibular", "popliteofibular"),
    ("gastronomes", "gastrocnemius"),
    ("eminance", "eminence"),
    ("quadricep ", "quadriceps "),
    ("mesocondyle", "condyle"),
    ("ligamentous", "ligament"),
)

#: Function words and generic locators that carry no anatomical content.
STOPWORDS: frozenset[str] = frozenset(
    "the a an in at to on its with for from level part parts aspect aspects side sides region "
    "regions area areas zone zones site sites portion portions surrounding adjacent between "
    "around both visualized visualised under by such as is are structure element component "
    "vicinity surfacing of".split()
)

#: Vague quantifiers. Dropped separately from :data:`STOPWORDS` because
#: :func:`canonicalise` can be asked to keep them.
HEDGE_WORDS: frozenset[str] = frozenset("other remaining various multiple general regional rest".split())

#: Irregular plurals that suffix stripping cannot reach.
LATIN_PLURALS: dict[str, str] = {
    "menisci": "meniscus",
    "retinacula": "retinaculum",
    "retinaculums": "retinaculum",
    "plicae": "plica",
    "fascicles": "fascicle",
    "eminences": "eminence",
    "condyles": "condyle",
    "epicondyles": "epicondyle",
    "structures": "structure",
    "elements": "element",
    "components": "component",
    "bursas": "bursa",
    "articulations": "articulation",
    "planes": "plane",
    "thighs": "thigh",
    "legs": "leg",
    "facets": "facet",
}

# Singular words that merely end in `s`. Stripping it would silently rename a
# structure: `quadriceps` to `quadricep`, `biceps` to `bicep`, `pes anserinus`
# to `pe anserinu`.
PROTECTED_PLURALS: frozenset[str] = frozenset(
    "quadriceps biceps femoris pes anserinus osseous synchondroses hoffa baker".split()
)

#: Laterality words, dropped by :func:`canonicalise`: the study images one knee,
#: so "right" tells you nothing about *where* in that knee.
SIDE_WORDS: frozenset[str] = frozenset({"right", "left", "bilateral"})

#: English synonyms for medial and lateral. Folded, or `outer compartment`
#: silently loses its side and lands in the unspecified bucket.
INNER_OUTER: dict[str, str] = {
    "inner": "medial",
    "internal": "medial",
    "outer": "lateral",
    "external": "lateral",
}

# Head nouns strong enough to stand alone as a site. `split_components` uses
# these to decide whether a comma-separated fragment already names a structure
# or needs to borrow the shared head noun from a neighbour.
STRONG_HEADS: frozenset[str] = frozenset(
    "meniscus meniscal ligament joint articulation cartilage patella tendon space condyle marrow "
    "tibia bone osseous bony retinaculum retinacular plateau compartment mechanism apparatus pad "
    "trochlea tissue bursa synovium synovial band tract cyst femur muscle musculature cavity "
    "fabella capsule plica metaphysis epiphysis diaphysis shaft apophysis tubercle tuberosity "
    "fibula nerve artery vein vessel vessels subcutis fat epicondyle aponeurosis sheath membrane "
    "fluid effusion alignment tracking distance fracture fragment recess fascicle gastrocnemius "
    "semimembranosus semitendinosus popliteus anserinus anserine soleus quadriceps biceps "
    "iliotibial hamstring thigh leg peroneal vascular neurovascular cartilaginous chondral "
    "osteochondral cellular adipose fatty meniscocapsular parameniscal knee bodies interface "
    "subcutaneous cortex".split()
)

# Sub-parts that name a location *within* a structure. Not a site on their own
# ("posterior horn" is not a site), but enough to donate a head noun to a
# fragment that has none.
WEAK_HEADS: frozenset[str] = frozenset(
    "horn root body segment facet complex head attachment insertion surface margin fossa notch "
    "groove plane bundle corner eminence spine apex pole ridge trochlear patellar femoral tibial "
    "articular bursal".split()
)

# Modifier classes. When a fragment borrows a head noun from a neighbour it must
# not borrow the neighbour's *modifiers* of a class it already carries -- in
# `medial and lateral meniscus` the `lateral` fragment inherits `meniscus` but
# must not also inherit `medial`, or both fragments become MENISCUS_BOTH.
_LATERALITY_WORDS: frozenset[str] = frozenset(
    "medial lateral inner outer internal external medially laterally fibular tibial femoral".split()
)
_POSITION_WORDS: frozenset[str] = frozenset(
    "anterior posterior proximal distal superior inferior posteromedial posterolateral "
    "anteromedial anterolateral posterosuperior superolateral central deep superficial mid".split()
)
_NAMED_STRUCTURE_WORDS: frozenset[str] = frozenset(
    "quadriceps patellar patella popliteus biceps femoris semimembranosus semitendinosus "
    "gastrocnemius anserinus anserine iliotibial hamstring soleus cruciate collateral".split()
)
_COMPARTMENT_WORDS: frozenset[str] = frozenset(
    "femorotibial tibiofemoral patellofemoral femoropatellar retropatellar tibiofibular".split()
)
MODIFIER_CLASSES: tuple[frozenset[str], ...] = (
    _LATERALITY_WORDS,
    _POSITION_WORDS,
    _NAMED_STRUCTURE_WORDS,
    _COMPARTMENT_WORDS,
)

#: Every cluster :func:`clusters_of` can return, including the ``ZZ_`` residual
#: buckets and the two synthetic names that only appear when a fold is switched
#: off. Kept explicit so ``kappa.labels.label_map`` can be checked for gaps:
#: an unmapped cluster contributes no labels and does so silently.
CLUSTER_NAMES: frozenset[str] = frozenset(
    {
        "ACL",
        "BONE_CONDYLE_PLATEAU_UNSPEC",
        "BONE_FIBULA_FABELLA",
        "BONE_INTERCONDYLAR",
        "BONE_LATERAL_FEMORAL_CONDYLE",
        "BONE_LATERAL_TIBIAL_PLATEAU",
        "BONE_MARROW_LATERAL",
        "BONE_MARROW_MEDIAL",
        "BONE_MARROW_UNSPECIFIED",
        "BONE_MEDIAL_FEMORAL_CONDYLE",
        "BONE_MEDIAL_TIBIAL_PLATEAU",
        "BONE_METAPHYSIS_EPIPHYSIS",
        "BONE_SUBCHONDRAL",
        "BONE_TIBIAL_TUBEROSITY",
        "BONE_TROCHLEA",
        "BONE_UNSPECIFIED",
        "BURSA",
        "CARTILAGE_FULLY_UNSPECIFIED",
        "CARTILAGE_LATERAL_TIBIOFEMORAL",
        "CARTILAGE_MEDIAL_TIBIOFEMORAL",
        "CARTILAGE_MULTICOMPARTMENT",
        "CARTILAGE_PATELLOFEMORAL",
        "CARTILAGE_TIBIOFEMORAL_SIDE_UNSPEC",
        "COLLATERAL_BOTH",
        "COLLATERAL_SIDE_UNSPECIFIED",
        "COMPARTMENT_LATERAL",
        "COMPARTMENT_MEDIAL",
        "COMPARTMENT_PATELLOFEMORAL",
        "COMPARTMENT_UNSPECIFIED",
        "CRUCIATE_BOTH",
        "CRUCIATE_UNSPECIFIED",
        "CYST_POPLITEAL_BAKER",
        "EXTENSOR_MECHANISM",
        "FAT_PAD_HOFFA_INFRAPATELLAR",
        "FAT_PAD_OTHER",
        "GASTROCNEMIUS_SEMIMEMBRANOSUS",
        "ILIOTIBIAL_BAND",
        "JOINT_CAPSULE",
        "JOINT_LATERAL_TIBIOFEMORAL",
        "JOINT_MARGIN_ambiguous",
        "JOINT_MEDIAL_TIBIOFEMORAL",
        "JOINT_PATELLOFEMORAL",
        "JOINT_SPACE_EFFUSION",
        "JOINT_TIBIOFIBULAR",
        "LCL",
        "LIGAMENT_PATELLAR_UNFOLDED",
        "LIGAMENT_UNSPECIFIED",
        "MCL",
        "MENISCOFEMORAL_LIGAMENT",
        "MENISCUS_BOTH",
        "MENISCUS_LATERAL",
        "MENISCUS_MEDIAL",
        "MENISCUS_SIDE_UNSPECIFIED",
        "MUSCLE",
        "NEUROVASCULAR",
        "OTHER",
        "PARAMENISCAL_REGION",
        "PATELLA",
        "PATELLAR_RETINACULUM",
        "PATELLOFEMORAL_LIGAMENT_MPFL",
        "PCL",
        "POPLITEAL_FOSSA",
        "POSTEROLATERAL_CORNER",
        "POSTEROMEDIAL_OBLIQUE_LIGAMENT",
        "SOFT_TISSUE",
        "SYNOVIUM_PLICA",
        "TENDON_BICEPS_FEMORIS",
        "TENDON_PATELLAR",
        "TENDON_PES_ANSERINUS_HAMSTRING",
        "TENDON_POPLITEUS",
        "TENDON_QUADRICEPS",
        "TENDON_QUAD_AND_PATELLAR",
        "TENDON_UNSPECIFIED",
        "ZZ_NONANAT_finding_in_anatomy_field",
        "ZZ_NONANAT_measurement",
        "ZZ_OUTSIDE_KNEE",
        "ZZ_UNMAPPABLE_vague",
        "ZZ_WHOLE_KNEE_nonlocalising",
    }
)

_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.;:]+$")
_POSSESSIVE_RE = re.compile(r"'s\b")
_TT_TG_RE = re.compile(r"\btt tg\b")
_PARENTHETICAL_RE = re.compile(r"\(([^)]*)\)")
_PARENS_RE = re.compile(r"[()]")
_SHORT_WORD_RE = re.compile(r"\b([a-z]{2,5})\b")
_COMPONENT_SPLIT_RE = re.compile(r",| and | or |/")


def tokens(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    NFKC first: the corpus contains full-width and ligature forms that would
    otherwise survive the non-word strip as separate tokens.
    """
    return _NON_WORD_RE.sub(" ", unicodedata.normalize("NFKC", text).lower()).split()


def preprocess(anatomy: str) -> str:
    """Normalise one raw anatomy string into a flat lowercase phrase.

    Expands abbreviations, drops parenthetical glosses, repairs known typos and
    removes possessives, so that the many surface forms of one site collapse
    before any clustering decision is made.
    """
    text = _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", anatomy)).strip().lower()
    text = text.replace("’", "'")
    text = _POSSESSIVE_RE.sub("", text)
    text = text.replace("&", " and ")
    text = _TRAILING_PUNCT_RE.sub("", text)
    text = text.replace("-", " ")
    # Before the parentheses are stripped: `TT-TG` is a measurement, and its
    # expansion is what routes it to the non-anatomical bucket.
    text = _TT_TG_RE.sub("tibial tubercle trochlear groove", text)
    # A parenthetical is a gloss ("anterior cruciate ligament (ACL)") and its
    # content is redundant -- unless dropping it empties the string, which
    # happens when the whole anatomy field is parenthesised.
    without_parens = _PARENTHETICAL_RE.sub(" ", text)
    text = without_parens if without_parens.strip() else _PARENS_RE.sub(" ", text)
    text = _SHORT_WORD_RE.sub(lambda m: ABBREVIATIONS.get(m.group(0), m.group(0)), text)
    for wrong, right in TYPO_FIXES:
        text = text.replace(wrong, right)
    return _WHITESPACE_RE.sub(" ", text).strip()


def singularise(token: str) -> str:
    """Fold a plural token onto its singular, conservatively.

    Suffix stripping only fires when the result is a token the corpus actually
    uses, which is what stops `pes` becoming `pe` and `osseous` becoming
    `osseou`.
    """
    if token in PROTECTED_PLURALS:
        return token
    if token in LATIN_PLURALS:
        return LATIN_PLURALS[token]
    if token.endswith("es") and token[:-2] in INDUCED_VOCABULARY:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and token[:-1] in INDUCED_VOCABULARY:
        return token[:-1]
    return token


def split_components(phrase: str) -> list[list[str]]:
    """Split a preprocessed phrase into one token bag per anatomical site.

    Two shapes occur, and they split differently. ``cartilage of the medial and
    lateral compartments`` shares its head noun *before* the sites, so the head
    is distributed over them; ``medial meniscus, ACL`` is a flat list. After
    splitting, a fragment with no head noun of its own borrows one from a
    neighbour -- searching forward first, because English puts the shared head
    last ("medial and lateral meniscus").
    """
    if " of " in phrase:
        head, _, site = phrase.partition(" of ")
        parts = [x.strip() for x in _COMPONENT_SPLIT_RE.split(site) if x.strip()]
        base = [f"{head} {x}" for x in parts] if len(parts) > 1 else [phrase]
    else:
        base = [x.strip() for x in _COMPONENT_SPLIT_RE.split(phrase) if x.strip()]
        if not base:
            base = [phrase]
    bags = [[singularise(t) for t in x.split()] for x in base]
    if len(bags) == 1:
        return bags

    def has_strong_head(bag: list[str]) -> bool:
        return any(t in STRONG_HEADS for t in bag)

    def has_any_head(bag: list[str]) -> bool:
        return any(t in STRONG_HEADS or t in WEAK_HEADS for t in bag)

    for index, bag in enumerate(bags):
        if has_strong_head(bag):
            continue
        donor = next(
            (bags[j] for j in range(index + 1, len(bags)) if has_any_head(bags[j]) and bags[j] != bag),
            None,
        )
        if donor is None:
            donor = next(
                (bags[j] for j in range(index - 1, -1, -1) if has_any_head(bags[j]) and bags[j] != bag),
                None,
            )
        if donor is None:
            continue
        # Inherit the donor's head noun but none of its modifiers in a class
        # this bag already fills, or `medial and lateral meniscus` collapses.
        own_classes = [c for c in MODIFIER_CLASSES if any(t in c for t in bag)]
        inherited = [t for t in donor if not any(t in c for c in own_classes)]
        bags[index] = bag + inherited
    return bags


def canonicalise(bag: list[str], *, inout: bool = True, hedge: bool = True) -> frozenset[str]:
    """Reduce one token bag to the set of tokens that can decide a cluster.

    A set, not a list: after stopword removal the cascade only ever asks whether
    a token is present, and duplicates from head-noun inheritance would
    otherwise be indistinguishable from emphasis.
    """
    kept = [t for t in bag if t not in STOPWORDS]
    if hedge:
        kept = [t for t in kept if t not in HEDGE_WORDS]
    kept = [t for t in kept if t not in SIDE_WORDS]
    if inout:
        kept = [INNER_OUTER.get(t, t) for t in kept]
    return frozenset(kept)


def cluster_of_bag(bag: frozenset[str]) -> str:
    """Assign one canonicalised token bag to exactly one cluster.

    **The order of these tests is the algorithm.** Every branch is written to
    fire on a bag that an earlier branch has already declined, so the cascade
    reads as a priority list: non-anatomical content is removed first, then the
    structures whose identity is unambiguous (meniscus, cruciate, collateral),
    then the tissue-type questions where a site alone is ambiguous (cartilage,
    marrow, bone), then the containers (joint space, compartment, bursa), and
    only then the generic fallbacks. Reordering any pair changes which cluster a
    real string lands in -- `ligament` before `cartilage` is why `patellofemoral
    ligament` is the MPFL and not patellofemoral cartilage. Do not rewrite this
    as a lookup table or a rule list sorted by anything but this order.
    """

    def has(*candidates: str) -> bool:
        return any(t in bag for t in candidates)

    medial = has("medial", "medially")
    lateral = has("lateral", "laterally")
    anterior = has("anterior", "anteromedial", "anterolateral")
    posterior = has("posterior", "posteromedial", "posterolateral", "posterosuperior")
    patellofemoral = has(
        "patellofemoral", "femoropatellar", "retropatellar", "trochlea", "trochlear",
        "patella", "patellar", "ridge",
    )

    # Non-anatomical content that arrived in the anatomy field. Removed first
    # because these strings otherwise match a structure and invent a finding on
    # it: `TT-TG distance` names the trochlea and is a measurement.
    if has("alignment", "tracking", "distance") or ("tubercle" in bag and "groove" in bag):
        return "ZZ_NONANAT_measurement"
    if has("fracture", "fragment", "bodies", "effusion", "fluid") and not has(
        "bursa", "space", "cyst", "cavity", "recess"
    ):
        return "ZZ_NONANAT_finding_in_anatomy_field"
    if (
        has(
            "perilesional", "retrofemoral", "perigonal", "peritrochanteric", "subpatellar",
            "osteochondral", "intermuscular",
        )
        and len(bag) <= 3
        and not has("tissue", "plane", "muscle")
    ):
        return "ZZ_UNMAPPABLE_vague"
    if bag <= frozenset({"anterior", "posterior", "articular", "medial", "lateral"}):
        return "ZZ_UNMAPPABLE_vague"
    if has("thigh", "leg", "peritrochanteric", "acromial", "tibialis") and not has(
        "meniscus", "ligament", "cartilage", "patella"
    ):
        return "ZZ_OUTSIDE_KNEE"
    if (
        bag
        and bag
        <= frozenset(
            {"knee", "anterior", "posterior", "medial", "lateral", "superior", "inferior",
             "joint", "alignment"}
        )
        and "joint" not in bag
    ):
        return "ZZ_WHOLE_KNEE_nonlocalising"
    if not bag:
        return "ZZ_UNMAPPABLE_vague"

    if has("meniscus", "meniscal"):
        if medial and lateral:
            return "MENISCUS_BOTH"
        if medial:
            return "MENISCUS_MEDIAL"
        if lateral:
            return "MENISCUS_LATERAL"
        return "MENISCUS_SIDE_UNSPECIFIED"
    if "cruciate" in bag:
        if anterior and posterior:
            return "CRUCIATE_BOTH"
        if anterior:
            return "ACL"
        if posterior:
            return "PCL"
        return "CRUCIATE_UNSPECIFIED"
    if "collateral" in bag:
        # `fibular collateral ligament` is the LCL under another name, which is
        # what `fold_fibular` in `clusters_of` can switch off.
        if medial and (lateral or "fibular" in bag):
            return "COLLATERAL_BOTH"
        if medial:
            return "MCL"
        if lateral or "fibular" in bag:
            return "LCL"
        return "COLLATERAL_SIDE_UNSPECIFIED"
    if has("popliteofibular", "meniscopopliteal", "fascicle") or (
        "corner" in bag and (posterior or lateral)
    ):
        return "POSTEROLATERAL_CORNER"
    if has("retinaculum", "retinacular"):
        return "PATELLAR_RETINACULUM"
    if "ligament" in bag and has("patellofemoral", "femoropatellar") and not has("cartilage"):
        return "PATELLOFEMORAL_LIGAMENT_MPFL"
    if "ligament" in bag and has("patellar", "patella") and not has("cartilage", "retinaculum"):
        # The patellar *ligament* and the patellar *tendon* are the same
        # structure; `fold_patlig` in `clusters_of` can keep them apart.
        return "TENDON_PATELLAR"
    if "ligament" in bag and has("meniscofemoral", "humphrey", "wrisberg"):
        return "MENISCOFEMORAL_LIGAMENT"
    if "ligament" in bag and "oblique" in bag:
        return "POSTEROMEDIAL_OBLIQUE_LIGAMENT"
    if "ligament" in bag and not has("cartilage", "tendon", "bone", "marrow"):
        return "LIGAMENT_UNSPECIFIED"

    if has("cartilage", "cartilaginous", "chondral"):
        tibiofemoral = has(
            "femorotibial", "tibiofemoral", "femoral", "tibial", "femur", "tibia", "condyle",
            "plateau", "compartment",
        )
        if (
            patellofemoral
            and (tibiofemoral or medial or lateral)
            and (
                medial
                or lateral
                or has("femorotibial", "tibiofemoral", "femur", "tibia", "condyle", "plateau")
            )
        ):
            return "CARTILAGE_MULTICOMPARTMENT"
        if patellofemoral:
            return "CARTILAGE_PATELLOFEMORAL"
        if medial and lateral:
            return "CARTILAGE_MULTICOMPARTMENT"
        if medial:
            return "CARTILAGE_MEDIAL_TIBIOFEMORAL"
        if lateral:
            return "CARTILAGE_LATERAL_TIBIOFEMORAL"
        if tibiofemoral:
            return "CARTILAGE_TIBIOFEMORAL_SIDE_UNSPEC"
        return "CARTILAGE_FULLY_UNSPECIFIED"
    if (
        "articular" in bag
        and has("surface", "interface")
        and not has("joint", "space", "bone", "cartilage", "patella", "condyle", "plateau")
    ):
        return "CARTILAGE_FULLY_UNSPECIFIED"
    if "margin" in bag:
        return "JOINT_MARGIN_ambiguous"

    if "marrow" in bag:
        if medial and not lateral:
            return "BONE_MARROW_MEDIAL"
        if lateral and not medial:
            return "BONE_MARROW_LATERAL"
        return "BONE_MARROW_UNSPECIFIED"
    if has("subchondral", "subcortical", "subarticular"):
        return "BONE_SUBCHONDRAL"
    if has("condyle", "epicondyle", "plateau"):
        if medial and not lateral and has("femoral", "femur"):
            return "BONE_MEDIAL_FEMORAL_CONDYLE"
        if lateral and not medial and has("femoral", "femur"):
            return "BONE_LATERAL_FEMORAL_CONDYLE"
        if medial and not lateral and has("tibial", "tibia"):
            return "BONE_MEDIAL_TIBIAL_PLATEAU"
        if lateral and not medial and has("tibial", "tibia"):
            return "BONE_LATERAL_TIBIAL_PLATEAU"
        return "BONE_CONDYLE_PLATEAU_UNSPEC"
    if has("eminence", "spine", "notch", "intercondylar") or (
        "fossa" in bag and "popliteal" not in bag
    ):
        return "BONE_INTERCONDYLAR"
    if has("tubercle", "tuberosity", "apophysis"):
        return "BONE_TIBIAL_TUBEROSITY"
    if has("fibula", "fabella") or ("head" in bag and "fibular" in bag):
        return "BONE_FIBULA_FABELLA"
    if has("metaphysis", "epiphysis", "diaphysis", "shaft", "synchondroses", "cortex"):
        return "BONE_METAPHYSIS_EPIPHYSIS"
    if has("trochlea", "trochlear", "groove") and not has("tendon", "cartilage"):
        return "BONE_TROCHLEA"
    if has("bone", "osseous", "bony"):
        return "BONE_UNSPECIFIED"
    if has("patella", "patellar", "facet", "ridge") and not has(
        "tendon", "bursa", "fat", "fatty", "adipose", "bursal"
    ):
        return "PATELLA"

    if has("mechanism", "apparatus") or ("extensor" in bag and not has("tendon")):
        return "EXTENSOR_MECHANISM"
    if has("tendon", "aponeurosis", "sheath", "insertion", "attachment", "extensor") or has(
        "quadriceps", "femoris", "biceps", "anserinus", "anserine", "popliteus",
        "semimembranosus", "semitendinosus", "iliotibial", "hamstring", "band", "tract",
        "gastrocnemius", "soleus",
    ):
        if "extensor" in bag:
            return "EXTENSOR_MECHANISM"
        if "quadriceps" in bag and has("patellar", "patella"):
            return "TENDON_QUAD_AND_PATELLAR"
        if "quadriceps" in bag:
            return "TENDON_QUADRICEPS"
        if has("patellar", "patella"):
            return "TENDON_PATELLAR"
        if "popliteus" in bag:
            return "TENDON_POPLITEUS"
        if has("biceps", "femoris"):
            return "TENDON_BICEPS_FEMORIS"
        if has("anserinus", "anserine", "semimembranosus", "semitendinosus"):
            return "TENDON_PES_ANSERINUS_HAMSTRING"
        if has("iliotibial", "band", "tract"):
            return "ILIOTIBIAL_BAND"
        if has("gastrocnemius", "soleus"):
            return "GASTROCNEMIUS_SEMIMEMBRANOSUS"
        return "TENDON_UNSPECIFIED"

    if "cyst" in bag:
        return "CYST_POPLITEAL_BAKER"
    if has("bursa", "bursal"):
        return "BURSA"
    if has("synovium", "synovial", "plica", "membrane"):
        return "SYNOVIUM_PLICA"
    if "capsule" in bag:
        return "JOINT_CAPSULE"
    if "popliteal" in bag and not has(
        "artery", "vein", "nerve", "vessel", "vessels", "neurovascular", "bundle"
    ):
        return "POPLITEAL_FOSSA"
    if has("space", "recess", "cavity", "surface", "interface", "joint", "articulation"):
        if has("tibiofibular"):
            return "JOINT_TIBIOFIBULAR"
        if patellofemoral and not (medial or lateral) and not has("femorotibial", "tibiofemoral"):
            return "JOINT_PATELLOFEMORAL"
        if has("femorotibial", "tibiofemoral") and medial and not lateral:
            return "JOINT_MEDIAL_TIBIOFEMORAL"
        if has("femorotibial", "tibiofemoral") and lateral and not medial:
            return "JOINT_LATERAL_TIBIOFEMORAL"
        return "JOINT_SPACE_EFFUSION"
    if "compartment" in bag:
        # The patellofemoral test comes first, and it is guarded exactly as
        # `JOINT_PATELLOFEMORAL` above is, so the two stay consistent. Without
        # it a bag reading `patellofemoral compartment` fell to
        # COMPARTMENT_UNSPECIFIED, which the label map calls *ambiguous* for
        # all three OA labels -- so under polarity-conditioned fan-out a
        # positive finding was masked instead of setting PF OA. Measured over
        # the survey's anatomy table: 5 distinct strings, 10 findings.
        #
        # A bag carrying BOTH a patellofemoral token and a side (or a
        # tibiofemoral compartment name) deliberately falls through to the
        # medial/lateral tests below and keeps its previous cluster. Such a bag
        # is a compound the splitter failed to separate, and inventing a
        # combined cluster for it would be a second, unmeasured change.
        if patellofemoral and not (medial or lateral) and not has("femorotibial", "tibiofemoral"):
            return "COMPARTMENT_PATELLOFEMORAL"
        if medial and not lateral:
            return "COMPARTMENT_MEDIAL"
        if lateral and not medial:
            return "COMPARTMENT_LATERAL"
        return "COMPARTMENT_UNSPECIFIED"
    if has("fat", "pad", "fatty", "adipose"):
        if has("hoffa", "infrapatellar"):
            return "FAT_PAD_HOFFA_INFRAPATELLAR"
        return "FAT_PAD_OTHER"
    if has("neurovascular", "nerve", "artery", "vein", "vessel", "vessels", "vascular"):
        return "NEUROVASCULAR"
    if has("muscle", "musculature", "muscular"):
        return "MUSCLE"
    if has("tissue", "subcutis", "subcutaneous", "cellular", "plane", "connective"):
        return "SOFT_TISSUE"
    if has("parameniscal", "meniscocapsular"):
        return "PARAMENISCAL_REGION"
    if has("femur", "tibia", "femoral", "tibial"):
        return "BONE_UNSPECIFIED"
    if has("prepatellar", "suprapatellar", "infrapatellar"):
        return "SOFT_TISSUE"
    if "complex" in bag:
        return "ZZ_UNMAPPABLE_vague"
    return "OTHER"


def clusters_of(
    anatomy: str,
    *,
    inout: bool = True,
    fold_fibular: bool = True,
    fold_patlig: bool = True,
    expand_extensor: bool = False,
) -> frozenset[str]:
    """Cluster one raw anatomy string into zero or more canonical clusters.

    More than one cluster comes back whenever the string names more than one
    site, which is the common case for the phrases that decide a side-specific
    label.

    The four flags are ablation switches over synonym folds that are judgement
    calls, not corpus facts, so a run can measure what each fold is worth
    instead of assuming:

    ``inout``
        Fold `inner`/`outer` onto medial/lateral. Off, they lose their side.
    ``fold_fibular``
        Treat `fibular collateral ligament` as the LCL. Off, it becomes
        ``COLLATERAL_SIDE_UNSPECIFIED``.
    ``fold_patlig``
        Treat the patellar *ligament* as the patellar tendon. Off, it becomes
        ``LIGAMENT_PATELLAR_UNFOLDED``, which the label map maps to nothing --
        that entry exists only to serve this flag.
    ``expand_extensor``
        Replace ``EXTENSOR_MECHANISM`` with its three members. On, one finding
        on "the extensor mechanism" becomes direct evidence about the
        quadriceps tendon, the patellar tendon and the patella.
    """
    clusters: list[str] = []
    for bag in split_components(preprocess(anatomy)):
        canonical = canonicalise(bag, inout=inout)
        cluster = cluster_of_bag(canonical)
        if (
            not fold_fibular
            and cluster == "LCL"
            and "fibular" in canonical
            and not ("lateral" in canonical or "laterally" in canonical)
        ):
            cluster = "COLLATERAL_SIDE_UNSPECIFIED"
        if (
            not fold_patlig
            and cluster == "TENDON_PATELLAR"
            and "ligament" in canonical
            and "tendon" not in canonical
        ):
            cluster = "LIGAMENT_PATELLAR_UNFOLDED"
        if expand_extensor and cluster == "EXTENSOR_MECHANISM":
            clusters.extend(["TENDON_QUADRICEPS", "TENDON_PATELLAR", "PATELLA"])
            continue
        clusters.append(cluster)
    return frozenset(clusters)
