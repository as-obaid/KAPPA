"""The port of the clusterer must differ from the original in five known ways only.

``kappa.labels.anatomy_map`` is a port of ``_recovered/cluster4.py``, the only
copy of the code behind the measured macro balanced accuracy of 0.765 on the 58
anchors. A port is worth nothing unless the drift is bounded, and "mostly
agrees" is not a bound: a single reordered branch in the cascade moves real
strings between clusters, and the label map turns some of those moves into a
different label. So this test runs both implementations over **every** anatomy
string the corpus contains and pins the difference to an enumerated list.

Exactly five strings and ten findings are allowed to move, and they are named
below. They are the patellofemoral-compartment fix: ``patellofemoral
compartment`` used to fall to ``COMPARTMENT_UNSPECIFIED``, which the label map
calls ambiguous for all three OA labels, so under polarity-conditioned fan-out a
positive finding on it was masked instead of setting PF OA. Anything else that
moves is a regression, and this test is what says so.

This is the **one sanctioned place that loads from ``_recovered/``**. Nothing in
``src/kappa/`` may import from that directory; it is a read-only reference, and
the reference is only useful if something compares against it. The directory is
derived from competition data and is not published, so every test that reads it
skips when it is absent; the unit-level companion in ``tests/test_anatomy_map.py``
still runs.
"""

from __future__ import annotations

import collections
import csv
import functools
import os
import re
from pathlib import Path

import pytest

from kappa.labels.anatomy_map import (
    ABBREVIATIONS,
    CLUSTER_NAMES,
    INDUCED_VOCABULARY,
    clusters_of,
    tokens,
)

RECOVERED = Path(__file__).resolve().parents[1] / "src" / "kappa" / "labels" / "_recovered"

#: The original clusterer and the corpus it reads. Neither is published, so the tests
#: that compare against them skip in a public checkout instead of failing.
needs_recovered = pytest.mark.skipif(
    not ((RECOVERED / "cluster4.py").is_file() and (RECOVERED / "anatomy_raw.tsv").is_file()),
    reason="src/kappa/labels/_recovered/ (original clusterer and anatomy corpus) is not published",
)

#: Measured input size. Asserted rather than assumed: the differential means
#: nothing if the corpus file it reads has been truncated.
N_DISTINCT_STRINGS = 1010
N_FINDINGS = 3962

# The complete, deliberate difference between the original and the port: raw
# anatomy string -> (findings, original clusters, ported clusters). Written out
# in full so that a sixth difference, or a different sixth cluster, fails.
KNOWN_DIFFERENCES: dict[str, tuple[int, frozenset[str], frozenset[str]]] = {
    "patellofemoral compartment": (
        6,
        frozenset({"COMPARTMENT_UNSPECIFIED"}),
        frozenset({"COMPARTMENT_PATELLOFEMORAL"}),
    ),
    # The corpus misspells it, and the typo variant has to move with the
    # correctly spelled one or the fix covers only part of the evidence.
    "patelofemoral compartment": (
        1,
        frozenset({"COMPARTMENT_UNSPECIFIED"}),
        frozenset({"COMPARTMENT_PATELLOFEMORAL"}),
    ),
    "lateral and patellofemoral compartments": (
        1,
        frozenset({"COMPARTMENT_LATERAL", "COMPARTMENT_UNSPECIFIED"}),
        frozenset({"COMPARTMENT_LATERAL", "COMPARTMENT_PATELLOFEMORAL"}),
    ),
    "medial tibiofemoral and patellofemoral compartments": (
        1,
        frozenset({"COMPARTMENT_MEDIAL", "COMPARTMENT_UNSPECIFIED"}),
        frozenset({"COMPARTMENT_MEDIAL", "COMPARTMENT_PATELLOFEMORAL"}),
    ),
    "medial femoral tibial and patellofemoral compartments": (
        1,
        frozenset({"COMPARTMENT_MEDIAL", "COMPARTMENT_UNSPECIFIED"}),
        frozenset({"COMPARTMENT_MEDIAL", "COMPARTMENT_PATELLOFEMORAL"}),
    ),
}

N_CHANGED_STRINGS = 5
N_CHANGED_FINDINGS = 10


@functools.lru_cache(maxsize=1)
def _original() -> dict[str, object]:
    """Load ``cluster4.py``'s induction functions without running its main loop.

    Replicates ``_recovered/mapper.py`` exactly, including how it isolates the
    functions: the file is one script that induces its vocabulary, defines the
    clusterer and then prints a report, so mapper.py splits the source on the
    literal first line of the report loop and executes only the prefix. The
    ``chdir`` is needed because ``cluster4.py`` opens ``anatomy_raw.tsv`` by
    relative path, and it is undone whatever happens so no other test inherits
    a moved working directory.
    """
    source = (RECOVERED / "cluster4.py").read_text(encoding="utf-8")
    prefix, marker, _ = source.partition("cl=collections.defaultdict")
    assert marker, "cluster4.py no longer contains the split marker mapper.py relies on"
    namespace: dict[str, object] = {}
    previous = os.getcwd()
    os.chdir(RECOVERED)
    try:
        exec(compile(prefix, "cluster4.py", "exec"), namespace)  # noqa: S102
    finally:
        os.chdir(previous)
    return namespace


def _original_clusters_of(
    anatomy: str,
    *,
    inout: bool = True,
    fold_fibular: bool = True,
    fold_patlig: bool = True,
    expand_extensor: bool = False,
) -> frozenset[str]:
    """``_recovered/mapper.py``'s ``clusters_of``, verbatim in behaviour."""
    namespace = _original()
    pre, split_components, canon, cluster = (
        namespace["pre"], namespace["split_components"], namespace["canon"], namespace["C"]
    )
    out: list[str] = []
    for bag in split_components(pre(anatomy)):
        canonical = canon(bag, inout=inout)
        name = cluster(canonical)
        if (
            not fold_fibular
            and name == "LCL"
            and "fibular" in canonical
            and not ("lateral" in canonical or "laterally" in canonical)
        ):
            name = "COLLATERAL_SIDE_UNSPECIFIED"
        if (
            not fold_patlig
            and name == "TENDON_PATELLAR"
            and "ligament" in canonical
            and "tendon" not in canonical
        ):
            name = "LIGAMENT_PATELLAR_UNFOLDED"
        if expand_extensor and name == "EXTENSOR_MECHANISM":
            out.extend(["TENDON_QUADRICEPS", "TENDON_PATELLAR", "PATELLA"])
            continue
        out.append(name)
    return frozenset(out)


@functools.lru_cache(maxsize=1)
def _corpus() -> tuple[tuple[str, int], ...]:
    """Every distinct raw anatomy string in the survey, with its finding count."""
    with open(RECOVERED / "anatomy_raw.tsv", encoding="utf-8") as fh:
        return tuple(
            (row["anatomy"], int(row["n"])) for row in csv.DictReader(fh, delimiter="\t")
        )


@needs_recovered
class TestTheCorpusUnderTest:
    def test_the_input_is_the_measured_corpus(self):
        corpus = _corpus()
        assert len(corpus) == N_DISTINCT_STRINGS
        assert sum(n for _, n in corpus) == N_FINDINGS

    def test_every_anatomy_string_is_ascii(self):
        # Measured: 0 of 1,010 anatomy strings carry a non-ASCII codepoint, so
        # the clusterer's Latin-only tokeniser is sound *for this model's
        # output*. It is not a guarantee about any other model -- see
        # `tests/test_anatomy_map.py::TestLookalikeFolding`.
        offenders = [a for a, _ in _corpus() if not a.isascii()]
        assert offenders == []


@needs_recovered
class TestInducedConstantsAreTheCorpusOnes:
    """The port freezes what the original induced at import. Prove they match.

    The frozen copies are what stop the depluralisation in ``singularise``
    depending on which findings happen to be in the batch. Frozen and *wrong*
    would be worse than recomputed, so the constants are re-derived here from
    the same corpus file and compared.
    """

    def test_induced_vocabulary_matches_the_corpus_token_set(self):
        assert set(_original()["CT"]) == set(INDUCED_VOCABULARY)

    def test_tokens_is_the_function_that_induced_the_vocabulary(self):
        # `tokens` has no caller inside `anatomy_map` -- `preprocess` does its
        # own regex work -- so without this it looks like decoration and the
        # next reader deletes it. It is the induction function, and the frozen
        # constant is exactly its output over the corpus.
        assert {t for anatomy, _ in _corpus() for t in tokens(anatomy)} == set(
            INDUCED_VOCABULARY
        )

    def test_abbreviation_table_matches_the_corpus_glosses(self):
        assert _original()["ABBREV"] == ABBREVIATIONS

    def test_the_corpus_glosses_the_non_english_abbreviations_itself(self):
        # VKB is Turkish for the ACL, HKB Turkish for the PCL, LCA
        # Spanish/French for the ACL. None was hand-written: each was harvested
        # from a report that glossed it inline, which a hand-written English
        # list would have missed.
        harvested = {
            abbreviation
            for row in _corpus()
            for abbreviation in re.findall(
                r"[A-Za-z][A-Za-z \-]{4,}?\s*\(([A-Z][A-Z\-]{1,5})\)", row[0]
            )
        }
        assert {"VKB", "HKB", "LCA"} <= harvested


@needs_recovered
class TestDifferentialOverTheWholeCorpus:
    def test_only_the_enumerated_strings_change_cluster(self):
        changed = {
            anatomy: (n, _original_clusters_of(anatomy), clusters_of(anatomy))
            for anatomy, n in _corpus()
            if _original_clusters_of(anatomy) != clusters_of(anatomy)
        }
        assert set(changed) == set(KNOWN_DIFFERENCES), (
            f"unexpected drift: {sorted(set(changed) - set(KNOWN_DIFFERENCES))}; "
            f"no longer drifting: {sorted(set(KNOWN_DIFFERENCES) - set(changed))}"
        )
        assert changed == KNOWN_DIFFERENCES

    def test_exactly_five_strings_and_ten_findings_change(self):
        changed = [
            (anatomy, n)
            for anatomy, n in _corpus()
            if _original_clusters_of(anatomy) != clusters_of(anatomy)
        ]
        assert len(changed) == N_CHANGED_STRINGS
        assert sum(n for _, n in changed) == N_CHANGED_FINDINGS

    def test_every_difference_is_the_patellofemoral_compartment_fix(self):
        # The shape of the intended change, asserted independently of the
        # enumeration above: one cluster moves, from the unspecified
        # compartment to the patellofemoral one, and nothing else does.
        for anatomy, (_, old, new) in KNOWN_DIFFERENCES.items():
            assert old - new == {"COMPARTMENT_UNSPECIFIED"}, anatomy
            assert new - old == {"COMPARTMENT_PATELLOFEMORAL"}, anatomy

    @pytest.mark.parametrize(
        "flags",
        [
            {"inout": False},
            {"fold_fibular": False},
            {"fold_patlig": False},
            {"expand_extensor": True},
            {"inout": False, "fold_fibular": False, "fold_patlig": False,
             "expand_extensor": True},
        ],
    )
    def test_the_ablation_switches_introduce_no_further_drift(self, flags):
        # Each flag is an ablation the runs will actually use, so the port has
        # to agree with the original under every one of them, not only under
        # the defaults.
        changed = [
            (anatomy, n)
            for anatomy, n in _corpus()
            if _original_clusters_of(anatomy, **flags) != clusters_of(anatomy, **flags)
        ]
        assert len(changed) == N_CHANGED_STRINGS, changed
        assert sum(n for _, n in changed) == N_CHANGED_FINDINGS

    def test_the_port_emits_no_cluster_outside_the_declared_vocabulary(self):
        # CLUSTER_NAMES is what `label_map` is checked for gaps against, so a
        # cluster the clusterer can emit but that set omits would contribute no
        # labels, silently.
        emitted = {
            cluster
            for anatomy, _ in _corpus()
            for flags in ({}, {"fold_fibular": False}, {"fold_patlig": False},
                          {"expand_extensor": True})
            for cluster in clusters_of(anatomy, **flags)
        }
        assert emitted <= CLUSTER_NAMES, sorted(emitted - CLUSTER_NAMES)

    def test_the_finding_weighted_cluster_distribution_is_unchanged(self):
        # A per-string differential can pass while the corpus-level totals move,
        # if two strings swap clusters. Count findings per cluster on both sides
        # and allow only the ten findings the fix moves.
        def distribution(fn) -> collections.Counter:
            counts: collections.Counter = collections.Counter()
            for anatomy, n in _corpus():
                for cluster in fn(anatomy):
                    counts[cluster] += n
            return counts

        old, new = distribution(_original_clusters_of), distribution(clusters_of)
        assert new - old == collections.Counter({"COMPARTMENT_PATELLOFEMORAL": 10})
        assert old - new == collections.Counter({"COMPARTMENT_UNSPECIFIED": 10})


class TestTheMisspelledVariant:
    """Needs only the port, so it runs whether or not ``_recovered/`` is present."""

    def test_the_misspelled_variant_moves_with_the_correctly_spelled_one(self):
        assert clusters_of("patelofemoral compartment") == frozenset(
            {"COMPARTMENT_PATELLOFEMORAL"}
        )
        assert clusters_of("patelofemoral compartment") == clusters_of("patellofemoral compartment")
