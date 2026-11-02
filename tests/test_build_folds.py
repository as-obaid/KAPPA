"""The fold-build gate: an unbalanceable split must not be written as GATE PASSED.

``kappa.data.folds.assign_folds`` *warns* rather than raises when a group is
larger than one fold's share, and a warning changes no exit code. So the script
printed "GATE PASSED" and wrote the table for a split the library had just
called unbalanceable -- the exact class of silent wrong answer the gate exists
to catch, reproduced by the gate.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kappa.constants import LABELS
from kappa.data.folds import assign_folds
from kappa.data.paths import repo_root
from kappa.labels.label_map import MASK_SUFFIX


def _load_build_folds():
    """Import scripts/build_folds.py by path -- scripts/ is not a package."""
    path = repo_root() / "scripts" / "build_folds.py"
    spec = importlib.util.spec_from_file_location("_build_folds_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_folds_script = _load_build_folds()


def _degenerate_table(n: int = 40) -> pd.DataFrame:
    """Every row in one group, so no split can balance it."""
    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        "patient_id": ["same"] * n,
        "report_group": ["same"] * n,
        "scanner_fingerprint": ["same"] * n,
        "language": ["english"] * n,
    }
    for label in LABELS:
        data[label] = rng.integers(0, 2, size=n).astype(float)
        data[label + MASK_SUFFIX] = np.ones(n, dtype=bool)
    return pd.DataFrame(data)


class TestTheBalanceGate:
    def test_the_matched_substring_is_what_assign_folds_actually_says(self):
        # The gate matches on message text because the warning has no dedicated
        # category. If that message is reworded, this fails rather than the gate
        # silently never firing again.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assign_folds(_degenerate_table(), n_folds=5)
        messages = [str(w.message) for w in caught]
        assert any(build_folds_script._BALANCE_WARNING in m for m in messages), messages

    def test_an_unbalanceable_build_is_counted_as_a_failure(self, capsys):
        result, unbalanced = build_folds_script._build_gated(
            lambda: assign_folds(_degenerate_table(), n_folds=5)
        )
        assert unbalanced == 1
        assert result is not None
        # Re-emitted, not swallowed: capturing without printing would trade a
        # silent pass for a silent gate.
        err = capsys.readouterr().err
        assert build_folds_script._BALANCE_WARNING in err
        assert "GATE FAILED" in err

    def test_a_balanceable_build_is_not_counted(self, capsys):
        rng = np.random.default_rng(0)
        n = 200
        data: dict[str, object] = {
            "patient_id": [f"p{i}" for i in range(n)],
            "report_group": [f"g{i}" for i in range(n)],
            "scanner_fingerprint": [f"s{i % 7}" for i in range(n)],
            "language": ["english"] * n,
        }
        for label in LABELS:
            data[label] = rng.integers(0, 2, size=n).astype(float)
            data[label + MASK_SUFFIX] = np.ones(n, dtype=bool)
        _, unbalanced = build_folds_script._build_gated(
            lambda: assign_folds(pd.DataFrame(data), n_folds=5)
        )
        assert unbalanced == 0
        assert "GATE FAILED" not in capsys.readouterr().err

    def test_the_real_corpus_does_not_trip_the_gate(self):
        # Measured on the shipped table: largest scanner group 327, largest patient
        # group 37, against a fold share of 881. Pinned so that a future regrouping that
        # makes the shipped split unbalanceable fails here rather than at build time.
        path = repo_root() / "data" / "folds" / "folds_v1.csv"
        if not path.is_file():
            pytest.skip("folds_v1.csv not present")
        folds = pd.read_csv(path)
        share = len(folds) / 5
        for column in ("patient_group", "scanner_group"):
            largest = int(folds[column].value_counts().max())
            assert largest <= share, f"{column}: {largest} > one fold's share {share:.0f}"
