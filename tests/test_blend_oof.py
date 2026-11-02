"""The two ways an OOF blend goes wrong without looking wrong.

Both failures this file guards are silent, and both move the number in the
*flattering* direction, so neither is caught by reading the result.

**Blending configs that disagree on the fold assignment is a leak.** A study held
out by one model was trained on by another, so the blend reads its own training
data and scores higher for it. Nothing about the output says so -- the tables have
the same shape, the same studies and the same columns. ``verify_agreement`` must
therefore refuse, not warn, and it must refuse on each of the four things that can
differ independently: study set, fold column, truth and mask.

**Rank-normalising globally instead of within each fold mixes five models' scales.**
The fold *is* the model: study 7's logit comes from the fold-0 checkpoint and study
900's from the fold-3 checkpoint, and those networks never agreed on a scale. The
trap survives because it is invisible in the per-fold mean -- the map is monotone
within a fold, so the per-fold macro cannot move at all. It shows only on the pooled
table, which is exactly the estimator nobody checks.

Everything here runs on small synthetic frames. The real parquets live on purgeable
cluster storage, so a test that needed them would be a test that stops running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from blend_oof import (  # noqa: E402
    FOLDS,
    FoldDisagreement,
    HeldOutSplitTouched,
    assert_no_test_overlap,
    discover_complete,
    load_oof,
    per_fold_macro,
    per_fold_rank_average,
    per_fold_rank_normalise,
    score,
    verify_agreement,
)
from kappa.constants import LABELS  # noqa: E402

UID = "StudyInstanceUID"
TRUE = [f"{label}__true" for label in LABELS]
MASK = [f"{label}__mask" for label in LABELS]
N_PER_FOLD = 40


def _uids(n: int) -> list[str]:
    return [f"1.2.826.0.0.{i:06d}" for i in range(n)]


def _frame(seed: int, *, fold_offsets: dict[int, float] | None = None) -> pd.DataFrame:
    """One config's pooled OOF table: five folds of ``N_PER_FOLD`` studies each.

    ``fold_offsets`` shifts a fold's logits by a constant. That is the whole of the
    scale problem in one parameter: it leaves every within-fold ordering untouched,
    so no per-fold statistic can see it, while the pooled ranking is destroyed.
    """
    n = N_PER_FOLD * len(FOLDS)
    fold = np.repeat(np.array(FOLDS), N_PER_FOLD)
    # Truth and the split are properties of the DATA, so every config shares them and
    # they do not move with ``seed``. Only the predictions are a property of the model.
    truth = np.random.default_rng(0).integers(0, 2, size=(n, len(LABELS))).astype(np.float64)
    rng = np.random.default_rng(1000 + seed)
    # A signal a rank metric can find, plus noise, so the AUC is neither 0.5 nor 1.0.
    scores = truth * 1.5 + rng.normal(0.0, 1.0, size=truth.shape)
    if fold_offsets:
        for key, offset in fold_offsets.items():
            scores[fold == key] += offset
    frame = pd.DataFrame({UID: _uids(n), "fold": fold, "seed": 0})
    for column, label in enumerate(LABELS):
        frame[label] = scores[:, column]
        frame[f"{label}__true"] = truth[:, column]
        frame[f"{label}__mask"] = True
    return frame


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame["fold"].to_numpy(),
        frame[TRUE].to_numpy(dtype=np.float64),
        frame[MASK].to_numpy(dtype=bool),
        frame[list(LABELS)].to_numpy(dtype=np.float64),
    )


# ------------------------------------------------------- the fold-agreement refusal


def test_agreeing_configs_pass() -> None:
    tables = {"a": _frame(0), "b": _frame(1), "c": _frame(2)}
    verify_agreement(tables, "a")  # must not raise


def test_differing_fold_assignment_is_refused() -> None:
    """The leak case. Same studies, same truth, same mask -- a different partition."""
    tables = {"a": _frame(0), "b": _frame(1)}
    rolled = tables["b"].copy()
    rolled["fold"] = np.roll(rolled["fold"].to_numpy(), N_PER_FOLD)
    tables["b"] = rolled
    with pytest.raises(FoldDisagreement, match="FOLD ASSIGNMENT"):
        verify_agreement(tables, "a")


def test_differing_study_set_is_refused() -> None:
    tables = {"a": _frame(0), "b": _frame(1).iloc[:-1].reset_index(drop=True)}
    with pytest.raises(FoldDisagreement, match="study set"):
        verify_agreement(tables, "a")


def test_renamed_study_is_refused_even_at_the_same_length() -> None:
    tables = {"a": _frame(0), "b": _frame(1)}
    ids = tables["b"][UID].to_numpy().copy()
    ids[3] = "1.2.826.0.0.999999"
    tables["b"][UID] = ids
    with pytest.raises(FoldDisagreement, match="study set"):
        verify_agreement(tables, "a")


def test_differing_truth_is_refused() -> None:
    """A rebuilt label table means the two configs were never scored against the same
    target, so their difference is not a modelling difference."""
    tables = {"a": _frame(0), "b": _frame(1)}
    tables["b"].loc[0, TRUE[0]] = 1.0 - tables["b"].loc[0, TRUE[0]]
    with pytest.raises(FoldDisagreement, match="truth"):
        verify_agreement(tables, "a")


def test_differing_mask_is_refused() -> None:
    """The mask is each label's denominator, so a difference changes what the macro
    is a mean over without changing anything visible about the table."""
    tables = {"a": _frame(0), "b": _frame(1)}
    tables["b"].loc[0, MASK[0]] = False
    with pytest.raises(FoldDisagreement, match="mask"):
        verify_agreement(tables, "a")


def test_nan_truth_is_not_a_mismatch_against_itself() -> None:
    """98.7% of raw label cells are empty and NaN != NaN. Comparing truth naively
    would report every config as disagreeing with every other."""
    tables = {"a": _frame(0), "b": _frame(1)}
    for name in tables:
        tables[name].loc[0, TRUE[0]] = np.nan
        tables[name].loc[0, MASK[0]] = False
    verify_agreement(tables, "a")


# ----------------------------------------------------- per-fold vs global rank scale


def test_global_rank_normalisation_wrecks_the_pooled_macro() -> None:
    """Per-fold beats global on the pooled table, and by a lot when scales differ.

    The offsets here are deliberately larger than the signal, which is the honest
    version of what five independently trained checkpoints do: every study in the
    shifted fold outranks every study elsewhere, so pooled negatives beat pooled
    positives and the metric collapses.
    """
    frame = _frame(7, fold_offsets={0: -6.0, 1: -3.0, 3: 3.0, 4: 6.0})
    fold, truth, mask, logits = _arrays(frame)

    per_fold = score(truth, mask, per_fold_rank_normalise(logits, fold)).macro
    # The global path: one ranking over the pooled table, i.e. what rank_average does
    # when it is handed the whole OOF table instead of one fold at a time.
    from kappa.evaluation.metrics import rank_normalise

    global_rank = score(truth, mask, rank_normalise(logits)).macro

    assert per_fold > global_rank + 0.05, (per_fold, global_rank)


def test_the_per_fold_mean_cannot_see_the_scale_mixing() -> None:
    """Why the trap survives: rank-normalising is monotone inside each fold, so every
    per-fold macro is bit-identical either way. Only the pooled number moves."""
    from kappa.evaluation.metrics import rank_normalise

    frame = _frame(7, fold_offsets={0: -6.0, 4: 6.0})
    fold, truth, mask, logits = _arrays(frame)

    per_fold = per_fold_macro(truth, mask, fold, per_fold_rank_normalise(logits, fold))
    globally = per_fold_macro(truth, mask, fold, rank_normalise(logits))
    raw = per_fold_macro(truth, mask, fold, logits)

    np.testing.assert_allclose(per_fold, raw)
    np.testing.assert_allclose(globally, raw)


def test_per_fold_rank_average_is_a_mean_of_per_fold_normalised_members() -> None:
    """The blend arithmetic has one implementation: ``rank_average``, applied per fold."""
    members = [_frame(seed) for seed in (0, 1, 2)]
    fold = members[0]["fold"].to_numpy()
    logits = [frame[list(LABELS)].to_numpy(dtype=np.float64) for frame in members]
    normalised = [per_fold_rank_normalise(one, fold) for one in logits]

    blended = per_fold_rank_average(normalised, fold)

    np.testing.assert_allclose(blended, np.mean(normalised, axis=0))


def test_blending_scale_mismatched_members_needs_the_per_fold_path() -> None:
    """End to end: three members whose folds sit on different scales. Blended per fold
    the pooled macro survives; blended globally it does not."""
    from kappa.evaluation.metrics import rank_average

    offsets = {0: -5.0, 2: 4.0}
    frames = [_frame(seed, fold_offsets=offsets) for seed in (11, 12, 13)]
    fold, truth, mask, _ = _arrays(frames[0])
    logits = [frame[list(LABELS)].to_numpy(dtype=np.float64) for frame in frames]

    per_fold = score(
        truth, mask, per_fold_rank_average([per_fold_rank_normalise(x, fold) for x in logits], fold)
    ).macro
    globally = score(truth, mask, rank_average(logits)).macro

    assert per_fold > globally + 0.05, (per_fold, globally)


# ----------------------------------------------------------- the held-out test split


def _folds_csv(path: Path, dev_uids: list[str], test_uids: list[str]) -> Path:
    rows = [{UID: uid, "split": "dev", "fold": i % 5} for i, uid in enumerate(dev_uids)]
    rows += [{UID: uid, "split": "test", "fold": -1} for uid in test_uids]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_test_studies_in_an_oof_table_are_refused(tmp_path: Path) -> None:
    """The 685 held-out studies are spent the first time they are scored."""
    dev, test = _uids(20), [f"9.9.{i}" for i in range(5)]
    csv = _folds_csv(tmp_path / "folds_v2.csv", dev, test)
    with pytest.raises(HeldOutSplitTouched, match="held-out test studies"):
        assert_no_test_overlap(np.array(dev + test[:1]), csv)


def test_clean_oof_table_reports_the_held_out_count(tmp_path: Path) -> None:
    dev, test = _uids(20), [f"9.9.{i}" for i in range(5)]
    csv = _folds_csv(tmp_path / "folds_v2.csv", dev, test)
    assert assert_no_test_overlap(np.array(dev), csv) == 5


def test_a_missing_fold_table_is_not_a_pass(tmp_path: Path) -> None:
    """No fold table means no evidence the test split was left alone, which is not
    the same thing as evidence that it was."""
    with pytest.raises(FileNotFoundError):
        assert_no_test_overlap(np.array(_uids(3)), tmp_path / "absent.csv")


# ------------------------------------------------------------------ disk discovery


def test_only_complete_five_fold_runs_are_discovered(tmp_path: Path) -> None:
    """Four folds of a config is not a 5-fold model. It is a table over a different
    set of studies, and blending it would score the blend on rows a member never
    predicted."""
    runs = tmp_path / "runs"
    for name, folds in (("whole", FOLDS), ("partial", FOLDS[:4]), ("empty", ())):
        (runs / name).mkdir(parents=True)
        for k in folds:
            _frame(0).head(1).to_parquet(runs / name / f"oof_seed0_fold{k}.parquet")

    assert discover_complete(runs) == ["whole"]


def test_load_oof_sorts_by_study_uid(tmp_path: Path) -> None:
    """Five independent jobs write the five folds in whatever order they finish, so
    concatenation order is a property of the queue and not of the data. Sorting is
    what makes two configs row-comparable at all."""
    runs = tmp_path / "runs" / "cfg"
    runs.mkdir(parents=True)
    frame = _frame(3)
    for k in FOLDS:
        part = frame[frame["fold"] == k]
        part.iloc[::-1].to_parquet(runs / f"oof_seed0_fold{k}.parquet")

    loaded = load_oof(runs.parent, "cfg")

    assert loaded[UID].is_monotonic_increasing
    assert len(loaded) == len(frame)
