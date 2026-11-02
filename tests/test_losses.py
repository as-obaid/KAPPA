"""NaN must not reach the loss, and the loss must actually be asymmetric.

Two failures motivate every test here, and both are silent.

The first is arithmetic. 98.7% of raw label cells are NaN in this project, and
``BCEWithLogitsLoss`` over a raw label array returns NaN, back-propagates NaN and
NaNs the weights inside one step -- with no exception, no warning and no diverging
curve, just a fold that looks disappointing. The masking idiom ``(per_cell *
mask).sum() / mask.sum()`` does not save you, because ``0.0 * nan`` is ``nan``. So
the refusal has to be unconditional over the whole tensor, and it has to be a
``raise`` rather than an ``assert``, since ``python -O`` deletes ``assert``. There is
a test below that runs a subprocess under ``-O`` for exactly that reason.

The second is that a loss which returns a plausible positive scalar looks like a
working loss. Asymmetric Loss is chosen over focal precisely because focal applies
one focusing exponent to both classes and so suppresses the rare positives it meant
to rescue; with the rarest labels here that is the whole game. A test that only
checks the output is a finite positive number does not test any of that, so these
compare the positive and negative terms against each other, against plain BCE, and
against the paper's reference implementation (arXiv 2009.14119) inlined below.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the modelling extra is optional")

import torch.nn.functional as F  # noqa: E402  (must follow the importorskip)

from kappa.constants import LABELS, N_LABELS  # noqa: E402
from kappa.training.losses import LOSSES, AsymmetricLoss, MaskedBCE, build_loss  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    """The caller's environment plus this checkout on the path.

    Inherited rather than emptied because ``import torch`` needs whatever the host
    put in the environment to find its libraries, and this test is about ``-O``.
    """
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}


def _batch(rows: int = 6, seed: int = 0, rate: float = 0.3, scale: float = 3.0):
    """Logits, 0/1 targets and a partly-False mask, shaped like the pinned batch."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, N_LABELS, generator=generator) * scale
    targets = (torch.rand(rows, N_LABELS, generator=generator) < rate).float()
    mask = torch.rand(rows, N_LABELS, generator=generator) > 0.4
    return logits, targets, mask


def _cell(logit: float, target: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One kept cell, so a loss value is that cell's loss and nothing else."""
    return (
        torch.tensor([[logit]]),
        torch.tensor([[target]]),
        torch.ones(1, 1, dtype=torch.bool),
    )


def _reference_asl(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-cell ASL transcribed from the paper's published implementation.

    Kept verbatim in shape, including the ``eps`` clamps this repository's version
    replaces with ``logsigmoid``, so that the two can only agree by computing the
    same quantity rather than by sharing a helper.
    """
    xs_pos = torch.sigmoid(x)
    xs_neg = 1.0 - xs_pos
    if clip > 0.0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)
    los_pos = y * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg
    if gamma_neg > 0.0 or gamma_pos > 0.0:
        pt = xs_pos * y + xs_neg * (1.0 - y)
        gamma = gamma_pos * y + gamma_neg * (1.0 - y)
        loss = loss * torch.pow(1.0 - pt, gamma)
    return -loss


ALL_LOSSES = [AsymmetricLoss, MaskedBCE]


class TestNanRefusal:
    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_a_nan_target_raises_even_where_the_mask_is_false(self, factory):
        # The trap: 0.0 * nan is nan, so masking a NaN target does not remove it.
        # Checking only the kept cells would let this batch through and end the run.
        logits = torch.zeros(1, 2)
        targets = torch.tensor([[0.0, float("nan")]])
        mask = torch.tensor([[True, False]])
        with pytest.raises(ValueError, match="targets contain NaN"):
            factory()(logits, targets, mask)

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_a_nan_target_in_a_kept_cell_raises(self, factory):
        logits, targets, mask = _cell(0.0, 0.0)
        targets = torch.tensor([[float("nan")]])
        with pytest.raises(ValueError, match="targets contain NaN"):
            factory()(logits, targets, mask)

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_a_nan_logit_raises_and_blames_the_model_not_the_labels(self, factory):
        logits, targets, mask = _cell(0.0, 1.0)
        logits = torch.tensor([[float("nan")]])
        with pytest.raises(ValueError, match="diverged"):
            factory()(logits, targets, mask)

    @pytest.mark.parametrize("logit", [float("inf"), float("-inf")])
    @pytest.mark.parametrize("target", [0.0, 1.0])
    def test_an_infinite_logit_raises_rather_than_leaking_inf_or_nan(self, logit, target):
        # A -inf logit against a 0.0 target is 0 * inf, i.e. NaN, and used to be
        # reported as a bug inside the loss. An infinity is a diverged model.
        logits, targets, mask = _cell(logit, target)
        with pytest.raises(ValueError, match="not finite"):
            AsymmetricLoss()(logits, targets, mask)

    @pytest.mark.parametrize("bad", [-1.0, 2.0, float("inf")])
    def test_a_sentinel_target_outside_zero_one_raises(self, bad):
        # A -1 "unknown" sentinel is worse than NaN: it yields a *smaller* loss than
        # a correct cell, so training converges happily on nothing.
        logits, targets, mask = _cell(0.0, 0.0)
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            AsymmetricLoss()(logits, torch.tensor([[bad]]), mask)

    def test_the_refusal_survives_python_dash_o(self):
        # An `assert` here would be stripped by -O and the check would vanish in
        # exactly the deployment that runs longest.
        program = (
            "import torch\n"
            "from kappa.training.losses import AsymmetricLoss\n"
            "assert False, 'asserts are live, so this test proves nothing'\n"
        )
        control = subprocess.run(
            [sys.executable, "-O", "-c", program],
            capture_output=True,
            text=True,
            env=_subprocess_env(),
        )
        assert control.returncode == 0, "the -O subprocess did not strip asserts"

        program = (
            "import torch\n"
            "from kappa.training.losses import AsymmetricLoss\n"
            "targets = torch.tensor([[float('nan')]])\n"
            "try:\n"
            "    dropped = torch.zeros(1, 1, dtype=torch.bool)\n"
            "    AsymmetricLoss()(torch.zeros(1, 1), targets, dropped)\n"
            "except ValueError as exc:\n"
            # The message matters: the downstream NaN gate would also raise here, and
            # that would hide the loss of the target check itself.
            "    raise SystemExit(0 if 'targets contain NaN' in str(exc) else str(exc))\n"
            "raise SystemExit('a NaN target passed under -O')\n"
        )
        checked = subprocess.run(
            [sys.executable, "-O", "-c", program],
            capture_output=True,
            text=True,
            env=_subprocess_env(),
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_the_loss_is_finite_for_saturated_logits_of_both_signs(self, factory):
        logits = torch.tensor([[-80.0, -20.0, 0.0, 20.0, 80.0]])
        mask = torch.ones_like(logits, dtype=torch.bool)
        for target in (0.0, 1.0):
            loss = factory()(logits, torch.full_like(logits, target), mask)
            assert torch.isfinite(loss), (factory.__name__, target, loss)

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_no_nan_leaves_the_loss_across_a_random_sweep(self, factory):
        criterion = factory()
        for seed in range(25):
            logits, targets, mask = _batch(rows=8, seed=seed, scale=12.0)
            assert torch.isfinite(criterion(logits, targets, mask))

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_mismatched_shapes_raise(self, factory):
        criterion = factory()
        with pytest.raises(ValueError, match="targets"):
            criterion(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool))
        with pytest.raises(ValueError, match="mask"):
            criterion(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2, 4, dtype=torch.bool))


class TestMasking:
    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_the_masked_loss_equals_the_mean_over_the_kept_cells_alone(self, factory):
        logits, targets, mask = _batch(seed=1)
        criterion = factory()
        got = criterion(logits, targets, mask)
        per_cell = (
            _reference_asl(logits, targets)
            if factory is AsymmetricLoss
            else F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        )
        assert got.item() == pytest.approx(per_cell[mask].mean().item(), rel=1e-6)

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_the_denominator_is_the_kept_cell_count_not_the_element_count(self, factory):
        logits, targets, mask = _batch(seed=2)
        assert mask.sum() < mask.numel(), "this test needs some cells masked out"
        criterion = factory()
        got = criterion(logits, targets, mask)
        per_cell = (
            _reference_asl(logits, targets)
            if factory is AsymmetricLoss
            else F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        )
        over_numel = (per_cell * mask).sum() / mask.numel()
        assert got.item() != pytest.approx(over_numel.item(), rel=1e-6)
        assert got.item() == pytest.approx(per_cell[mask].mean().item(), rel=1e-6)

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_a_masked_cell_cannot_change_the_loss(self, factory):
        logits, targets, mask = _batch(seed=3)
        dropped = (~mask).nonzero()[0]
        row, column = int(dropped[0]), int(dropped[1])
        criterion = factory()
        before = criterion(logits, targets, mask)
        vandalised_logits = logits.clone()
        vandalised_logits[row, column] = 40.0
        vandalised_targets = targets.clone()
        vandalised_targets[row, column] = 1.0 - vandalised_targets[row, column]
        after = criterion(vandalised_logits, vandalised_targets, mask)
        assert after.item() == pytest.approx(before.item(), rel=1e-9)

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_an_all_masked_batch_is_exactly_zero_not_nan(self, factory):
        # It happens: only 58 of 4,407 studies carry labels, so a batch drawn from
        # the unlabelled majority has every cell masked. NaN here ends the run.
        logits, targets, _ = _batch(seed=4)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        loss = factory()(logits, targets, mask)
        assert loss.item() == 0.0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_an_all_masked_batch_still_back_propagates_a_zero_gradient(self, factory):
        # The zero must keep its graph, or gradient accumulation raises on a scalar
        # with no history and the epoch dies on a technicality instead.
        logits, targets, _ = _batch(seed=5)
        logits = logits.detach().requires_grad_(True)
        loss = factory()(logits, targets, torch.zeros_like(targets, dtype=torch.bool))
        assert loss.requires_grad
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert float(logits.grad.abs().max()) == 0.0

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_an_empty_batch_is_zero_rather_than_a_zero_division(self, factory):
        loss = factory()(
            torch.zeros(0, N_LABELS), torch.zeros(0, N_LABELS),
            torch.zeros(0, N_LABELS, dtype=torch.bool),
        )
        assert loss.item() == 0.0

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_a_float_mask_and_a_bool_mask_agree(self, factory):
        logits, targets, mask = _batch(seed=6)
        criterion = factory()
        assert criterion(logits, targets, mask.float()).item() == pytest.approx(
            criterion(logits, targets, mask).item()
        )

    @pytest.mark.parametrize("factory", ALL_LOSSES)
    def test_double_precision_is_preserved(self, factory):
        logits, targets, mask = _batch(seed=7)
        loss = factory()(logits.double(), targets.double(), mask)
        assert loss.dtype == torch.float64
        assert torch.isfinite(loss)


class TestAsymmetry:
    def test_an_easy_negative_is_down_weighted_far_more_than_an_easy_positive(self):
        # The single claim that distinguishes ASL from BCE and from focal. Same
        # confidence on both sides: the negative must nearly vanish, the positive
        # must not. Plain BCE scores these two cells identically.
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        easy_positive = criterion(*_cell(2.0, 1.0)).item()
        easy_negative = criterion(*_cell(-2.0, 0.0)).item()
        assert easy_negative < easy_positive / 1000.0

        bce = MaskedBCE()
        mirrored = bce(*_cell(-2.0, 0.0)).item()
        assert bce(*_cell(2.0, 1.0)).item() == pytest.approx(mirrored, rel=1e-5)

    def test_gamma_pos_zero_leaves_a_positive_cell_at_its_full_bce_weight(self):
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        for logit in (-3.0, -0.5, 0.0, 1.5, 4.0):
            assert criterion(*_cell(logit, 1.0)).item() == pytest.approx(
                MaskedBCE()(*_cell(logit, 1.0)).item(), rel=1e-5
            )

    def test_a_negative_cell_is_never_up_weighted_by_the_focusing_term(self):
        plain = MaskedBCE()
        focused = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.0)
        for logit in (-6.0, -2.0, -0.1, 0.0, 3.0):
            assert focused(*_cell(logit, 0.0)).item() <= plain(*_cell(logit, 0.0)).item() + 1e-6

    def test_raising_gamma_neg_monotonically_shrinks_the_negative_term(self):
        losses = [
            AsymmetricLoss(gamma_neg=gamma, gamma_pos=0.0, clip=0.0)(*_cell(-1.5, 0.0)).item()
            for gamma in (0.0, 1.0, 2.0, 4.0, 8.0)
        ]
        assert losses == sorted(losses, reverse=True)
        assert losses[0] > losses[-1]

    def test_raising_gamma_neg_leaves_the_positive_term_untouched(self):
        values = {
            AsymmetricLoss(gamma_neg=gamma, gamma_pos=0.0)(*_cell(1.0, 1.0)).item()
            for gamma in (0.0, 2.0, 8.0)
        }
        assert len(values) == 1

    def test_clip_removes_an_already_confident_negative_entirely(self):
        # The hard threshold, which is the second half of the asymmetry: a negative
        # the model scores below `clip` contributes exactly nothing.
        clipped = AsymmetricLoss(gamma_neg=4.0, clip=0.05)
        assert clipped(*_cell(-6.0, 0.0)).item() == 0.0
        assert AsymmetricLoss(gamma_neg=4.0, clip=0.0)(*_cell(-6.0, 0.0)).item() > 0.0

    def test_clip_does_not_touch_a_positive_cell(self):
        with_clip = AsymmetricLoss(clip=0.4)(*_cell(0.7, 1.0)).item()
        without_clip = AsymmetricLoss(clip=0.0)(*_cell(0.7, 1.0)).item()
        assert with_clip == pytest.approx(without_clip, rel=1e-6)

    def test_equal_gammas_are_symmetric_again_which_is_the_focal_behaviour_avoided(self):
        # Sanity check on the mechanism: the asymmetry comes only from the split
        # exponent, so equalising the two exponents restores the symmetry that
        # makes focal loss the wrong choice here.
        symmetric = AsymmetricLoss(gamma_neg=2.0, gamma_pos=2.0, clip=0.0)
        assert symmetric(*_cell(2.0, 1.0)).item() == pytest.approx(
            symmetric(*_cell(-2.0, 0.0)).item(), rel=1e-5
        )

    def test_zero_gammas_and_zero_clip_collapse_to_masked_bce(self):
        logits, targets, mask = _batch(seed=8)
        bare = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, clip=0.0)
        assert bare(logits, targets, mask).item() == pytest.approx(
            MaskedBCE()(logits, targets, mask).item(), rel=1e-5
        )

    def test_it_matches_the_papers_reference_implementation(self):
        for seed in range(10):
            logits, targets, mask = _batch(rows=8, seed=seed)
            got = AsymmetricLoss()(logits, targets, mask).item()
            expected = _reference_asl(logits, targets)[mask].mean().item()
            assert got == pytest.approx(expected, rel=1e-6, abs=1e-8)

    def test_it_matches_the_reference_for_non_default_hyperparameters(self):
        logits, targets, mask = _batch(seed=9)
        for gamma_neg, gamma_pos, clip in [(2.0, 1.0, 0.1), (0.0, 0.0, 0.2), (4.0, 0.0, 0.0)]:
            got = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
            expected = _reference_asl(logits, targets, gamma_neg, gamma_pos, clip)
            assert got(logits, targets, mask).item() == pytest.approx(
                expected[mask].mean().item(), rel=1e-5, abs=1e-8
            )

    def test_the_loss_is_non_negative(self):
        for seed in range(10):
            logits, targets, mask = _batch(rows=8, seed=seed, scale=6.0)
            assert AsymmetricLoss()(logits, targets, mask).item() >= 0.0

    @pytest.mark.parametrize(
        "kwargs", [{"gamma_neg": -1.0}, {"gamma_pos": -0.5}, {"clip": 1.0}, {"clip": -0.1}]
    )
    def test_impossible_hyperparameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            AsymmetricLoss(**kwargs)

    def test_the_repr_states_the_hyperparameters_a_run_actually_used(self):
        text = repr(AsymmetricLoss(gamma_neg=3.0, gamma_pos=1.0, clip=0.1))
        assert "gamma_neg=3.0" in text
        assert "gamma_pos=1.0" in text
        assert "clip=0.1" in text


class TestGradients:
    def test_the_gradient_reaches_the_logits_of_kept_cells_only(self):
        logits, targets, mask = _batch(seed=10)
        logits = logits.detach().requires_grad_(True)
        AsymmetricLoss()(logits, targets, mask).backward()
        assert torch.isfinite(logits.grad).all()
        assert float(logits.grad[~mask].abs().max()) == 0.0
        assert float(logits.grad[mask].abs().max()) > 0.0

    @pytest.mark.parametrize("gamma_pos", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("logit", [-100.0, -20.0, 20.0, 100.0])
    @pytest.mark.parametrize("target", [0.0, 1.0])
    def test_the_gradient_stays_finite_at_saturation(self, gamma_pos, logit, target):
        # The focusing term is (1 - p_true) ** gamma, and p_true saturates to exactly
        # 1.0 in float32, so the base hits 0. pow(0, 0) must not produce a NaN
        # gradient, or the guard on the returned scalar would pass while the
        # optimiser received NaN.
        logits = torch.tensor([[logit]], requires_grad=True)
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=gamma_pos)
        criterion(logits, torch.tensor([[target]]), torch.ones(1, 1, dtype=torch.bool)).backward()
        assert torch.isfinite(logits.grad).all(), logits.grad

    def test_a_positive_cell_pulls_its_logit_upwards(self):
        logits = torch.tensor([[0.0]], requires_grad=True)
        AsymmetricLoss()(logits, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool)).backward()
        assert float(logits.grad) < 0.0

    def test_a_negative_cell_pushes_its_logit_downwards(self):
        logits = torch.tensor([[0.0]], requires_grad=True)
        AsymmetricLoss()(logits, torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool)).backward()
        assert float(logits.grad) > 0.0


class TestMaskedBCE:
    def test_it_shares_the_asymmetric_loss_signature_so_the_ablation_is_a_config_change(self):
        logits, targets, mask = _batch(seed=11)
        for criterion in (AsymmetricLoss(), MaskedBCE()):
            loss = criterion(logits, targets, mask)
            assert loss.shape == ()
            assert torch.isfinite(loss)

    def test_it_matches_binary_cross_entropy_with_logits_on_the_kept_cells(self):
        logits, targets, mask = _batch(seed=12)
        expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        assert MaskedBCE()(logits, targets, mask).item() == pytest.approx(
            expected[mask].mean().item(), rel=1e-6
        )

    def test_pos_weight_scales_the_positive_term_and_leaves_negatives_alone(self):
        weights = torch.ones(1) * 3.0
        weighted = MaskedBCE(pos_weight=weights)
        plain = MaskedBCE()
        assert weighted(*_cell(0.2, 1.0)).item() == pytest.approx(
            3.0 * plain(*_cell(0.2, 1.0)).item(), rel=1e-5
        )
        assert weighted(*_cell(0.2, 0.0)).item() == pytest.approx(
            plain(*_cell(0.2, 0.0)).item(), rel=1e-6
        )

    def test_pos_weight_is_a_buffer_so_it_travels_with_the_module(self):
        criterion = MaskedBCE(pos_weight=torch.ones(N_LABELS) * 2.0)
        assert "pos_weight" in dict(criterion.named_buffers())
        assert MaskedBCE().pos_weight is None


class TestBuildLoss:
    def test_the_registry_holds_both_arms_of_the_ablation(self):
        assert set(LOSSES) == {"asl", "bce"}

    def test_it_defaults_to_the_asymmetric_loss(self):
        assert isinstance(build_loss(), AsymmetricLoss)

    def test_it_builds_the_bce_arm_by_name(self):
        assert isinstance(build_loss("bce"), MaskedBCE)

    def test_it_forwards_hyperparameters(self):
        criterion = build_loss("asl", gamma_neg=6.0, clip=0.2)
        assert criterion.gamma_neg == 6.0
        assert criterion.clip == 0.2

    def test_an_unknown_name_lists_the_names_that_do_exist(self):
        with pytest.raises(ValueError, match=r"known: \['asl', 'bce'\]"):
            build_loss("focal")


class TestPerLabelGammaNeg:
    """``gamma_neg`` per label, because the derived table's rarity is per label.

    The scalar encodes "positives are rare". Measured on the derived table
    (n=4,407) that holds for Fracture at 7.4% positive and inverts for
    Effusion at 58.8%, where a scalar gamma_neg focuses down the *rarer* class.
    Each label is 1/12 of the macro score, so that is paid at full price.
    """

    def test_a_uniform_vector_is_exactly_the_scalar(self):
        # The equivalence that makes the default path provably unchanged: if this
        # ever drifts, every baseline number silently stops being comparable.
        logits, targets, mask = _random_cells(seed=0)
        scalar = AsymmetricLoss(gamma_neg=4.0)
        vector = AsymmetricLoss(gamma_neg=4.0, gamma_neg_by_label=[4.0] * N_LABELS)
        assert torch.allclose(scalar(logits, targets, mask), vector(logits, targets, mask), atol=1e-7)

    def test_a_zero_vector_matches_a_zero_scalar(self):
        logits, targets, mask = _random_cells(seed=1)
        zeros = AsymmetricLoss(gamma_neg=4.0, gamma_neg_by_label=[0.0] * N_LABELS)
        assert torch.allclose(
            zeros(logits, targets, mask),
            AsymmetricLoss(gamma_neg=0.0)(logits, targets, mask),
            atol=1e-7,
        )

    def test_the_vector_overrides_the_scalar(self):
        logits, targets, mask = _random_cells(seed=2)
        scalar = AsymmetricLoss(gamma_neg=4.0)
        vector = AsymmetricLoss(gamma_neg=4.0, gamma_neg_by_label=[0.0] * N_LABELS)
        assert not torch.allclose(scalar(logits, targets, mask), vector(logits, targets, mask))

    def test_the_exponent_lands_on_the_named_column_only(self):
        # One column's exponent must not move any other column's loss. Scored one
        # column at a time so a vector applied along the wrong axis is visible.
        logits, targets, mask = _random_cells(seed=3)
        gammas = [4.0] * N_LABELS
        gammas[LABELS.index("Effusion")] = 0.0
        vector = AsymmetricLoss(gamma_neg=4.0, gamma_neg_by_label=gammas)
        scalar = AsymmetricLoss(gamma_neg=4.0)
        for column in range(N_LABELS):
            only = torch.zeros_like(mask)
            only[:, column] = mask[:, column]
            a = scalar(logits, targets, only)
            b = vector(logits, targets, only)
            if column == LABELS.index("Effusion"):
                assert not torch.allclose(a, b), "the overridden column did not change"
            else:
                assert torch.allclose(a, b, atol=1e-7), f"{LABELS[column]} moved"

    def test_a_vector_of_the_wrong_length_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="one per label"):
            AsymmetricLoss(gamma_neg_by_label=[1.0] * 5)

    def test_a_vector_that_disagrees_with_the_logits_is_refused_at_forward(self):
        # It would broadcast rather than raise, handing every column someone else's
        # exponent and converging to a wrong number quietly.
        criterion = AsymmetricLoss(gamma_neg_by_label=[1.0] * N_LABELS)
        wide = torch.zeros(2, N_LABELS + 1)
        with pytest.raises(ValueError, match="broadcasts"):
            criterion(wide, torch.zeros_like(wide), torch.ones_like(wide, dtype=torch.bool))

    def test_a_negative_exponent_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            AsymmetricLoss(gamma_neg_by_label=[-1.0] * N_LABELS)

    def test_the_vector_is_saved_with_the_checkpoint(self):
        # Reloading with the wrong gammas trains perfectly well and is a different
        # experiment, so the vector has to be part of the state, not the code.
        criterion = AsymmetricLoss(gamma_neg_by_label=[float(i) for i in range(N_LABELS)])
        assert "gamma_neg_by_label" in criterion.state_dict()
        restored = AsymmetricLoss(gamma_neg_by_label=[0.0] * N_LABELS)
        restored.load_state_dict(criterion.state_dict())
        assert restored.gamma_neg_by_label.tolist() == [float(i) for i in range(N_LABELS)]

    def test_the_scalar_attribute_is_still_a_float(self):
        # build_loss(...).gamma_neg is read as a number elsewhere.
        assert AsymmetricLoss(gamma_neg=6.0).gamma_neg == 6.0
        assert AsymmetricLoss(gamma_neg=6.0, gamma_neg_by_label=[1.0] * N_LABELS).gamma_neg == 6.0

    def test_the_repr_names_the_per_label_vector(self):
        text = repr(AsymmetricLoss(gamma_neg_by_label=[1.0] * N_LABELS))
        assert "per-label" in text

    def test_it_still_refuses_nan_targets(self):
        criterion = AsymmetricLoss(gamma_neg_by_label=[1.0] * N_LABELS)
        targets = torch.zeros(2, N_LABELS)
        targets[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            criterion(
                torch.zeros(2, N_LABELS), targets, torch.ones(2, N_LABELS, dtype=torch.bool)
            )


def _random_cells(seed: int):
    """Logits, targets and a full mask over the twelve real labels."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(16, N_LABELS, generator=generator)
    targets = (torch.rand(16, N_LABELS, generator=generator) > 0.5).to(torch.float32)
    return logits, targets, torch.ones(16, N_LABELS, dtype=torch.bool)
