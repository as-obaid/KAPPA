"""Weight averaging: the arithmetic, the restore, and the precision trap.

The failure this module is written against is not a crash. An EMA accumulated in
bf16 with decay=0.999 has an update term about 1e-3 of the value, which is below
bf16's ~3 decimal digits, so the shadow stops moving and the "EMA" becomes an
expensive copy of the initial weights -- a real number supporting a false
conclusion, which is the shape of failure this project keeps hitting.
"""

from __future__ import annotations

import pytest

# The modelling extra is optional (pyproject [project.optional-dependencies]).
# Without this the module fails to COLLECT on a torch-free machine, which reads
# as a broken suite rather than an absent optional dependency.
pytest.importorskip("torch")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from kappa.config import loads_config
from kappa.training.averaging import (
    ExponentialMovingAverage,
    LastNAverage,
    build_averager,
)


def _model(fill: float = 0.0) -> nn.Module:
    model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(fill)
    return model


def _set(model: nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def _bn3d_model(fill: float = 0.0) -> nn.Module:
    # nn.BatchNorm3d specifically: this is the layer r3d_18 carries 20 of, and
    # the exact case the module docstring warns about.
    model = nn.Sequential(nn.Conv3d(2, 3, kernel_size=1), nn.BatchNorm3d(3))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(fill)
    return model


def _set_bn_buffers(model: nn.Module, value: float) -> None:
    bn = model[1]
    with torch.no_grad():
        bn.running_mean.fill_(value)
        bn.running_var.fill_(value)


class TestExponentialMovingAverage:
    def test_one_step_moves_the_shadow_by_one_minus_decay(self):
        model = _model(0.0)
        ema = ExponentialMovingAverage(model, decay=0.75)
        _set(model, 1.0)
        ema.after_step(model)
        assert ema.state_dict()["0.weight"].flatten()[0].item() == pytest.approx(0.25)

    def test_it_converges_toward_a_held_value(self):
        model = _model(0.0)
        ema = ExponentialMovingAverage(model, decay=0.5)
        _set(model, 1.0)
        for _ in range(20):
            ema.after_step(model)
        assert ema.state_dict()["0.weight"].flatten()[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_it_lags_the_raw_weights(self):
        # The entire point: if the shadow equalled the weights there would be no
        # averaging, and a decay bug that made it track exactly would look fine.
        model = _model(0.0)
        ema = ExponentialMovingAverage(model, decay=0.9)
        _set(model, 1.0)
        ema.after_step(model)
        assert ema.state_dict()["0.weight"].flatten()[0].item() < 0.5

    def test_the_shadow_is_fp32_even_from_a_bf16_model(self):
        # The precision trap in the module docstring, asserted rather than trusted.
        model = _model(0.0).to(torch.bfloat16)
        ema = ExponentialMovingAverage(model, decay=0.999)
        assert all(t.dtype == torch.float32 for t in ema.state_dict().values())

    def test_a_small_update_still_moves_a_bf16_model_s_shadow(self):
        # Started near 1.0 on purpose. bf16 has 8 mantissa bits, so its ulp around
        # 1.0 is about 0.0078; with decay=0.999 the update term is 1e-3 of the gap,
        # and 1.0 + 0.001 rounds straight back to 1.0. A shadow accumulated in the
        # model's dtype therefore stops moving and the "EMA" becomes an expensive
        # copy of the initial weights -- with no error and a normal loss curve.
        # Starting from 0.0 would hide this: 0.0 + 0.001 is exactly representable.
        model = _model(1.0).to(torch.bfloat16)
        ema = ExponentialMovingAverage(model, decay=0.999)
        _set(model, 2.0)
        before = ema.state_dict()["0.weight"].flatten()[0].item()
        ema.after_step(model)
        after = ema.state_dict()["0.weight"].flatten()[0].item()
        assert before == pytest.approx(1.0)
        assert after > before, "the shadow did not move; it underflowed"
        assert after == pytest.approx(1.001, abs=1e-6)

    def test_it_reports_whether_it_has_seen_anything(self):
        model = _model()
        ema = ExponentialMovingAverage(model, decay=0.9)
        assert not ema.ready()
        ema.after_step(model)
        assert ema.ready() and ema.steps == 1

    def test_an_out_of_range_decay_is_refused(self):
        for decay in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                ExponentialMovingAverage(_model(), decay=decay)


class TestLastNAverage:
    def test_it_averages_exactly_the_last_n_snapshots(self):
        model = _model()
        averager = LastNAverage(model, n=2)
        for value in (2.0, 4.0, 6.0):
            _set(model, value)
            averager.after_epoch(model)
        assert averager.state_dict()["0.weight"].flatten()[0].item() == pytest.approx(5.0)

    def test_it_does_not_keep_more_snapshots_than_it_needs(self):
        # n copies of a 50M-parameter model beside the activations is how a run that
        # trains fine dies at the last epoch.
        model = _model()
        averager = LastNAverage(model, n=2)
        for _ in range(10):
            averager.after_epoch(model)
        assert averager.observed == 2

    def test_a_short_run_averages_what_it_has_and_says_so(self):
        # "the last 5 epochs" describing a mean of 2 is a wrong number in a table.
        model = _model()
        averager = LastNAverage(model, n=5)
        _set(model, 3.0)
        averager.after_epoch(model)
        assert averager.observed == 1
        assert averager.state_dict()["0.weight"].flatten()[0].item() == pytest.approx(3.0)

    def test_it_refuses_to_average_nothing(self):
        with pytest.raises(RuntimeError, match="no snapshots"):
            LastNAverage(_model(), n=2).state_dict()

    def test_snapshots_are_held_on_the_cpu(self):
        model = _model()
        averager = LastNAverage(model, n=1)
        averager.after_epoch(model)
        assert all(t.device.type == "cpu" for t in averager.state_dict().values())

    def test_a_zero_window_is_refused(self):
        with pytest.raises(ValueError):
            LastNAverage(_model(), n=0)


class TestApplyAndRestore:
    """Training continues from the raw trajectory, so the swap must be reversible."""

    def test_the_averaged_weights_are_visible_inside_the_block(self):
        model = _model(0.0)
        averager = LastNAverage(model, n=1)
        _set(model, 5.0)
        averager.after_epoch(model)
        _set(model, 9.0)
        with averager.applied(model):
            assert model[0].weight.flatten()[0].item() == pytest.approx(5.0)

    def test_the_raw_weights_come_back_afterwards(self):
        # An averaged weight left in place would make every later step descend from
        # a point the optimiser's state does not describe.
        model = _model(0.0)
        averager = LastNAverage(model, n=1)
        _set(model, 5.0)
        averager.after_epoch(model)
        _set(model, 9.0)
        with averager.applied(model):
            pass
        assert model[0].weight.flatten()[0].item() == pytest.approx(9.0)

    def test_the_weights_are_restored_even_if_the_block_raises(self):
        model = _model(0.0)
        averager = LastNAverage(model, n=1)
        _set(model, 5.0)
        averager.after_epoch(model)
        _set(model, 9.0)
        with pytest.raises(RuntimeError):
            with averager.applied(model):
                raise RuntimeError("validation blew up")
        assert model[0].weight.flatten()[0].item() == pytest.approx(9.0)

    def test_integer_buffers_are_left_alone(self):
        # BatchNorm's num_batches_tracked is an int64 counter. Averaging it is
        # meaningless, and copying a float into it would raise or truncate.
        model = _model()
        averager = LastNAverage(model, n=1)
        averager.after_epoch(model)
        assert "1.num_batches_tracked" not in averager.state_dict()
        with averager.applied(model):
            assert model[1].num_batches_tracked.dtype == torch.int64


class TestBatchNormRunningStatsAreCopiedNotAveraged:
    """averaging.py:27 promises running statistics are copied, not averaged.

    ``_float_params`` used to decide what to average by ``is_floating_point()``,
    and BatchNorm's ``running_mean``/``running_var`` are float32, so they were
    swept in alongside the real parameters. This is the r3d_18 case: 20
    ``BatchNorm3d`` layers whose running stats would silently become an average
    of snapshots instead of the live model's own statistics.
    """

    def test_ema_leaves_running_stats_at_the_live_value(self):
        model = _bn3d_model(0.0)
        _set_bn_buffers(model, 1.0)
        ema = ExponentialMovingAverage(model, decay=0.75)
        _set(model, 1.0)
        _set_bn_buffers(model, 5.0)
        ema.after_step(model)
        # Moved again, after the after_step that would have shadowed it: this
        # is the value that must survive into the applied state, because
        # buffers are never shadowed in the first place.
        _set_bn_buffers(model, 9.0)
        with ema.applied(model):
            assert model[1].running_mean.flatten()[0].item() == pytest.approx(9.0)
            assert model[1].running_var.flatten()[0].item() == pytest.approx(9.0)
            # The trainable Conv3d weight IS averaged (0.75*0 + 0.25*1 = 0.25,
            # the same arithmetic as test_one_step_moves_the_shadow_by_one_minus_decay):
            # the fix cannot be satisfied by disabling averaging entirely.
            assert model[0].weight.flatten()[0].item() == pytest.approx(0.25)

    def test_last_n_average_leaves_running_stats_at_the_live_value(self):
        model = _bn3d_model(0.0)
        averager = LastNAverage(model, n=2)
        for value in (1.0, 3.0):
            _set_bn_buffers(model, value)
            _set(model, value)
            averager.after_epoch(model)
        # The live value after both snapshots -- not 2.0, the mean of the two.
        _set_bn_buffers(model, 9.0)
        with averager.applied(model):
            assert model[1].running_mean.flatten()[0].item() == pytest.approx(9.0)
            assert model[1].running_var.flatten()[0].item() == pytest.approx(9.0)
            # Trainable weights still average to the mean of the snapshots
            # (1.0, 3.0) -> 2.0, so the fix is not "stop averaging anything".
            assert model[0].weight.flatten()[0].item() == pytest.approx(2.0)


class TestBuildAverager:
    def test_the_default_config_asks_for_none(self):
        assert build_averager(loads_config("name: t"), _model()) is None

    def test_ema_is_selected_by_its_own_key(self):
        built = build_averager(loads_config("name: t\noptim: {ema_decay: 0.99}\n"), _model())
        assert isinstance(built, ExponentialMovingAverage) and built.decay == 0.99

    def test_last_n_is_selected_by_its_own_key(self):
        built = build_averager(loads_config("name: t\noptim: {swa_last_n: 3}\n"), _model())
        assert isinstance(built, LastNAverage) and built.n == 3

    def test_each_scheme_labels_its_own_checkpoint(self):
        # The averaged weights are saved to a separate file keyed by this label;
        # a shared label would have EMA overwrite SWA.
        ema = build_averager(loads_config("name: t\noptim: {ema_decay: 0.99}\n"), _model())
        swa = build_averager(loads_config("name: t\noptim: {swa_last_n: 3}\n"), _model())
        assert ema.label != swa.label
