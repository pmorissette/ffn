import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", object])
@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("annualize", [False, True])
def test_sortino_preserves_clipped_downside(dtype, as_frame, annualize):
    index = pd.date_range("2024-01-01", periods=6)
    returns = pd.Series([np.nan, 0.1, -0.3, 0.0, 0.2, np.nan], index=index, dtype=dtype, name="asset")
    if as_frame:
        returns = pd.concat([returns, returns * 0, returns.abs(), returns * np.nan], axis=1)
    original = returns.copy()
    risk_free = pd.Series([0.0, 0.01, 0.02, np.nan, 0.03, 0.0], index=index[::-1], dtype=dtype)
    er = returns.to_excess_returns(risk_free, nperiods=252)
    downside_mean = (er.clip(upper=0.0) ** 2).mean()
    if as_frame and downside_mean.dtype == object:
        # Object-dtype NumPy sqrt is unsupported by the existing DataFrame path.
        with pytest.raises(TypeError, match="sqrt"):
            ffn.calc_sortino_ratio(returns, rf=risk_free, nperiods=252, annualize=annualize)
        pd.testing.assert_frame_equal(returns, original)
        return
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.divide(er.mean(), np.sqrt(downside_mean))
    if annualize:
        expected *= np.sqrt(252)

    actual = ffn.calc_sortino_ratio(returns, rf=risk_free, nperiods=252, annualize=annualize)

    if as_frame:
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
        pd.testing.assert_frame_equal(returns, original)
    else:
        assert actual == expected
        pd.testing.assert_series_equal(returns, original)


@pytest.mark.parametrize("values", [[], [np.nan] * 4, [0.0] * 4, [0.01] * 4, [-0.01] * 4])
@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_sortino_preserves_degenerate_series(values, dtype):
    returns = pd.Series(values, dtype=dtype)
    er = returns.to_excess_returns(0.0, nperiods=252)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.divide(er.mean(), np.sqrt((er.clip(upper=0.0) ** 2).mean())) * np.sqrt(252)

    actual = ffn.calc_sortino_ratio(returns, nperiods=252)

    if pd.isna(expected):
        assert pd.isna(actual)
    else:
        assert actual == expected


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
@pytest.mark.parametrize("risk_free_kind", ["zero", "scalar", "numpy_scalar", "prices"])
@pytest.mark.parametrize("annualization_factor", [None, 365])
def test_performance_stats_reuses_excess_returns(monkeypatch, dtype, risk_free_kind, annualization_factor):
    rng = np.random.default_rng(71)
    index = pd.bdate_range("2018-01-02", periods=1512)
    prices = pd.Series(100 * np.exp(rng.normal(0.0002, 0.01, len(index)).cumsum()), index=index, dtype=dtype, name="asset")
    prices.iloc[10::43] = np.nan
    original = prices.copy()
    risk_free = {"zero": 0.0, "scalar": 0.05, "numpy_scalar": np.float32(0.05)}.get(risk_free_kind)
    if risk_free_kind == "prices":
        rf_index = pd.date_range(index[0] - pd.Timedelta(days=2), index[-1] + pd.Timedelta(days=2))
        risk_free = pd.Series(100 * 1.0001 ** np.arange(len(rf_index)), index=rf_index, dtype=dtype)
        risk_free.iloc[7::31] = np.nan
    original_rf = risk_free.copy() if isinstance(risk_free, pd.Series) else risk_free
    calls = []
    to_excess_returns = pd.Series.to_excess_returns

    def record_excess_returns(returns, rf, nperiods=None):
        calls.append(nperiods)
        return to_excess_returns(returns, rf, nperiods=nperiods)

    monkeypatch.setattr(pd.Series, "to_excess_returns", record_excess_returns)

    stats = ffn.PerformanceStats(prices, rf=risk_free, annualization_factor=annualization_factor)

    assert calls == [stats.annualization_factor, 12, 1]
    for frequency, returns, periods, offset in [
        ("daily", stats.returns, stats.annualization_factor, None),
        ("monthly", stats.monthly_returns, 12, ffn.core._MonthEnd),
        ("yearly", stats.yearly_returns, 1, ffn.core._YearEnd),
    ]:
        rf = risk_free
        if isinstance(rf, pd.Series):
            rf = (rf.resample(offset).last() if offset else rf).to_returns()
        er = to_excess_returns(returns, rf, nperiods=periods)
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe = np.divide(er.mean(), er.std(ddof=1)) * np.sqrt(periods)
            sortino = np.divide(er.mean(), np.sqrt((er.clip(upper=0.0) ** 2).mean())) * np.sqrt(periods)
        assert getattr(stats, frequency + "_sharpe") == sharpe
        assert getattr(stats, frequency + "_sortino") == sortino
    pd.testing.assert_series_equal(prices, original)
    if isinstance(risk_free, pd.Series):
        pd.testing.assert_series_equal(risk_free, original_rf)


def test_performance_stats_preserves_zero_yearly_volatility():
    prices = pd.Series(100.0, index=pd.bdate_range("2018-01-02", periods=1512))

    stats = ffn.PerformanceStats(prices, rf=0.05)

    assert stats.yearly_vol == 0
    assert np.isnan(stats.yearly_sharpe)
    assert stats.yearly_sortino == ffn.calc_sortino_ratio(stats.yearly_returns, rf=0.05, nperiods=1)
