import numpy as np
import pandas as pd
import pytest

import ffn


def test_sharpe_rejects_series_without_dispersion():
    """Treat constant effective returns as undefined across supported dtypes."""
    for dtype in ("float32", "float64", "Float32", "Float64"):
        for value in (-0.001, 1e-7, 1.0):
            for length in (3, 250, 5000):
                returns = pd.Series([value] * length, dtype=dtype)
                original = returns.copy()

                # The range is an exact oracle even when the sample standard deviation
                # accumulates floating-point residue for identical stored values.
                assert returns.max() - returns.min() == 0
                for risk_free in (0.0, 0.05):
                    for actual in (
                        ffn.calc_sharpe(returns, rf=risk_free, nperiods=252),
                        returns.calc_sharpe(rf=risk_free, nperiods=252),
                        returns.calc_sharpe_ratio(rf=risk_free, nperiods=252),
                    ):
                        assert pd.isna(actual)
                pd.testing.assert_series_equal(returns, original)


def test_sharpe_uses_column_local_excess_dispersion():
    """Measure dispersion per column after aligning a Series risk-free input."""
    index = pd.date_range("2026-01-01", periods=8, freq="D")
    risk_free = pd.Series(np.linspace(0.0001, 0.0008, 6), index=index[:6])
    excess_returns = pd.DataFrame(
        {
            "constant": [0.001] * 6,
            "missing_constant": [0.001, np.nan, 0.001, 0.001, np.nan, 0.001],
            "zero": [0.0] * 6,
            "all_missing": [np.nan] * 6,
            "varying": [0.001, 0.002, 0.001, 0.003, -0.001, 0.002],
        },
        index=index[:6],
    )
    returns = excess_returns.add(risk_free, axis="index").reindex(index)
    original_returns = returns.copy()
    original_risk_free = risk_free.copy()
    expected_varying = excess_returns["varying"].mean() / excess_returns["varying"].std(ddof=1) * np.sqrt(252)

    expected = pd.Series(
        [np.nan, np.nan, np.nan, np.nan, expected_varying],
        index=returns.columns,
    )
    for result in (
        returns.calc_sharpe(rf=risk_free, nperiods=252),
        returns.calc_sharpe_ratio(rf=risk_free, nperiods=252),
    ):
        actual = pd.Series(result, index=returns.columns)
        pd.testing.assert_series_equal(actual, expected)

    pd.testing.assert_frame_equal(returns, original_returns)
    pd.testing.assert_series_equal(risk_free, original_risk_free)


def test_sharpe_preserves_quiet_dispersion():
    """Keep a finite Sharpe ratio for genuinely varying low-volatility data."""
    returns = pd.Series(1.0 + np.random.default_rng(1).normal(0.0, 1e-12, 5000))

    for scale in (1.0, 1e-7):
        scaled = returns * scale
        expected = scaled.mean() / scaled.std(ddof=1) * np.sqrt(252)
        actual = ffn.calc_sharpe(scaled, nperiods=252)

        assert np.isfinite(actual)
        assert actual == expected


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"])
@pytest.mark.parametrize("as_frame", [False, True])
def test_sharpe_integer_dispersion_does_not_overflow(dtype, as_frame):
    minimum = np.iinfo(dtype.lower()).min
    returns = pd.Series([minimum, 0, -minimum - 1], dtype=dtype, name="varying")
    constant = pd.Series([minimum] * 3, dtype=dtype, name="constant")
    if as_frame:
        missing = pd.Series([pd.NA] * 3, dtype="Float64", name="all_missing")
        returns = pd.concat([returns, constant, missing], axis=1)
    original = returns.copy()
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    if as_frame:
        expected.iloc[1] = np.nan

    actual = ffn.calc_sharpe(returns, rf=0, nperiods=252)

    if as_frame:
        pd.testing.assert_series_equal(actual, expected)
        pd.testing.assert_frame_equal(returns, original)
    else:
        assert actual == expected
        assert pd.isna(ffn.calc_sharpe(constant, rf=0, nperiods=252))
        pd.testing.assert_series_equal(returns, original)


def test_performance_stats_rejects_no_dispersion_sharpe():
    """Keep zero volatility and undefined Sharpe consistent in performance stats."""
    prices = pd.Series(2.0 ** np.arange(5), index=pd.bdate_range("2026-01-05", periods=5), name="asset")
    original = prices.copy()

    stats = ffn.PerformanceStats(prices)

    assert stats.daily_vol == 0
    assert np.isnan(stats.daily_sharpe)
    pd.testing.assert_series_equal(prices, original)


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
