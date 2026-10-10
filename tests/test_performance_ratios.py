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


@pytest.mark.parametrize("freq,daily", [("B", True), ("W-FRI", False), (ffn.core._MonthEnd, False), (pd.offsets.QuarterEnd(), False), (pd.offsets.YearEnd(), False)])
def test_start_row_one_day_early_keeps_data_frequency(freq, daily):
    """bt prepends a row a day before the first date; that one gap must not make weekly or monthly prices daily."""
    index = pd.date_range("2020-01-07", periods=60, freq=freq)
    prices = pd.Series(100 * np.cumprod(1 + 0.02 * np.sin(np.arange(60))), index=index, name="asset")
    start_row = pd.Series([prices.iloc[0]], index=[index[0] - pd.Timedelta(days=1)], name="asset")

    plain = ffn.PerformanceStats(prices)
    stats = ffn.PerformanceStats(pd.concat([start_row, prices]))

    for name in ("daily_mean", "daily_vol", "daily_sharpe", "daily_sortino", "daily_skew", "daily_kurt", "best_day", "worst_day"):
        assert np.isfinite(getattr(stats, name)) == daily
    for frequency in ("monthly", "yearly"):
        for field in ("mean", "vol", "sharpe", "sortino", "skew", "kurt"):
            name = frequency + "_" + field
            assert np.isclose(getattr(stats, name), getattr(plain, name), equal_nan=True)


@pytest.mark.parametrize("start", ["2024-01-04", "2024-01-05"])
@pytest.mark.parametrize("annualization_factor", [None, 365])
def test_short_business_day_series_keeps_daily_stats(start, annualization_factor):
    dates = pd.DatetimeIndex(pd.bdate_range(start, periods=3).values)
    prices = pd.Series([100.0, 102.0, 101.0], index=dates, name="asset")
    original = prices.copy(deep=True)
    returns = prices.to_returns()
    periods = 252 if annualization_factor is None else annualization_factor

    stats = ffn.PerformanceStats(prices, annualization_factor=annualization_factor)

    assert stats.daily_mean == pytest.approx(returns.mean() * periods)
    assert stats.daily_vol == pytest.approx(returns.std(ddof=1) * np.sqrt(periods))
    assert stats.daily_sharpe == pytest.approx(ffn.calc_sharpe(returns, nperiods=periods))
    assert stats.daily_sortino == pytest.approx(ffn.calc_sortino_ratio(returns, nperiods=periods))
    assert stats.best_day == returns.max()
    assert stats.worst_day == returns.min()
    pd.testing.assert_series_equal(prices, original)


@pytest.mark.parametrize("spacing", [2, 3, 5, 10])
@pytest.mark.parametrize("prepend", [False, True])
def test_sparse_yearly_prices_keep_long_term_returns(spacing, prepend):
    dates = pd.date_range("1970-01-01", periods=6, freq=pd.offsets.YearEnd(spacing))
    prices = pd.Series([100.0, 105.0, 103.0, 110.0, 108.0, 115.0], index=dates, name="asset")
    if prepend:
        start_row = pd.Series([prices.iloc[0]], index=[dates[0] - pd.Timedelta(days=1)], name="asset")
        prices = pd.concat([start_row, prices])
    original = prices.copy(deep=True)

    stats = ffn.PerformanceStats(prices)

    for frequency in ("daily", "monthly", "yearly"):
        for field in ("mean", "vol", "sharpe", "sortino", "skew", "kurt"):
            assert pd.isna(getattr(stats, frequency + "_" + field))
    for years, field in ((3, "three_year"), (5, "five_year"), (10, "ten_year")):
        if spacing <= years:
            window = prices.loc[dates[-1] - pd.DateOffset(years=years) :]
            assert getattr(stats, field) == pytest.approx(ffn.calc_cagr(window))
        else:
            assert pd.isna(getattr(stats, field))
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


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    ],
)
@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("risk_free_value", [0, 1])
def test_sortino_integer_arithmetic_does_not_overflow(dtype, as_frame, risk_free_value):
    limits = np.iinfo(dtype.lower())
    values = np.array([limits.min, 0, limits.max], dtype=dtype.lower())
    returns = pd.Series(values, dtype=dtype, name="varying")
    if as_frame:
        constant = pd.Series(np.full(3, limits.min, dtype=dtype.lower()), dtype=dtype, name="constant")
        missing = pd.Series([pd.NA] * 3, dtype="Float64", name="all_missing")
        returns = pd.concat([returns, constant, missing], axis=1)
    original = returns.copy()
    risk_free = pd.Series([risk_free_value] * 3, dtype=dtype)

    # Promote before subtracting or squaring values that cannot be represented
    # in their original fixed-width integer dtype.
    excess = returns.astype("float64").sub(risk_free.astype("float64"), axis="index")
    downside = excess.where(excess < 0, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.divide(excess.mean(), np.sqrt((downside**2).mean()))

    actual = ffn.calc_sortino_ratio(returns, rf=risk_free, nperiods=252, annualize=False)

    if as_frame:
        pd.testing.assert_series_equal(actual, expected)
        pd.testing.assert_frame_equal(returns, original)
    else:
        assert actual == expected
        pd.testing.assert_series_equal(returns, original)


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
            if offset:
                rf = rf.resample(offset).last().to_returns()
            else:
                # Daily excess returns compare both holdings over the same observed asset interval.
                # rf.to_returns() on rf's own calendar paired Fri->Mon asset returns with Sun->Mon rf
                # returns. Read rf prices at the asset's endpoints; missing endpoints stay NaN.
                rf_endpoints = rf.reindex(returns.index)
                rf = rf_endpoints / rf_endpoints.shift(1) - 1
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


def _daily_riskfree_prices(growth):
    index = pd.date_range("2024-01-01", "2024-03-31")
    return pd.Series(100.0 * growth ** np.arange(len(index)), index=index, name="rf")


@pytest.mark.parametrize("growth", [1.0001, 0.9999])
@pytest.mark.parametrize("calendar", ["daily", "business", "gapped_business", "intraday_utc"])
def test_performance_stats_same_holding_as_riskfree_has_no_daily_excess(calendar, growth):
    """Holding the risk-free asset itself earns exactly zero excess over every asset interval."""
    risk_free = _daily_riskfree_prices(growth)
    prices = risk_free.rename("same_holding")
    if calendar != "daily":
        prices = prices[prices.index.dayofweek < 5]
    if calendar == "gapped_business":
        prices = prices.drop(prices.index[[4, 10, 23]])
    if calendar == "intraday_utc":
        risk_free = risk_free.tz_localize("UTC")
        prices = prices.tz_localize("UTC")
        prices.index = prices.index + pd.Timedelta(hours=16)
    original_rf = risk_free.copy()

    # Both holdings have identical endpoint prices; checked with plain float division.
    for start, end in zip(prices.index[:-1], prices.index[1:]):
        rf_return = float(risk_free[end.normalize()]) / float(risk_free[start.normalize()]) - 1
        assert float(prices[end]) / float(prices[start]) - 1 - rf_return == 0

    stats = ffn.PerformanceStats(prices, rf=risk_free, annualization_factor=252)

    # The daily branch ran on nonzero asset returns.
    assert np.sign(stats.daily_mean) == np.sign(growth - 1)
    assert np.isnan(stats.daily_sharpe)
    assert np.isnan(stats.daily_sortino)

    # GroupStats children rebuilt for a narrower window read rf at the window's endpoints.
    group = ffn.GroupStats(prices)
    group.set_riskfree_rate(risk_free)
    group.set_date_range(start=prices.index[5])
    assert np.isnan(group["same_holding"].daily_sharpe)
    assert np.isnan(group["same_holding"].daily_sortino)
    pd.testing.assert_series_equal(risk_free, original_rf)


def test_performance_stats_daily_riskfree_prices_match_asset_interval_oracle():
    """Nonzero daily excess returns use rf prices at each observed asset interval's endpoints."""
    risk_free = _daily_riskfree_prices(1.0002)
    index = pd.bdate_range("2024-01-02", periods=16).delete(5)
    prices = pd.Series(100.0 * np.cumprod(np.tile([1.004, 0.997, 1.001, 0.998, 1.003], 3)), index=index, name="asset")
    # An unavailable rf price removes both adjacent intervals instead of being filled.
    risk_free[index[7]] = np.nan

    excess = []
    for start, end in zip(index[:-1], index[1:]):
        rf_start, rf_end = float(risk_free[start]), float(risk_free[end])
        if np.isnan(rf_start) or np.isnan(rf_end):
            continue
        excess.append((float(prices[end]) / float(prices[start]) - 1) - (rf_end / rf_start - 1))
    n = len(excess)
    mean = sum(excess) / n
    std = (sum((x - mean) ** 2 for x in excess) / (n - 1)) ** 0.5
    downside = (sum(min(x, 0.0) ** 2 for x in excess) / n) ** 0.5
    assert n == len(index) - 3
    assert min(excess) < 0 < max(excess)

    stats = ffn.PerformanceStats(prices, rf=risk_free, annualization_factor=252)

    assert stats.daily_sharpe == pytest.approx(mean / std * 252**0.5, rel=1e-9)
    assert stats.daily_sortino == pytest.approx(mean / downside * 252**0.5, rel=1e-9)


def test_performance_stats_rejects_mixed_timezone_riskfree_prices():
    risk_free = _daily_riskfree_prices(1.0001)
    prices = risk_free[risk_free.index.dayofweek < 5].rename("asset").tz_localize("UTC")

    with pytest.raises(TypeError, match="tz-naive"):
        ffn.PerformanceStats(prices, rf=risk_free)


def _periodic_timezone_prices(calendar="business", dtype="float64"):
    # Late New York observations fall in the next UTC date, including at month/year ends.
    dates = (pd.bdate_range("2020-01-01", periods=1000) + pd.Timedelta(hours=23, minutes=30)).tz_localize("America/New_York")
    rng = np.random.default_rng(20261007)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates))), index=dates, name="asset", dtype=dtype)
    risk_free = pd.Series(100 * 1.0001 ** np.arange(len(dates)), index=dates, name="rf", dtype=dtype)
    if calendar != "business":
        last_positions = {}
        for position, date in enumerate(dates):
            key = (date.year, date.month) if calendar == "monthly" else date.year
            last_positions[key] = position
        prices = prices.iloc[list(last_positions.values())]
    return prices, risk_free


def _calendar_risk_ratios(prices, risk_free, periods):
    """Derive paired calendar returns without ffn ratios, pandas resampling, or label subtraction."""
    endpoints = []
    for series in (prices, risk_free):
        values = {}
        for date, value in series.items():
            # The asset's calendar defines both holdings' periods, irrespective of RF storage zone.
            local_date = date.tz_convert(prices.index.tz)
            key = (local_date.year, local_date.month) if periods == 12 else local_date.year
            if pd.notna(value):
                values[key] = float(value)
        endpoints.append(values)
    asset, rf = endpoints
    keys = list(asset)
    excess = []
    for start, end in zip(keys[:-1], keys[1:]):
        # Missing RF bins exclude both adjacent intervals; never bridge or fill them.
        if start in rf and end in rf:
            excess.append((asset[end] / asset[start] - 1) - (rf[end] / rf[start] - 1))
    mean = sum(excess) / len(excess)
    deviation = (sum((value - mean) ** 2 for value in excess) / (len(excess) - 1)) ** 0.5
    downside = (sum(min(value, 0.0) ** 2 for value in excess) / len(excess)) ** 0.5
    return mean / deviation * periods**0.5, mean / downside * periods**0.5


@pytest.mark.parametrize("calendar", ["business", "monthly", "yearly"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("rf_timezone", ["America/New_York", "UTC", "Asia/Tokyo"])
def test_performance_stats_periodic_riskfree_timezone_oracle(calendar, dtype, rf_timezone):
    """Equivalent instants retain asset-local endpoint prices through DST and calendar boundaries."""
    prices, risk_free = _periodic_timezone_prices(calendar, dtype)
    risk_free = risk_free.tz_convert(rf_timezone)
    original_prices, original_rf = prices.copy(), risk_free.copy()

    stats = ffn.PerformanceStats(prices, rf=risk_free, annualization_factor=365)

    frequencies = [("yearly", 1)] if calendar == "yearly" else [("monthly", 12), ("yearly", 1)]
    for frequency, periods in frequencies:
        expected = _calendar_risk_ratios(prices, risk_free, periods)
        for ratio, value in zip(("sharpe", "sortino"), expected):
            assert getattr(stats, frequency + "_" + ratio) == pytest.approx(value, rel=1e-10, abs=1e-12)
    if calendar == "yearly":
        assert np.isnan(stats.monthly_sharpe)
    assert stats.rf is risk_free
    pd.testing.assert_series_equal(prices, original_prices)
    pd.testing.assert_series_equal(risk_free, original_rf)


@pytest.mark.parametrize("rf_timezone", ["America/New_York", "UTC"])
def test_performance_stats_periodic_riskfree_missing_calendar_bins(rf_timezone):
    """Partial RF coverage and an absent month remove intervals without filling or shifting prices."""
    prices, risk_free = _periodic_timezone_prices()
    risk_free = risk_free.loc["2020-07":]
    risk_free = risk_free[~((risk_free.index.year == 2021) & (risk_free.index.month == 5))].tz_convert(rf_timezone)
    original_rf = risk_free.copy()
    stats = prices.calc_perf_stats(risk_free_rate=risk_free)

    for frequency, periods in [("monthly", 12), ("yearly", 1)]:
        for ratio, value in zip(("sharpe", "sortino"), _calendar_risk_ratios(prices, risk_free, periods)):
            assert getattr(stats, frequency + "_" + ratio) == pytest.approx(value, rel=1e-10, abs=1e-12)
    pd.testing.assert_series_equal(risk_free, original_rf)


def test_performance_stats_periodic_riskfree_keeps_distinct_observation_times():
    """A real time shift across calendar boundaries must not be mistaken for a storage-zone change."""
    prices, risk_free = _periodic_timezone_prices()
    shifted = risk_free.set_axis(risk_free.index + pd.Timedelta(hours=2)).tz_convert("UTC")
    original_rf = shifted.copy()
    stats = ffn.PerformanceStats(prices, rf=shifted)

    for frequency, periods in [("monthly", 12), ("yearly", 1)]:
        expected = _calendar_risk_ratios(prices, shifted, periods)
        unshifted = _calendar_risk_ratios(prices, risk_free, periods)
        for ratio, value, control in zip(("sharpe", "sortino"), expected, unshifted):
            actual = getattr(stats, frequency + "_" + ratio)
            assert actual == pytest.approx(value, rel=1e-10, abs=1e-12)
            assert actual != pytest.approx(control, rel=1e-10, abs=1e-12)
    pd.testing.assert_series_equal(shifted, original_rf)


def test_group_stats_periodic_riskfree_timezone_rebuilds():
    """RF updates and date-range rebuilds retain each child's calendar and original RF series."""
    prices, risk_free = _periodic_timezone_prices()
    frame = pd.concat([prices, prices.iloc[1::2].rename("gapped")], axis=1)
    original = frame.copy()
    risk_free = risk_free.tz_convert("UTC")
    original_rf = risk_free.copy()
    stats = frame.calc_stats(annualization_factor=365)
    stats.set_riskfree_rate(risk_free)

    for start in (None, prices.index[110]):
        stats.set_date_range(start=start)
        for name, child in stats.items():
            selected = frame[name].dropna().loc[start:]
            for frequency, periods in [("monthly", 12), ("yearly", 1)]:
                for ratio, value in zip(("sharpe", "sortino"), _calendar_risk_ratios(selected, risk_free, periods)):
                    field = frequency + "_" + ratio
                    assert getattr(child, field) == pytest.approx(value, rel=1e-10, abs=1e-12)
                    assert stats.stats.loc[field, name] == getattr(child, field)
            assert child.rf is risk_free
            assert child.annualization_factor == 365
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(risk_free, original_rf)


@pytest.mark.parametrize("calendar", ["monthly", "yearly"])
@pytest.mark.parametrize("naive_input", ["asset", "rf"])
def test_performance_stats_periodic_riskfree_rejects_mixed_timezones(calendar, naive_input):
    """Non-daily dispatch must preserve naive/aware rejection and staged-update atomicity."""
    prices, risk_free = _periodic_timezone_prices(calendar)
    if naive_input == "asset":
        prices = prices.tz_localize(None)
    else:
        risk_free = risk_free.tz_localize(None)
    stats = ffn.PerformanceStats(prices, rf=0.03)
    original_state = stats.__dict__.copy()
    original_rf = risk_free.copy()

    with pytest.raises(TypeError, match="tz-naive"):
        stats.set_riskfree_rate(risk_free)

    assert stats.__dict__.keys() == original_state.keys()
    for name, value in original_state.items():
        assert stats.__dict__[name] is value, name
    pd.testing.assert_series_equal(risk_free, original_rf)
