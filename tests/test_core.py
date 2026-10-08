import ffn
import pandas as pd
import numpy as np
from pytest import approx, fixture, mark, raises, warns
from numpy.testing import assert_almost_equal as aae
from packaging.version import Version


@fixture
def df():
    try:
        df = pd.read_csv("tests/data/test_data.csv", index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        try:
            df = pd.read_csv("data/test_data.csv", index_col=0, parse_dates=True)
        except FileNotFoundError as e2:
            raise (str(e2))
    return df


@fixture
def ts(df):
    return df["AAPL"].iloc[0:10]


def test_mtd_ytd(df):
    data = df["AAPL"]

    # Intramonth
    prices = data[pd.to_datetime("2004-12-10"): pd.to_datetime("2004-12-25")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, -0.0175, 4)
    assert mtd_actual == ytd_actual

    # Year change - first month
    prices = data[pd.to_datetime("2004-12-10"): pd.to_datetime("2005-01-15")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, 0.0901, 4)
    assert mtd_actual == ytd_actual

    # Year change - second month
    prices = data[pd.to_datetime("2004-12-10"): pd.to_datetime("2005-02-15")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, 0.1497, 4)
    aae(ytd_actual, 0.3728, 4)

    # Single day
    prices = data[[pd.to_datetime("2004-12-10")]]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    assert mtd_actual == ytd_actual == 0


def test_mtd_uses_current_period_when_prior_month_is_unavailable():
    """Ignore an empty prior month when the current month has enough prices."""
    prices = pd.Series(
        [90.0, 100.0, 110.0],
        index=pd.to_datetime(["2024-01-31", "2024-03-01", "2024-03-15"]),
    )
    monthly_prices = prices.resample(ffn.core._MonthEnd).last()

    # Adding unrelated January history must not change March's 110 / 100 - 1 return.
    expected = 0.1
    assert np.isclose(ffn.calc_mtd(prices, monthly_prices), expected)
    assert np.isclose(ffn.PerformanceStats(prices).mtd, expected)


def test_ytd_uses_current_period_when_prior_year_is_unavailable():
    """Ignore an empty prior year when the current year has enough prices."""
    prices = pd.Series(
        [90.0, 100.0, 110.0],
        index=pd.to_datetime(["2019-12-31", "2021-01-01", "2021-01-15"]),
    )
    yearly_prices = prices.resample(ffn.core._YearEnd).last()

    # Adding unrelated 2019 history must not change 2021's 110 / 100 - 1 return.
    expected = 0.1
    assert np.isclose(ffn.calc_ytd(prices, yearly_prices), expected)
    assert np.isclose(ffn.PerformanceStats(prices).ytd, expected)


def test_mtd_ytd_select_current_period_fallback_per_column():
    """Retain each available prior endpoint in mixed DataFrame inputs."""
    cases = (
        (ffn.calc_mtd, ffn.core._MonthEnd, ["2024-01-31", "2024-02-29", "2024-03-01", "2024-03-15"]),
        (ffn.calc_ytd, ffn.core._YearEnd, ["2019-12-31", "2020-12-31", "2021-01-01", "2021-01-15"]),
    )
    for calculator, frequency, dates in cases:
        daily_prices = pd.DataFrame(
            {
                "gap": [90.0, np.nan, 100.0, 110.0],
                "prior": [10.0, 20.0, 30.0, 35.0],
                "single": [90.0, np.nan, np.nan, 110.0],
            },
            index=pd.to_datetime(dates),
        )
        period_prices = daily_prices.resample(frequency).last()

        # gap uses 110 / 100 - 1; prior keeps 35 / 20 - 1; single stays unavailable.
        expected = pd.Series({"gap": 0.1, "prior": 0.75, "single": np.nan})
        pd.testing.assert_series_equal(calculator(daily_prices, period_prices), expected)


def test_mtd_ytd_keep_unavailable_with_one_current_period_price():
    """Do not manufacture a zero return from one current-period observation."""
    cases = (
        (ffn.calc_mtd, ffn.core._MonthEnd, ["2024-01-31", "2024-03-15"]),
        (ffn.calc_ytd, ffn.core._YearEnd, ["2019-12-31", "2021-01-15"]),
    )
    for calculator, frequency, dates in cases:
        daily_prices = pd.Series([90.0, 110.0], index=pd.to_datetime(dates))
        period_prices = daily_prices.resample(frequency).last()

        assert pd.isna(calculator(daily_prices, period_prices))


@mark.parametrize("dtype", ["float64", "Float64"])
@mark.parametrize("timezone", [None, "America/New_York"])
@mark.parametrize(
    "padding_dates",
    [
        ["2024-02-20"],
        ["2024-03-15"],
        ["2024-03-15", "2024-04-15"],
        ["2025-03-15"],
        ["2025-03-15", "2026-03-15"],
    ],
    ids=["same-period", "next-month", "future-months", "next-year", "future-years"],
)
def test_performance_stats_current_period_returns_ignore_future_missing_bins(dtype, timezone, padding_dates):
    """Reporting-calendar padding cannot move the observed return period."""
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-15", "2024-02-01", "2024-02-15"]).tz_localize(timezone)
    observed = pd.Series([80, 90, 95, 100, 110], index=dates, dtype=dtype, name="asset")
    padding = pd.Series(np.nan, index=pd.to_datetime(padding_dates).tz_localize(timezone), dtype=dtype, name="asset")
    prices = pd.concat([observed, padding])
    original = prices.copy()

    stats = ffn.PerformanceStats(prices)

    # February uses January's last price; 2024 uses the last price of 2023.
    assert stats.mtd == approx(110 / 95 - 1)
    assert stats.ytd == approx(110 / 80 - 1)
    assert stats.end == dates[-1]
    # Other reports intentionally retain unavailable month/year bins.
    pd.testing.assert_series_equal(stats.monthly_prices, prices.resample(ffn.core._MonthEnd).last())
    pd.testing.assert_series_equal(stats.yearly_prices, prices.resample(ffn.core._YearEnd).last())
    pd.testing.assert_series_equal(stats.monthly_returns, stats.monthly_prices.to_returns())
    pd.testing.assert_series_equal(prices, original)


@mark.parametrize(
    "field, dates, padding_date",
    [
        ("mtd", ["2024-01-31", "2024-03-01", "2024-03-15"], "2024-05-15"),
        ("ytd", ["2019-12-31", "2021-01-01", "2021-01-15"], "2023-03-15"),
    ],
)
@mark.parametrize("current_count", [1, 2])
def test_performance_stats_current_period_padding_preserves_missing_prior_fallback(field, dates, padding_date, current_count):
    """Padding must preserve both the observed fallback and its two-price guard."""
    prices = pd.Series([90, 100, 110], index=pd.to_datetime(dates), dtype="Float64", name="asset")
    if current_count == 1:
        prices.iloc[1] = pd.NA
    prices.loc[pd.Timestamp(padding_date)] = pd.NA

    actual = getattr(ffn.PerformanceStats(prices), field)

    if current_count == 1:
        assert pd.isna(actual)
    else:
        # The prior period is empty: use 110 / 100, not the older 90 price.
        assert actual == approx(0.1)


@mark.parametrize("end", ["2024-02-29", "2024-12-31"])
@mark.parametrize("separate_days", [False, True])
def test_performance_stats_current_period_padding_respects_daily_cardinality(end, separate_days):
    """Calendar-end selection must not turn one sampled day into a defined return."""
    last = pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(hours=16)
    first = last - (pd.Timedelta(days=1) if separate_days else pd.Timedelta(hours=1))
    prices = pd.Series([100, 110, np.nan], index=pd.DatetimeIndex([first, last, last + pd.DateOffset(years=1)]), name="asset")

    stats = ffn.PerformanceStats(prices)

    assert stats.total_return == approx(0.1)
    if separate_days:
        assert stats.mtd == approx(0.1)
        assert stats.ytd == approx(0.1)
    else:
        assert pd.isna(stats.mtd)
        assert pd.isna(stats.ytd)


def test_performance_stats_current_period_padding_public_reports_and_range_reset():
    """The observed-period correction reaches public reports and range rebuilds."""
    prices = pd.Series(
        [80, 90, 95, 100, 110, np.nan],
        index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-15", "2024-02-01", "2024-02-15", "2026-03-15"]),
        name="asset",
    )
    original = prices.copy()
    results = (ffn.PerformanceStats(prices), ffn.calc_perf_stats(prices), ffn.calc_stats(prices), prices.calc_perf_stats(), prices.calc_stats())

    for stats in results:
        for field, expected in (("mtd", 110 / 95 - 1), ("ytd", 110 / 80 - 1)):
            assert getattr(stats, field) == approx(expected)
            assert stats.stats[field] == approx(expected)
            assert stats.lookback_returns[field] == approx(expected)
            assert f"{field.upper()},{expected:.2%}" in stats.to_csv().splitlines()
        stats.set_date_range(start="2024-02-01")
        assert stats.mtd == approx(0.1)
        assert stats.ytd == approx(0.1)
        stats.set_date_range()
        assert stats.mtd == approx(110 / 95 - 1)
        assert stats.ytd == approx(110 / 80 - 1)

    pd.testing.assert_series_equal(prices, original)


def test_to_returns_ts(ts):
    data = ts
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual.iloc[0])
    aae(actual.iloc[1], -0.019, 3)
    aae(actual.iloc[9], -0.022, 3)


def test_to_returns_df(df):
    data = df
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.iloc[0]))
    aae(actual["AAPL"].iloc[1], -0.019, 3)
    aae(actual["AAPL"].iloc[9], -0.022, 3)
    aae(actual["MSFT"].iloc[1], -0.011, 3)
    aae(actual["MSFT"].iloc[9], -0.014, 3)
    aae(actual["C"].iloc[1], -0.012, 3)
    aae(actual["C"].iloc[9], 0.004, 3)


def test_to_log_returns_ts(ts):
    data = ts
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual.iloc[0])
    aae(actual.iloc[1], -0.019, 3)
    aae(actual.iloc[9], -0.022, 3)


def test_to_log_returns_df(df):
    data = df
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.iloc[0]))
    aae(actual["AAPL"].iloc[1], -0.019, 3)
    aae(actual["AAPL"].iloc[9], -0.022, 3)
    aae(actual["MSFT"].iloc[1], -0.011, 3)
    aae(actual["MSFT"].iloc[9], -0.014, 3)
    aae(actual["C"].iloc[1], -0.012, 3)
    aae(actual["C"].iloc[9], 0.004, 3)


def test_to_log_returns_preserves_dataframe_missing_gaps():
    """Require both adjacent prices before calculating each log return."""
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            "complete": [100.0, 110.0, 121.0, 133.1],
            "leading_gap": [np.nan, 50.0, 55.0, 60.5],
            "internal_gap": [20.0, np.nan, 22.0, 24.2],
            "trailing_gap": [30.0, 33.0, np.nan, np.nan],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=index,
    )
    change = np.log(1.1)
    expected = pd.DataFrame(
        {
            "complete": [np.nan, change, change, change],
            "leading_gap": [np.nan, np.nan, change, change],
            "internal_gap": [np.nan, np.nan, np.nan, change],
            "trailing_gap": [np.nan, change, np.nan, np.nan],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=index,
    )

    pd.testing.assert_frame_equal(ffn.to_log_returns(prices), expected)
    pd.testing.assert_frame_equal(prices.to_log_returns(), expected)


def test_to_log_returns_preserves_nullable_series_missing_gaps():
    """Preserve nullable missing prices instead of filling across them."""
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, pd.NA, 110.0, 121.0], index=index, dtype="Float64", name="asset")
    expected = pd.Series(
        [pd.NA, pd.NA, pd.NA, np.log(1.1)],
        index=index,
        dtype="Float64",
        name="asset",
    )

    pd.testing.assert_series_equal(ffn.to_log_returns(prices), expected)
    pd.testing.assert_series_equal(prices.to_log_returns(), expected)


def test_to_price_index(df):
    data = df
    rets = data.to_returns()
    actual = rets.to_price_index()

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 100, 3)
    aae(actual["MSFT"].iloc[0], 100, 3)
    aae(actual["C"].iloc[0], 100, 3)
    aae(actual["AAPL"].iloc[9], 91.366, 3)
    aae(actual["MSFT"].iloc[9], 95.191, 3)
    aae(actual["C"].iloc[9], 101.199, 3)

    actual = rets.to_price_index(start=1)

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 1, 3)
    aae(actual["MSFT"].iloc[0], 1, 3)
    aae(actual["C"].iloc[0], 1, 3)
    aae(actual["AAPL"].iloc[9], 0.914, 3)
    aae(actual["MSFT"].iloc[9], 0.952, 3)
    aae(actual["C"].iloc[9], 1.012, 3)


def test_to_price_index_mixed_leading_nan_dataframe():
    returns = pd.DataFrame(
        {
            "missing_first": [np.nan, 0.10, -0.05],
            "valid_first": [0.20, -0.10, 0.05],
        },
        index=pd.date_range("2024-01-08", periods=3, freq="B"),
    )

    start = 250
    prices = returns.to_price_index(start=start)

    assert len(prices) == len(returns) + 1
    assert prices.index.is_unique
    assert prices.index[0] == returns.index[0] - pd.offsets.BusinessDay()
    assert np.allclose(prices.iloc[0].to_numpy(), start)

    roundtrip = prices.to_returns()
    assert np.allclose(roundtrip["valid_first"].iloc[1:].to_numpy(), returns["valid_first"].to_numpy())
    assert np.isclose(roundtrip["missing_first"].iloc[1], 0)
    assert np.allclose(roundtrip["missing_first"].iloc[2:].to_numpy(), returns["missing_first"].iloc[1:].to_numpy())


def test_to_price_index_mixed_leading_nan_range_index():
    returns = pd.DataFrame({"missing_first": [np.nan, 0.10], "valid_first": [0.20, -0.10]})

    prices = returns.to_price_index()

    assert prices.index.equals(pd.Index([-1, 0, 1]))
    assert np.allclose(prices.to_returns().iloc[1:].to_numpy(), returns.fillna(0).to_numpy())


def test_to_price_index_preserves_duplicate_columns():
    returns = pd.DataFrame(
        [[np.nan, 0.20], [0.10, -0.10]],
        columns=["duplicate", "duplicate"],
        index=pd.date_range("2024-01-01", periods=2),
    )

    prices = returns.to_price_index()

    assert prices.columns.equals(returns.columns)
    assert np.allclose(prices.to_returns().iloc[1:].to_numpy(), returns.fillna(0).to_numpy())


def test_to_price_index_nullable_series():
    returns = pd.Series([pd.NA, 0.10], dtype="Float64")

    prices = returns.to_price_index()

    assert prices.index.equals(returns.index)
    assert np.allclose(prices.to_numpy(dtype=float), [100, 110])


def test_to_price_index_preserves_first_return_in_stats():
    returns = pd.Series(
        [0.10, -0.05, 0.02, 0.03],
        index=pd.date_range("2024-01-08", periods=4, freq="B"),
        name="strategy",
    )

    prices = returns.to_price_index()
    stats = prices.calc_stats()

    assert prices.index.is_unique
    assert np.allclose(stats.returns.dropna(), returns)
    assert np.isclose(stats.daily_mean, returns.mean() * 252)
    assert np.isclose(stats.total_return, np.prod(1 + returns) - 1)


@mark.parametrize("as_frame", [False, True])
@mark.parametrize("length", [1, 2])
@mark.parametrize(
    "index",
    [
        pd.Index(["first", "second"]),
        pd.Index([10, 20]),
        pd.period_range("2024-01", periods=2, freq="M"),
        pd.timedelta_range("1 day", periods=2),
        pd.CategoricalIndex(["first", "second"]),
        pd.MultiIndex.from_tuples([("asset", 1), ("asset", 2)]),
    ],
    ids=["string", "integer", "period", "timedelta", "categorical", "multi"],
)
def test_to_price_index_rejects_unsupported_baseline_index(index, length, as_frame):
    """A mixed first row still needs a distinct label for the baseline."""
    returns = pd.Series([0.1, 0.2], index=index, name="asset").iloc[:length]
    if as_frame:
        returns = pd.concat([returns * np.nan, returns], axis=1)
    original = returns.copy(deep=True)

    for convert in (ffn.to_price_index, lambda data: data.to_price_index()):
        with raises(TypeError, match="DatetimeIndex or RangeIndex"):
            convert(returns)
        if as_frame:
            pd.testing.assert_frame_equal(returns, original)
        else:
            pd.testing.assert_series_equal(returns, original)


@mark.parametrize("as_frame", [False, True])
@mark.parametrize(
    "index",
    [
        pd.Index(["first", "second"]),
        pd.Index([10, 20]),
        pd.period_range("2024-01", periods=2, freq="M"),
        pd.timedelta_range("1 day", periods=2),
        pd.CategoricalIndex(["first", "second"]),
        pd.MultiIndex.from_tuples([("asset", 1), ("asset", 2)]),
    ],
    ids=["string", "integer", "period", "timedelta", "categorical", "multi"],
)
def test_to_price_index_reuses_leading_missing_baseline_for_any_index(index, as_frame):
    """An existing missing baseline needs no synthesized label or index restriction."""
    returns = pd.Series([pd.NA, 0.2], index=index, name="asset", dtype="Float64")
    if as_frame:
        returns = pd.concat([returns, returns], axis=1)
    original = returns.copy(deep=True)

    for convert in (ffn.to_price_index, lambda data, start: data.to_price_index(start=start)):
        prices = convert(returns, start=250)
        pd.testing.assert_index_equal(prices.index, returns.index)
        assert np.allclose(prices.iloc[0].to_numpy(dtype=float) if as_frame else prices.iloc[0], 250)
        assert np.allclose(prices.iloc[1].to_numpy(dtype=float) if as_frame else prices.iloc[1], 300)
        if as_frame:
            pd.testing.assert_index_equal(prices.columns, returns.columns)
            pd.testing.assert_frame_equal(returns, original)
        else:
            assert prices.name == returns.name
            pd.testing.assert_series_equal(returns, original)


@mark.parametrize("as_frame", [False, True])
@mark.parametrize(
    "index, baseline",
    [
        (pd.RangeIndex(2), -1),
        (pd.RangeIndex(5, 9, 2), 3),
        (pd.RangeIndex(5, 1, -2), 7),
        (pd.date_range("2024-01-08", periods=2, freq="B"), pd.Timestamp("2024-01-05")),
        (pd.DatetimeIndex(["2024-01-08", "2024-01-09", "2024-01-10"]), pd.Timestamp("2024-01-07")),
        (pd.DatetimeIndex(["2024-01-08", "2024-01-11"]), pd.Timestamp("2024-01-05")),
        (pd.DatetimeIndex(["2024-01-08"]), pd.Timestamp("2024-01-07")),
        (pd.date_range("2024-01-08", periods=2, tz="UTC"), pd.Timestamp("2024-01-07", tz="UTC")),
    ],
    ids=["range", "stepped-range", "descending-range", "business", "inferred", "irregular", "single", "timezone"],
)
def test_to_price_index_supported_baselines_preserve_first_return(index, baseline, as_frame):
    """Probe every predecessor branch with nullable returns and an independent price path."""
    returns = pd.Series([0.1, 0.2, -0.1][: len(index)], index=index, name="asset", dtype="Float64")
    if as_frame:
        returns = pd.concat([returns, returns], axis=1)
    original = returns.copy(deep=True)
    expected = np.array([250.0, 275.0, 330.0, 297.0][: len(index) + 1])

    for convert in (ffn.to_price_index, lambda data, start: data.to_price_index(start=start)):
        prices = convert(returns, start=250)
        assert len(prices) == len(returns) + 1
        assert prices.index.is_unique
        assert prices.index[0] == baseline
        assert prices.index[1:].equals(returns.index)
        actual = prices.to_numpy(dtype=float)
        assert np.allclose(actual, expected[:, None] if as_frame else expected)
        assert np.allclose(prices.to_returns().iloc[1:].to_numpy(dtype=float), returns.to_numpy(dtype=float))
        if as_frame:
            pd.testing.assert_index_equal(prices.columns, returns.columns)
            pd.testing.assert_frame_equal(returns, original)
        else:
            assert prices.name == returns.name
            pd.testing.assert_series_equal(returns, original)


def test_rebase(df):
    data = df
    actual = data.rebase()

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 100, 3)
    aae(actual["MSFT"].iloc[0], 100, 3)
    aae(actual["C"].iloc[0], 100, 3)
    aae(actual["AAPL"].iloc[9], 91.366, 3)
    aae(actual["MSFT"].iloc[9], 95.191, 3)
    aae(actual["C"].iloc[9], 101.199, 3)


def test_rebase_uses_each_columns_first_valid_price():
    """Rebase every DataFrame column from its own first observed price."""
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            "complete": [100.0, 110.0, 121.0, 133.1],
            "leading_gap": [np.nan, 50.0, 55.0, 60.5],
            "internal_gap": [20.0, np.nan, 22.0, 24.2],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=index,
    )
    expected = pd.DataFrame(
        {
            "complete": [100.0, 110.0, 121.0, 133.1],
            "leading_gap": [np.nan, 100.0, 110.0, 121.0],
            "internal_gap": [100.0, np.nan, 110.0, 121.0],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=index,
    )

    module_result = ffn.rebase(prices)
    method_result = prices.rebase()

    assert isinstance(module_result, pd.DataFrame)
    assert isinstance(method_result, pd.DataFrame)
    pd.testing.assert_frame_equal(module_result, expected)
    pd.testing.assert_frame_equal(method_result, expected)


def test_rebase_nullable_series_uses_first_valid_price():
    """Preserve nullable gaps while rebasing from the first observed price."""
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series(
        [pd.NA, 50.0, 55.0, 60.5], index=index, dtype="Float64", name="asset"
    )
    expected = pd.Series(
        [pd.NA, 100.0, 110.0, 121.0], index=index, dtype="Float64", name="asset"
    )

    module_result = ffn.rebase(prices)
    method_result = prices.rebase()

    assert isinstance(module_result, pd.Series)
    assert isinstance(method_result, pd.Series)
    pd.testing.assert_series_equal(module_result, expected)
    pd.testing.assert_series_equal(method_result, expected)


def test_to_drawdown_series_ts(ts):
    data = ts
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual.iloc[0], 0, 3)
    aae(actual.iloc[1], -0.019, 3)
    aae(actual.iloc[9], -0.086, 3)


def test_to_drawdown_series_df(df):
    data = df
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 0, 3)
    aae(actual["MSFT"].iloc[0], 0, 3)
    aae(actual["C"].iloc[0], 0, 3)

    aae(actual["AAPL"].iloc[1], -0.019, 3)
    aae(actual["MSFT"].iloc[1], -0.011, 3)
    aae(actual["C"].iloc[1], -0.012, 3)

    aae(actual["AAPL"].iloc[9], -0.086, 3)
    aae(actual["MSFT"].iloc[9], -0.048, 3)
    aae(actual["C"].iloc[9], -0.029, 3)


def test_to_drawdown_series_handles_dataframe_gaps():
    prices = pd.DataFrame(
        {
            "leading_gap": pd.Series([pd.NA, 100.0, 90.0, pd.NA, 110.0], dtype="Float64"),
            "internal_gap": pd.Series([50.0, 45.0, pd.NA, 55.0, 44.0], dtype="Float64"),
        }
    )
    expected = pd.DataFrame(
        {
            "leading_gap": pd.Series([pd.NA, 0.0, -0.1, -0.1, 0.0], dtype="Float64"),
            "internal_gap": pd.Series([0.0, -0.1, -0.1, 0.0, -0.2], dtype="Float64"),
        }
    )
    original = prices.copy()

    pd.testing.assert_frame_equal(ffn.to_drawdown_series(prices), expected)
    pd.testing.assert_frame_equal(prices.to_drawdown_series(), expected)
    pd.testing.assert_frame_equal(prices, original)


def test_max_drawdown_ts(ts):
    data = ts
    actual = data.calc_max_drawdown()

    aae(actual, -0.086, 3)


def test_max_drawdown_df(df):
    data = df
    data = data[0:10]
    actual = data.calc_max_drawdown()

    aae(actual["AAPL"], -0.086, 3)
    aae(actual["MSFT"], -0.048, 3)
    aae(actual["C"], -0.033, 3)


def test_year_frac():
    actual = ffn.year_frac(pd.to_datetime("2004-03-10"), pd.to_datetime("2004-03-29"))
    # not exactly the same as excel but close enough
    aae(actual, 0.0520, 4)


def test_cagr_ts(ts):
    data = ts
    actual = data.calc_cagr()
    aae(actual, -0.921, 3)


def test_cagr_df(df):
    data = df
    actual = data.calc_cagr()
    aae(actual["AAPL"], 0.440, 3)
    aae(actual["MSFT"], 0.041, 3)
    aae(actual["C"], -0.205, 3)


@mark.parametrize("calculate", [ffn.calc_cagr, ffn.calc_total_return], ids=["cagr", "total-return"])
@mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", "Int64", "object"])
def test_return_helpers_preserve_complete_endpoints(calculate, dtype):
    dates = pd.date_range("2020-01-01", periods=3, freq="YS")
    prices = pd.DataFrame({"a": [100, None, 121], "b": [50, None, 55]}, index=dates, dtype=dtype)
    prices.columns.name = "asset"
    original = prices.copy()
    expected = prices.iloc[-1] / prices.iloc[0]
    if calculate is ffn.calc_cagr:
        years = (dates[-1] - dates[0]) / pd.Timedelta(days=365.25)
        expected = expected ** (1 / years)
    expected = expected - 1

    pd.testing.assert_series_equal(calculate(prices), expected)
    pd.testing.assert_frame_equal(prices, original)


@mark.parametrize("calculate", [ffn.calc_cagr, ffn.calc_total_return], ids=["cagr", "total-return"])
@mark.parametrize("missing", [[None, None, None], [None, 100.0, None]], ids=["all-missing", "one-observation"])
def test_return_helpers_preserve_object_frame_arithmetic(calculate, missing):
    dates = pd.date_range("2020-01-01", periods=3, freq="YS")
    prices = pd.DataFrame({"valid": [100.0, 110.0, 121.0], "missing": missing}, index=dates, dtype=object)
    original = prices.copy()
    expected = prices.iloc[-1] / prices.iloc[0]
    if calculate is ffn.calc_cagr:
        years = (dates[-1] - dates[0]) / pd.Timedelta(days=365.25)
        expected = expected ** (1 / years)
    expected = expected - 1

    pd.testing.assert_series_equal(calculate(prices), expected)
    pd.testing.assert_frame_equal(prices, original)


@mark.parametrize("calculate", [ffn.calc_cagr, ffn.calc_total_return], ids=["cagr", "total-return"])
@mark.parametrize("dtype", ["Float64", pd.SparseDtype(float)], ids=["nullable", "sparse"])
def test_return_helpers_preserve_ragged_frame_dtypes(calculate, dtype):
    dates = pd.date_range("2020-01-01", periods=3, freq="YS")
    prices = pd.DataFrame({"a": [None, 100.0, 121.0], "b": [100.0, 110.0, None]}, index=dates, dtype=dtype)
    original = prices.copy()
    expected = pd.Series([121.0, 110.0], index=prices.columns, dtype=dtype) / 100.0
    if calculate is ffn.calc_cagr:
        years = np.array([(dates[2] - dates[1]), (dates[1] - dates[0])]) / pd.Timedelta(days=365.25)
        expected = expected ** (1 / years)
    expected = expected - 1

    pd.testing.assert_series_equal(calculate(prices), expected)
    pd.testing.assert_frame_equal(prices, original)


def test_calc_cagr_uses_observed_price_endpoints():
    dates = pd.date_range("2020-01-01", periods=4, freq="YS", tz="UTC")
    prices = pd.DataFrame(
        {
            "a": [np.nan, 100.0, 121.0, np.nan],
            "b": [100.0, 110.0, np.nan, np.nan],
            "internal_gap": [100.0, np.nan, np.nan, 121.0],
            "one_observation": [np.nan, 100.0, np.nan, np.nan],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )
    original = prices.copy()

    # Each CAGR uses the elapsed time between that column's observed endpoints.
    year_a = (dates[2] - dates[1]) / pd.Timedelta(days=365.25)
    year_b = (dates[1] - dates[0]) / pd.Timedelta(days=365.25)
    year_internal = (dates[3] - dates[0]) / pd.Timedelta(days=365.25)
    expected = pd.Series(
        {
            "a": (121.0 / 100.0) ** (1 / year_a) - 1,
            "b": (110.0 / 100.0) ** (1 / year_b) - 1,
            "internal_gap": (121.0 / 100.0) ** (1 / year_internal) - 1,
            "one_observation": np.nan,
            "all_missing": np.nan,
        }
    )

    pd.testing.assert_series_equal(ffn.calc_cagr(prices), expected)
    pd.testing.assert_series_equal(prices.calc_cagr(), expected)
    assert np.isclose(ffn.calc_cagr(prices["a"]), expected["a"])
    assert np.isclose(prices["a"].calc_cagr(), expected["a"])
    assert np.isclose(ffn.calc_cagr(prices["a"].astype("Float64")), expected["a"])
    assert np.isclose(ffn.PerformanceStats(prices["a"]).cagr, expected["a"])
    pd.testing.assert_frame_equal(prices, original)


def test_merge():
    a = pd.Series(index=pd.date_range("2010-01-01", periods=5), data=100, name="a")
    b = pd.Series(index=pd.date_range("2010-01-02", periods=5), data=200, name="b")
    actual = ffn.merge(a, b)

    assert "a" in actual
    assert "b" in actual
    assert len(actual) == 6
    assert len(actual.columns) == 2
    assert np.isnan(actual["a"].iloc[-1])
    assert np.isnan(actual["b"].iloc[0])
    assert actual["a"].iloc[0] == 100
    assert actual["a"].iloc[1] == 100
    assert actual["b"].iloc[-1] == 200
    assert actual["b"].iloc[1] == 200

    old = actual
    old.columns = ["c", "d"]

    actual = ffn.merge(old, a, b)

    assert "a" in actual
    assert "b" in actual
    assert "c" in actual
    assert "d" in actual
    assert len(actual) == 6
    assert len(actual.columns) == 4
    assert np.isnan(actual["a"].iloc[-1])
    assert np.isnan(actual["b"].iloc[0])
    assert actual["a"].iloc[0] == 100
    assert actual["a"].iloc[1] == 100
    assert actual["b"].iloc[-1] == 200
    assert actual["b"].iloc[1] == 200


def test_calc_inv_vol_weights(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna()
    actual = ffn.core.calc_inv_vol_weights(rets)

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.218, 3)
    aae(actual["MSFT"], 0.464, 3)
    aae(actual["C"], 0.318, 3)


def test_calc_inv_vol_weights_object_regression_204(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna().astype(object)
    actual = ffn.core.calc_inv_vol_weights(rets)

    aae(actual["AAPL"], 0.218, 3)
    aae(actual["MSFT"], 0.464, 3)
    aae(actual["C"], 0.318, 3)


@mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", "object"])
@mark.parametrize("constant", [0.0, 0.1])
def test_calc_inv_vol_weights_excludes_constant_columns(dtype, constant):
    returns = pd.DataFrame(
        {
            "constant": pd.Series([constant] * 30, dtype=dtype),
            "variable": pd.Series(np.linspace(-0.075, 0.075, 30), dtype=dtype),
        }
    )
    original = returns.copy()
    expected = pd.Series([np.nan, 1.0], index=returns.columns)

    # Identical stored returns have exact zero dispersion even when std retains residue.
    assert returns["constant"].nunique(dropna=True) == 1
    for actual in (
        ffn.calc_inv_vol_weights(returns),
        returns.calc_inv_vol_weights(),
    ):
        pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", "object"])
def test_calc_inv_vol_weights_excludes_insufficient_observations(dtype):
    returns = pd.DataFrame(
        {
            "missing": pd.Series([np.nan, np.nan, np.nan], dtype=dtype),
            "single": pd.Series([0.1, np.nan, np.nan], dtype=dtype),
            "variable": pd.Series([-1e-7, 0.0, 1e-7], dtype=dtype),
            "twice_variable": pd.Series([-2e-7, 0.0, 2e-7], dtype=dtype),
        }
    )
    original = returns.copy()
    expected = pd.Series([np.nan, np.nan, 2.0 / 3.0, 1.0 / 3.0], index=returns.columns)

    for actual in (
        ffn.calc_inv_vol_weights(returns),
        returns.calc_inv_vol_weights(),
    ):
        pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original)


def test_calc_mean_var_weights(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna()
    actual = ffn.core.calc_mean_var_weights(rets)

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.000, 3)
    aae(actual["MSFT"], 0.000, 3)
    aae(actual["C"], 1.000, 3)


@mark.parametrize(
    ("covar_method", "dtype"),
    [("ledoit-wolf", "float64"), ("standard", "float64"), ("standard", "Float64")],
)
@mark.parametrize("use_pandas_method", [False, True], ids=["package", "pandas"])
def test_calc_mean_var_weights_passes_numpy_operands_to_solver(monkeypatch, covar_method, dtype, use_pandas_method):
    returns = pd.DataFrame(
        [[0.01, 0.03], [0.02, -0.01], [-0.01, 0.02]],
        columns=["first", "second"],
        dtype=dtype,
    )
    original = returns.copy()

    def fake_minimize(fitness, weights, args, **kwargs):
        expected_returns, covariance, risk_free = args
        assert isinstance(expected_returns, np.ndarray)
        assert isinstance(covariance, np.ndarray)
        assert risk_free == 0.0001
        assert kwargs["bounds"] == [(0.2, 0.8), (0.2, 0.8)]

        # Exercise the real objective so the conversion cannot change its arithmetic.
        expected_mean = sum(np.asarray(returns, dtype=float).mean(axis=0) * weights)
        expected_variance = np.dot(np.dot(weights, covariance), weights)
        assert fitness(weights, *args) == -(expected_mean - risk_free) / np.sqrt(expected_variance)

        return type("Result", (), {"success": True, "x": weights})()

    monkeypatch.setattr(ffn.core, "minimize", fake_minimize)
    calculate = returns.calc_mean_var_weights if use_pandas_method else ffn.calc_mean_var_weights
    args = () if use_pandas_method else (returns,)

    result = calculate(*args, covar_method=covar_method, weight_bounds=(0.2, 0.8), rf=0.0001)

    pd.testing.assert_series_equal(result, pd.Series([0.5, 0.5], index=returns.columns))
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize(
    ("covar_method", "dtype"),
    [(method, dtype) for method in ("ledoit-wolf", "standard") for dtype in ("float32", "float64", "Float32", "Float64")]
    + [("ledoit-wolf", object), ("standard", pd.SparseDtype("float64", 0))],
)
def test_calc_mean_var_weights_preserves_objective_summation(monkeypatch, dtype, covar_method):
    returns = pd.DataFrame(
        np.array([[0.0, 1e16 - 2, -1e16], [1.0, 1e16, -1e16 + 2], [2.0, 1e16 + 2, -1e16 - 2]]) * 2.0**-60,
        columns=["small", "positive", "negative"],
        dtype=dtype,
    )
    original = returns.copy(deep=True)

    def fake_minimize(fitness, weights, args, **kwargs):
        _, covariance, risk_free = args
        # Preserve Series iteration: Python floats and NumPy scalars take
        # different built-in sum paths on Python 3.12 and newer.
        expected_mean = sum(returns.mean() * weights)
        expected_variance = np.dot(np.dot(weights, covariance), weights)
        assert expected_variance > 0
        assert fitness(weights, *args) == -(expected_mean - risk_free) / np.sqrt(expected_variance)
        return type("Result", (), {"success": True, "x": weights})()

    monkeypatch.setattr(ffn.core, "minimize", fake_minimize)
    result = ffn.calc_mean_var_weights(returns, covar_method=covar_method)

    pd.testing.assert_series_equal(result, pd.Series([1.0 / 3] * 3, index=returns.columns))
    pd.testing.assert_frame_equal(returns, original)


def test_calc_mean_var_weights_preserves_nullable_missing_failure():
    returns = pd.DataFrame(
        {
            "missing": pd.Series([pd.NA, pd.NA, pd.NA], dtype="Float64"),
            "valid": pd.Series([0.01, -0.01, 0.02], dtype="Float64"),
        }
    )

    # SciPy 1.18 rejects the non-scalar objective before the older NA comparison.
    with raises((TypeError, ValueError), match="boolean value of NA is ambiguous|objective function must return a scalar value"):
        ffn.calc_mean_var_weights(returns, covar_method="standard")


@mark.parametrize("covar_method", ["ledoit-wolf", "standard"])
@mark.parametrize("use_pandas_method", [False, True], ids=["package", "pandas"])
def test_calc_mean_var_weights_rejects_duplicate_columns(covar_method, use_pandas_method):
    returns = pd.DataFrame(
        [[0.01, 0.03], [0.02, -0.01], [-0.01, 0.02]],
        columns=["same", "same"],
    )
    original = returns.copy()
    calculate = returns.calc_mean_var_weights if use_pandas_method else ffn.calc_mean_var_weights
    args = () if use_pandas_method else (returns,)

    with raises(ValueError, match="returns columns must be unique"):
        calculate(*args, covar_method=covar_method)

    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("return_scale", [1e-6, 1.0, 1e6])
def test_standard_covariance_rejects_indefinite_before_solver(monkeypatch, return_scale):
    returns = return_scale * pd.DataFrame(
        [
            [2.0, np.nan, 0.0],
            [np.nan, -1.0, -3.0],
            [-3.0, -3.0, np.nan],
            [np.nan, 1.0, 3.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.0, 3.0, np.nan],
            [2.0, 1.0, -3.0],
        ],
        columns=list("ABC"),
    )
    original = returns.copy(deep=True)
    covariance = returns.cov().to_numpy(dtype=float)

    # A negative determinant independently disproves positive semidefiniteness here.
    assert np.linalg.det(covariance) < 0

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("solver must not receive an indefinite covariance")

    monkeypatch.setattr(ffn.core, "minimize", unexpected_solver)
    monkeypatch.setattr(ffn.core, "_erc_weights_ccd", unexpected_solver)

    with raises(ValueError, match="standard covariance matrix must be positive semidefinite"):
        ffn.calc_mean_var_weights(returns, covar_method="standard")
    with raises(ValueError, match="standard covariance matrix must be positive semidefinite"):
        returns.calc_erc_weights(covar_method="standard")

    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("return_scale", [1e-160, 1.0, np.sqrt(8e307)])
@mark.parametrize("method", ["mean-var", "ccd", "slsqp"])
def test_standard_covariance_rejects_indefinite_at_extreme_scales(monkeypatch, return_scale, method):
    returns = return_scale * pd.DataFrame([[1.0, 1.0], [-1.0, -1.0], [0.0, np.nan], [np.nan, 0.0]], columns=["A", "B"])
    original = returns.copy(deep=True)
    covariance = returns.cov().to_numpy()
    assert np.isfinite(covariance).all()
    # A negative portfolio variance disproves PSD without relying on eigenvalues.
    direction = np.array([1.0, -1.0])
    assert direction @ (covariance / np.abs(covariance).max()) @ direction < 0

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("solver must not receive an indefinite covariance")

    monkeypatch.setattr(ffn.core, "minimize", unexpected_solver)
    monkeypatch.setattr(ffn.core, "_erc_weights_ccd", unexpected_solver)
    monkeypatch.setattr(ffn.core, "_erc_weights_slsqp", unexpected_solver)

    with raises(ValueError, match="standard covariance matrix must be positive semidefinite"):
        if method == "mean-var":
            ffn.calc_mean_var_weights(returns, covar_method="standard")
        else:
            returns.calc_erc_weights(covar_method="standard", risk_parity_method=method)

    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("scale", [1e-300, 1e-12, 1.0, 1e12, 1e308])
def test_standard_covariance_allows_rounding_negative_eigenvalue(monkeypatch, scale):
    returns = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["A", "B"])
    epsilon = np.finfo(float).eps
    # Create a one-epsilon negative eigenvalue to exercise the scale-aware roundoff allowance.
    covariance = pd.DataFrame(
        np.array([[1.0, 1.0 + epsilon], [1.0 + epsilon, 1.0]]) * scale,
        index=returns.columns,
        columns=returns.columns,
    )
    assert np.linalg.eigvalsh(covariance.to_numpy())[0] < 0
    calls = []

    monkeypatch.setattr(pd.DataFrame, "cov", lambda self: covariance)

    def fake_minimize(*args, **kwargs):
        calls.append("mean-var")
        return type("Result", (), {"success": True, "x": np.array([0.5, 0.5])})()

    def fake_ccd(*args, **kwargs):
        calls.append("erc")
        return np.array([0.5, 0.5])

    monkeypatch.setattr(ffn.core, "minimize", fake_minimize)
    monkeypatch.setattr(ffn.core, "_erc_weights_ccd", fake_ccd)

    ffn.calc_mean_var_weights(returns, covar_method="standard")
    returns.calc_erc_weights(covar_method="standard")

    assert calls == ["mean-var", "erc"]


@mark.parametrize("values", [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.25], [0.25, 1.0]]])
@mark.parametrize("scale", [1e-300, 1.0, 1e308])
def test_standard_covariance_preserves_valid_matrix(monkeypatch, values, scale):
    returns = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["A", "B"])
    covariance = pd.DataFrame(np.array(values) * scale, index=returns.columns, columns=returns.columns)
    original = covariance.copy(deep=True)
    monkeypatch.setattr(pd.DataFrame, "cov", lambda self: covariance)

    actual = ffn.core._calc_standard_covariance(returns)

    np.testing.assert_array_equal(actual, original.to_numpy())
    pd.testing.assert_frame_equal(covariance, original)


def test_calc_erc_weights(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna()

    actual = ffn.core.calc_erc_weights(rets)

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.270, 3)
    aae(actual["MSFT"], 0.374, 3)
    aae(actual["C"], 0.356, 3)

    actual = ffn.core.calc_erc_weights(
        rets, covar_method="ledoit-wolf", risk_parity_method="slsqp", tolerance=1e-9
    )

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.270, 3)
    aae(actual["MSFT"], 0.374, 3)
    aae(actual["C"], 0.356, 3)

    actual = ffn.core.calc_erc_weights(
        rets, covar_method="standard", risk_parity_method="ccd", tolerance=1e-9
    )

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.234, 3)
    aae(actual["MSFT"], 0.409, 3)
    aae(actual["C"], 0.356, 3)

    actual = ffn.core.calc_erc_weights(
        rets, covar_method="standard", risk_parity_method="slsqp", tolerance=1e-9
    )

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.234, 3)
    aae(actual["MSFT"], 0.409, 3)
    aae(actual["C"], 0.356, 3)


@mark.parametrize("covar_method", ["ledoit-wolf", "standard"])
@mark.parametrize("use_pandas_method", [False, True], ids=["package", "pandas"])
def test_calc_erc_weights_rejects_duplicate_columns_before_covariance(monkeypatch, covar_method, use_pandas_method):
    """Reject ambiguous labels before either covariance path through both public APIs."""
    returns = pd.DataFrame(
        [[0.01, 0.03], [0.02, -0.01], [-0.01, 0.02]],
        columns=["same", "same"],
    )
    original = returns.copy()

    def unexpected_covariance(*args, **kwargs):
        raise AssertionError("covariance must not be calculated")

    monkeypatch.setattr(pd.DataFrame, "cov", unexpected_covariance)
    monkeypatch.setattr(ffn.core.sklearn.covariance, "ledoit_wolf", unexpected_covariance)
    calculate = returns.calc_erc_weights if use_pandas_method else ffn.calc_erc_weights
    args = () if use_pandas_method else (returns,)

    with raises(ValueError, match="returns columns must be unique"):
        calculate(*args, covar_method=covar_method)

    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize(
    "risk_weights, message",
    [
        (np.array([0.5, 0.5]), "one value per return column"),
        (np.array([0.4, 0.3, 0.2, 0.1]), "one value per return column"),
        (np.array([[0.8, 0.1, 0.1]]), "one value per return column"),
        (np.array([1.0, -0.1, 0.1]), "finite and nonnegative with a positive total"),
        (np.array([0.0, 0.0, 0.0]), "finite and nonnegative with a positive total"),
        (np.array([0.8, np.nan, 0.2]), "finite and nonnegative with a positive total"),
        (np.array([0.8, np.inf, 0.2]), "finite and nonnegative with a positive total"),
    ],
    ids=["short", "long", "two-dimensional", "negative", "zero", "nan", "infinite"],
)
def test_calc_erc_weights_rejects_invalid_risk_weights_before_covariance(monkeypatch, risk_weights, message):
    """Reject malformed targets before covariance or solver work begins."""
    returns = pd.DataFrame(
        [[0.01, 0.02, 0.03], [0.02, -0.01, 0.01], [-0.01, 0.01, 0.02]],
        columns=list("ABC"),
    )
    original_returns = returns.copy()
    original_risk_weights = risk_weights.copy()

    def unexpected_operation(*args, **kwargs):
        raise AssertionError("covariance and solver work must not start")

    monkeypatch.setattr(pd.DataFrame, "cov", unexpected_operation)
    monkeypatch.setattr(ffn.core.sklearn.covariance, "ledoit_wolf", unexpected_operation)
    monkeypatch.setattr(ffn.core, "_erc_weights_ccd", unexpected_operation)
    monkeypatch.setattr(ffn.core, "_erc_weights_slsqp", unexpected_operation)

    with raises(ValueError, match=message):
        ffn.calc_erc_weights(returns, risk_weights=risk_weights, covar_method="standard")

    pd.testing.assert_frame_equal(returns, original_returns)
    np.testing.assert_array_equal(risk_weights, original_risk_weights)


@mark.parametrize("dtype", ["object", "Float32", "Float64"])
@mark.parametrize("as_array", [False, True])
@mark.parametrize("risk_parity_method", ["ccd", "slsqp"])
@mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
def test_calc_erc_weights_accepts_numeric_object_and_nullable_targets(dtype, as_array, risk_parity_method, covar_method):
    signs = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    returns = pd.DataFrame(signs * [0.01, 0.02, 0.04], columns=list("ABC"))
    target = pd.Series([0.8, 0.2, 0.0], dtype=dtype)
    target = target.to_numpy() if as_array else target
    original = target.copy()
    expected = ffn.calc_erc_weights(returns, risk_weights=np.asarray(target, dtype=float), covar_method=covar_method, risk_parity_method=risk_parity_method)

    actual = returns.calc_erc_weights(risk_weights=target, covar_method=covar_method, risk_parity_method=risk_parity_method)

    pd.testing.assert_series_equal(actual, expected)
    if as_array:
        np.testing.assert_array_equal(target, original)
    else:
        pd.testing.assert_series_equal(target, original)


@mark.parametrize(
    "risk_weights",
    [
        np.array([0.8 + 1j, 0.1, 0.1]),
        np.array([0.8, 0.1, 0.1], dtype=complex),
        np.array([0.8 + 1j, 0.1, 0.1], dtype=object),
        np.array([np.complex128(0.8 + 1j), 0.1, 0.1], dtype=object),
        np.array([0.8, None, 0.2], dtype=object),
        pd.Series([0.8, pd.NA, 0.2], dtype="Float64"),
        np.array([0.8, "invalid", 0.2], dtype=object),
        np.array([0.8, "invalid", 0.2]),
    ],
    ids=["complex", "complex-real", "complex-object", "numpy-complex-object", "none", "nullable-missing", "nonnumeric-object", "nonnumeric-string"],
)
@mark.parametrize("covar_method", ["standard", "ledoit-wolf"])
def test_calc_erc_weights_rejects_nonreal_or_missing_targets_before_covariance(monkeypatch, risk_weights, covar_method):
    returns = pd.DataFrame([[0.01, 0.02, 0.03], [0.02, -0.01, 0.01]], columns=list("ABC"))
    original = risk_weights.copy()

    def unexpected_operation(*args, **kwargs):
        raise AssertionError("covariance and solver work must not start")

    monkeypatch.setattr(pd.DataFrame, "cov", unexpected_operation)
    monkeypatch.setattr(ffn.core.sklearn.covariance, "ledoit_wolf", unexpected_operation)
    monkeypatch.setattr(ffn.core, "_erc_weights_ccd", unexpected_operation)
    monkeypatch.setattr(ffn.core, "_erc_weights_slsqp", unexpected_operation)

    with raises(ValueError, match="risk_weights"):
        ffn.calc_erc_weights(returns, risk_weights=risk_weights, covar_method=covar_method)

    if isinstance(risk_weights, pd.Series):
        pd.testing.assert_series_equal(risk_weights, original)
    else:
        np.testing.assert_array_equal(risk_weights, original)


@mark.parametrize("risk_parity_method", ["ccd", "slsqp"])
@mark.parametrize("use_pandas_method", [False, True], ids=["package", "pandas"])
def test_calc_erc_weights_accepts_list_initial_weights(risk_parity_method, use_pandas_method):
    """Treat documented list initial weights like equivalent NumPy arrays."""
    signs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    returns = pd.DataFrame(signs * np.array([0.01, 0.02, 0.04]), columns=list("ABC"))
    initial_weights = [0.5, 0.3, 0.2]
    expected = ffn.calc_erc_weights(
        returns,
        initial_weights=np.array(initial_weights),
        covar_method="standard",
        risk_parity_method=risk_parity_method,
        tolerance=1e-9,
    )
    calculate = returns.calc_erc_weights if use_pandas_method else ffn.calc_erc_weights
    args = () if use_pandas_method else (returns,)

    actual = calculate(
        *args,
        initial_weights=initial_weights,
        covar_method="standard",
        risk_parity_method=risk_parity_method,
        tolerance=1e-9,
    )

    pd.testing.assert_series_equal(actual, expected)
    assert initial_weights == [0.5, 0.3, 0.2]


def test_calc_erc_weights_slsqp_honors_risk_target():
    """Honor a non-equal target through both public paths and covariance methods."""
    target = np.array([0.8, 0.1, 0.1])
    covariance = np.array([[0.04, 0.012, 0.008], [0.012, 0.09, 0.018], [0.008, 0.018, 0.16]])
    returns = pd.DataFrame(
        np.random.default_rng(20260909).multivariate_normal(np.zeros(3), covariance, 500),
        columns=list("ABC"),
    )
    original_returns = returns.copy()
    original_target = target.copy()

    covariance_cases = (
        ("standard", returns.cov().to_numpy(dtype=float)),
        ("ledoit-wolf", ffn.core.sklearn.covariance.ledoit_wolf(returns)[0]),
    )
    for covar_method, estimated_covariance in covariance_cases:
        assert isinstance(estimated_covariance, np.ndarray)
        # CCD is the accepted control: it already applies the same relative target.
        ccd = ffn.calc_erc_weights(
            returns,
            risk_weights=target,
            covar_method=covar_method,
            risk_parity_method="ccd",
            tolerance=1e-9,
        )
        assert isinstance(ccd, pd.Series)
        ccd_weights = ccd.to_numpy(dtype=float)
        ccd_contributions = ccd_weights * (estimated_covariance @ ccd_weights)
        # Allow CCD's established iterative accuracy; SLSQP is checked more tightly below.
        np.testing.assert_allclose(ccd_contributions / ccd_contributions.sum(), target, atol=5e-6)

        for actual in (
            ffn.calc_erc_weights(
                returns,
                risk_weights=target,
                covar_method=covar_method,
                risk_parity_method="slsqp",
                tolerance=1e-9,
            ),
            returns.calc_erc_weights(
                risk_weights=target,
                covar_method=covar_method,
                risk_parity_method="slsqp",
                tolerance=1e-9,
            ),
        ):
            assert isinstance(actual, pd.Series)
            weights = actual.to_numpy(dtype=float)
            # Reconstruct the contribution shares independently from the optimizer objective.
            contributions = weights * (estimated_covariance @ weights)
            np.testing.assert_allclose(contributions / contributions.sum(), target, atol=1e-6)
            assert isinstance(actual.index, pd.Index)
            pd.testing.assert_index_equal(actual.index, returns.columns)
            assert actual.name == "erc"
            assert np.isfinite(weights).all()
            assert (weights >= 0).all()
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)

    pd.testing.assert_frame_equal(returns, original_returns)
    np.testing.assert_array_equal(target, original_target)


@mark.parametrize("risk_parity_method", ["ccd", "slsqp"])
def test_calc_erc_weights_matches_diagonal_risk_target(risk_parity_method):
    """Match the diagonal oracle across equivalent target and return scales."""
    signs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    returns = pd.DataFrame(signs * np.array([0.01, 0.02, 0.04]), columns=list("ABC"))
    target = np.array([0.8, 0.1, 0.1])
    original_returns = returns.copy()
    original_target = target.copy()

    # Orthogonal centered columns give a diagonal sample covariance and a closed form.
    covariance = returns.cov().to_numpy(dtype=float)
    expected = np.sqrt(target / np.diag(covariance))
    expected /= expected.sum()
    equivalent_targets = (
        target,
        target.tolist(),
        target * 10,
        target.astype("float32"),
        np.array([8_000_000_000_000_000_000, 1_000_000_000_000_000_000, 1_000_000_000_000_000_000], dtype="int64"),
        np.array([3.2e38, 4e37, 4e37], dtype="float32"),
        np.array([1.6e308, 2e307, 2e307]),
    )
    for return_scale in (1.0, 1e-4):
        for scaled_target in equivalent_targets:
            original_scaled_target = scaled_target.copy()
            actual = ffn.calc_erc_weights(
                returns * return_scale,
                risk_weights=scaled_target,
                covar_method="standard",
                risk_parity_method=risk_parity_method,
                tolerance=1e-9,
            )
            assert isinstance(actual, pd.Series)
            np.testing.assert_allclose(actual.to_numpy(dtype=float), expected, atol=1e-5)
            np.testing.assert_array_equal(scaled_target, original_scaled_target)

    pd.testing.assert_frame_equal(returns, original_returns)
    np.testing.assert_array_equal(target, original_target)


@mark.parametrize("risk_parity_method", ["ccd", "slsqp"])
def test_calc_erc_weights_preserves_zero_risk_target(risk_parity_method):
    """Preserve a valid zero-risk component through both solver branches."""
    signs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    returns = pd.DataFrame(signs * np.array([0.01, 0.02, 0.04]), columns=list("ABC"))
    target = np.array([0.8, 0.2, 0.0])
    covariance = returns.cov().to_numpy(dtype=float)

    actual = ffn.calc_erc_weights(
        returns,
        risk_weights=target,
        covar_method="standard",
        risk_parity_method=risk_parity_method,
        tolerance=1e-9,
    )

    weights = actual.to_numpy(dtype=float)
    contributions = weights * (covariance @ weights)
    np.testing.assert_allclose(contributions / contributions.sum(), target, atol=1e-5)
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)


def test_calc_total_return(df):
    prc = df.iloc[0:11]
    actual = prc.calc_total_return()

    assert len(actual) == 3
    aae(actual["AAPL"], -0.079, 3)
    aae(actual["MSFT"], -0.038, 3)
    aae(actual["C"], 0.012, 3)


def test_calc_total_return_uses_observed_price_endpoints():
    dates = pd.date_range("2020-01-01", periods=4, freq="YS")
    prices = pd.DataFrame(
        {
            "a": [np.nan, 100.0, 121.0, np.nan],
            "b": [100.0, 110.0, np.nan, np.nan],
            "internal_gap": [100.0, np.nan, np.nan, 121.0],
            "one_observation": [np.nan, 100.0, np.nan, np.nan],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )
    original = prices.copy()
    expected = pd.Series(
        {
            "a": 0.21,
            "b": 0.10,
            "internal_gap": 0.21,
            "one_observation": np.nan,
            "all_missing": np.nan,
        }
    )

    pd.testing.assert_series_equal(ffn.calc_total_return(prices), expected)
    pd.testing.assert_series_equal(prices.calc_total_return(), expected)
    assert np.isclose(ffn.calc_total_return(prices["a"]), expected["a"])
    assert np.isclose(prices["a"].calc_total_return(), expected["a"])
    assert np.isclose(ffn.calc_total_return(prices["a"].astype("Float64")), expected["a"])
    assert np.isclose(ffn.PerformanceStats(prices["a"]).total_return, expected["a"])
    pd.testing.assert_frame_equal(prices, original)


def test_get_num_days_required():
    actual = ffn.core.get_num_days_required(pd.DateOffset(months=3), perc_required=1.0)
    assert actual >= 60

    actual = ffn.core.get_num_days_required(
        pd.DateOffset(months=3), perc_required=1.0, period="m"
    )
    assert actual >= 3


def test_asfreq_actual():
    a = pd.Series(
        {pd.to_datetime("2010-02-27"): 100, pd.to_datetime("2010-03-25"): 200}
    )
    actual = a.asfreq_actual(freq=ffn.core._MonthEnd, method="ffill")

    assert len(actual) == 1
    assert "2010-02-27" in actual


def test_asfreq_actual_preserves_timezone():
    """Test that actual-frequency conversion keeps timezone-aware labels."""
    index = pd.to_datetime(["2020-01-30 16:00", "2020-02-27 16:00", "2020-03-30 16:00"]).tz_localize("UTC")
    prices = pd.Series([10, 20, 30], index=index, name="asset")
    expected = pd.Series([10, 20], index=index[:2], name="asset")

    actual = ffn.asfreq_actual(prices, ffn.core._MonthEnd)

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected)


def test_asfreq_actual_preserves_dataframe_columns():
    """Test that an existing dt column remains ordinary user data."""
    index = pd.to_datetime(["2020-01-30", "2020-02-27", "2020-03-30"])
    frame = pd.DataFrame(
        {
            "dt": pd.Series([1, pd.NA, 3], index=index, dtype="Int64"),
            "asset": [10.0, np.nan, 30.0],
        },
        index=index,
    )
    expected = frame.head(2)

    actual = frame.asfreq_actual(ffn.core._MonthEnd)

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected)


def test_to_monthly():
    a = pd.Series(range(100), index=pd.date_range("2010-01-01", periods=100))
    # to test for actual dates
    a["2010-01-31"] = np.nan
    a = a.dropna()

    actual = a.to_monthly()

    assert len(actual) == 3
    assert "2010-01-30" in actual
    assert actual["2010-01-30"] == 29


def test_to_monthly_preserves_falsey_series_name():
    """Test that actual monthly dates do not replace a falsey Series name."""
    index = pd.to_datetime(["2020-01-30", "2020-02-27", "2020-03-30"])
    prices = pd.Series([10, 20, 30], index=index, name=0)
    expected = pd.Series([10, 20], index=index[:2], name=0)

    actual = prices.to_monthly()

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected)


def test_drop_duplicate_cols():
    a = pd.Series(index=pd.date_range("2010-01-01", periods=5), data=100, name="a")
    # second version of a w/ less data
    a2 = pd.Series(index=pd.date_range("2010-01-02", periods=4), data=900, name="a")
    b = pd.Series(index=pd.date_range("2010-01-02", periods=5), data=200, name="b")
    data = ffn.merge(a, a2, b)
    original = data.copy()
    expected = ffn.merge(a, b)

    assert data["a"].shape[1] == 2
    assert len(data.columns) == 3

    actual = data.drop_duplicate_cols()

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(data, original)

    # The returned selection must not expose the caller's values to mutation.
    actual.iloc[0, 0] = -1
    pd.testing.assert_frame_equal(data, original)


def test_drop_duplicate_cols_keeps_first_tied_column():
    data = pd.DataFrame(
        [[1, 10, 100, 1000], [2, 20, 200, 2000]],
        columns=["a", "b", "a", "a"],
    )
    expected = data.iloc[:, :2]

    # All three a columns tie, so retain the first one in first-label order.
    actual = data.drop_duplicate_cols()

    pd.testing.assert_frame_equal(actual, expected)


def test_drop_duplicate_cols_preserves_unique_columns():
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    original = data.copy()

    actual = data.drop_duplicate_cols()
    pd.testing.assert_frame_equal(actual, original)

    actual.iloc[0, 0] = -1

    pd.testing.assert_frame_equal(data, original)


@mark.parametrize("dtype", ["Float64", "Int64"])
@mark.parametrize("duplicate", [False, True], ids=["unique", "duplicate"])
@mark.parametrize("use_pandas_method", [False, True], ids=["package", "pandas"])
def test_drop_duplicate_cols_isolates_nullable_values(dtype, duplicate, use_pandas_method):
    columns = ["a", "b", "a"] if duplicate else ["a", "b", "c"]
    data = pd.DataFrame([[1, 10, 100], [pd.NA, 20, 200]], columns=columns, dtype=dtype)
    original = data.copy()
    keep = [2, 1] if duplicate else [0, 1, 2]
    expected = data.iloc[:, keep].copy()

    actual = data.drop_duplicate_cols() if use_pandas_method else ffn.drop_duplicate_cols(data)

    pd.testing.assert_frame_equal(actual, expected)
    actual.iloc[0, 0] = -1
    pd.testing.assert_frame_equal(data, original)

    result_snapshot = actual.copy()
    data.iloc[1, keep[0]] = -2
    pd.testing.assert_frame_equal(actual, result_snapshot)


@mark.parametrize("label, dtype", [("only", "float64"), (7, "Float64"), (("group", "only"), "Int64")])
@mark.parametrize("use_pandas_method", [False, True])
@mark.parametrize("plot", [False, True])
def test_calc_clusters_single_asset(label, dtype, use_pandas_method, plot):
    plt = ffn.core.plt
    returns = pd.DataFrame({label: pd.Series([1, 3, 2, 4], dtype=dtype)})
    original = returns.copy(deep=True)
    figures = set(plt.get_fignums())

    try:
        actual = returns.calc_clusters(plot=plot) if use_pandas_method else ffn.calc_clusters(returns, plot=plot)

        # There is exactly one partition of a singleton set, regardless of its label.
        assert actual == {0: [label]}
        pd.testing.assert_frame_equal(returns, original)
        created = set(plt.get_fignums()) - figures
        assert len(created) == int(plot)
        if plot:
            figure = plt.figure(created.pop())
            figure.canvas.draw()
            ax = figure.axes[0]
            np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0.0, 0.0]])
            assert [text.get_text() for text in ax.texts] == [str(label)]
    finally:
        for number in set(plt.get_fignums()) - figures:
            plt.close(number)


@mark.parametrize("dtype", ["float64", "Float64", "Int64"])
def test_calc_clusters_single_asset_uses_observed_sample_without_models(monkeypatch, dtype):
    returns = pd.DataFrame({"only": pd.Series([None, 1, 2, None], dtype=dtype)})
    original = returns.copy(deep=True)

    def unexpected_model(*args, **kwargs):
        raise AssertionError("A valid singleton does not require fitting a model")

    monkeypatch.setattr(ffn.core.sklearn.manifold, "MDS", unexpected_model)
    monkeypatch.setattr(ffn.core.sklearn.cluster, "KMeans", unexpected_model)

    assert ffn.calc_clusters(returns) == {0: ["only"]}
    assert ffn.calc_clusters(returns.dropna()) == {0: ["only"]}
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("values", [[], [1.0], [1.0, 1.0], [np.nan, np.nan], [np.nan, 1.0]])
@mark.parametrize("plot", [False, True])
@mark.parametrize("dtype", ["float64", "Float64"])
def test_calc_clusters_rejects_undefined_single_asset_correlation(values, plot, dtype):
    returns = pd.DataFrame({"only": pd.Series(values, dtype=dtype)})
    original = returns.copy(deep=True)
    figures = ffn.core.plt.get_fignums()

    with raises(ValueError):
        ffn.calc_clusters(returns, plot=plot)

    pd.testing.assert_frame_equal(returns, original)
    assert ffn.core.plt.get_fignums() == figures


@mark.parametrize("n", [None, 1])
def test_calc_clusters_two_assets_preserves_model_selection(n):
    returns = pd.DataFrame({"a": [-2.0, -1.0, 0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0, -1.0, -2.0]})
    original = returns.copy(deep=True)

    actual = ffn.calc_clusters(returns, n=n)

    # Cluster numbers are model labels; the asset partition is the invariant.
    expected = {frozenset({"a"}), frozenset({"b"})} if n is None else {frozenset({"a", "b"})}
    assert {frozenset(assets) for assets in actual.values()} == expected
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("missing_first", [False, True])
@mark.parametrize("plot", [False, True])
def test_calc_clusters_rejects_all_missing_object_asset(missing_first, plot):
    returns = pd.DataFrame({"observed": [1.0, 3.0, 2.0, 4.0], "missing": pd.Series([None] * 4, dtype=object)})
    if missing_first:
        returns = returns[["missing", "observed"]]
    original = returns.copy(deep=True)
    figures = set(ffn.core.plt.get_fignums())

    try:
        # pandas 1.5 drops the object column from corr(), but this is not a singleton input.
        with raises(ValueError):
            ffn.calc_clusters(returns, plot=plot)

        pd.testing.assert_frame_equal(returns, original)
        assert set(ffn.core.plt.get_fignums()) == figures
    finally:
        for number in set(ffn.core.plt.get_fignums()) - figures:
            ffn.core.plt.close(number)


@mark.parametrize("columns, n", [(1, 1), (1, 0), (1, 2), (2, None), (0, None)])
def test_calc_clusters_preserves_other_model_paths(monkeypatch, columns, n):
    returns = pd.DataFrame(np.tile([1.0, 2.0, 3.0], (columns, 1)).T)
    original = returns.copy(deep=True)

    def model_boundary(*args, **kwargs):
        raise RuntimeError("existing model path")

    monkeypatch.setattr(ffn.core.sklearn.manifold, "MDS", model_boundary)
    with raises(RuntimeError, match="existing model path"):
        ffn.calc_clusters(returns, n=n)
    pd.testing.assert_frame_equal(returns, original)


def test_calc_ftca_avoids_rebuilding_labelled_working_frames(monkeypatch):
    class Correlation(pd.DataFrame):
        def __getitem__(self, key):
            raise AssertionError("Finite FTCA working sets should not rebuild labelled frames")

    corr = Correlation([[1.0, 0.9, 0.2, 0.1], [0.9, 1.0, 0.3, 0.2], [0.2, 0.3, 1.0, 0.8], [0.1, 0.2, 0.8, 1.0]], index=list("abcd"), columns=list("abcd"))
    returns = pd.DataFrame(columns=corr.columns)
    monkeypatch.setattr(pd.DataFrame, "corr", lambda self: corr)

    # b has the highest mean and d the lowest; each absorbs its strongly correlated peer.
    assert ffn.calc_ftca(returns) == {1: ["b", "a"], 2: ["d", "c"]}


@mark.parametrize(
    "threshold, expected",
    [(0.5, {1: ["a", "b"], 2: ["c"]}), (np.nextafter(0.5, 0.0), {1: ["a", "b", "c"]}), (0.75, {1: ["b"], 2: ["a"], 3: ["c"]})],
)
@mark.parametrize("use_pandas_method", [False, True])
def test_calc_ftca_preserves_threshold_ties_and_member_order(monkeypatch, threshold, expected, use_pandas_method):
    corr = pd.DataFrame([[1.0, 0.75, 0.25], [0.75, 1.0, 0.75], [0.25, 0.75, 1.0]], index=list("abc"), columns=list("abc"))
    snapshot = corr.copy(deep=True)
    returns = pd.DataFrame(columns=corr.columns)
    monkeypatch.setattr(pd.DataFrame, "corr", lambda self: corr)

    # a and c tie for the lowest mean; c's mean correlation with seeds a/b is exactly 0.5.
    actual = returns.calc_ftca(threshold) if use_pandas_method else ffn.calc_ftca(returns, threshold)

    assert actual == expected
    assert list(actual) == list(expected)
    pd.testing.assert_frame_equal(corr, snapshot)


@mark.parametrize("dtype", ["float64", "float32", "Float64", "Int64", "object"])
@mark.parametrize("use_pandas_method", [False, True])
def test_calc_ftca_preserves_numeric_samples_and_labels(dtype, use_pandas_method):
    columns = pd.Index([7, 2, 9], name="asset")
    returns = pd.DataFrame([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], columns=columns, dtype=dtype)
    original = returns.copy(deep=True)

    if dtype == "object" and Version(pd.__version__) < Version("2.0"):
        # pandas 1.5 excludes object columns by default; preserve its empty universe and warning.
        with warns(FutureWarning, match="numeric_only"):
            actual = returns.calc_ftca() if use_pandas_method else ffn.calc_ftca(returns)
        assert actual == {}
        pd.testing.assert_frame_equal(returns, original)
        return

    actual = returns.calc_ftca() if use_pandas_method else ffn.calc_ftca(returns)

    # Orthogonal columns tie in mean correlation; high then low are assigned before the remainder.
    assert actual == {1: [columns[2]], 2: [columns[0]], 3: [list(columns)[1]]}
    assert type(actual[1][0]) is type(columns[2])
    assert type(actual[3][0]) is type(list(columns)[1])
    pd.testing.assert_frame_equal(returns, original)
    actual[1].append("new member")
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("large_label", [2**60 + 1, -(2**60 + 1), 2**64 - 1])
@mark.parametrize("threshold", [0.1, -0.6])
@mark.parametrize("use_pandas_method", [False, True])
def test_calc_ftca_preserves_mixed_numeric_label_precision(monkeypatch, large_label, threshold, use_pandas_method):
    labels = pd.Index([large_label, 1.5, "c"], dtype=object, name="asset")
    corr = pd.DataFrame([[1.0, -0.5, 0.5], [-0.5, 1.0, -0.75], [0.5, -0.75, 1.0]], index=labels, columns=labels)
    original = corr.copy(deep=True)
    returns = pd.DataFrame(columns=labels)
    monkeypatch.setattr(pd.DataFrame, "corr", lambda self: corr)

    # The high/low seeds are a large integer and a float; a combined lookup can round the integer.
    expected = {1: [large_label, "c"], 2: [1.5]} if threshold == 0.1 else {1: [1.5, large_label, "c"]}
    actual = returns.calc_ftca(threshold) if use_pandas_method else ffn.calc_ftca(returns, threshold)

    assert actual == expected
    assert list(actual) == list(expected)
    pd.testing.assert_frame_equal(corr, original)


@mark.parametrize("numeric_only", [False, True])
@mark.parametrize("all_object", [False, True])
def test_calc_ftca_preserves_correlation_column_selection(monkeypatch, numeric_only, all_object):
    returns = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "object": pd.Series([2.0, 4.0, 6.0], dtype=object)})
    if all_object:
        returns = returns.astype(object)
    original = returns.copy(deep=True)
    native_corr = pd.DataFrame.corr
    monkeypatch.setattr(pd.DataFrame, "corr", lambda self: native_corr(self, numeric_only=numeric_only))

    # Exercise both supported pandas defaults explicitly, including a zero- or one-column result.
    expected = ({} if all_object else {1: ["numeric"]}) if numeric_only else {1: ["numeric", "object"]}
    assert ffn.calc_ftca(returns) == expected
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("values, expected", [([], {}), ([np.nan], {1: ["only"]}), ([1.0, 1.0], {1: ["only"]})])
def test_calc_ftca_preserves_empty_and_single_asset_inputs(values, expected):
    returns = pd.DataFrame({"only": values}) if values else pd.DataFrame()
    original = returns.copy(deep=True)

    assert ffn.calc_ftca(returns) == expected
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("dtype", ["float64", "Float64", "object"])
@mark.parametrize("missing_row", [False, True])
def test_calc_ftca_assigns_shared_members_to_high_seed_first(dtype, missing_row):
    high = np.array([1, 1, -1, -1])
    low = np.array([1, -1, 1, -1])
    returns = pd.DataFrame({**{f"high_{i}": high for i in range(6)}, "shared": high + low, "low": low}, dtype=dtype)
    if missing_row:
        returns.loc[len(returns)] = np.nan
    original = returns.copy(deep=True)

    if dtype == "object" and Version(pd.__version__) < Version("2.0"):
        # The legacy numeric-only default removes every object column before seed assignment.
        with warns(FutureWarning, match="numeric_only"):
            actual = ffn.calc_ftca(returns)
        assert actual == {}
        pd.testing.assert_frame_equal(returns, original)
        return

    # Preserve pandas' native seed choice: quicksort's order among ties varies across runtimes.
    high_seed = returns.corr().mean().sort_values().index[-1]
    high_members = [label for label in returns.columns[:6] if label != high_seed]
    # Orthogonal seeds split; shared correlates sqrt(0.5) with both and belongs to high first.
    assert ffn.calc_ftca(returns) == {1: [high_seed] + high_members + ["shared"], 2: ["low"]}
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("all_missing", [False, True])
def test_calc_ftca_preserves_nonfinite_correlation_fallback(all_missing):
    returns = pd.DataFrame({"a": [np.nan, np.nan, np.nan] if all_missing else [1.0, 1.0, 1.0], "b": [np.nan, np.nan, np.nan] if all_missing else [1.0, 2.0, 3.0]})
    original = returns.copy(deep=True)

    # Native skip-NaN mean ordering puts an undefined seed last, even when every mean is undefined.
    expected = {1: ["b"], 2: ["a"]} if all_missing else {1: ["a"], 2: ["b"]}
    assert ffn.calc_ftca(returns) == expected
    pd.testing.assert_frame_equal(returns, original)


def test_calc_ftca_preserves_hierarchical_labels():
    columns = pd.MultiIndex.from_tuples([("group", "a"), ("group", "b")])
    returns = pd.DataFrame([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], columns=columns)
    original = returns.copy(deep=True)

    assert ffn.calc_ftca(returns) == {1: [("group", "a"), ("group", "b")]}
    pd.testing.assert_frame_equal(returns, original)


def test_calc_ftca_preserves_missing_label_normalization():
    returns = pd.DataFrame([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], columns=[pd.NA, "b"])
    original = returns.copy(deep=True)
    expected_label = returns.corr().mean().sort_values().index[0]

    actual = ffn.calc_ftca(returns)

    # Preserve pandas' seed label: older object indexes retain pd.NA; string indexes normalize it.
    assert list(actual) == [1]
    assert len(actual[1]) == 2
    if expected_label is pd.NA:
        assert actual[1][0] is pd.NA
    else:
        assert type(actual[1][0]) is type(expected_label)
        assert np.isnan(actual[1][0])
    assert actual[1][1] == "b"
    pd.testing.assert_frame_equal(returns, original)


def test_calc_ftca_preserves_duplicate_label_rejection():
    returns = pd.DataFrame([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], columns=["a", "a"])
    original = returns.copy(deep=True)

    with raises(ValueError):
        ffn.calc_ftca(returns)

    pd.testing.assert_frame_equal(returns, original)


def test_calc_ftca_preserves_array_threshold():
    returns = pd.DataFrame([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], columns=["a", "b"])
    original = returns.copy(deep=True)

    assert ffn.calc_ftca(returns, np.array([0.5])) == {1: ["a", "b"]}
    pd.testing.assert_frame_equal(returns, original)


def test_limit_weights():
    w = {"a": 0.3, "b": 0.1, "c": 0.05, "d": 0.05, "e": 0.5}
    actual_exp = {"a": 0.3, "b": 0.2, "c": 0.1, "d": 0.1, "e": 0.3}
    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0
    for k in actual_exp:
        assert actual[k] == actual_exp[k]

    w = pd.Series(w)
    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0
    for k in actual_exp:
        assert actual[k] == actual_exp[k]

    w = pd.Series({"a": 0.29, "b": 0.1, "c": 0.06, "d": 0.05, "e": 0.5})

    assert w.sum() == 1.0

    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0

    assert all(x <= 0.3 for x in actual)

    aae(actual["a"], 0.300, 3)
    aae(actual["b"], 0.190, 3)
    aae(actual["c"], 0.114, 3)
    aae(actual["d"], 0.095, 3)
    aae(actual["e"], 0.300, 3)


def test_limit_weights_preserves_precision():
    """Preserve the original budget during proportional redistribution."""
    weights = pd.Series([0.50006, 0.29997, 0.19997], index=["a", "b", "c"])
    expected = pd.Series(
        [0.5, 0.3000060007200864, 0.1999939992799136],
        index=weights.index,
    )

    actual = ffn.core.limit_weights(weights, limit=0.5)

    pd.testing.assert_series_equal(actual, expected)
    assert actual.sum() == 1.0
    assert actual.max() <= 0.5


def test_limit_weights_accepts_exact_feasible_boundary():
    size = 49
    limit = 1.0 / size
    values = np.arange(1, size + 1, dtype=float)
    weights = pd.Series(values / values.sum(), index=[f"asset_{i}" for i in range(size)])
    original = weights.copy(deep=True)
    expected = pd.Series(np.full(size, limit), index=weights.index)

    actual = ffn.limit_weights(weights, limit=limit)

    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(weights, original)
    np.testing.assert_allclose(actual.sum(), 1.0)
    assert actual.max() <= limit


@mark.parametrize("size", [49, 98])
@mark.parametrize("concentrated", [False, True])
@mark.parametrize("dtype", ["float64", "Float64"])
@mark.parametrize("as_dict", [False, True])
def test_limit_weights_redistributes_to_zero_weights_at_feasible_boundary(size, concentrated, dtype, as_dict):
    limit = 1.0 / size
    values = np.arange(size, dtype=float)
    if concentrated:
        values[:-1] = 0.0
    weights = pd.Series(values / values.sum(), index=[f"asset_{i}" for i in range(size)], dtype=dtype)
    original = weights.copy(deep=True)
    source = weights.to_dict() if as_dict else weights
    expected = pd.Series(limit, index=weights.index, dtype="float64" if as_dict else dtype)

    actual = ffn.limit_weights(source, limit=limit)

    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(weights, original)
    if as_dict:
        assert source == original.to_dict()
    assert actual.notna().all()
    np.testing.assert_allclose(actual.sum(), 1.0)
    assert actual.max() <= limit


def test_limit_weights_rejects_nearby_infeasible_boundary():
    size = 5
    feasible_limit = 1.0 / size
    weights = pd.Series(np.full(size, feasible_limit))

    with np.testing.assert_raises_regex(ValueError, "1 / limit"):
        ffn.limit_weights(weights, limit=np.nextafter(feasible_limit, 0.0))


@mark.parametrize("values, limit, expected", [([1.0, 0.0, 0.0], 0.4, [0.4, 0.3, 0.3]), ([0.5, 0.5, 0.0], 0.5, [0.5, 0.5, 0.0])])
def test_limit_weights_handles_only_zero_weights_below_limit(values, limit, expected):
    weights = pd.Series(values)

    actual = ffn.limit_weights(weights, limit=limit)

    pd.testing.assert_series_equal(actual, pd.Series(expected))
    pd.testing.assert_series_equal(weights, pd.Series(values))


@mark.parametrize(
    "weights",
    [
        pd.Series([0.6, 0.4, pd.NA], index=["a", "b", "c"], dtype="Float64"),
        pd.Series([0.6, 0.4, np.nan], index=["a", "b", "c"]),
        pd.Series([0.6, 0.4, np.inf], index=["a", "b", "c"], dtype="Float64"),
        pd.Series([0.6, 0.4, -np.inf], index=["a", "b", "c"]),
        pd.Series([0.6, 0.4, pd.NA], index=["a", "b", "c"], dtype=object),
    ],
    ids=["nullable-missing", "float-nan", "nullable-positive-inf", "float-negative-inf", "object-missing"],
)
def test_limit_weights_rejects_nonfinite_series(weights):
    original = weights.copy(deep=True)

    with np.testing.assert_raises_regex(ValueError, "finite"):
        ffn.limit_weights(weights, limit=0.5)

    pd.testing.assert_series_equal(weights, original)


@mark.parametrize("invalid", [np.nan, pd.NA, None, np.inf, -np.inf])
def test_limit_weights_rejects_nonfinite_dict(invalid):
    weights = {"a": 0.6, "b": 0.4, "c": invalid}
    original = pd.Series(weights)

    with np.testing.assert_raises_regex(ValueError, "finite"):
        ffn.limit_weights(weights, limit=0.5)

    pd.testing.assert_series_equal(pd.Series(weights), original)


def test_limit_weights_preserves_nullable_input_during_redistribution():
    weights = pd.Series([0.6, 0.3, 0.1], index=["a", "b", "c"], dtype="Float64")
    original = weights.copy(deep=True)
    expected = pd.Series([0.5, 0.37499999999999994, 0.125], index=weights.index, dtype="Float64")

    # This valid case reaches redistribution's internal mutations; they must
    # remain isolated from the caller by the function's copy.
    actual = ffn.limit_weights(weights, limit=0.5)

    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(weights, original)
    assert actual.sum() == 1.0
    assert actual.max() <= 0.5


def test_random_weights():
    PANDAS_VERSION = Version(pd.__version__)
    PANDAS_210 = PANDAS_VERSION >= Version("2.1.0")

    select_map = "map"
    if not PANDAS_210:
        select_map = "applymap"

    n = 10
    bounds = (0.0, 1.0)
    tot = 1.0000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (getattr(df, select_map)(lambda x: (x >= low and x <= high))
            .all().all())

    n = 4
    bounds = (0.0, 0.25)
    tot = 1.0000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (
        getattr(df, select_map)(lambda x: (np.round(x, 2) >= low
                                           and np.round(x, 2) <= high))
        .all()
        .all()
    )

    n = 7
    bounds = (0.0, 0.25)
    tot = 0.8000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (
        getattr(df, select_map)(lambda x: (np.round(x, 2) >= low
                                           and np.round(x, 2) <= high))
        .all()
        .all()
    )

    n = 10
    bounds = (-0.25, 0.25)
    tot = 0.0
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (
        getattr(df, select_map)(lambda x: (np.round(x, 2) >= low
                                           and np.round(x, 2) <= high))
        .all()
        .all()
    )


def test_random_weights_throws_error():
    try:
        ffn.random_weights(2, (0.0, 0.25), 1.0)
        assert False
    except ValueError:
        assert True

    try:
        ffn.random_weights(10, (0.5, 0.25), 1.0)
        assert False
    except ValueError:
        assert True

    try:
        ffn.random_weights(10, (0.5, 0.75), 0.2)
        assert False
    except ValueError:
        assert True


@mark.parametrize(
    "values",
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    ],
    ids=["square", "wide", "tall"],
)
@mark.parametrize("path", ["package", "pandas"])
def test_plot_heatmap_labels_each_cell(values, path, recwarn):
    """Map every rectangular cell to its matching text and coordinates."""
    data = pd.DataFrame(
        values,
        index=[f"row-{i}" for i in range(len(values))],
        columns=[f"column-{i}" for i in range(len(values[0]))],
    )
    original = data.copy(deep=True)
    expected = {(column + 0.5, row + 0.5): format(data.iloc[row, column], ".1f") for row in range(data.shape[0]) for column in range(data.shape[1])}

    ffn.core.plt.close("all")
    if path == "package":
        result = ffn.plot_heatmap(data, show_legend=False, label_fmt=".1f")
    else:
        result = data.plot_heatmap(show_legend=False, label_fmt=".1f")
    result.gcf().canvas.draw()

    axes = result.gca()
    actual = {text.get_position(): text.get_text() for text in axes.texts}
    assert result is ffn.core.plt
    assert actual == expected
    assert [tick.get_text() for tick in axes.get_xticklabels()] == list(data.columns)
    assert [tick.get_text() for tick in axes.get_yticklabels()] == list(data.index)
    pd.testing.assert_frame_equal(data, original)
    assert not recwarn
    ffn.core.plt.close("all")


def test_plot_heatmap_can_disable_cell_labels(recwarn):
    data = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    ffn.core.plt.close("all")
    result = ffn.plot_heatmap(data, show_legend=False, show_labels=False)
    result.gcf().canvas.draw()

    assert len(result.gca().texts) == 0
    assert not recwarn
    ffn.core.plt.close("all")


def test_rollapply():
    a = pd.Series([1, 2, 3, 4, 5])

    actual = a.rollapply(3, np.mean)

    assert np.isnan(actual[0])
    assert np.isnan(actual[1])
    assert actual[2] == 2
    assert actual[3] == 3
    assert actual[4] == 4

    b = pd.DataFrame({"a": a, "b": a})

    actual = b.rollapply(3, np.mean)

    assert all(np.isnan(actual.iloc[0]))
    assert all(np.isnan(actual.iloc[1]))
    assert all(actual.iloc[2] == 2)
    assert all(actual.iloc[3] == 3)
    assert all(actual.iloc[4] == 4)


@mark.parametrize(
    "dtype,columns",
    [
        ("float64", pd.Index(["A", "B"], name="asset")),
        ("Float64", pd.Index(["A", "B"], name="asset")),
        ("Int64", pd.Index([1, 2], name="asset")),
        ("float64", pd.Index([("A", 1), ("B", 2)], tupleize_cols=False, name="asset")),
        ("Float64", pd.MultiIndex.from_tuples([("A", 1), ("B", 2)], names=["asset", "leg"])),
        ("float64", pd.Index([np.nan, "B"], dtype=object, name="asset")),
        ("float64", pd.Index([True, False], name="asset")),
        ("float64", pd.IntervalIndex.from_tuples([(0, 2), (1, 4)], name="asset")),
        ("float64", pd.CategoricalIndex(["A", "B"], name="asset")),
        ("float64", pd.date_range("2020-01-01", periods=2, tz="UTC", name="asset")),
        ("float64", pd.timedelta_range("1D", periods=2, name="asset")),
        ("float64", pd.period_range("2020-01", periods=2, freq="M", name="asset")),
    ],
    ids=[
        "ordinary",
        "nullable-float",
        "nullable-integer-labels",
        "tuple-labels",
        "multiindex",
        "missing-label",
        "boolean-labels",
        "overlapping-intervals",
        "categorical",
        "datetime",
        "timedelta",
        "period",
    ],
)
@mark.parametrize("reversed_row", [False, True], ids=["same-order", "reversed"])
@mark.parametrize("pandas_method", [False, True], ids=["module", "pandas"])
def test_rollapply_preserves_labeled_callback_rows(dtype, columns, reversed_row, pandas_method, recwarn):
    """Callback order must not move an asset's totals to another output column."""
    import math

    data = pd.DataFrame(
        [[1, 10], [2, None], [3, 30], [4, 40]],
        index=pd.date_range("2020-01-01", periods=4, tz="UTC", name="observed"),
        columns=columns,
        dtype=dtype,
    )
    original = data.copy(deep=True)
    for window in [1, 2, len(data), len(data) + 1]:
        calls = []

        def callback(sample, calls=calls):
            row = sample.sum()
            if reversed_row:
                row = row.iloc[::-1]
            calls.append((sample.index.copy(), sample.columns.copy(), row, row.copy(deep=True)))
            return row

        actual = data.rollapply(window, callback) if pandas_method else ffn.rollapply(data, window, callback)
        expected = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
        for end in range(window - 1, len(data)):
            # Scalar source-column sums are independent of the callback's Series order.
            expected.iloc[end] = [math.fsum(float(value) for value in data.iloc[end - window + 1 : end + 1, column] if not pd.isna(value)) for column in range(len(data.columns))]
        pd.testing.assert_frame_equal(actual, expected)
        assert len(calls) == max(0, len(data) - window + 1)
        for end, (index, labels, row, snapshot) in enumerate(calls, start=window - 1):
            pd.testing.assert_index_equal(index, data.index[end - window + 1 : end + 1])
            pd.testing.assert_index_equal(labels, data.columns)
            pd.testing.assert_series_equal(row, snapshot)
        pd.testing.assert_frame_equal(data, original)
    assert not recwarn


@mark.parametrize("kind", ["array", "list", "scalar"])
@mark.parametrize("pandas_method", [False, True], ids=["module", "pandas"])
def test_rollapply_preserves_positional_callback_rows(kind, pandas_method):
    """Unlabeled results retain positional assignment and scalar broadcasting."""
    data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]})

    def callback(sample):
        if kind == "scalar":
            return 7.0
        # The reverse order is intentional: unlabeled values must stay positional.
        values = [sample["B"].sum(), sample["A"].sum()]
        return np.array(values) if kind == "array" else values

    actual = data.rollapply(2, callback) if pandas_method else ffn.rollapply(data, 2, callback)
    values = [[np.nan, np.nan], [7.0, 7.0], [7.0, 7.0]] if kind == "scalar" else [[np.nan, np.nan], [30.0, 3.0], [50.0, 5.0]]
    pd.testing.assert_frame_equal(actual, pd.DataFrame(values, columns=data.columns))


@mark.parametrize(
    "columns,row,expected",
    [
        (["A", "B"], pd.Series([3.0], index=["A"]), [3.0, 3.0]),
        (["A", "B"], pd.Series([30.0, 3.0], index=["B", "C"]), [30.0, 3.0]),
        (["A", "B"], pd.Series([30.0, 3.0], index=["A", "A"]), [30.0, 3.0]),
        (["A", "A"], pd.Series([30.0, 3.0], index=["B", "A"]), [30.0, 3.0]),
        (["A", "B"], pd.Series([30.0, 3.0], index=pd.IntervalIndex.from_tuples([(1, 4), (0, 2)])), [30.0, 3.0]),
        # Interval lookup can match contained points; those are not matching row labels.
        ([3, 1], pd.Series([30.0, 3.0], index=pd.IntervalIndex.from_tuples([(0, 2), (2, 4)])), [30.0, 3.0]),
        ([3, 1.5], pd.Series([30.0, 3.0], index=pd.IntervalIndex.from_tuples([(0, 2), (1, 4)])), [30.0, 3.0]),
    ],
    ids=["partial", "missing-extra", "duplicate-row", "duplicate-columns", "unrelated-interval", "inexact-interval", "ambiguous-containment"],
)
@mark.parametrize("pandas_method", [False, True], ids=["module", "pandas"])
def test_rollapply_preserves_unmatched_callback_rows(columns, row, expected, pandas_method):
    """Alignment must not redefine incomplete, duplicate or inexact row-label policy."""
    data = pd.DataFrame([[1.0, 10.0], [2.0, 20.0]], columns=columns)
    snapshot = row.copy(deep=True)
    actual = data.rollapply(2, lambda sample: row) if pandas_method else ffn.rollapply(data, 2, lambda sample: row)
    pd.testing.assert_frame_equal(actual, pd.DataFrame([[np.nan, np.nan], expected], columns=data.columns))
    pd.testing.assert_series_equal(row, snapshot)


def test_rollapply_preserves_extra_callback_row_rejection():
    data = pd.DataFrame({"A": [1.0, 2.0], "B": [10.0, 20.0]})
    original = data.copy(deep=True)
    with raises(ValueError):
        ffn.rollapply(data, 2, lambda sample: pd.Series([3.0, 30.0, 300.0], index=["A", "B", "C"]))
    pd.testing.assert_frame_equal(data, original)


def test_winsorize():
    x = pd.Series(range(20), dtype="float")
    res = x.winsorize(limits=0.05)
    assert res.iloc[0] == 1
    assert res.iloc[-1] == 18

    # make sure initial values still intact
    assert x.iloc[0] == 0
    assert x.iloc[-1] == 19

    x = pd.DataFrame(
        {
            "a": pd.Series(range(20), dtype="float"),
            "b": pd.Series(range(20), dtype="float"),
        }
    )
    res = x.winsorize(axis=0, limits=0.05)

    assert res["a"].iloc[0] == 1
    assert res["b"].iloc[0] == 1
    assert res["a"].iloc[-1] == 18
    assert res["b"].iloc[-1] == 18

    assert x["a"].iloc[0] == 0
    assert x["b"].iloc[0] == 0
    assert x["a"].iloc[-1] == 19
    assert x["b"].iloc[-1] == 19


@mark.parametrize("dtype", ["Float32", "Float64", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"])
@mark.parametrize("use_pandas_method", [False, True], ids=["package", "pandas"])
def test_winsorize_nullable_series(dtype, use_pandas_method):
    index = pd.Index(range(11), name="row")
    values = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 100, pd.NA], index=index, dtype=dtype)
    # Ten observed values and ten-percent limits replace one value at each tail.
    expected = pd.Series([1, 1, 2, 3, 4, 5, 6, 7, 8, 8, pd.NA], index=index, dtype=dtype)
    original = values.copy()
    calculate = values.winsorize if use_pandas_method else ffn.winsorize
    args = () if use_pandas_method else (values,)

    actual = calculate(*args, limits=0.1)

    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(values, original)


@mark.parametrize("dtype", ["Float64", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"])
@mark.parametrize("axis", [0, 1])
def test_winsorize_nullable_dataframe(axis, dtype):
    values = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 100, pd.NA], dtype=dtype)
    expected_values = pd.Series([1, 1, 2, 3, 4, 5, 6, 7, 8, 8, pd.NA], dtype=dtype)
    # The all-missing slice must bypass SciPy while the observed slice is winsorized.
    data = pd.DataFrame({"observed": values, "all_missing": pd.Series(pd.NA, index=values.index, dtype=dtype)})
    expected = pd.DataFrame({"observed": expected_values, "all_missing": pd.Series(pd.NA, index=values.index, dtype=dtype)})
    if axis == 1:
        data = data.T
        expected = expected.T
    original = data.copy()

    actual = data.winsorize(axis=axis, limits=0.1)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(data, original)
    # Applying along either axis must not expose the caller's extension storage.
    actual.iloc[0, 0] = 2
    pd.testing.assert_frame_equal(data, original)
    snapshot = actual.copy()
    data.iloc[0, 0] = 3
    pd.testing.assert_frame_equal(actual, snapshot)


@mark.parametrize("dtype", ["Int8", "UInt8", "Int64", "UInt64"])
@mark.parametrize("limits", [0.0, 0.2, (0.2, None), (None, 0.2)])
def test_winsorize_nullable_integer_cut_points(dtype, limits):
    bounds = np.iinfo(pd.Series(dtype=dtype).dtype.numpy_dtype)
    raw = [int(bounds.max), int(bounds.min), int(bounds.max) - 1, int(bounds.min) + 1, 1, 0]
    index = pd.Index(["f", "e", "d", "c", "b", "a", "missing"], name="row")
    values = pd.Series(raw + [pd.NA], index=index, dtype=dtype, name="observations")
    original = values.copy()
    # Winsorization selects observed order statistics, not interpolated quantiles.
    # Python integers keep wide cut points exact where a float conversion would round.
    ordered = sorted(raw)
    lower, upper = (limits, limits) if np.isscalar(limits) else limits
    low = ordered[int(len(raw) * (lower or 0))]
    high = ordered[len(raw) - int(len(raw) * (upper or 0)) - 1]
    expected = pd.Series([min(max(value, low), high) for value in raw] + [pd.NA], index=index, dtype=dtype, name=values.name)

    actual = ffn.winsorize(values, limits=limits)

    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(values, original)
    # Neither returned extension storage nor caller storage may alias the other.
    actual.iloc[0] = 2
    pd.testing.assert_series_equal(values, original)
    snapshot = actual.copy()
    values.iloc[1] = 3
    pd.testing.assert_series_equal(actual, snapshot)


@mark.parametrize("dtype", ["Int8", "UInt8", "Int64", "UInt64"])
@mark.parametrize("raw", [[], [pd.NA, pd.NA]], ids=["empty", "all-missing"])
def test_winsorize_nullable_integer_without_observations(dtype, raw):
    values = pd.Series(raw, dtype=dtype, name="observations")
    original = values.copy()

    actual = values.winsorize(limits=0.1)

    # No observed cut point exists; retain the original dtype, mask and labels.
    pd.testing.assert_series_equal(actual, original)
    pd.testing.assert_series_equal(values, original)


def test_rescale():
    x = pd.Series(range(10), dtype="float")
    res = x.rescale()

    assert res.iloc[0] == 0
    assert res.iloc[4] == (4.0 - 0.0) / (9.0 - 0.0)
    assert res.iloc[-1] == 1

    assert x.iloc[0] == 0
    assert x.iloc[4] == 4
    assert x.iloc[-1] == 9

    x = pd.DataFrame(
        {
            "a": pd.Series(range(10), dtype="float"),
            "b": pd.Series(range(10), dtype="float"),
        }
    )
    res = x.rescale(axis=0)

    assert res["a"].iloc[0] == 0
    assert res["a"].iloc[4] == (4.0 - 0.0) / (9.0 - 0.0)
    assert res["a"].iloc[-1] == 1
    assert res["b"].iloc[0] == 0
    assert res["b"].iloc[4] == (4.0 - 0.0) / (9.0 - 0.0)
    assert res["b"].iloc[-1] == 1

    assert x["a"].iloc[0] == 0
    assert x["a"].iloc[4] == 4
    assert x["a"].iloc[-1] == 9
    assert x["b"].iloc[0] == 0
    assert x["b"].iloc[4] == 4
    assert x["b"].iloc[-1] == 9


def test_rescale_dataframe_axis_1():
    x = pd.DataFrame({"a": [1.0, 2.0, 5.0], "b": [3.0, 4.0, 1.0], "c": [5.0, 0.0, 3.0]})
    res = x.rescale(axis=1)

    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ["a", "b", "c"]
    assert res.index.equals(x.index)
    assert res.loc[0].tolist() == [0.0, 0.5, 1.0]
    assert res.loc[2].tolist() == [1.0, 0.0, 0.5]
    assert (res.dtypes == "float64").all()


@mark.parametrize("dtype", ["int64", "uint64", "float32", "float64", "Int64", "Float64"])
@mark.parametrize("axis", [0, 1, "index", "columns"])
@mark.parametrize("bounds", [(0.0, 1.0), (-0.5, 2.5)])
def test_rescale_dataframe_preserves_interpolated_values(dtype, axis, bounds):
    data = pd.DataFrame(
        [[0, 2, 6], [6, 0, 2], [2, 6, 0]],
        index=pd.Index(["third", "first", "second"], name="observation"),
        columns=pd.Index(["c", "a", "b"], name="asset"),
        dtype=dtype,
    )
    original = data.copy(deep=True)
    lower, upper = bounds
    expected = data.astype(float) / 6 * (upper - lower) + lower

    actual = data.rescale(min=lower, max=upper, axis=axis)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(data, original)


@mark.parametrize("axis", [0, 1])
@mark.parametrize("labels", ["duplicate", "multiindex"])
def test_rescale_dataframe_preserves_labels(axis, labels):
    if labels == "duplicate":
        index = pd.Index(["row", "row", "other"], name="observation")
        columns = pd.Index(["asset", "asset", "other"], name="asset")
    else:
        index = pd.MultiIndex.from_tuples([("b", 2), ("a", 1), ("a", 2)], names=["group", "row"])
        columns = pd.MultiIndex.from_tuples([("z", 2), ("y", 1), ("y", 2)], names=["group", "asset"])
    data = pd.DataFrame([[0.0, 2.0, 6.0], [6.0, 0.0, 2.0], [2.0, 6.0, 0.0]], index=index, columns=columns)
    original = data.copy(deep=True)

    actual = ffn.rescale(data, axis=axis)

    pd.testing.assert_frame_equal(actual, data / 6)
    pd.testing.assert_frame_equal(data, original)


# A falsey non-string name guards against conditional metadata propagation.
@mark.parametrize("name", [None, "asset", 0], ids=["unnamed", "string-name", "falsey-name"])
@mark.parametrize("method_name", ["winsorize", "rescale"])
@mark.parametrize("use_pandas_method", [False, True], ids=["package", "pandas"])
def test_series_value_transform_preserves_name(name, method_name, use_pandas_method):
    values = pd.Series(range(10), dtype="float", name=name)
    calculate = getattr(values, method_name) if use_pandas_method else getattr(ffn, method_name)
    args = () if use_pandas_method else (values,)

    actual = calculate(*args)

    assert actual.name == name


def test_annualize():
    assert ffn.annualize(0.1, 60) == (1.1 ** (1.0 / (60.0 / 365)) - 1)


def test_calc_sortino_ratio(df):
    rf = 0
    p = 1
    r = df.to_returns()
    a = r.calc_sortino_ratio(rf=rf, nperiods=p)
    er = r.to_excess_returns(rf, p)
    negative_returns = er.clip(upper=0.0)
    downside_deviation = np.sqrt((negative_returns**2).mean())
    assert np.allclose(
        a, (er.mean() - rf) / downside_deviation * np.sqrt(p)
    )


def test_calc_sortino_ratio_infers_periods_before_deannualizing_risk_free_rate():
    """Infer a return frequency before validating an annualized scalar rate."""
    index = pd.date_range("2025-01-01", periods=20, freq="B")
    series = pd.Series(np.linspace(-0.02, 0.03, 20), index=index, name="strategy")
    frame = pd.DataFrame({"strategy": series, "scaled": series * 0.8})
    rf = 0.05
    nperiods = 252

    # Calculate the per-period hurdle directly so the expected ratios do not
    # depend on the frequency-inference or deannualization paths under test.
    period_rf = (1.0 + rf) ** (1.0 / nperiods) - 1.0
    for returns in (series, frame):
        excess_returns = returns - period_rf
        downside_deviation = np.sqrt((excess_returns.clip(upper=0.0) ** 2).mean())
        expected = excess_returns.mean() / downside_deviation * np.sqrt(nperiods)

        assert np.allclose(ffn.calc_sortino_ratio(returns, rf=rf), expected)
        for result in (
            returns.calc_sortino_ratio(rf=rf),
            returns.calc_sortino(rf=rf),
        ):
            assert isinstance(result, (float, pd.Series))
            assert np.allclose(result, expected)


def test_calc_sortino_ratio_requires_periods_for_uninferrable_scalar_rate():
    """Reject an annualized scalar rate when no period count can be inferred."""
    returns = pd.Series(
        [-0.02, 0.01, 0.03, -0.01], index=["a", "b", "c", "d"]
    )

    with np.testing.assert_raises(ValueError):
        ffn.calc_sortino_ratio(returns, rf=0.05)
    with np.testing.assert_raises(ValueError):
        returns.calc_sortino_ratio(rf=0.05)


def test_calc_sortino_ratio_is_order_invariant():
    # Both the mean and the downside deviation are symmetric functions of the
    # sample, so reordering the same returns must not change the ratio.
    idx = pd.date_range("2026-01-31", periods=4, freq=ffn.core._MonthEnd)
    negative_first = pd.Series([-0.10, 0.02, 0.01, 0.03], index=idx)
    negative_last = pd.Series([0.02, 0.01, 0.03, -0.10], index=idx)

    assert np.isclose(
        negative_first.calc_sortino_ratio(annualize=False), -0.2
    )
    assert np.isclose(
        negative_last.calc_sortino_ratio(annualize=False), -0.2
    )


def test_calc_sortino_ratio_counts_first_period_downside():
    # A series whose only losing period comes first still has downside risk.
    idx = pd.date_range("2026-01-31", periods=4, freq=ffn.core._MonthEnd)
    returns = pd.Series([-0.10, 0.02, 0.01, 0.03], index=idx)

    assert np.isfinite(returns.calc_sortino_ratio(annualize=False))


def test_calc_sortino_ratio_ignores_leading_nan(df):
    # Returns built from prices carry a leading NaN, which pandas already skips
    # in both the mean and the downside deviation.
    r = df.to_returns()

    assert np.allclose(
        r.calc_sortino_ratio(annualize=False),
        r[1:].calc_sortino_ratio(annualize=False),
    )

def test_to_ulcer_index_is_in_percentage_points():
    # 100 -> 90 -> 100 has drawdowns of 0%, -10%, 0%, so the ulcer index is
    # sqrt(mean([0, 100, 0])) = sqrt(100 / 3)
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 90.0, 100.0], index=idx)

    assert np.isclose(prices.to_ulcer_index(), np.sqrt(100 / 3))


def test_to_ulcer_index_without_drawdown_is_zero():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)

    assert np.isclose(prices.to_ulcer_index(), 0.0)


def test_to_ulcer_index_ignores_a_gap_in_the_price_series():
    # A missing close is a missing observation, not a new high water mark.
    # Before the ffill, np.maximum.accumulate propagated the NaN over the
    # whole tail and Series.mean() skipped those rows, so the result was the
    # ulcer index of only the prefix before the gap.
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    prices = pd.Series([100.0, 90.0, 100.0, 80.0, 60.0, 100.0], index=idx)

    gapped = prices.copy()
    gapped.iloc[2] = np.nan

    # Carrying the last known price across the gap is the honest reading, and
    # is what to_drawdown_series already does.
    assert np.isclose(gapped.to_ulcer_index(), gapped.ffill().to_ulcer_index())

    # The old behaviour returned the ulcer index of the prefix alone, which
    # missed the -40% drawdown that happens after the gap.
    assert gapped.to_ulcer_index() > prices.iloc[:2].to_ulcer_index()


def test_to_ulcer_index_gap_does_not_report_zero_risk():
    # A NaN in the second position used to truncate the series to a single
    # observation, giving an ulcer index of exactly 0.0 for a series that
    # halves.
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    gapped = pd.Series([100.0, np.nan, 50.0, 50.0], index=idx)

    assert gapped.to_ulcer_index() > 0.0


def test_to_ulcer_index_ignores_a_leading_gap():
    # A leading NaN is an observation that has not happened yet, not a price
    # of NaN, so it drops out of the average instead of poisoning it. This is
    # what to_drawdown_series already does with the same input.
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([np.nan, 100.0, 90.0, 100.0], index=idx)

    assert np.isclose(prices.to_ulcer_index(), prices.dropna().to_ulcer_index())


def test_to_ulcer_index_handles_an_ndarray_gap_like_a_series():
    # The docstring advertises ndarray support and promises that gaps are
    # ignored, so an ndarray must not return NaN where a Series does not.
    values = [100.0, 90.0, np.nan, 60.0, 100.0]

    assert np.isclose(
        ffn.core.to_ulcer_index(np.array(values)),
        pd.Series(values).to_ulcer_index(),
    )


def test_to_ulcer_index_is_per_column_for_a_dataframe():
    # Every other ffn measure reduces per column; this one pooled every
    # column into a single scalar, while to_ulcer_performance_index returned
    # a per-column Series divided by that pooled value.
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    df = pd.DataFrame({"a": [100.0, 90.0, 100.0], "b": [100.0, 50.0, 100.0]}, index=idx)

    result = df.to_ulcer_index()

    assert isinstance(result, pd.Series)
    assert np.isclose(result["a"], df["a"].to_ulcer_index())
    assert np.isclose(result["b"], df["b"].to_ulcer_index())


def test_to_ulcer_index_unchanged_without_gaps():
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    prices = pd.Series([100.0, 110, 105, 120, 90, 95, 130, 125], index=idx)

    max_values = np.maximum.accumulate(prices)
    drawdowns = ((prices - max_values) / max_values) * 100
    expected = np.sqrt(np.mean(np.square(drawdowns)))

    assert np.isclose(prices.to_ulcer_index(), expected)


def test_to_ulcer_performance_index_matches_ulcer_index_scale():
    # The ulcer index is expressed in percentage points, so the excess return
    # must be too. Prices rise 10% over 365 days, an annualized return of
    # 1.1 ** (365.25 / 365) - 1, against an ulcer index of sqrt(100 / 3).
    idx = pd.DatetimeIndex(["2025-01-01", "2025-07-02", "2026-01-01"])
    prices = pd.Series([100.0, 90.0, 110.0], index=idx)

    expected = (1.1 ** (365.25 / 365) - 1) * 100 / np.sqrt(100 / 3)

    assert np.isclose(prices.to_ulcer_performance_index(), expected)


def test_to_ulcer_performance_index_ignores_a_gap_in_the_price_series():
    # Both halves of the ratio must see the same prices. The numerator used
    # the raw series, whose gap makes two of the three returns NaN and leaves
    # the mean return as the single +100% recovery, while the denominator's
    # ulcer index was already computed over the filled series.
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    gapped = pd.Series([100.0, np.nan, 50.0, 100.0], index=idx)

    assert np.isclose(
        gapped.to_ulcer_performance_index(),
        gapped.ffill().to_ulcer_performance_index(),
    )


def test_to_ulcer_performance_index_is_dimensionally_consistent():
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    prices = pd.Series([100.0, 110, 105, 120, 90, 95, 130, 125], index=idx)

    upi = prices.to_ulcer_performance_index()
    annualized_return_pct = prices.calc_cagr() * 100

    assert np.isclose(upi * prices.to_ulcer_index(), annualized_return_pct)


@mark.parametrize("risk_free", [0.0, 0.05])
def test_to_ulcer_performance_index_does_not_depend_on_sampling_frequency(risk_free):
    # The same year of prices sampled daily and monthly, with the same first
    # and last dates. The Ulcer Index barely moves, and the annualized excess
    # return is identical. A per-period mean return in the numerator made the
    # daily UPI about 20x smaller than the monthly one.
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    t = np.arange(len(idx))
    daily = pd.Series(100 * np.exp(0.08 * t / 365) * (1 - 0.15 * np.exp(-(((t - 150) / 30) ** 2))), index=idx)
    monthly = daily[daily.index.is_month_start | (daily.index == daily.index[-1])]

    numerator_daily = daily.to_ulcer_performance_index(rf=risk_free, nperiods=365) * daily.to_ulcer_index()
    numerator_monthly = monthly.to_ulcer_performance_index(rf=risk_free, nperiods=12) * monthly.to_ulcer_index()

    assert np.isclose(numerator_daily, numerator_monthly)
    assert np.isclose(numerator_daily, (daily.calc_cagr() - risk_free) * 100)


def _diff_series(n, mean=0.001, std=0.01):
    # A return series against a flat benchmark, with the differential's mean and
    # standard deviation pinned so the information ratio is exactly mean / std
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return (
        pd.Series(mean + std * raw, index=idx),
        pd.Series(np.zeros(n), index=idx),
    )


def test_calc_prob_mom_increases_with_sample_size():
    """Test that the same per-period edge observed for longer gives more confidence"""
    probs = []
    for n in (10, 50, 250, 1000):
        returns, benchmark = _diff_series(n)
        assert np.isclose(returns.calc_information_ratio(benchmark), 0.1)
        probs.append(returns.calc_prob_mom(benchmark))

    for earlier, later in zip(probs, probs[1:]):
        assert later > earlier

    assert probs[0] < 0.7
    assert probs[-1] > 0.99


def test_calc_prob_mom_matches_one_sample_t_test():
    """Test that the probability is the t CDF of the information ratio scaled by sqrt(n)"""
    # 250 observations at an information ratio of 0.1 give a t statistic of
    # 0.1 * sqrt(250) = 1.5811 on 249 degrees of freedom
    returns, benchmark = _diff_series(250)

    assert np.isclose(returns.calc_prob_mom(benchmark), 0.942442, atol=1e-6)


def test_calc_prob_mom_without_an_edge_is_half():
    """Test that a series compared against itself carries no information"""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(np.linspace(0.001, 0.002, 100), index=idx)

    assert np.isclose(returns.calc_prob_mom(returns), 0.5)


def test_calc_prob_mom_ignores_unaligned_observations():
    """Test that NaNs and non-overlapping dates don't count toward the sample size"""
    returns, benchmark = _diff_series(250)
    padded = returns.copy()
    padded.iloc[0] = np.nan  # e.g. the leading NaN from to_returns()
    short_benchmark = benchmark.iloc[:100]

    # Only the 99 dates that are non-NaN in both series carry information, so
    # the result must match the computation restricted to that window
    expected = returns.iloc[1:100].calc_prob_mom(benchmark.iloc[1:100])

    assert np.isclose(padded.calc_prob_mom(short_benchmark), expected)


def test_calc_prob_mom_dataframe_against_series_benchmark():
    """Test that a Series benchmark aligns on the index, not on the columns"""
    returns, benchmark = _diff_series(250)
    frame = pd.DataFrame({"a": returns, "b": returns * 2})

    result = frame.calc_prob_mom(benchmark)

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["a", "b"]
    # Each column must match the value its own Series carries
    for col in frame:
        assert np.isclose(result[col], frame[col].calc_prob_mom(benchmark))


def test_calc_prob_mom_dataframe_ignores_unaligned_observations():
    """Test that the per-column sample size still excludes NaN and non-overlapping dates"""
    returns, benchmark = _diff_series(250)
    padded = returns.copy()
    padded.iloc[0] = np.nan  # e.g. the leading NaN from to_returns()
    frame = pd.DataFrame({"a": returns, "b": padded})
    short_benchmark = benchmark.iloc[:100]

    result = frame.calc_prob_mom(short_benchmark)

    # Column "b" only overlaps on 99 dates, column "a" on 100
    assert np.isclose(result["a"], returns.iloc[:100].calc_prob_mom(short_benchmark))
    assert np.isclose(result["b"], returns.iloc[1:100].calc_prob_mom(benchmark.iloc[1:100]))


def test_calc_prob_mom_series_against_dataframe_benchmark():
    """Test that a DataFrame benchmark aligns on the index, not on the columns"""
    returns, benchmark = _diff_series(250)
    frame = pd.DataFrame({"a": benchmark, "b": benchmark + 0.0005})

    result = returns.calc_prob_mom(frame)

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["a", "b"]
    # Each benchmark column must match the value it carries on its own
    for col in frame:
        assert np.isclose(result[col], returns.calc_prob_mom(frame[col]))


def test_calc_prob_mom_dataframe_against_dataframe_benchmark():
    """Test that a column-wise result stays labelled by the frame's columns"""
    returns, benchmark = _diff_series(250)
    frame = pd.DataFrame({"a": returns, "b": returns * 2})
    benchmarks = pd.DataFrame({"a": benchmark, "b": benchmark + 0.0005})

    result = frame.calc_prob_mom(benchmarks)

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["a", "b"]
    for col in frame:
        assert np.isclose(result[col], frame[col].calc_prob_mom(benchmarks[col]))


def test_calmar_ratio(df):
    cagr = df.calc_cagr()
    mdd = df.calc_max_drawdown()

    a = df.calc_calmar_ratio()
    assert np.allclose(a, cagr / abs(mdd))


def test_calc_stats(df):
    # test twelve_month_win_perc divide by zero
    prices = df.C["2010-10-01":"2011-08-01"]
    stats = ffn.calc_stats(prices).stats
    assert pd.isnull(stats["twelve_month_win_perc"])
    prices = df.C["2009-10-01":"2011-08-01"]
    stats = ffn.calc_stats(prices).stats
    assert not pd.isnull(stats["twelve_month_win_perc"])

    # test yearly_sharpe divide by zero
    prices = df.C["2009-01-01":"2012-01-01"]
    stats = ffn.calc_stats(prices).stats
    assert "yearly_sharpe" in stats.index

    prices[prices > 0.0] = 1.0
    # throws warnings
    stats = ffn.calc_stats(prices).stats
    assert pd.isnull(stats["yearly_sharpe"])


def test_twelve_month_win_perc_uses_twelve_month_window():
    # 13 month end prices => exactly one full twelve month window,
    # 2020-01-31 -> 2021-01-31, which returns 101 / 100 - 1 = +1%.
    # The eleven month window 2020-01-31 -> 2020-12-31 is 99 / 100 - 1 = -1%,
    # so measuring the wrong window flips the result.
    index = pd.date_range("2020-01-31", periods=13, freq=ffn.core._MonthEnd)
    prices = pd.Series([100.0] * 11 + [99.0, 101.0], index=index)

    stats = ffn.calc_stats(prices).stats

    assert stats["twelve_month_win_perc"] == 1.0

    # 12 month end prices are only eleven monthly returns, which is not
    # enough for a twelve month window
    stats = ffn.calc_stats(prices.iloc[:12]).stats
    assert pd.isnull(stats["twelve_month_win_perc"])

    # Missing endpoint prices do not represent losing windows and are excluded.
    full_index = pd.date_range("2020-01-31", periods=14, freq=ffn.core._MonthEnd)
    prices = pd.Series(range(100, 113), index=full_index.delete(1))
    stats = ffn.calc_stats(prices).stats
    assert stats["twelve_month_win_perc"] == 1.0


def test_calc_sharpe(df):
    x = pd.Series()
    assert np.isnan(x.calc_sharpe(annualize=False))

    r = df.to_returns()

    res = r.calc_sharpe(annualize=False)
    assert np.allclose(res, r.mean() / r.std())

    res = r.calc_sharpe(rf=0.05, nperiods=252)
    drf = ffn.deannualize(0.05, 252)
    ar = r - drf
    assert np.allclose(res, ar.mean() / ar.std() * np.sqrt(252))


def test_calc_expected_max_sharpe():
    # No dispersion, or a single trial, means no selection to correct for
    assert ffn.calc_expected_max_sharpe(1, 0.5) == 0.0
    assert ffn.calc_expected_max_sharpe(50, 0.0) == 0.0

    # The hurdle grows with the number of trials and scales with their dispersion
    assert ffn.calc_expected_max_sharpe(100, 0.5) > ffn.calc_expected_max_sharpe(10, 0.5)
    aae(
        ffn.calc_expected_max_sharpe(10, 1.0) * 0.5,
        ffn.calc_expected_max_sharpe(10, 0.5),
    )


def test_calc_prob_backtest_overfitting():
    np.random.seed(0)
    n_periods, n_trials = 640, 40
    index = pd.date_range(start="2015-01-01", periods=n_periods, freq="D")

    # Pure noise: selecting the in-sample best carries no information, so the
    # winner's out-of-sample rank is uniform and PBO sits near one half.
    noise = pd.DataFrame(np.random.normal(0, 0.01, (n_periods, n_trials)), index=index)
    pbo_noise = ffn.calc_prob_backtest_overfitting(noise, n_blocks=8)
    assert 0.15 < pbo_noise < 0.85

    # One genuinely skilled trial: the in-sample winner keeps winning out of
    # sample, so PBO collapses towards zero. This also guards the rank
    # direction: with the ranking inverted, this case reports near one.
    skilled = noise.copy()
    skilled[7] = skilled[7] + 0.004
    pbo_skilled = ffn.calc_prob_backtest_overfitting(skilled, n_blocks=8)
    assert pbo_skilled < 0.1
    assert pbo_noise - pbo_skilled > 0.15

    # Accessible as a DataFrame method, with full diagnostics on request
    full = skilled.calc_prob_backtest_overfitting(n_blocks=8, full_output=True)
    assert full["pbo"] == pbo_skilled
    assert len(full["logits"]) == 70  # C(8, 4) combinations
    assert 0.0 < full["mean_oos_rank"] < 1.0

    # Input validation
    with np.testing.assert_raises(ValueError):
        ffn.calc_prob_backtest_overfitting(noise, n_blocks=7)
    with np.testing.assert_raises(TypeError):
        ffn.calc_prob_backtest_overfitting(noise[3], n_blocks=8)
    with np.testing.assert_raises(ValueError):
        ffn.calc_prob_backtest_overfitting(noise.iloc[:4], n_blocks=8)


@mark.parametrize("n_blocks", [2, 4, 6, 8])
def test_calc_prob_backtest_overfitting_pairs_default_folds(monkeypatch, n_blocks):
    from math import comb

    returns = pd.DataFrame(np.random.default_rng(42).normal(0, 0.01, (65, 5)))
    original = returns.copy()

    def sharpe(series):
        std = series.std(ddof=1)
        return series.mean() / std if std > 0 and series.max() > series.min() else np.nan

    expected = returns.calc_prob_backtest_overfitting(n_blocks=n_blocks, metric=sharpe, full_output=True)
    calls = []
    std = pd.DataFrame.std

    def counted_std(sample, *args, **kwargs):
        calls.append(len(sample))
        return std(sample, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "std", counted_std)
    result = ffn.calc_prob_backtest_overfitting(returns, n_blocks=n_blocks, full_output=True)

    # Each subset is evaluated once, then reused when its complement is in-sample.
    assert len(calls) == comb(n_blocks, n_blocks // 2)
    assert set(calls) == {(len(returns) // n_blocks) * n_blocks // 2}
    pd.testing.assert_series_equal(result["logits"], expected["logits"], check_exact=True)
    assert result["pbo"] == expected["pbo"]
    assert result["mean_oos_rank"] == expected["mean_oos_rank"]
    pd.testing.assert_frame_equal(returns, original)


def test_calc_prob_backtest_overfitting_default_fold_order():
    returns = pd.DataFrame([[3, 1, 2], [4, 2, 3], [4, 3, 2], [5, 4, 3], [1, 4, 3], [2, 5, 4], [2, 3, 4], [3, 4, 5]])

    result = returns.calc_prob_backtest_overfitting(n_blocks=4, full_output=True)

    # AB, AC, AD, BC, BD, CD: compare positive mean squared / sample variance.
    # AC's out-of-sample winner ties another trial; CD's in-sample tie selects column 1.
    ranks = np.array([0.25, 0.375, 0.25, 0.25, 0.25, 0.25])
    np.testing.assert_array_equal(result["logits"], np.log(ranks / (1.0 - ranks)))
    assert result["mean_oos_rank"] == ranks.mean()
    assert result["pbo"] == 1.0


@mark.parametrize("order", ["permuted", "descending"])
def test_calc_prob_backtest_overfitting_rejects_nonchronological_dates(order):
    returns = pd.DataFrame(
        np.random.default_rng(18).normal(0, 0.01, (24, 5)),
        index=pd.date_range("2020-01-01", periods=24, tz="UTC"),
    )
    if order == "permuted":
        returns = returns.sample(frac=1, random_state=4)
    else:
        returns = returns.iloc[::-1]
    original = returns.copy()

    with raises(ValueError, match="trial_returns index must be monotonic increasing"):
        ffn.calc_prob_backtest_overfitting(returns, n_blocks=6)

    pd.testing.assert_frame_equal(returns, original)


def test_calc_prob_backtest_overfitting_validates_truncated_tail():
    index = pd.date_range("2020-01-01", periods=25, tz="UTC")
    returns = pd.DataFrame(np.arange(125).reshape(25, 5), index=index)
    returns.index = index[:-1].append(pd.DatetimeIndex([index[0] - pd.Timedelta(days=1)]))

    # The last row would be truncated for six blocks, but remains part of the caller's input.
    with raises(ValueError, match="trial_returns index must be monotonic increasing"):
        ffn.calc_prob_backtest_overfitting(returns, n_blocks=6)


def test_calc_prob_backtest_overfitting_accepts_duplicate_dates():
    returns = pd.DataFrame(
        np.random.default_rng(18).normal(0, 0.01, (24, 5)),
        index=pd.date_range("2020-01-01", periods=24),
    )
    expected = ffn.calc_prob_backtest_overfitting(returns.reset_index(drop=True), n_blocks=6)
    returns.index = returns.index[:5].append(pd.DatetimeIndex([returns.index[4]])).append(returns.index[6:])

    # Equal neighboring dates remain monotonic under the accepted datetime-index policy.
    assert ffn.calc_prob_backtest_overfitting(returns, n_blocks=6) == expected


def test_calc_prob_backtest_overfitting_exact_ranks():
    returns = pd.DataFrame([[3, 1, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]])

    result = returns.calc_prob_backtest_overfitting(n_blocks=4, metric=pd.Series.mean, full_output=True)

    # AB, AC, AD, BC, BD, CD; in-sample ties select the first column.
    ranks = np.array([0.25, 0.5, 0.25, 0.25, 0.5, 0.375])
    np.testing.assert_allclose(result["logits"], np.log(ranks / (1 - ranks)))
    aae(result["mean_oos_rank"], ranks.mean())
    assert result["pbo"] == 1.0


def test_calc_prob_backtest_overfitting_identical_trials():
    returns = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

    result = returns.calc_prob_backtest_overfitting(n_blocks=2, full_output=True)

    np.testing.assert_array_equal(result["logits"], [0.0, 0.0])
    assert result["mean_oos_rank"] == 0.5
    assert result["pbo"] == 1.0  # The existing PBO convention includes the median.


@mark.parametrize("value", [0.0, 0.1, np.nan])
def test_calc_prob_backtest_overfitting_undefined_full_output(value):
    returns = pd.DataFrame(value, index=range(40), columns=["a", "b"])

    assert np.isnan(returns.calc_prob_backtest_overfitting(n_blocks=2))
    result = returns.calc_prob_backtest_overfitting(n_blocks=2, full_output=True)

    assert set(result) == {"pbo", "logits", "mean_oos_rank"}
    assert np.isnan(result["pbo"])
    assert np.isnan(result["mean_oos_rank"])
    assert len(result["logits"]) == 2
    assert result["logits"].isna().all()


@mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@mark.parametrize("trial", ["a", "b"])
def test_calc_prob_backtest_overfitting_nonfinite_metric(invalid, trial):
    returns = pd.DataFrame({"a": [11, 12, 13, 14], "b": [1, 2, 3, 4]})

    def metric(series):
        if series.iloc[0] == returns[trial].iloc[2]:
            return invalid
        return series.mean()

    result = returns.calc_prob_backtest_overfitting(n_blocks=2, metric=metric, full_output=True)

    assert np.isnan(result["pbo"])
    assert np.isnan(result["mean_oos_rank"])
    assert len(result["logits"]) == 2
    assert result["logits"].isna().all()


def test_calc_prob_backtest_overfitting_retains_undefined_folds():
    returns = pd.DataFrame({"a": [1, 1, 1, 1, 3, 4, 5, 6], "b": [1, 2, 3, 4, 1, 2, 1, 2]})

    result = returns.calc_prob_backtest_overfitting(n_blocks=4, full_output=True)

    assert len(result["logits"]) == 6
    assert result["logits"].iloc[[0, 5]].isna().all()
    assert np.isfinite(result["logits"].iloc[1:5]).all()
    assert np.isnan(result["pbo"])
    assert np.isnan(result["mean_oos_rank"])


@mark.parametrize("n_blocks", [np.int64(4), np.uint64(4)])
def test_calc_prob_backtest_overfitting_custom_metric_preserves_labels(n_blocks):
    returns = pd.DataFrame(
        np.arange(18, dtype=float).reshape(9, 2),
        index=pd.date_range("2020-01-01", periods=9, freq="2D", tz="UTC"),
        columns=["a", "b"],
    )
    seen = []

    def metric(series):
        assert isinstance(series.index, pd.DatetimeIndex)
        assert series.index.is_monotonic_increasing
        assert returns.index[-1] not in series.index  # Truncate the ninth row.
        pd.testing.assert_series_equal(series, returns.loc[series.index, series.name])
        seen.append(series)
        return series.mean()

    assert returns.calc_prob_backtest_overfitting(n_blocks=n_blocks, metric=metric) == 0.0
    assert len(seen) == 24  # Six folds, two halves, two trials.
    assert all(len(series) == 4 for series in seen)
    # Stateful metrics can distinguish repeated halves and their evaluation order.
    folds = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 6, 7], [2, 3, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7]]
    expected = []
    for rows in folds:
        complement = [i for i in range(8) if i not in rows]
        expected.extend(returns.iloc[half][column] for half in (rows, complement) for column in returns)
    for actual, sample in zip(seen, expected):
        pd.testing.assert_series_equal(actual, sample)


@mark.parametrize("n_blocks", [0, 1, 3, -2, 2.0, 2.5, "2", None, np.nan, np.inf, True, False])
def test_calc_prob_backtest_overfitting_invalid_block_count(n_blocks):
    returns = pd.DataFrame(np.arange(16).reshape(8, 2))
    with np.testing.assert_raises_regex(ValueError, "n_blocks"):
        returns.calc_prob_backtest_overfitting(n_blocks=n_blocks)


def test_calc_prob_backtest_overfitting_custom_metric_stops_early():
    returns = pd.DataFrame(np.arange(128).reshape(64, 2))
    calls = []

    def metric(series):
        calls.append(series.name)
        raise RuntimeError("stop after the first sample")

    # A callback can stop a large fold search before its output would fit in memory.
    with raises(RuntimeError, match="stop after the first sample"):
        returns.calc_prob_backtest_overfitting(n_blocks=64, metric=metric)
    assert calls == [0]


def test_calc_prob_backtest_overfitting_invalid_metric():
    returns = pd.DataFrame(np.arange(8).reshape(4, 2))
    with np.testing.assert_raises_regex(TypeError, "metric"):
        returns.calc_prob_backtest_overfitting(n_blocks=2, metric=1)
    with np.testing.assert_raises_regex(ValueError, "scalar"):
        returns.calc_prob_backtest_overfitting(n_blocks=2, metric=lambda series: series.to_numpy())


def test_calc_prob_backtest_overfitting_single_trial():
    with np.testing.assert_raises(ValueError):
        ffn.calc_prob_backtest_overfitting(pd.DataFrame({"a": range(8)}), n_blocks=2)


def test_calc_deflated_sharpe_ratio():
    np.random.seed(0)
    n_trials, n_periods = 40, 1000
    index = pd.date_range(start="2015-01-01", periods=n_periods, freq="D")
    # A search over trials that have no skill whatsoever
    trials = pd.DataFrame(np.random.normal(0, 0.01, (n_periods, n_trials)), index=index)
    sharpes = trials.calc_sharpe()
    winner = trials[sharpes.idxmax()]

    dsr = ffn.calc_deflated_sharpe_ratio(winner, sharpes)
    assert 0 <= dsr <= 1
    # The winner of a skill-less search must not survive deflation ...
    assert dsr < 0.95
    # ... though it looks significant when its selection is ignored
    assert ffn.calc_deflated_sharpe_ratio(winner, [sharpes.max()]) > dsr

    # More trials set a higher hurdle, hence a lower probability
    assert ffn.calc_deflated_sharpe_ratio(winner, sharpes[:10]) > dsr

    # Attached to pandas objects like the other metrics
    aae(winner.calc_deflated_sharpe_ratio(sharpes), dsr)


@mark.parametrize("invalid", [np.nan, np.inf, -np.inf, pd.NA, None])
@mark.parametrize("annualized_trials", [True, False])
@mark.parametrize("dtype", [None, object, "Float32", "Float64"])
def test_calc_deflated_sharpe_ratio_nonfinite_trials(invalid, annualized_trials, dtype):
    returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 250))
    trials = [0.0, 0.5, invalid]
    if dtype is not None:
        trials = pd.Series(trials, dtype=dtype)
    original = trials.copy()

    result = ffn.calc_deflated_sharpe_ratio(
        returns,
        trials,
        nperiods=252,
        annualized_trials=annualized_trials,
    )

    assert np.isnan(result)
    assert np.isnan(returns.calc_deflated_sharpe_ratio(trials, nperiods=252, annualized_trials=annualized_trials))
    if isinstance(trials, pd.Series):
        pd.testing.assert_series_equal(trials, original)
    else:
        assert trials == original


def test_calc_deflated_sharpe_ratio_scalar_trial():
    returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 250))

    # Scalar inputs take the same public coercion path through a zero-dimensional
    # NumPy array, which must be validated before pandas boxes it as an object.
    assert np.isfinite(ffn.calc_deflated_sharpe_ratio(returns, 0.5, nperiods=252))
    for invalid in (np.nan, np.inf, -np.inf, pd.NA, None):
        assert np.isnan(ffn.calc_deflated_sharpe_ratio(returns, invalid, nperiods=252))


def test_calc_deflated_sharpe_ratio_nonfinite_calculated_trials():
    trial_returns = pd.DataFrame(
        {
            "varying_a": [0.01, -0.02, 0.03, 0.0],
            "constant": [0.01] * 4,
            "varying_b": [-0.01, 0.02, -0.005, 0.015],
        }
    )
    trial_sharpes = trial_returns.calc_sharpe(nperiods=252)
    original = trial_sharpes.copy()

    # A constant evaluated trial has no Sharpe ratio, so the full search cannot
    # define the dispersion used by the deflated-Sharpe hurdle.
    assert trial_sharpes.isna().equals(pd.Series([False, True, False], index=trial_sharpes.index))
    result = trial_returns["varying_a"].calc_deflated_sharpe_ratio(
        trial_sharpes,
        nperiods=252,
    )

    assert np.isnan(result)
    pd.testing.assert_series_equal(trial_sharpes, original)


def test_calc_deflated_sharpe_ratio_ignores_missing_returns():
    """Exclude missing returns from the deflated Sharpe sample size."""
    observed = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 250))
    trial_sharpes = pd.Series([0.0, 0.1, 0.2, 0.3])

    # Independently evaluating the documented formula over 250 observations gives
    # this probability; changing only their missing-value representation cannot change it.
    expected = 0.9061936787765773
    padding = pd.Series([np.nan] * len(observed))
    internal = observed.repeat(2).reset_index(drop=True)
    internal.iloc[1::2] = np.nan
    padded_cases = {
        "leading": pd.concat([padding, observed], ignore_index=True),
        "trailing": pd.concat([observed, padding], ignore_index=True),
        "internal": internal,
    }

    assert np.isclose(
        ffn.calc_deflated_sharpe_ratio(observed, trial_sharpes, nperiods=252),
        expected,
    )
    for name, padded in padded_cases.items():
        module_result = ffn.calc_deflated_sharpe_ratio(
            padded,
            trial_sharpes,
            nperiods=252,
        )
        assert np.isclose(module_result, expected), name
        method_result = padded.calc_deflated_sharpe_ratio(trial_sharpes, nperiods=252)
        assert method_result == module_result, name


def test_calc_deflated_sharpe_ratio_series_rf_sets_the_sample_size():
    """A Series risk-free rate decides how many observations the statistic sees."""
    # An rf that covers part of the return index aligns the rest away, so the
    # Sharpe ratio, skew and kurtosis are all taken over fewer observations
    # than the return series holds. The sample size has to follow them.
    index = pd.date_range("2020-01-01", periods=250, freq="B")
    returns = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 250), index=index)
    rf = pd.Series(0.0001, index=index[:100])
    trial_sharpes = pd.Series([0.0, 0.1, 0.2, 0.3])

    actual = ffn.calc_deflated_sharpe_ratio(returns, trial_sharpes, rf=rf, nperiods=252)
    expected = ffn.calc_deflated_sharpe_ratio(returns.iloc[:100], trial_sharpes, rf=rf, nperiods=252)

    assert np.isclose(actual, expected)


def test_calc_information_ratio_dataframe():
    returns = pd.DataFrame(
        {
            "varying": [0.03, 0.01, -0.02, 0.04],
            "constant": [0.01, 0.01, 0.01, 0.01],
        }
    )
    benchmark = pd.DataFrame(
        {
            "varying": [0.01, 0.0, -0.01, 0.02],
            "constant": [0.01, 0.01, 0.01, 0.01],
        }
    )

    actual = returns.calc_information_ratio(benchmark)
    difference = returns - benchmark
    expected = difference.mean() / difference.std(ddof=1)
    expected["constant"] = 0.0

    pd.testing.assert_series_equal(actual, expected)


def test_calc_information_ratio_dataframe_with_series_benchmark():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "fund_a": [0.03, 0.01, -0.02, 0.04],
            "fund_b": [0.02, 0.02, -0.01, 0.03],
        },
        index=index,
    )
    benchmark = pd.Series([0.01, 0.0, -0.01, 0.02], index=index)

    actual = returns.calc_information_ratio(benchmark)
    difference = returns.sub(benchmark, axis="index")
    expected = difference.mean() / difference.std(ddof=1)

    pd.testing.assert_series_equal(actual, expected)


def test_calc_information_ratio_series_with_dataframe_benchmark():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.Series([0.03, 0.01, -0.02, 0.04], index=index)
    benchmark = pd.DataFrame(
        {
            "bench_a": [0.01, 0.0, -0.01, 0.02],
            "bench_b": [0.02, 0.01, 0.0, 0.01],
        },
        index=index,
    )

    actual = returns.calc_information_ratio(benchmark)
    difference = benchmark.rsub(returns, axis="index")
    expected = difference.mean() / difference.std(ddof=1)

    pd.testing.assert_series_equal(actual, expected)


def test_calc_deflated_sharpe_ratio_zero_dispersion():
    sharpes = [0.5, 1.0, 1.5, 2.0]

    # A constant series has no dispersion, so no Sharpe ratio and no deflated one.
    # Its standard deviation is floating-point residue rather than an exact zero, so
    # the ratio comes out finite (~4.6e15) and reaches the deflation arithmetic, which
    # answered 1.0 -- certainty of an edge, from the one input that cannot show one.
    # Checked across values and lengths, not at one point: the residue depends on both,
    # so a guard calibrated on a single series passes while still leaking elsewhere.
    for value in (1e-7, 1e-4, 0.001, 0.01, 1.0, 100.0):
        for n in (3, 10, 250, 5000):
            flat = pd.Series([value] * n)
            assert np.isnan(
                ffn.calc_deflated_sharpe_ratio(flat, sharpes)
            ), f"leaked at value={value}, n={n}"
    assert np.isnan(ffn.calc_deflated_sharpe_ratio(pd.Series([0.0] * 250), sharpes))

    # The guard is relative to the scale of the data: a real but very quiet series
    # still gets a number.
    quiet = pd.Series(np.random.default_rng(1).normal(0, 1e-8, 250))
    assert 0 <= ffn.calc_deflated_sharpe_ratio(quiet, sharpes) <= 1

    # A long, quiet series around a nonzero mean still has real dispersion. The
    # zero-dispersion threshold must not grow with the sample size and swallow it.
    quiet_nonzero = pd.Series(1.0 + np.random.default_rng(1).normal(0, 1e-12, 5000))
    assert 0 <= ffn.calc_deflated_sharpe_ratio(quiet_nonzero, sharpes, nperiods=252) <= 1

    # Dispersion is measured after subtracting a series risk-free rate, matching the
    # returns used to calculate Sharpe. Check both directions of that distinction.
    rf = pd.Series(np.linspace(0.0001, 0.0002, 250))
    constant_excess = rf + 0.001
    assert np.isnan(ffn.calc_deflated_sharpe_ratio(constant_excess, sharpes, rf=rf, nperiods=252))

    variable_excess = pd.Series([0.001] * 250)
    assert 0 <= ffn.calc_deflated_sharpe_ratio(variable_excess, sharpes, rf=rf, nperiods=252) <= 1


def test_deannualize():
    res = ffn.deannualize(0.05, 252)
    assert np.allclose(res, np.power(1.05, 1 / 252.0) - 1)


def test_to_excess_returns(df):
    """Verify scalar risk-free rates preserve existing excess-return behavior."""
    rf = 0.05
    r = df.to_returns()

    assert np.allclose(r.to_excess_returns(0), r, equal_nan=True)

    # A float risk-free rate is annualized only when a period count is available.
    assert np.allclose(
        r.to_excess_returns(rf, nperiods=252),
        r.to_excess_returns(ffn.deannualize(rf, 252)),
        equal_nan=True,
    )

    assert np.allclose(r.to_excess_returns(rf), r - rf, equal_nan=True)


def test_to_excess_returns_dataframe_with_series_risk_free():
    """Align a time-indexed risk-free Series across DataFrame rows."""
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "fund_a": [0.03, 0.01, -0.02, 0.04],
            "fund_b": [0.02, 0.02, -0.01, 0.03],
        },
        index=index,
    )
    risk_free = pd.Series([0.01, 0.03, 0.0, 0.04], index=index)
    expected = pd.DataFrame(
        {
            "fund_a": returns["fund_a"] - risk_free,
            "fund_b": returns["fund_b"] - risk_free,
        },
        index=index,
    )

    # Exercise both public paths before checking metrics built on excess returns.
    pd.testing.assert_frame_equal(ffn.to_excess_returns(returns, risk_free), expected)
    pd.testing.assert_frame_equal(returns.to_excess_returns(risk_free), expected)

    # Each ratio is calculated independently from the expected aligned frame.
    expected_sharpe = expected.mean() / expected.std(ddof=1)
    pd.testing.assert_series_equal(ffn.calc_sharpe(returns, rf=risk_free, annualize=False), expected_sharpe)
    pd.testing.assert_series_equal(returns.calc_sharpe(rf=risk_free, annualize=False), expected_sharpe)

    downside_deviation = np.sqrt((expected.clip(upper=0.0) ** 2).mean())
    expected_sortino = expected.mean() / downside_deviation
    pd.testing.assert_series_equal(
        ffn.calc_sortino_ratio(returns, rf=risk_free, annualize=False),
        expected_sortino,
    )
    pd.testing.assert_series_equal(
        returns.calc_sortino_ratio(rf=risk_free, annualize=False),
        expected_sortino,
    )


def test_numpy_floating_risk_free_rates_match_python_float():
    """Apply annualized-rate semantics to every NumPy floating scalar."""
    index = pd.date_range("2025-01-01", periods=20, freq="B")
    series = pd.Series(np.linspace(-0.02, 0.03, 20), index=index, name="strategy")
    frame = pd.DataFrame({"strategy": series, "scaled": series * 0.8})
    prices = (1.0 + series).cumprod() * 100.0
    annual_rate = 0.0625
    nperiods = 252

    for scalar_name in ("float16", "float32", "float64"):
        risk_free = getattr(np, scalar_name)(annual_rate)
        # Derive the per-period hurdle independently of ffn's type-dispatch and
        # deannualization paths, using the exact value represented by each dtype.
        period_rate = (1.0 + float(risk_free)) ** (1.0 / nperiods) - 1.0

        for returns in (series, frame):
            expected_excess = returns - period_rate
            expected_sharpe = expected_excess.mean() / expected_excess.std(ddof=1) * np.sqrt(nperiods)
            downside_deviation = np.sqrt((expected_excess.clip(upper=0.0) ** 2).mean())
            expected_sortino = expected_excess.mean() / downside_deviation * np.sqrt(nperiods)

            for actual in (
                ffn.to_excess_returns(returns, risk_free, nperiods=nperiods),
                returns.to_excess_returns(risk_free, nperiods=nperiods),
                ffn.to_excess_returns(returns, risk_free),
                returns.to_excess_returns(risk_free),
            ):
                assert (actual == expected_excess).to_numpy().all()

            for actual in (
                ffn.calc_sharpe(returns, rf=risk_free, nperiods=nperiods),
                returns.calc_sharpe(rf=risk_free, nperiods=nperiods),
                returns.calc_sharpe_ratio(rf=risk_free, nperiods=nperiods),
                ffn.calc_sharpe(returns, rf=risk_free),
                returns.calc_sharpe(rf=risk_free),
                returns.calc_sharpe_ratio(rf=risk_free),
            ):
                if isinstance(actual, pd.Series):
                    assert (actual == expected_sharpe).all()
                else:
                    assert actual == expected_sharpe

            for actual in (
                ffn.calc_sortino_ratio(returns, rf=risk_free, nperiods=nperiods),
                returns.calc_sortino_ratio(rf=risk_free, nperiods=nperiods),
                returns.calc_sortino(rf=risk_free, nperiods=nperiods),
                ffn.calc_sortino_ratio(returns, rf=risk_free),
                returns.calc_sortino_ratio(rf=risk_free),
                returns.calc_sortino(rf=risk_free),
            ):
                if isinstance(actual, pd.Series):
                    assert (actual == expected_sortino).all()
                else:
                    assert actual == expected_sortino

        drawdowns = prices / prices.cummax() - 1.0
        ulcer_index = ((drawdowns * 100.0) ** 2).mean() ** 0.5
        years = (prices.index[-1] - prices.index[0]).total_seconds() / 31557600
        growth = prices.iloc[-1] / prices.iloc[0]
        expected_upi = (growth ** (1.0 / years) - 1.0 - float(risk_free)) * 100.0 / ulcer_index
        aae(
            ffn.to_ulcer_performance_index(prices, rf=risk_free, nperiods=nperiods),
            expected_upi,
        )
        aae(
            prices.to_ulcer_performance_index(rf=risk_free, nperiods=nperiods),
            expected_upi,
        )
        aae(
            ffn.to_ulcer_performance_index(prices, rf=risk_free),
            expected_upi,
        )
        aae(
            prices.to_ulcer_performance_index(rf=risk_free),
            expected_upi,
        )


def test_numpy_floating_risk_free_rates_require_periods():
    """Reject nonzero NumPy floating rates when periods are unavailable."""
    # A non-datetime index leaves the observation frequency deliberately
    # uninferrable, so each public ratio must require nperiods explicitly.
    returns = pd.Series([-0.02, 0.01, 0.03, -0.01], index=["a", "b", "c", "d"])
    prices = pd.Series([100.0, 98.0, 99.0, 102.0], index=["a", "b", "c", "d"])

    for scalar_name in ("float16", "float32", "float64"):
        risk_free = getattr(np, scalar_name)(0.0625)
        with np.testing.assert_raises(ValueError):
            ffn.calc_sharpe(returns, rf=risk_free)
        with np.testing.assert_raises(ValueError):
            ffn.calc_sortino_ratio(returns, rf=risk_free)
        with np.testing.assert_raises(ValueError):
            ffn.to_ulcer_performance_index(prices, rf=risk_free)


def test_numpy_floating_risk_free_rates_work_in_stats():
    """Keep NumPy floating rates on scalar statistics and output paths."""
    import contextlib
    import io

    annual_rate = 0.0625
    # Four years of business-daily prices exercise the daily, monthly, and
    # yearly PerformanceStats risk-free branches in one deterministic sample.
    index = pd.bdate_range("2020-01-02", periods=1000)
    returns = np.tile([-0.0125, -0.00625, 0.003125, 0.009375, 0.015625], 200)
    asset_a_prices = pd.Series(
        (1.0 + returns).cumprod() * 100.0,
        index=index,
        name="asset_a",
    )
    prices = pd.DataFrame(
        {
            "asset_a": asset_a_prices,
            "asset_b": (1.0 + returns * 0.8).cumprod() * 100.0,
        },
        index=index,
    )
    expected_stats = ffn.PerformanceStats(asset_a_prices, rf=annual_rate)

    for scalar_name in ("float16", "float32", "float64"):
        risk_free = getattr(np, scalar_name)(annual_rate)
        stats = ffn.PerformanceStats(asset_a_prices, rf=risk_free)

        for field in (
            "daily_sharpe",
            "daily_sortino",
            "monthly_sharpe",
            "monthly_sortino",
            "yearly_sharpe",
            "yearly_sortino",
        ):
            assert np.isclose(getattr(stats, field), getattr(expected_stats, field))

        assert stats.stats["rf"] == risk_free
        csv = stats.to_csv()
        assert csv is not None
        assert "Risk-free rate,6.25%" in csv
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            stats.display()
        assert "Annual risk-free rate considered: 6.25%" in output.getvalue()

        group = ffn.GroupStats(prices)
        group.set_riskfree_rate(risk_free)
        assert (group.stats.loc["rf"] == annual_rate).all()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            group.display()
        risk_free_row = next(line for line in output.getvalue().splitlines() if line.startswith("Risk-free rate"))
        assert risk_free_row.split() == ["Risk-free", "rate", "6.25%", "6.25%"]

        # Zero still exercises scalar classification even though its
        # deannualized value is numerically unchanged.
        zero_rate = getattr(np, scalar_name)(0.0)
        zero_stats = ffn.PerformanceStats(asset_a_prices, rf=zero_rate)
        expected_zero = ffn.PerformanceStats(asset_a_prices, rf=0.0)
        assert zero_stats.daily_sharpe == expected_zero.daily_sharpe


def _assert_attribute_graph_unchanged(obj, expected):
    """Require a rejected state transition to preserve every attribute object."""
    assert obj.__dict__.keys() == expected.keys()
    for name, value in expected.items():
        assert obj.__dict__[name] is value, name


def test_performance_stats_failed_riskfree_update_preserves_state():
    """Reject incompatible risk-free prices without partially updating statistics."""
    dates = pd.date_range("2024-01-01", periods=8, tz="UTC")
    prices = pd.Series([100.0, 101.0, 99.0, 102.0, 104.0, 103.0, 106.0, 108.0], index=dates, name="asset")
    invalid_rf = pd.Series(np.linspace(100.0, 101.0, len(dates)), index=dates.tz_localize(None), name="rf")
    original_rf = invalid_rf.copy()
    stats = ffn.PerformanceStats(prices, rf=0.03, annualization_factor=365)
    original_state = stats.__dict__.copy()

    with raises(TypeError, match="tz-naive and tz-aware"):
        stats.set_riskfree_rate(invalid_rf)

    _assert_attribute_graph_unchanged(stats, original_state)
    pd.testing.assert_series_equal(invalid_rf, original_rf)


def test_group_stats_failed_riskfree_update_preserves_all_children():
    """Stage every child so one rejected update cannot leave a partially changed group."""
    dates = pd.date_range("2024-01-01", periods=8, tz="UTC")
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 99.0, 102.0, 104.0, 103.0, 106.0, 108.0],
            "B": [80.0, 79.0, 81.0, 82.0, 81.0, 84.0, 83.0, 86.0],
        },
        index=dates,
    )
    invalid_rf = pd.Series(np.linspace(100.0, 101.0, len(dates)), index=dates.tz_localize(None), name="rf")
    original_rf = invalid_rf.copy()
    stats = ffn.GroupStats(prices, annualization_factor=365)
    stats.set_riskfree_rate(0.03)
    original_state = stats.__dict__.copy()
    original_children = {name: stats[name] for name in stats._names}
    original_child_states = {name: child.__dict__.copy() for name, child in original_children.items()}

    with raises(TypeError, match="tz-naive and tz-aware"):
        stats.set_riskfree_rate(invalid_rf)

    _assert_attribute_graph_unchanged(stats, original_state)
    assert all(stats[name] is original_children[name] for name in stats._names)
    for name in stats._names:
        _assert_attribute_graph_unchanged(stats[name], original_child_states[name])
    pd.testing.assert_series_equal(invalid_rf, original_rf)

    # A later valid transition must still commit through the existing child objects.
    stats.set_riskfree_rate(0.05)
    expected = ffn.GroupStats(prices, annualization_factor=365)
    expected.set_riskfree_rate(0.05)
    assert all(stats[name] is original_children[name] for name in stats._names)
    pd.testing.assert_frame_equal(stats.stats, expected.stats)
    pd.testing.assert_frame_equal(stats.lookback_returns, expected.lookback_returns)


def test_group_stats_later_riskfree_failure_preserves_all_children(monkeypatch):
    """Do not commit an earlier staged child when a later child rejects the update."""
    dates = pd.date_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 99.0, 102.0, 104.0, 103.0, 106.0, 108.0],
            "B": [80.0, 79.0, 81.0, 82.0, 81.0, 84.0, 83.0, 86.0],
        },
        index=dates,
    )
    stats = ffn.GroupStats(prices, annualization_factor=365)
    stats.set_riskfree_rate(0.03)
    original_state = stats.__dict__.copy()
    original_children = {name: stats[name] for name in stats._names}
    original_child_states = {name: child.__dict__.copy() for name, child in original_children.items()}
    original_setter = ffn.core.PerformanceStats.set_riskfree_rate
    updated_names = []

    def reject_second_child(child, rf):
        updated_names.append(child.name)
        if child.name == "B":
            raise RuntimeError("second child rejected the risk-free rate")
        original_setter(child, rf)

    monkeypatch.setattr(ffn.core.PerformanceStats, "set_riskfree_rate", reject_second_child)

    with raises(RuntimeError, match="second child rejected"):
        stats.set_riskfree_rate(0.05)

    assert updated_names == ["A", "B"]
    _assert_attribute_graph_unchanged(stats, original_state)
    assert all(stats[name] is original_children[name] for name in stats._names)
    for name in stats._names:
        _assert_attribute_graph_unchanged(stats[name], original_child_states[name])


def test_set_riskfree_rate(df):
    r = df.to_returns()

    performanceStats = ffn.PerformanceStats(df["MSFT"])
    groupStats = ffn.GroupStats(df)
    daily_returns = df["MSFT"].resample("D").last().dropna().pct_change()

    aae(
        performanceStats.daily_sharpe,
        daily_returns.dropna().mean() / (daily_returns.dropna().std()) * (np.sqrt(252)),
        3,
    )

    aae(performanceStats.daily_sharpe, groupStats["MSFT"].daily_sharpe, 3)

    monthly_returns = df["MSFT"].resample(ffn.core._MonthEnd).last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        monthly_returns.dropna().mean()
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample(ffn.core._YearEnd).last().pct_change()
    aae(
        performanceStats.yearly_sharpe,
        yearly_returns.dropna().mean() / (yearly_returns.dropna().std()) * (np.sqrt(1)),
        3,
    )
    aae(performanceStats.yearly_sharpe, groupStats["MSFT"].yearly_sharpe, 3)

    performanceStats.set_riskfree_rate(0.02)
    groupStats.set_riskfree_rate(0.02)

    daily_returns = df["MSFT"].pct_change()
    aae(
        performanceStats.daily_sharpe,
        np.mean(daily_returns.dropna() - 0.02 / 252)
        / (daily_returns.dropna().std())
        * (np.sqrt(252)),
        3,
    )
    aae(performanceStats.daily_sharpe, groupStats["MSFT"].daily_sharpe, 3)

    monthly_returns = df["MSFT"].resample(ffn.core._MonthEnd).last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        np.mean(monthly_returns.dropna() - 0.02 / 12)
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample(ffn.core._YearEnd).last().pct_change()
    aae(
        performanceStats.yearly_sharpe,
        np.mean(yearly_returns.dropna() - 0.02 / 1)
        / (yearly_returns.dropna().std())
        * (np.sqrt(1)),
        3,
    )
    aae(performanceStats.yearly_sharpe, groupStats["MSFT"].yearly_sharpe, 3)

    rf = np.zeros(df.shape[0])
    # annual rf is 2%
    rf[1:] = 0.02 / 252
    rf[0] = 0.0
    # convert to price series
    rf = 100 * np.cumprod(1 + pd.Series(data=rf, index=df.index, name="rf"))

    performanceStats.set_riskfree_rate(rf)
    groupStats.set_riskfree_rate(rf)

    daily_returns = df["MSFT"].pct_change()
    rf_daily_returns = rf.pct_change()
    aae(
        performanceStats.daily_sharpe,
        np.mean(daily_returns - rf_daily_returns)
        / (daily_returns.dropna().std())
        * (np.sqrt(252)),
        3,
    )
    aae(performanceStats.daily_sharpe, groupStats["MSFT"].daily_sharpe, 3)

    monthly_returns = df["MSFT"].resample(ffn.core._MonthEnd).last().pct_change()
    rf_monthly_returns = rf.resample(ffn.core._MonthEnd).last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        np.mean(monthly_returns - rf_monthly_returns)
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample(ffn.core._YearEnd).last().pct_change()
    rf_yearly_returns = rf.resample(ffn.core._YearEnd).last().pct_change()
    aae(
        performanceStats.yearly_sharpe,
        np.mean(yearly_returns - rf_yearly_returns)
        / (yearly_returns.dropna().std())
        * (np.sqrt(1)),
        3,
    )
    aae(performanceStats.yearly_sharpe, groupStats["MSFT"].yearly_sharpe, 3)


def test_group_stats_preserves_scalar_riskfree_rate_across_date_ranges():
    """Preserve a scalar risk-free rate when GroupStats rebuilds its children."""
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 99.0, 102.0, 104.0, 103.0, 106.0, 108.0],
            "B": [80.0, 79.0, 81.0, 82.0, 81.0, 84.0, 83.0, 86.0],
        },
        index=pd.date_range("2024-01-01", periods=8),
    )
    annual_rate = 0.05
    annualization_factor = 252
    # Convert the annual scalar independently using the established compounding convention.
    period_rate = (1.0 + annual_rate) ** (1.0 / annualization_factor) - 1.0
    start = prices.index[2]
    stats = ffn.GroupStats(prices, annualization_factor=annualization_factor)
    stats.set_riskfree_rate(annual_rate)

    # Exercise both a partial rebuild and the no-argument full-range reset.
    # Derive the expected ratios without reusing ffn's risk-free helpers.
    for range_start, windowed_prices in ((start, prices.loc[start:]), (None, prices)):
        stats.set_date_range(start=range_start)

        for name in ("A", "B"):
            returns = windowed_prices[name].pct_change()
            expected_sharpe = (returns - period_rate).mean() / returns.std(ddof=1) * np.sqrt(annualization_factor)
            child = stats[name]
            assert child is not None
            assert child.rf == annual_rate
            assert -1e-12 < child.daily_sharpe - expected_sharpe < 1e-12

        assert (stats.stats.loc["rf"] == annual_rate).all()


def test_group_stats_preserves_series_riskfree_rate_across_date_ranges():
    """Preserve risk-free prices when GroupStats rebuilds its children."""
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 99.0, 102.0, 104.0, 103.0, 106.0, 108.0],
            "B": [80.0, 79.0, 81.0, 82.0, 81.0, 84.0, 83.0, 86.0],
        },
        index=pd.date_range("2024-01-01", periods=8),
    )
    annualization_factor = 252
    risk_free_prices = pd.Series(
        100.0 * np.cumprod(np.full(len(prices), 1.0 + 0.03 / annualization_factor)),
        index=prices.index,
        name="rf",
    )
    # Retain caller-owned input so the reconstruction can also prove non-mutation.
    original_risk_free_prices = risk_free_prices.copy()
    risk_free_returns = risk_free_prices.pct_change()
    start = prices.index[2]
    stats = ffn.GroupStats(prices, annualization_factor=annualization_factor)
    stats.set_riskfree_rate(risk_free_prices)

    # Exercise both a partial rebuild and the no-argument full-range reset.
    # Derive the expected ratios without reusing ffn's risk-free helpers.
    for range_start, windowed_prices in ((start, prices.loc[start:]), (None, prices)):
        stats.set_date_range(start=range_start)

        for name in ("A", "B"):
            returns = windowed_prices[name].pct_change()
            expected_sharpe = (returns - risk_free_returns).mean() / returns.std(ddof=1) * np.sqrt(annualization_factor)
            child = stats[name]
            assert child is not None
            assert isinstance(child.rf, pd.Series)
            pd.testing.assert_series_equal(child.rf, risk_free_prices)
            assert -1e-12 < child.daily_sharpe - expected_sharpe < 1e-12

    pd.testing.assert_series_equal(risk_free_prices, original_risk_free_prices)


def test_performance_stats(df):
    ps = ffn.PerformanceStats(df["AAPL"])

    pd.testing.assert_series_equal(ps.log_returns, ps.daily_prices.to_log_returns())
    num_stats = len(ps.stats.keys())
    num_unique_stats = len(ps.stats.keys().drop_duplicates())
    assert num_stats == num_unique_stats


def test_statistics_specifications_preserve_rows_and_fresh_lists():
    expected = [
        ("start", "Start", "dt"),
        ("end", "End", "dt"),
        ("rf", "Risk-free rate", "p"),
        (None, None, None),
        ("total_return", "Total Return", "p"),
        ("cagr", "CAGR", "p"),
        ("max_drawdown", "Max Drawdown", "p"),
        ("calmar", "Calmar Ratio", "n"),
        (None, None, None),
        ("mtd", "MTD", "p"),
        ("three_month", "3m", "p"),
        ("six_month", "6m", "p"),
        ("ytd", "YTD", "p"),
        ("one_year", "1Y", "p"),
        ("three_year", "3Y (ann.)", "p"),
        ("five_year", "5Y (ann.)", "p"),
        ("ten_year", "10Y (ann.)", "p"),
        ("incep", "Since Incep. (ann.)", "p"),
        (None, None, None),
        ("daily_sharpe", "Daily Sharpe", "n"),
        ("daily_sortino", "Daily Sortino", "n"),
        ("daily_mean", "Daily Mean (ann.)", "p"),
        ("daily_vol", "Daily Vol (ann.)", "p"),
        ("daily_skew", "Daily Skew", "n"),
        ("daily_kurt", "Daily Kurt", "n"),
        ("best_day", "Best Day", "p"),
        ("worst_day", "Worst Day", "p"),
        (None, None, None),
        ("monthly_sharpe", "Monthly Sharpe", "n"),
        ("monthly_sortino", "Monthly Sortino", "n"),
        ("monthly_mean", "Monthly Mean (ann.)", "p"),
        ("monthly_vol", "Monthly Vol (ann.)", "p"),
        ("monthly_skew", "Monthly Skew", "n"),
        ("monthly_kurt", "Monthly Kurt", "n"),
        ("best_month", "Best Month", "p"),
        ("worst_month", "Worst Month", "p"),
        (None, None, None),
        ("yearly_sharpe", "Yearly Sharpe", "n"),
        ("yearly_sortino", "Yearly Sortino", "n"),
        ("yearly_mean", "Yearly Mean", "p"),
        ("yearly_vol", "Yearly Vol", "p"),
        ("yearly_skew", "Yearly Skew", "n"),
        ("yearly_kurt", "Yearly Kurt", "n"),
        ("best_year", "Best Year", "p"),
        ("worst_year", "Worst Year", "p"),
        (None, None, None),
        ("avg_drawdown", "Avg. Drawdown", "p"),
        ("avg_drawdown_days", "Avg. Drawdown Days", "n"),
        ("avg_up_month", "Avg. Up Month", "p"),
        ("avg_down_month", "Avg. Down Month", "p"),
        ("win_year_perc", "Win Year %", "p"),
        ("twelve_month_win_perc", "Win 12m %", "p"),
    ]
    expected_group = expected.copy()
    expected_group[5:5] = [
        ("daily_sharpe", "Daily Sharpe", "n"),
        ("daily_sortino", "Daily Sortino", "n"),
    ]

    performance_stats = object.__new__(ffn.PerformanceStats)
    group_stats = dict.__new__(ffn.GroupStats)

    assert performance_stats._stats() == expected
    assert group_stats._stats() == expected_group

    performance_stats._stats().append(("extra", "Extra", "n"))
    group_stats._stats().clear()

    assert performance_stats._stats() == expected
    assert group_stats._stats() == expected_group


def _assert_csv_row_width(output, sep, expected_width):
    import csv
    import io

    rows = csv.reader(io.StringIO(output), delimiter=sep)
    assert {len(row) for row in rows} == {expected_width}


@mark.parametrize(
    "label",
    [-7, 0, 3.5, np.int64(7), np.uint64(2**63), np.float32(2.5), np.float64(-3.25), "fund"],
    ids=["integer", "zero", "float", "numpy-integer", "numpy-unsigned", "numpy-float32", "numpy-float64", "string"],
)
@mark.parametrize("grouped", [False, True], ids=["performance", "group"])
@mark.parametrize("sep", [",", ";"], ids=["comma", "semicolon"])
@mark.parametrize("series_rf", [False, True], ids=["scalar-riskfree", "series-riskfree"])
def test_statistics_to_csv_serializes_numeric_labels(label, grouped, sep, series_rf, tmp_path):
    """Changing strategy labels must affect only header text, including file exports."""
    import csv
    import io

    prices = pd.Series([100.0, 101.0, 99.0, 104.0, 103.0, 108.0], index=pd.date_range("2020-01-01", periods=6), name=label)
    if grouped:
        prices = pd.concat([prices, pd.Series([100.0, 99.0, 102.0, 101.0, 105.0, 104.0], index=prices.index, name="peer")], axis=1)
        control_prices = prices.copy()
        control_prices.columns = [str(label), "peer"]
        stats = ffn.GroupStats(prices)
        control = ffn.GroupStats(control_prices)
        labels = [label, "peer"]
    else:
        stats = ffn.PerformanceStats(prices)
        control = ffn.PerformanceStats(prices.rename(str(label)))
        labels = [label]

    risk_free = pd.Series([100.0, 100.1, 100.2, 100.3, 100.4, 100.5], index=prices.index) if series_rf else 0.03
    stats.set_riskfree_rate(risk_free)
    control.set_riskfree_rate(risk_free)
    original_prices = prices.copy(deep=True)
    original_stats = stats.stats.copy(deep=True)

    output = stats.to_csv(sep=sep)
    # A string-label control protects every metric, separator, and blank-row byte.
    assert output == control.to_csv(sep=sep)
    rows = list(csv.reader(io.StringIO(output), delimiter=sep))
    assert rows[0] == ["Stat"] + [str(name) for name in labels]
    assert {len(row) for row in rows} == {len(labels) + 1}
    assert ["Total Return", "8.00%"] + (["4.00%"] if grouped else []) in rows
    assert [""] * (len(labels) + 1) in rows

    path = tmp_path / "statistics.csv"
    assert stats.to_csv(sep=sep, path=path) is None
    assert path.read_text() == output
    # Compare native file bytes so platform newline translation remains unchanged.
    control_path = tmp_path / "control.csv"
    control.to_csv(sep=sep, path=control_path)
    assert path.read_bytes() == control_path.read_bytes()
    if grouped:
        assert stats._names == labels
        assert stats[label].name == label
        pd.testing.assert_frame_equal(prices, original_prices)
        pd.testing.assert_frame_equal(stats.stats, original_stats)
    else:
        assert stats.name == label
        pd.testing.assert_series_equal(prices, original_prices)
        pd.testing.assert_series_equal(stats.stats, original_stats)


@mark.parametrize(
    "label",
    ["fund", "", "SMA(10,20)", "SMA(10;20)", '"fund"', 'fund "hedged"', "fund\nhedged", "fund\rhedged", "fund\r\nhedged", 'fund,;"hedged"\r\nnext'],
    ids=["ordinary", "empty", "comma", "semicolon", "leading-quote", "embedded-quote", "lf", "cr", "crlf", "combined"],
)
@mark.parametrize("grouped", [False, True], ids=["performance", "group"])
@mark.parametrize("sep", [",", ";"], ids=["comma", "semicolon"])
@mark.parametrize("series_rf", [False, True], ids=["scalar-riskfree", "series-riskfree"])
@mark.parametrize("linesep", ["\n", "\r\n"], ids=["posix", "windows"])
def test_statistics_to_csv_quotes_strategy_labels(label, grouped, sep, series_rf, linesep, tmp_path, monkeypatch):
    """Quoted labels round-trip while metric bytes and native file conventions stay intact."""
    import csv
    import io
    import os

    native_open = open

    def platform_open(path, mode, newline=None):
        return native_open(path, mode, newline=linesep if newline is None else newline)

    # Exercise Windows text translation even when the tests run on a POSIX host.
    monkeypatch.setattr(ffn.core, "open", platform_open, raising=False)
    monkeypatch.setattr(os, "linesep", linesep)

    prices = pd.Series([100.0, 101.0, 99.0, 104.0, 103.0, 108.0], index=pd.date_range("2020-01-01", periods=6), name=label)
    if grouped:
        # A second special label protects escaping beyond the first strategy column.
        peer = 'peer;"B"\r\nnext'
        prices = pd.concat([prices, pd.Series([100.0, 99.0, 102.0, 101.0, 105.0, 104.0], index=prices.index, name=peer)], axis=1)
        control_prices = prices.copy()
        control_prices.columns = ["control", "peer"]
        stats = ffn.GroupStats(prices)
        control = ffn.GroupStats(control_prices)
        labels = [label, peer]
    else:
        stats = ffn.PerformanceStats(prices)
        control = ffn.PerformanceStats(prices.rename("control"))
        labels = [label]
    risk_free = pd.Series([100.0, 100.1, 100.2, 100.3, 100.4, 100.5], index=prices.index) if series_rf else 0.03
    stats.set_riskfree_rate(risk_free)
    control.set_riskfree_rate(risk_free)
    original_prices = prices.copy(deep=True)
    original_stats = stats.stats.copy(deep=True)

    header = io.StringIO(newline="")
    # CRLF makes the independent writer quote either newline character; exports retain LF rows.
    csv.writer(header, delimiter=sep, lineterminator="\r\n").writerow(["Stat"] + labels)
    expected_header = header.getvalue()[:-2]
    control_output = control.to_csv(sep=sep)
    expected = expected_header + control_output[control_output.index("\n") :]
    output = stats.to_csv(sep=sep)
    assert output == expected
    rows = list(csv.reader(io.StringIO(output, newline=""), delimiter=sep))
    assert rows[0] == ["Stat"] + labels
    assert rows[1:] == list(csv.reader(io.StringIO(control_output), delimiter=sep))[1:]
    assert {len(row) for row in rows} == {len(labels) + 1}
    assert ["Total Return", "8.00%"] + (["4.00%"] if grouped else []) in rows
    assert [""] * (len(labels) + 1) in rows

    path = tmp_path / "statistics.csv"
    assert stats.to_csv(sep=sep, path=path) is None
    # Only record separators use platform line endings; quoted field content stays exact.
    expected_file = expected_header + control_output[control_output.index("\n") :].replace("\n", linesep)
    assert path.read_bytes().decode() == expected_file
    with path.open(newline="") as exported:
        file_rows = list(csv.reader(exported, delimiter=sep))
    assert file_rows[0] == ["Stat"] + labels
    assert {len(row) for row in file_rows} == {len(labels) + 1}
    if grouped:
        assert stats._names == labels
        assert [stats[name].name for name in labels] == labels
        pd.testing.assert_frame_equal(prices, original_prices)
        pd.testing.assert_frame_equal(stats.stats, original_stats)
    else:
        assert stats.name == label
        pd.testing.assert_series_equal(prices, original_prices)
        pd.testing.assert_series_equal(stats.stats, original_stats)


@mark.parametrize("sep", [",", ";"], ids=["comma", "semicolon"])
def test_performance_stats_to_csv_preserves_row_width(df, sep):
    stats = ffn.PerformanceStats(df["AAPL"])

    _assert_csv_row_width(stats.to_csv(sep=sep), sep, expected_width=2)


@mark.parametrize(
    "prices",
    (
        pd.Series([], index=pd.DatetimeIndex([]), dtype=float, name="asset"),
        pd.Series(
            [np.nan, np.nan],
            index=pd.date_range("2025-01-01", periods=2),
            name="asset",
        ),
        pd.Series(
            [pd.NA, pd.NA],
            index=pd.date_range("2025-01-01", periods=2),
            dtype="Float64",
            name="asset",
        ),
    ),
)
def test_performance_stats_rejects_prices_without_usable_values(prices):
    original = prices.copy()

    with raises(ValueError, match="at least one usable value"):
        ffn.PerformanceStats(prices)

    pd.testing.assert_series_equal(prices, original)


def test_performance_stats_public_helpers_reject_empty_prices():
    prices = pd.Series([], index=pd.DatetimeIndex([]), dtype=float, name="asset")
    constructors = (
        ffn.calc_perf_stats,
        ffn.calc_stats,
        lambda value: value.calc_perf_stats(),
        lambda value: value.calc_stats(),
    )

    for constructor in constructors:
        with raises(ValueError, match="at least one usable value"):
            constructor(prices)


def test_group_stats_rejects_empty_rows():
    prices = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["A", "B"], dtype=float)
    original = prices.copy()
    constructors = (
        lambda value: ffn.GroupStats(value),
        ffn.calc_stats,
        lambda value: value.calc_stats(),
    )

    for constructor in constructors:
        with raises(ValueError, match="at least one usable value"):
            constructor(prices)

    pd.testing.assert_frame_equal(prices, original)


@mark.parametrize("empty_column", ("A", "B"))
def test_group_stats_rejects_unusable_child(empty_column):
    prices = pd.DataFrame(
        {"A": [100.0, 101.0], "B": [50.0, 51.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    prices[empty_column] = np.nan
    original = prices.copy()

    # Exercise both child orders so rejection cannot depend on a prior valid child.
    with raises(ValueError, match="at least one usable value"):
        ffn.GroupStats(prices)
    with raises(ValueError, match="at least one usable value"):
        prices.calc_stats()

    pd.testing.assert_frame_equal(prices, original)


@mark.parametrize("positions", [[3, 2, 1, 0], [0, 2, 1, 3]], ids=["descending", "unsorted"])
def test_performance_stats_rejects_nonmonotonic_prices(positions):
    """Reject unsorted prices through every public Series statistics path."""
    index = pd.date_range("2020-12-31", periods=4, freq=ffn.core._YearEnd)
    prices = pd.Series([100.0, 110.0, 121.0, 133.1], index=index, name="asset").iloc[positions]
    original = prices.copy()

    for constructor in (ffn.PerformanceStats, ffn.calc_perf_stats, ffn.calc_stats):
        with raises(ValueError, match="prices index must be monotonic increasing"):
            constructor(prices)

    with raises(ValueError, match="prices index must be monotonic increasing"):
        prices.calc_perf_stats()
    with raises(ValueError, match="prices index must be monotonic increasing"):
        prices.calc_stats()
    pd.testing.assert_series_equal(prices, original)


def test_dataframe_stats_rejects_nonmonotonic_prices():
    """Reject an unsorted DataFrame through module and pandas entry points."""
    index = pd.date_range("2020-12-31", periods=4, freq=ffn.core._YearEnd)
    prices = pd.DataFrame({"asset": [133.1, 121.0, 110.0, 100.0]}, index=index[::-1])
    original = prices.copy()

    with raises(ValueError, match="prices index must be monotonic increasing"):
        ffn.GroupStats(prices)
    with raises(ValueError, match="prices index must be monotonic increasing"):
        ffn.calc_stats(prices)
    with raises(ValueError, match="prices index must be monotonic increasing"):
        prices.calc_stats()
    pd.testing.assert_frame_equal(prices, original)


def test_group_stats_rejects_nonmonotonic_component_before_merge():
    """Reject an unsorted component before GroupStats can normalize its order."""
    index = pd.date_range("2020-12-31", periods=4, freq=ffn.core._YearEnd)
    ascending = pd.Series([100.0, 110.0, 121.0, 133.1], index=index, name="ascending")
    descending = pd.Series([66.55, 60.5, 55.0, 50.0], index=index[::-1], name="descending")
    original_ascending = ascending.copy()
    original_descending = descending.copy()

    with raises(ValueError, match="prices index must be monotonic increasing"):
        ffn.GroupStats(ascending, descending)

    pd.testing.assert_series_equal(ascending, original_ascending)
    pd.testing.assert_series_equal(descending, original_descending)


@mark.parametrize(
    "index",
    [
        pd.date_range("2020-12-31", periods=4, freq=ffn.core._YearEnd),
        pd.to_datetime(["2020-12-31", "2021-12-31", "2021-12-31", "2023-12-31"]),
    ],
    ids=["ascending", "duplicate-dates"],
)
def test_performance_stats_accepts_monotonic_prices(index):
    """Preserve ascending and duplicate-date performance inputs."""
    prices = pd.Series([100.0, 110.0, 121.0, 133.1], index=index, name="asset")
    original = prices.copy()

    stats = ffn.PerformanceStats(prices)
    group = ffn.GroupStats(prices)
    group_stats = group["asset"]

    assert stats.total_return == approx(0.331)
    assert group_stats is not None
    assert group_stats.total_return == approx(0.331)
    pd.testing.assert_series_equal(prices, original)


def test_performance_stats_uses_observed_price_endpoints():
    """Use observed outer prices for endpoint dates and total return."""
    dates = pd.date_range("2025-01-01", periods=4, tz="UTC")
    price_series = (
        pd.Series([np.nan, 100.0, 105.0], index=dates[:3]),
        pd.Series([100.0, 105.0, np.nan], index=dates[:3]),
        pd.Series([pd.NA, 100.0, 105.0, pd.NA], index=dates, dtype="Float64"),
        pd.Series([np.nan, 100.0, 105.0, np.nan], index=pd.date_range("2025-01-01 09:00", periods=4, freq="h", tz="UTC")),
    )

    for prices in price_series:
        observed = prices.dropna()
        expected_total_return = observed.iloc[-1] / observed.iloc[0] - 1
        results = (
            ffn.PerformanceStats(prices),
            ffn.calc_perf_stats(prices),
            ffn.calc_stats(prices),
            prices.calc_perf_stats(),
            prices.calc_stats(),
        )

        for stats in results:
            assert isinstance(stats, ffn.PerformanceStats)
            assert stats.start == observed.index[0]
            assert stats.end == observed.index[-1]
            assert stats.total_return == expected_total_return


def test_performance_stats_date_range_reset_uses_observed_endpoints():
    """Restore observed endpoints when resetting a statistics date range."""
    dates = pd.date_range("2025-01-01", periods=5)
    prices = pd.Series([np.nan, 100.0, 105.0, 110.0, np.nan], index=dates)
    stats = ffn.PerformanceStats(prices)

    stats.set_date_range(start=dates[2], end=dates[3])
    stats.set_date_range()

    assert stats.start == dates[1]
    assert stats.end == dates[3]
    assert stats.total_return == 110.0 / 100.0 - 1


@mark.parametrize(
    ("start", "end"),
    (
        ("2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
        ("2024-01-02T00:00:00Z", "2024-01-02T00:00:00Z"),
    ),
)
def test_performance_stats_empty_date_range_preserves_state(start, end):
    dates = pd.date_range("2024-01-01", periods=4, tz="UTC")
    prices = pd.Series([100.0, np.nan, 102.0, 103.0], index=dates, name="asset")
    original_prices = prices.copy()
    stats = ffn.PerformanceStats(prices, rf=0.03, annualization_factor=365)
    original_scalars = (
        stats.start,
        stats.end,
        stats.total_return,
        stats.rf,
        stats.annualization_factor,
    )
    original_daily = stats.daily_prices.copy()
    original_monthly = stats.monthly_prices.copy()
    original_yearly = stats.yearly_prices.copy()
    original_stats = stats.stats.copy()
    original_lookbacks = stats.lookback_returns.copy()
    original_return_table = stats.return_table.copy()

    with raises(ValueError, match="no usable data"):
        stats.set_date_range(start=start, end=end)

    assert (
        stats.start,
        stats.end,
        stats.total_return,
        stats.rf,
        stats.annualization_factor,
    ) == original_scalars
    pd.testing.assert_series_equal(stats.daily_prices, original_daily)
    pd.testing.assert_series_equal(stats.monthly_prices, original_monthly)
    pd.testing.assert_series_equal(stats.yearly_prices, original_yearly)
    pd.testing.assert_series_equal(stats.stats, original_stats)
    pd.testing.assert_series_equal(stats.lookback_returns, original_lookbacks)
    pd.testing.assert_frame_equal(stats.return_table, original_return_table)
    pd.testing.assert_series_equal(prices, original_prices)


def test_performance_stats_empty_date_range_guard_accepts_one_price():
    dates = pd.date_range("2024-01-01", periods=3)
    stats = ffn.PerformanceStats(pd.Series([100.0, 101.0, 102.0], index=dates))

    stats.set_date_range(start=dates[1], end=dates[1])

    assert stats.start == dates[1]
    assert stats.end == dates[1]
    assert len(stats.daily_prices) == 1
    assert pd.isna(stats.total_return)


@mark.parametrize("empty_column", ("A", "B"))
def test_group_stats_empty_date_range_preserves_all_children(empty_column):
    dates = pd.date_range("2024-01-01", periods=6, tz="UTC")
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "B": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
        },
        index=dates,
    )
    prices.loc[dates[:2], empty_column] = np.nan
    original_prices = prices.copy()
    stats = ffn.GroupStats(prices, annualization_factor=365)
    stats.set_riskfree_rate(0.03)
    original_group_prices = stats.prices.copy()
    original_stats = stats.stats.copy()
    original_lookbacks = stats.lookback_returns.copy()
    original_children = {name: stats[name] for name in ("A", "B")}

    with raises(ValueError, match="no usable data"):
        stats.set_date_range(start=dates[0], end=dates[1])

    pd.testing.assert_frame_equal(stats.prices, original_group_prices)
    pd.testing.assert_frame_equal(stats.stats, original_stats)
    pd.testing.assert_frame_equal(stats.lookback_returns, original_lookbacks)
    assert all(stats[name] is original_children[name] for name in ("A", "B"))
    assert stats._riskfree_rate == 0.03
    assert stats._annualization_factor_override == 365
    pd.testing.assert_frame_equal(prices, original_prices)


def test_group_stats_empty_shared_date_range_keeps_usable_children():
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    first = pd.Series([100.0], index=dates[:1], name="first")
    second = pd.Series([200.0], index=dates[1:], name="second")
    stats = ffn.GroupStats(first, second)

    stats.set_date_range()

    assert stats.prices.empty
    pd.testing.assert_series_equal(stats["first"].prices, first)
    pd.testing.assert_series_equal(stats["second"].prices, second)


def test_group_stats_calc_stats(df):
    gs = df.calc_stats()

    num_stats = len(gs.stats.index)
    num_unique_stats = len(gs.stats.index.drop_duplicates())
    assert num_stats == num_unique_stats


def test_group_stats_integer_labels_take_precedence_over_positions(df):
    prices = df[["AAPL", "MSFT"]].rename(columns={"AAPL": 1, "MSFT": 0})
    stats = ffn.GroupStats(prices)

    # Reversed labels distinguish mapping lookup from positional lookup.
    assert stats[0] is dict.__getitem__(stats, 0)
    assert stats[1] is dict.__getitem__(stats, 1)


def test_group_stats_integer_lookup_falls_back_to_position(df):
    stats = ffn.GroupStats(df[["AAPL", "MSFT"]])

    assert stats[0] is stats["AAPL"]
    assert stats[-1] is stats["MSFT"]


@mark.parametrize("sep", [",", ";"], ids=["comma", "semicolon"])
def test_group_stats_to_csv_preserves_row_width(df, sep):
    prices = df[["AAPL", "MSFT"]].rename(columns={"AAPL": "fund", "MSFT": "peer-with-a-long-name"})
    stats = ffn.GroupStats(prices)

    # Unequal name lengths ensure blank-row width cannot follow serialized text length.
    _assert_csv_row_width(stats.to_csv(sep=sep), sep, expected_width=3)


def test_group_stats_to_csv_formats_series_riskfree_rate_as_unavailable(df):
    prices = df[["AAPL", "MSFT"]]
    risk_free_prices = pd.Series(np.linspace(100.0, 110.0, len(prices)), index=prices.index)
    stats = ffn.GroupStats(prices)

    assert "Risk-free rate,0.00%,0.00%" in stats.to_csv().splitlines()
    stats.set_riskfree_rate(risk_free_prices)

    output = stats.to_csv()

    # A price series has no single annual percentage to place in the summary row.
    assert "Risk-free rate,-,-" in output.splitlines()
    _assert_csv_row_width(output, ",", expected_width=3)


def test_calc_stats_annualization_factor(df):
    prices = df[["AAPL", "MSFT"]]
    stats = prices.calc_stats(annualization_factor=365)

    assert stats["AAPL"].annualization_factor == 365
    assert stats["MSFT"].annualization_factor == 365

    stats.set_date_range(start=prices.index[10])
    assert stats["AAPL"].annualization_factor == 365
    assert stats["MSFT"].annualization_factor == 365

    single_stats = prices["AAPL"].calc_stats(annualization_factor=365)
    assert single_stats.annualization_factor == 365


def test_group_stats_uses_each_series_own_calendar():
    # GH #155: a NaN row in one series should not drop that date from the
    # other series' individual stats
    dates = pd.date_range("2020-01-31", periods=24, freq=ffn.core._MonthEnd)
    np.random.seed(1)
    sym1 = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.01, 0.04, 24)), index=dates, name="SYM1"
    )
    sym2 = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.01, 0.04, 24)), index=dates, name="SYM2"
    )
    # SYM2 missing for three months in the middle
    sym2.iloc[9:12] = np.nan

    gs = ffn.GroupStats(sym1, sym2)

    # per-series stats match the single-series result
    aae(gs["SYM1"].monthly_sharpe, ffn.PerformanceStats(sym1).monthly_sharpe, 9)
    aae(gs["SYM1"].total_return, ffn.PerformanceStats(sym1).total_return, 9)
    aae(
        gs["SYM2"].monthly_sharpe, ffn.PerformanceStats(sym2.dropna()).monthly_sharpe, 9
    )

    # SYM1 keeps all of its own dates
    assert len(gs["SYM1"].prices) == 24

    # cross-sectional prices still use the common calendar
    assert len(gs.prices) == 21

    # date range slicing preserves the per-series calendar behaviour
    gs.set_date_range(start=dates[3])
    aae(
        gs["SYM1"].monthly_sharpe,
        ffn.PerformanceStats(sym1[dates[3]:]).monthly_sharpe,
        9,
    )
    gs.set_date_range()
    aae(gs["SYM1"].monthly_sharpe, ffn.PerformanceStats(sym1).monthly_sharpe, 9)


def test_group_stats_date_range_reset_restores_full_calendars():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    early = pd.Series(range(100, 105), index=dates[:5], name="EARLY")
    late = pd.Series(range(200, 206), index=dates[2:], name="LATE")

    gs = ffn.GroupStats(early, late)
    gs.set_date_range()

    assert gs["EARLY"].start == dates[0]
    assert gs["EARLY"].end == dates[4]
    assert gs["LATE"].start == dates[2]
    assert gs["LATE"].end == dates[-1]


def test_resample_returns(df):
    num_years = 30
    num_months = num_years * 12
    np.random.seed(0)
    returns = np.random.normal(loc=0.06 / 12, scale=0.20 / np.sqrt(12), size=num_months)
    returns = pd.Series(returns)

    sample_mean = np.mean(returns)

    sample_stats = ffn.resample_returns(returns, np.mean, seed=0, num_trials=100)

    resampled_mean = np.mean(sample_stats)
    std_resampled_means = np.std(sample_stats, ddof=1)

    # resampled statistics should be within 3 std devs of actual
    assert np.abs((sample_mean - resampled_mean) / std_resampled_means) < 3

    np.random.seed(0)
    returns = np.random.normal(
        loc=0.06 / 12, scale=0.20 / np.sqrt(12), size=num_months * 3
    ).reshape(num_months, 3)
    returns = pd.DataFrame(returns)

    sample_mean = np.mean(returns, axis=0)

    sample_stats = ffn.resample_returns(
        returns, lambda x: np.mean(x, axis=0), seed=0, num_trials=100
    )

    resampled_mean = np.mean(sample_stats, axis=0)
    std_resampled_means = np.std(sample_stats, ddof=1, axis=0)

    # resampled statistics should be within 3 std devs of actual
    assert np.all(np.abs((sample_mean - resampled_mean) / std_resampled_means) < 3)

    returns = df.to_returns().dropna()
    sample_mean = np.mean(returns, axis=0)

    sample_stats = ffn.resample_returns(
        returns, lambda x: np.mean(x, axis=0), seed=0, num_trials=100
    )

    resampled_mean = np.mean(sample_stats, axis=0)
    std_resampled_means = np.std(sample_stats, ddof=1, axis=0)

    assert np.all(np.abs((sample_mean - resampled_mean) / std_resampled_means) < 3)


def test_resample_returns_duplicate_labels():
    """Sample Series and DataFrame rows positionally when labels repeat."""
    returns = pd.Series([10.0, 20.0, 30.0], index=["a", "a", "b"])
    original = returns.copy()
    sample_sizes = ffn.resample_returns(returns, len, seed=0, num_trials=8)
    sample_stats = ffn.resample_returns(returns, np.sum, seed=0, num_trials=8)

    # Each expected value sums exactly three seeded row draws, including repeats.
    expected = np.array([40.0, 40.0, 40.0, 60.0, 80.0, 80.0, 60.0, 60.0])
    np.testing.assert_array_equal(sample_stats.to_numpy(dtype=float), expected)
    np.testing.assert_array_equal(sample_sizes.to_numpy(dtype=int), np.full(8, 3))
    pd.testing.assert_series_equal(returns, original)

    dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
    returns = pd.DataFrame({"first": [10.0, 20.0, 30.0], "second": [1.0, 2.0, 3.0]}, index=dates)
    original = returns.copy()
    sample_sizes = ffn.resample_returns(returns, len, seed=0, num_trials=8)
    sample_stats = ffn.resample_returns(returns, pd.DataFrame.sum, seed=0, num_trials=8)

    expected = pd.DataFrame({"first": expected, "second": expected / 10}, dtype=object)
    pd.testing.assert_frame_equal(sample_stats, expected)
    np.testing.assert_array_equal(sample_sizes.to_numpy(dtype=int), np.full((8, 2), 3))
    pd.testing.assert_frame_equal(returns, original)


@mark.parametrize("as_frame", [False, True])
@mark.parametrize(
    "index,positions,seed",
    [
        (pd.date_range("2024-01-31", periods=1, freq=ffn.core._MonthEnd, tz="UTC", name="date"), [0], 0),
        (pd.date_range("2024-01-01", periods=3, tz="UTC", name="date"), [2, 1, 0], 6),
        (pd.timedelta_range("0 days", periods=3, freq="D", name="elapsed"), [2, 1, 0], 6),
    ],
)
def test_resample_returns_preserves_sampled_index_metadata(as_frame, index, positions, seed):
    returns = pd.Series(np.arange(len(index), dtype=float), index=index, name="returns")
    if as_frame:
        returns = returns.to_frame()
    expected = returns.iloc[positions].copy()
    expected.index = pd.Index([index[position] for position in positions], name=index.name)

    def statistic(sample):
        # Label-based sampling did not infer frequency, which callbacks can use.
        assert sample.index.freq is None
        if as_frame:
            pd.testing.assert_frame_equal(sample, expected)
        else:
            pd.testing.assert_series_equal(sample, expected)
        return sample.sum()

    ffn.resample_returns(returns, statistic, seed=seed, num_trials=1)


@mark.parametrize("as_frame", [False, True])
@mark.parametrize("dtype", ["float64", "Float64"])
@mark.parametrize("seed", [0, 6, 2**32 - 3, np.int64(0), np.uint32(0)])
def test_resample_returns_preserves_seeded_callbacks(as_frame, dtype, seed):
    """Keep each seeded draw, callback order and value ownership intact."""
    returns = pd.Series([10, 20, 30], index=pd.Index(["a", "a", "b"], name="row"), dtype=dtype, name="return")
    if as_frame:
        returns = pd.DataFrame({"return": returns, "scaled": returns / 10})
        returns.columns.name = "asset"
    original = returns.copy(deep=True)
    expected_statistics = []
    calls = []

    def statistic(sample):
        # The independent draw oracle does not depend on pandas or ffn sampling.
        positions = np.random.RandomState(seed + len(calls)).choice(3, size=3, replace=True)
        expected = original.iloc[positions]
        if as_frame:
            pd.testing.assert_frame_equal(sample, expected)
        else:
            pd.testing.assert_series_equal(sample, expected)
        calls.append(positions)
        expected_statistics.append(expected.sum())
        sample.iloc[0] = 999
        return expected_statistics[-1]

    random_state = np.random.get_state()
    actual = ffn.resample_returns(returns, statistic, seed=seed, num_trials=3)

    assert len(calls) == 3
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], random_state[1])
    assert after[0] == random_state[0] and after[2:] == random_state[2:]
    if as_frame:
        expected = pd.DataFrame(expected_statistics, dtype=object)
        pd.testing.assert_frame_equal(actual, expected)
        pd.testing.assert_frame_equal(returns, original)
    else:
        pd.testing.assert_series_equal(actual, pd.Series(expected_statistics, dtype=float))
        pd.testing.assert_series_equal(returns, original)


@mark.parametrize("as_frame", [False, True])
@mark.parametrize("seed, trials", [(-1, 1), (2**32, 1), (0.5, 1), (2**32 - 1, 2)])
def test_resample_returns_preserves_native_rejections(as_frame, seed, trials):
    """Dependency-owned error forms may vary, but must match the native sampler."""
    size = 3
    returns = pd.Series(np.arange(size, dtype=float), name="return")
    if as_frame:
        returns = returns.to_frame()
    original = returns.copy(deep=True)
    calls = []

    with raises((ValueError, TypeError)) as native:
        ffn.core.resample(returns, returns.index, n_samples=size, random_state=seed + trials - 1)

    def statistic(sample):
        calls.append(len(sample))
        return sample.sum()

    with raises(type(native.value)) as actual:
        ffn.resample_returns(returns, statistic, seed=seed, num_trials=trials)
    assert str(actual.value) == str(native.value)
    # A valid maximum seed still calls the statistic before the next seed fails.
    assert calls == [size] * (trials - 1)
    if as_frame:
        pd.testing.assert_frame_equal(returns, original)
    else:
        pd.testing.assert_series_equal(returns, original)


@mark.parametrize("as_frame", [False, True])
def test_resample_returns_preserves_native_empty_behavior(as_frame):
    """Keep empty-input acceptance or rejection owned by the installed sampler."""
    returns = pd.Series([], dtype=float, name="return")
    if as_frame:
        returns = returns.to_frame()
    calls = []

    def statistic(sample):
        calls.append(len(sample))
        if as_frame:
            pd.testing.assert_frame_equal(sample, returns)
        else:
            pd.testing.assert_series_equal(sample, returns)
        return sample.sum()

    # Empty n_samples=0 is dependency-owned, rather than a new ffn policy.
    try:
        ffn.core.resample(returns, returns.index, n_samples=0, random_state=0)
    except (ValueError, TypeError) as native:
        with raises(type(native)) as actual:
            ffn.resample_returns(returns, statistic, num_trials=1)
        assert str(actual.value) == str(native)
        assert calls == []
    else:
        actual = ffn.resample_returns(returns, statistic, num_trials=1)
        assert calls == [0]
        assert np.asarray(actual.iloc[0]).sum() == 0
    assert returns.empty


@mark.parametrize("as_frame", [False, True])
@mark.parametrize("dtype", ["float64", "Float64"])
def test_resample_returns_stops_at_callback_error(as_frame, dtype):
    returns = pd.Series([1, 2, 3], dtype=dtype, name="return")
    if as_frame:
        returns = returns.to_frame()
    original = returns.copy(deep=True)
    calls = []

    def statistic(sample):
        calls.append(len(sample))
        sample.iloc[0] = 999
        if len(calls) == 2:
            raise RuntimeError("statistic failed")
        return sample.sum()

    with raises(RuntimeError, match="statistic failed"):
        ffn.resample_returns(returns, statistic, num_trials=5)
    assert calls == [3, 3]
    if as_frame:
        pd.testing.assert_frame_equal(returns, original)
    else:
        pd.testing.assert_series_equal(returns, original)


@mark.parametrize("as_frame", [False, True])
def test_resample_returns_callback_can_change_population(as_frame):
    """Each trial draws from current rows while retaining the initial sample size."""
    returns = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"], name="return")
    if as_frame:
        returns = returns.to_frame()
    calls = []

    def statistic(sample):
        calls.append(len(sample))
        if len(calls) == 1:
            returns.drop(index="a", inplace=True)
        return sample.sum()

    actual = ffn.resample_returns(returns, statistic, num_trials=2)

    # Seed 0 draws a,b,a; seed 1 draws c,c,b from the remaining population.
    assert calls == [3, 3]
    if as_frame:
        pd.testing.assert_frame_equal(actual, pd.DataFrame({"return": [4.0, 8.0]}, dtype=object))
    else:
        pd.testing.assert_series_equal(actual, pd.Series([4.0, 8.0]))
    assert returns.index.tolist() == ["b", "c"]


def test_monthly_returns():
    dates = [
        "31/12/2017",
        "5/1/2018",
        "9/1/2018",
        "13/1/2018",
        "17/1/2018",
        "21/1/2018",
        "25/1/2018",
        "29/1/2018",
        "2/2/2018",
        "6/2/2018",
        "10/2/2018",
        "14/2/2018",
        "18/2/2018",
        "22/2/2018",
        "26/2/2018",
        "1/5/2018",
        "5/5/2018",
        "9/5/2018",
        "13/5/2018",
        "17/5/2018",
        "21/5/2018",
        "25/5/2018",
        "29/5/2018",
        "2/6/2018",
        "6/6/2018",
        "10/6/2018",
        "14/6/2018",
        "18/6/2018",
        "22/6/2018",
        "26/6/2018",
    ]

    prices = [
        100,
        98,
        100,
        103,
        106,
        106,
        107,
        111,
        115,
        115,
        118,
        122,
        120,
        119,
        118,
        119,
        118,
        120,
        122,
        126,
        130,
        131,
        131,
        134,
        138,
        139,
        139,
        138,
        140,
        140,
    ]

    df1 = pd.DataFrame(
        prices, index=pd.to_datetime(dates, format="%d/%m/%Y"), columns=["Price"]
    )

    obj1 = ffn.PerformanceStats(df1["Price"])

    obj1.monthly_returns == df1["Price"].resample(ffn.core._MonthEnd).last().fillna(1.0).pct_change(fill_method=None)


def test_drawdown_details(df):
    drawdown = ffn.to_drawdown_series(df["MSFT"])
    drawdown_details = ffn.drawdown_details(drawdown)

    assert drawdown_details.loc[drawdown_details.index[1], "Length"] == 18

    num_years = 30
    num_months = num_years * 12
    np.random.seed(0)
    returns = np.random.normal(loc=0.06 / 12, scale=0.20 / np.sqrt(12), size=num_months)
    returns = pd.Series(np.cumprod(1 + returns))

    drawdown = ffn.to_drawdown_series(returns)
    drawdown_details = ffn.drawdown_details(drawdown, index_type=drawdown.index)


def test_infer_nperiods():
    daily = pd.DataFrame(np.random.randn(10),
                         index=pd.date_range(start='2018-01-01', periods=10, freq='D'))
    hourly = pd.DataFrame(np.random.randn(10),
                          index=pd.date_range(start='2018-01-01', periods=10, freq='h'))
    yearly = pd.DataFrame(np.random.randn(10),
                          index=pd.date_range(start='2018-01-01', periods=10, freq=ffn.core._YearEnd))
    monthly = pd.DataFrame(np.random.randn(10),
                           index=pd.date_range(start='2018-01-01', periods=10, freq=ffn.core._MonthEnd))
    minutely = pd.DataFrame(np.random.randn(10),
                            index=pd.date_range(start='2018-01-01', periods=10, freq='min'))
    secondly = pd.DataFrame(np.random.randn(10),
                            index=pd.date_range(start='2018-01-01', periods=10, freq='s'))

    minutely_30 = pd.DataFrame(np.random.randn(10),
                               index=pd.date_range(start='2018-01-01', periods=10, freq='30min'))

    not_known_vals = np.concatenate((pd.date_range(start='2018-01-01', periods=5, freq='1h').values,
                                     pd.date_range(start='2018-01-02', periods=5, freq='5h').values))

    not_known = pd.DataFrame(np.random.randn(10),
                             index=pd.DatetimeIndex(not_known_vals))

    assert ffn.core.infer_nperiods(daily) == ffn.core.TRADING_DAYS_PER_YEAR
    assert ffn.core.infer_nperiods(hourly) == ffn.core.TRADING_DAYS_PER_YEAR * 24
    assert ffn.core.infer_nperiods(minutely) == ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60
    assert ffn.core.infer_nperiods(secondly) == ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60 * 60
    assert ffn.core.infer_nperiods(monthly) == 12
    assert ffn.core.infer_nperiods(yearly) == 1
    expected_30min_periods = ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60 / 30
    assert ffn.core.infer_nperiods(minutely_30) == expected_30min_periods

    returns_30min = minutely_30.squeeze()
    expected_sharpe = returns_30min.mean() / returns_30min.std(ddof=1) * np.sqrt(expected_30min_periods)
    assert np.allclose(returns_30min.calc_sharpe(), expected_sharpe)

    descending_30min = minutely_30.sort_index(ascending=False).squeeze()
    assert ffn.core.infer_nperiods(descending_30min) == expected_30min_periods
    assert np.allclose(descending_30min.calc_sharpe(), expected_sharpe)
    assert ffn.core.infer_nperiods(not_known) is None


def test_infer_nperiods_business_weekly_quarterly():
    # pandas infers "B", "W-SUN" and "QE-DEC" for these; each used to fall
    # through to None, which silently disabled annualization.
    business = pd.DataFrame(np.random.randn(400),
                            index=pd.bdate_range(start='2018-01-01', periods=400))
    weekly = pd.DataFrame(np.random.randn(60),
                          index=pd.date_range(start='2018-01-07', periods=60, freq='W'))
    quarterly = pd.DataFrame(np.random.randn(20),
                             index=pd.date_range(start='2018-03-31', periods=20, freq=pd.offsets.QuarterEnd()))

    assert ffn.core.infer_nperiods(business) == ffn.core.TRADING_DAYS_PER_YEAR
    assert ffn.core.infer_nperiods(weekly) == 52
    assert ffn.core.infer_nperiods(quarterly) == 4

    # Multipliers still divide, and an anchor does not break parsing.
    biweekly = pd.DataFrame(np.random.randn(30),
                            index=pd.date_range(start='2018-01-07', periods=30, freq='2W'))
    assert ffn.core.infer_nperiods(biweekly) == 26


def test_infer_nperiods_distinguishes_subsecond_aliases_from_month_start():
    for frequency, periods_per_second in (("ms", 1_000), ("us", 1_000_000), ("ns", 1_000_000_000)):
        data = pd.Series(np.random.randn(10), index=pd.date_range("2018-01-01", periods=10, freq=frequency))
        expected = ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60 * 60 * periods_per_second
        assert ffn.core.infer_nperiods(data) == expected

    month_start = pd.Series(np.random.randn(10), index=pd.date_range("2018-01-01", periods=10, freq="MS"))
    assert ffn.core.infer_nperiods(month_start) == 12


def test_calc_sharpe_annualizes_business_daily():
    # A business-day price series is the ordinary case for equities, and its
    # Sharpe ratio must be scaled by sqrt(252) like a calendar-daily one.
    index = pd.bdate_range(start='2018-01-01', periods=756)
    returns = pd.Series(np.random.randn(756) / 100, index=index)

    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(ffn.core.TRADING_DAYS_PER_YEAR)
    assert np.allclose(returns.calc_sharpe(), expected)

    # Unannualized is unaffected by the frequency.
    assert np.allclose(returns.calc_sharpe(annualize=False),
                       returns.mean() / returns.std(ddof=1))


def test_calc_sortino_annualizes_weekly():
    index = pd.date_range(start='2018-01-07', periods=260, freq='W')
    returns = pd.Series(np.random.randn(260) / 100, index=index)

    downside = np.sqrt((returns.clip(upper=0.0) ** 2).mean())
    expected = returns.mean() / downside * np.sqrt(52)
    assert np.allclose(returns.calc_sortino_ratio(), expected)
