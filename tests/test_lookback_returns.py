import numpy as np
import pandas as pd
import pytest

import ffn


def _compounding(freq, end, rate=0.01):
    index = pd.date_range("2020-12-31", end, freq=freq)
    return pd.Series(100 * (1 + rate) ** np.arange(len(index)), index=index)


@pytest.mark.parametrize(
    "freq,end",
    [
        (pd.offsets.MonthEnd(), "2023-06-30"),
        (pd.offsets.MonthEnd(), "2023-11-30"),
        (pd.offsets.MonthEnd(), "2024-02-29"),
        (pd.offsets.BMonthEnd(), "2023-09-29"),
    ],
)
def test_month_end_lookbacks_span_whole_months(freq, end):
    # +1% a month, so 3, 6 and 12 months back are exactly 1.01 ** n - 1. Day-number
    # anchors put Jun 30 less three months on Mar 30, whose last month-end price is
    # Feb 28, so the returns covered one month too many.
    stats = ffn.PerformanceStats(_compounding(freq, end))

    assert np.isclose(stats.three_month, 1.01**3 - 1)
    assert np.isclose(stats.six_month, 1.01**6 - 1)
    assert np.isclose(stats.one_year, 1.01**12 - 1)


def test_quarterly_lookbacks_span_whole_quarters():
    stats = ffn.PerformanceStats(_compounding(pd.offsets.QuarterEnd(), "2023-06-30"))

    assert np.isclose(stats.three_month, 0.01)
    assert np.isclose(stats.six_month, 1.01**2 - 1)
    assert np.isclose(stats.one_year, 1.01**4 - 1)


def test_daily_lookbacks_ending_mid_month_keep_the_same_day():
    prices = pd.Series(np.linspace(100.0, 130.0, 900), index=pd.bdate_range("2021-01-04", periods=900))
    end = prices.index[-1]
    stats = ffn.PerformanceStats(prices)

    for months, value in ((3, stats.three_month), (6, stats.six_month), (12, stats.one_year)):
        anchor = prices[: end - pd.DateOffset(months=months)].iloc[-1]
        assert np.isclose(value, prices.iloc[-1] / anchor - 1)


def test_daily_lookbacks_ending_on_the_last_business_day_start_at_month_end():
    # Fri 2024-06-28 is the last business day of June, so the three-month return
    # starts from the last March close (Fri 2024-03-29), not from Mar 28.
    prices = pd.Series(np.linspace(100.0, 130.0, 900), index=pd.bdate_range(end="2024-06-28", periods=900))
    stats = ffn.PerformanceStats(prices)

    assert np.isclose(stats.three_month, prices.iloc[-1] / prices["2024-03-29"] - 1)


@pytest.mark.parametrize("tz", [None, "America/New_York"])
@pytest.mark.parametrize(
    "end,anchors",
    [
        ("2024-06-28", ("2024-03-31", "2023-12-31", "2023-06-30")),
        ("2024-06-29", ("2024-03-29", "2023-12-29", "2023-06-29")),
        ("2024-06-30", ("2024-03-31", "2023-12-31", "2023-06-30")),
        ("2025-08-30", ("2025-05-30", "2025-02-28", "2024-08-30")),
    ],
)
def test_lookback_prices_distinguishes_weekends_from_month_ends(end, anchors, tz):
    # Friday's business month-end and Sunday's calendar month-end use month buckets;
    # the intervening Saturday keeps its day number, including February clamping.
    dates = pd.date_range("2023-01-01", end, freq="D", tz=tz, name="date")
    prices = pd.Series(100.0 + np.arange(len(dates)), index=dates, name="asset")
    original = prices.copy(deep=True)

    for months, anchor in zip((3, 6, 12), anchors):
        expected = prices.loc[: pd.Timestamp(anchor, tz=tz)]
        actual = ffn.core._lookback_prices(prices, months)

        pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(prices, original)


@pytest.mark.parametrize("grouped", [False, True])
def test_weekend_lookback_uses_last_selected_observation(grouped):
    prices = pd.Series(
        [100.0, 100.0, 130.0, 160.0, 200.0, 999.0],
        index=pd.to_datetime(["2023-01-01", "2024-03-29", "2024-03-30", "2024-03-31", "2024-06-29", "2024-06-30"]),
        name="asset",
    )
    original = prices.copy(deep=True)
    if grouped:
        stats = prices.to_frame().calc_stats()
        stats.set_date_range(end="2024-06-29")
        child = stats["asset"]
    else:
        unavailable_tail = prices.copy()
        unavailable_tail.iloc[-1] = np.nan
        child = unavailable_tail.calc_perf_stats()

    # June 29 uses March 29's 100, not March 31's 160. The later 999 is unavailable
    # or outside the selected report, so neither it nor its month-end date may leak in.
    expected = 200 / 100 - 1
    assert child.three_month == pytest.approx(expected)
    assert child.lookback_returns["3m"] == pytest.approx(expected)
    assert child.end == pd.Timestamp("2024-06-29")
    if grouped:
        assert stats.lookback_returns.loc["3m", "asset"] == pytest.approx(expected)
    pd.testing.assert_series_equal(prices, original)
