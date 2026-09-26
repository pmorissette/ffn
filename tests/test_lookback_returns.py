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
