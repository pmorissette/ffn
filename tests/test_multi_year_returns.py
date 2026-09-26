import numpy as np
import pandas as pd
import pytest

import ffn


def _daily(start, end):
    index = pd.bdate_range(start, end)
    return pd.Series(np.linspace(100.0, 150.0, len(index)), index=index)


def test_three_year_needs_three_years_of_history():
    # 13 months that touch three calendar years used to report a "3Y (ann.)"
    # figure, which was just the since-inception CAGR.
    prices = pd.Series(
        [100.0, 110.0, 121.0, 133.1],
        index=pd.to_datetime(["2021-12-31", "2022-06-30", "2022-12-30", "2023-01-31"]),
    )
    stats = ffn.PerformanceStats(prices)

    assert np.isnan(stats.three_year)
    assert not np.isnan(stats.incep)


@pytest.mark.parametrize(
    "start,attr",
    [
        ("2021-12-20", "three_year"),
        ("2019-12-20", "five_year"),
        ("2015-12-21", "ten_year"),
    ],
)
def test_multi_year_returns_are_nan_with_less_history(start, attr):
    # 2.2, 4.2 and 8.2 years of prices that touch 3, 5 and 10 calendar years
    stats = ffn.PerformanceStats(_daily(start, "2024-03-01"))

    assert np.isnan(getattr(stats, attr))


@pytest.mark.parametrize("years,attr", [(3, "three_year"), (5, "five_year"), (10, "ten_year")])
def test_multi_year_returns_with_enough_history(years, attr):
    prices = _daily("2013-01-02", "2024-03-01")
    start = prices.index[-1] - pd.DateOffset(years=years)
    stats = ffn.PerformanceStats(prices)

    assert np.isclose(getattr(stats, attr), ffn.calc_cagr(prices[start:]))
