import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.fixture(params=[False, True], ids=["package", "pandas"])
def pain_index(request):
    if request.param:
        return lambda prices: prices.to_pain_index()
    return ffn.to_pain_index


@pytest.fixture(params=[False, True], ids=["package", "pandas"])
def pain_ratio(request):
    if request.param:
        return lambda prices, **kwargs: prices.to_pain_ratio(**kwargs)
    return ffn.to_pain_ratio


def test_pain_index_is_mean_absolute_drawdown_in_percentage_points(pain_index):
    # Drawdowns from the running peak of 100 are 0%, 20%, 10%, 0% and 5%,
    # so the pain index is 35 / 5 = 7 percentage points.
    prices = pd.Series([100.0, 80.0, 90.0, 100.0, 95.0])

    assert pain_index(prices) == pytest.approx(7.0)


def test_pain_index_without_drawdown_is_zero(pain_index):
    assert pain_index(pd.Series([100.0, 101.0, 102.0])) == pytest.approx(0.0)


def test_pain_index_is_not_larger_than_ulcer_index(pain_index):
    # Same path: the ulcer index is sqrt((400 + 100 + 25) / 5) = sqrt(105).
    prices = pd.Series([100.0, 80.0, 90.0, 100.0, 95.0])
    assert prices.to_ulcer_index() == pytest.approx(np.sqrt(105.0))

    rng = np.random.default_rng(7)
    walk = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(500, 4)), axis=0)))
    for column in walk:
        assert pain_index(walk[column]) <= walk[column].to_ulcer_index()


def test_pain_index_carries_prices_across_a_gap(pain_index):
    # The gap is filled with 90, so the drawdowns are 0%, 10%, 10% and 0%.
    gapped = pd.Series([100.0, 90.0, np.nan, 100.0])

    assert pain_index(gapped) == pytest.approx(5.0)
    assert pain_index(gapped) == pytest.approx(pain_index(gapped.ffill()))


def test_pain_index_ignores_a_leading_gap(pain_index):
    prices = pd.Series([np.nan, np.nan, 100.0, 90.0, 100.0])

    assert pain_index(prices) == pytest.approx(10.0 / 3.0)


def test_pain_index_handles_an_ndarray_like_a_series():
    values = [100.0, 80.0, np.nan, 100.0, 95.0]

    assert ffn.core.to_pain_index(np.array(values)) == pytest.approx(pd.Series(values).to_pain_index())


def test_pain_index_is_per_column_for_a_dataframe(pain_index):
    prices = pd.DataFrame({"a": [100.0, 80.0, 90.0, 100.0, 95.0], "b": [100.0, 90.0, 100.0, 100.0, 100.0]})

    result = pain_index(prices)

    pd.testing.assert_series_equal(result, pd.Series({"a": 7.0, "b": 2.0}))


def test_pain_ratio_divides_annual_return_by_pain_index(pain_ratio):
    # Two periods per year makes the two-step holding period exactly one year:
    # an annual return of 10%, minus 5% risk free, over a pain index of 10 / 3.
    prices = pd.Series([100.0, 90.0, 110.0])

    assert pain_ratio(prices, rf=0.05, nperiods=2) == pytest.approx(1.5)


def test_pain_ratio_is_negative_for_a_losing_year(pain_ratio):
    # One year, -5% total return, pain index of 7: -5 / 7.
    prices = pd.Series([100.0, 80.0, 90.0, 100.0, 95.0])

    assert pain_ratio(prices, nperiods=4) == pytest.approx(-5.0 / 7.0)


@pytest.mark.parametrize("risk_free", [0.0, 0.05, np.float32(0.1)])
def test_pain_ratio_uses_elapsed_calendar_time(pain_ratio, risk_free):
    dates = pd.to_datetime(["2025-01-01", "2025-07-02", "2026-01-01"])
    prices = pd.Series([100.0, 90.0, 110.0], index=dates)
    expected = (1.1 ** (365.25 / 365) - 1 - float(risk_free)) * 100 / (10.0 / 3.0)

    assert pain_ratio(prices, rf=risk_free) == pytest.approx(expected)


def test_pain_ratio_annualizes_risk_free_series_separately(pain_ratio):
    dates = pd.to_datetime(["2025-01-01", "2025-07-02", "2026-01-01"])
    prices = pd.Series([100.0, 90.0, 110.0], index=dates)
    # The return at the first price precedes the measured holding period.
    risk_free = pd.Series([0.5, 0.02, 0.03], index=dates)
    expected = (1.1 ** (365.25 / 365) - (1.02 * 1.03) ** (365.25 / 365)) * 100 / (10.0 / 3.0)

    assert pain_ratio(prices, rf=risk_free) == pytest.approx(expected)


def test_pain_ratio_shares_its_numerator_with_the_ulcer_performance_index(pain_ratio):
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=750, freq="B")
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=(750, 3)), axis=0)), index=dates)

    lhs = pain_ratio(prices, rf=0.02) * prices.to_pain_index()
    rhs = prices.to_ulcer_performance_index(rf=0.02) * prices.to_ulcer_index()

    pd.testing.assert_series_equal(lhs, rhs)


@pytest.mark.parametrize("values", [[], [100.0], [np.nan], [np.nan, np.nan]])
def test_pain_ratio_insufficient_prices_remain_unavailable(pain_ratio, values):
    prices = pd.Series(values, index=pd.date_range("2025", periods=len(values)), dtype=float)

    assert pd.isna(pain_ratio(prices))


def test_pain_ratio_undated_prices_require_periods(pain_ratio):
    with pytest.raises(ValueError, match="nperiods"):
        pain_ratio(pd.Series([100.0, 90.0, 110.0]))
