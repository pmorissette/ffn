import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.fixture(params=[False, True], ids=["package", "pandas"])
def calculate(request):
    if request.param:
        return lambda prices, **kwargs: prices.to_ulcer_performance_index(**kwargs)
    return ffn.to_ulcer_performance_index


@pytest.mark.parametrize("risk_free", [0.05, 0.1, np.float32(0.1)])
def test_subtracts_annual_risk_free_return(calculate, risk_free):
    dates = pd.to_datetime(["2025-01-01", "2025-07-02", "2026-01-01"])
    prices = pd.Series([100.0, 90.0, 110.0], index=dates)
    expected = (1.1 ** (365.25 / 365) - 1 - float(risk_free)) * 100 / np.sqrt(100 / 3)

    assert calculate(prices, rf=risk_free, nperiods=2) == pytest.approx(expected)
    assert calculate(prices, rf=risk_free) == pytest.approx(expected)


def test_annualizes_risk_free_series_separately(calculate):
    dates = pd.to_datetime(["2025-01-01", "2025-07-02", "2026-01-01"])
    prices = pd.Series([100.0, 90.0, 110.0], index=dates)
    # The return at the first price precedes the measured holding period.
    risk_free = pd.Series([0.5, 0.02, 0.03], index=dates, dtype="Float64")
    original = risk_free.copy()
    expected = (1.1 ** (365.25 / 365) - (1.02 * 1.03) ** (365.25 / 365)) * 100 / np.sqrt(100 / 3)

    assert calculate(prices, rf=risk_free) == pytest.approx(expected)
    pd.testing.assert_series_equal(risk_free, original)


@pytest.mark.parametrize("missing", ["all", "partial"])
def test_missing_risk_free_returns_remain_unavailable(calculate, missing):
    dates = pd.to_datetime(["2025-01-01", "2025-07-02", "2026-01-01"])
    prices = pd.Series([100.0, 90.0, 110.0], index=dates)
    risk_free = pd.Series([np.nan, 0.02, np.nan], index=dates)
    if missing == "all":
        risk_free.index = risk_free.index - pd.DateOffset(years=1)

    assert pd.isna(calculate(prices, rf=risk_free))


@pytest.mark.parametrize("values", [[], [100.0], [np.nan], [np.nan, np.nan]])
@pytest.mark.parametrize("as_frame", [False, True])
def test_insufficient_prices_remain_unavailable(calculate, values, as_frame):
    prices = pd.Series(values, index=pd.date_range("2025", periods=len(values)), dtype=float, name="asset")
    if as_frame:
        prices = prices.to_frame()

    result = calculate(prices)

    if as_frame:
        pd.testing.assert_series_equal(result, pd.Series([np.nan], index=prices.columns))
    else:
        assert pd.isna(result)


@pytest.mark.parametrize("periods", [2, 12])
def test_explicit_periods_support_undated_prices(calculate, periods):
    prices = pd.Series([100.0, 90.0, 110.0])
    expected = (1.1 ** (periods / 2) - 1 - 0.05) * 100 / np.sqrt(100 / 3)

    assert calculate(prices, rf=0.05, nperiods=periods) == pytest.approx(expected)


def test_undated_prices_require_periods(calculate):
    with pytest.raises(ValueError, match="nperiods"):
        calculate(pd.Series([100.0, 90.0, 110.0]))


def test_zero_price_does_not_corrupt_endpoint_growth(calculate):
    prices = pd.Series([100.0, 0.0, 110.0], index=pd.to_datetime(["2025-01-01", "2025-07-02", "2026-01-01"]))
    expected = (1.1 ** (365.25 / 365) - 1) * 100 / np.sqrt(10000 / 3)

    assert calculate(prices) == pytest.approx(expected)


def test_ragged_columns_use_their_own_risk_free_window(calculate):
    dates = pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"])
    prices = pd.DataFrame([[100.0, np.nan], [90.0, 100.0], [105.0, 90.0], [110.0, 120.0]], index=dates, columns=["asset", "asset"])
    risk_free = pd.Series([np.nan, np.nan, 0.02, 0.03], index=dates)
    original = prices.copy()
    years = (dates[-1] - dates[1]).total_seconds() / 31557600
    expected = pd.Series([np.nan, (1.2 ** (1 / years) - (1.02 * 1.03) ** (1 / years)) * 100 / np.sqrt(100 / 3)], index=prices.columns)

    pd.testing.assert_series_equal(calculate(prices, rf=risk_free), expected)
    pd.testing.assert_frame_equal(prices, original)
