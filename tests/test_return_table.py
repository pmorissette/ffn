import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_return_table_preserves_partial_years_and_missing_months(dtype, timezone):
    prices = pd.Series(
        [100, 125, 250, 125, 80, 100, 200, 250],
        index=pd.to_datetime(["2022-11-15", "2022-11-30", "2022-12-31", "2023-01-31", "2023-03-31", "2023-04-30", "2025-01-31", "2025-02-28"]).tz_localize(timezone),
        dtype=dtype,
        name="asset",
    )
    original = prices.copy()

    stats = ffn.PerformanceStats(prices)

    expected = pd.DataFrame(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 1, 1.5],
            [-0.5, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, -0.375],
            [0] * 13,
            [0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25],
        ],
        index=[2022, 2023, 2024, 2025],
        columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "YTD"],
        dtype=float,
    )
    pd.testing.assert_frame_equal(stats.return_table, expected)
    assert stats.pos_month_perc == 3 / 27
    assert stats.avg_up_month == 0.5
    assert stats.avg_down_month == -0.5
    pd.testing.assert_series_equal(prices, original)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_return_table_preserves_complete_year_precision(dtype):
    index = pd.date_range("2023-01-31", periods=12, freq=ffn.core._MonthEnd).insert(0, pd.Timestamp("2023-01-01"))
    prices = pd.Series(100 * 1.25 ** np.arange(13), index=index, dtype=dtype, name="asset")

    stats = ffn.PerformanceStats(prices)

    monthly_prices = prices.iloc[1:].to_numpy(dtype=dtype.lower())
    monthly_returns = monthly_prices / prices.iloc[:-1].to_numpy(dtype=dtype.lower()) - 1
    monthly_returns = np.array([float(prices.iloc[1]) / prices.iloc[0] - 1, *monthly_returns[1:]])
    expected_values = np.append(monthly_returns, np.prod(1 + monthly_returns) - 1)
    expected = pd.DataFrame([expected_values], index=[2023], columns=stats.return_table.columns)
    pd.testing.assert_frame_equal(stats.return_table, expected, check_exact=True)
