import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", object])
@pytest.mark.parametrize("values", [[np.nan, 100, 91, np.nan, 113, 84], [100, np.inf, 90, -np.inf, 80, 95], [np.nan] * 6, []])
@pytest.mark.parametrize("as_frame", [False, True])
def test_max_drawdown_matches_expanding_maximum(dtype, values, as_frame):
    prices = pd.Series(values, dtype=dtype, name="asset")
    if as_frame:
        prices = pd.concat([prices, prices], axis=1)
        prices.columns.name = "assets"
    original = prices.copy()
    expected = (prices / prices.expanding(min_periods=1).max()).min() - 1

    for actual in [ffn.calc_max_drawdown(prices), prices.calc_max_drawdown()]:
        if as_frame:
            pd.testing.assert_series_equal(actual, expected, check_exact=True)
        elif pd.isna(expected):
            assert pd.isna(actual)
        else:
            assert actual == expected
    if as_frame:
        pd.testing.assert_frame_equal(prices, original)
    else:
        pd.testing.assert_series_equal(prices, original)


def test_max_drawdown_preserves_large_integer_and_mixed_columns():
    prices = pd.DataFrame({"integer": [2**60 + 1, 2**60 + 7, 2**60 - 127], "float32": pd.Series([1, 1.3, 0.9], dtype="float32")})
    expected = (prices / prices.expanding(min_periods=1).max()).min() - 1

    pd.testing.assert_series_equal(ffn.calc_max_drawdown(prices), expected, check_exact=True)
