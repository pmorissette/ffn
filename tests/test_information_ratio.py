import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", object])
@pytest.mark.parametrize("duplicate_columns", [False, True])
def test_information_ratio_preserves_columnwise_results(dtype, duplicate_columns):
    index = pd.date_range("2024-01-01", periods=5)
    returns = pd.DataFrame(
        {
            "varying": [0.0, 0.2, -0.1, np.nan, 0.3],
            "constant": [0.1] * 5,
            "missing": [np.nan] * 5,
            "single": [np.nan, np.nan, 0.2, np.nan, np.nan],
            "zero": [0.0] * 5,
        },
        index=index,
        dtype=dtype,
    )
    if duplicate_columns:
        returns.columns = ["asset"] * len(returns.columns)
    returns.columns.name = "assets"
    original = returns.copy()
    benchmark = pd.Series(0.0, index=index, dtype=dtype)
    # pandas 1.x uses object-dtype reductions for nullable inputs.
    expected = pd.Series(
        [ffn.calc_information_ratio(returns.iloc[:, column], benchmark) for column in range(len(returns.columns))],
        index=returns.columns,
        dtype=object if returns.mean().dtype == object else float,
    )

    for actual in (ffn.calc_information_ratio(returns, benchmark), returns.calc_information_ratio(benchmark)):
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("dtype", ["float64", "Float64", object])
def test_information_ratio_preserves_empty_frames(shape, dtype):
    returns = pd.DataFrame(index=range(shape[0]), columns=range(shape[1]), dtype=dtype)

    result = ffn.calc_information_ratio(returns, returns)

    expected = pd.Series(0.0, index=returns.columns, dtype=object if returns.mean().dtype == object else float)
    pd.testing.assert_series_equal(result, expected)
