import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.mark.parametrize(
    "values, expected",
    [
        ([-0.1, -0.2, -0.05], [(0, 2, -0.2)]),
        ([-0.1, -0.2, 0.0], [(0, 2, -0.2)]),
        ([-0.1], [(0, 0, -0.1)]),
        ([-0.1, 0.0, -0.2, 0.0], [(0, 1, -0.1), (2, 3, -0.2)]),
        ([0.0, -0.1, 0.0], [(1, 2, -0.1)]),
    ],
)
def test_drawdown_details_includes_initial_drawdown(values, expected):
    index = pd.date_range("2024-01-01", periods=len(values))
    drawdown = pd.Series(values, index=index)
    original = drawdown.copy()

    result = ffn.drawdown_details(drawdown)

    assert result is not None
    assert len(result) == len(expected)
    for row, (start, end, minimum) in zip(result.itertuples(index=False), expected):
        assert row.Start == index[start]
        assert row.End == index[end]
        assert row.Length == end - start
        assert row.drawdown == minimum
    pd.testing.assert_series_equal(drawdown, original)


@pytest.mark.parametrize("values", [[], [0.0], [0.0, 0.0], [np.nan, 0.0]])
def test_drawdown_details_without_drawdowns(values):
    drawdown = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)), dtype=float)
    assert ffn.drawdown_details(drawdown) is None
