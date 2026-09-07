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


def test_drawdown_details_with_nullable_missing_initial_value():
    drawdown = pd.Series([pd.NA, 0.0], dtype="Float64")
    assert ffn.drawdown_details(drawdown, index_type=drawdown.index) is None


@pytest.mark.parametrize("dtype", [float, "Float64"])
def test_drawdown_details_aggregates_episode_minima_with_gaps(dtype):
    drawdown = pd.Series(
        [0.0, -0.1, np.nan, -0.3, 0.0, 0.0, -0.2, -0.1, 0.0],
        dtype=dtype,
    )

    result = ffn.drawdown_details(drawdown, index_type=drawdown.index)

    expected = pd.DataFrame(
        [(1, 4, 3, -0.3), (6, 8, 2, -0.2)],
        columns=("Start", "End", "Length", "drawdown"),
        dtype=object,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_drawdown_details_does_not_aggregate_past_episode_end():
    drawdown = pd.Series(
        [0.0, -0.1, 0.0, pd.NA, -0.2, 0.0],
        dtype="Float64",
    )

    result = ffn.drawdown_details(drawdown, index_type=drawdown.index)

    expected = pd.DataFrame(
        [(1, 2, 1, -0.1)],
        columns=("Start", "End", "Length", "drawdown"),
        dtype=object,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_drawdown_details_preserves_empty_reversed_episode():
    drawdown = pd.Series(
        [-0.2, 0.0, pd.NA, -0.2, 0.0, -0.01],
        dtype="Float64",
    )

    result = ffn.drawdown_details(drawdown, index_type=drawdown.index)

    assert result.iloc[1].Start == 5
    assert result.iloc[1].End == 4
    assert pd.isna(result.iloc[1].drawdown)


@pytest.mark.parametrize(
    "values, expected_minimum",
    [
        ([-0.1], -0.1),
        ([-0.1, -0.2], -0.2),
    ],
)
def test_drawdown_details_with_initial_drawdown_and_range_index(values, expected_minimum):
    drawdown = pd.Series(values)

    result = ffn.drawdown_details(drawdown, index_type=drawdown.index)

    assert result is not None
    assert len(result) == 1
    assert result.iloc[0].drawdown == expected_minimum
