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


@pytest.mark.parametrize("index_kind", ["datetime", "integer", "float32"])
def test_drawdown_details_preserves_timestamp_and_duration_types(index_kind):
    if index_kind == "datetime":
        index = pd.date_range("2024-03-09", periods=6, freq="17h", tz="America/New_York")
    elif index_kind == "integer":
        index = pd.Index([0, 2, 5, 9, 14, 20])
    else:
        index = pd.Index([0, 0.1, 0.3, 0.9, 1.4, 2.1], dtype="float32")
    drawdown = pd.Series([0.0, -0.1, -0.2, 0.0, -0.3, -0.1], index=index)
    original = drawdown.copy()
    index_type = pd.DatetimeIndex if index_kind == "datetime" else type(index)
    rows = []
    for start, end, minimum in [(1, 3, -0.2), (4, 5, -0.3)]:
        duration = index[end] - index[start]
        if index_kind == "datetime":
            duration = duration.days
        rows.append((index[start], index[end], duration, minimum))
    expected = pd.DataFrame(rows, columns=["Start", "End", "Length", "drawdown"], dtype=object)

    pd.testing.assert_frame_equal(ffn.drawdown_details(drawdown, index_type=index_type), expected, check_exact=True)
    pd.testing.assert_series_equal(drawdown, original)
