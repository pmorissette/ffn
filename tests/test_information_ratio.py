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


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", object])
def test_information_ratio_treats_rounding_residue_as_no_tracking_error(dtype):
    # (r + c) - r is c plus rounding residue that scales with r. Dividing by that
    # residue gave ratios near 1e15; it is the zero-tracking-error case.
    index = pd.date_range("2024-01-01", periods=250, freq="B")
    r = pd.Series(np.random.default_rng(0).normal(0.0, 0.02, 250), index=index, dtype=dtype)

    assert ffn.calc_information_ratio(r + 0.0005, r) == 0.0
    assert ffn.calc_information_ratio(pd.Series(0.001, index=index), pd.Series(0.0, index=index)) == 0.0
    assert ffn.calc_prob_mom(r + 0.0005, r) == 0.5

    frame = pd.DataFrame({"residue": r + 0.0005, "varying": r * 2})
    result = ffn.calc_information_ratio(frame, r)
    assert result["residue"] == 0.0
    assert np.isclose(result["varying"], r.mean() / r.std(ddof=1))


def test_information_ratio_keeps_a_small_real_tracking_error():
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01], index=index)
    diff = pd.Series([1e-9, -1e-9, 2e-9, 0.0, 1e-9, 3e-9], index=index)

    expected = diff.mean() / diff.std(ddof=1)

    assert np.isclose(ffn.calc_information_ratio(benchmark + diff, benchmark), expected, rtol=1e-4)


@pytest.mark.parametrize("metric", ["calc_information_ratio", "calc_prob_mom"])
@pytest.mark.parametrize("shape", ["frame_series", "series_frame", "frame_frame"])
@pytest.mark.parametrize("duplicate_columns", [False, True])
def test_tracking_error_tolerance_is_columnwise(metric, shape, duplicate_columns):
    quiet = pd.Series([1e-17, 2e-17, 4e-17, 3e-17])
    frame = pd.DataFrame({"quiet": quiet, "large": quiet * 1e16})
    if duplicate_columns:
        frame.columns = ["asset", "asset"]
    frame.columns.name = "assets"
    benchmark = pd.Series(0.0, index=frame.index)
    function = getattr(ffn, metric)
    expected = pd.Series([function(frame.iloc[:, i], benchmark) for i in range(2)], index=frame.columns)
    if shape == "series_frame":
        result = function(benchmark, -frame)
    else:
        if shape == "frame_frame":
            benchmark = frame * 0
        result = function(frame, benchmark)
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "metric,dtype",
    [("calc_information_ratio", "float64"), ("calc_information_ratio", "Float64"), ("calc_information_ratio", object), ("calc_prob_mom", "float64"), ("calc_prob_mom", "Float64")],
)
@pytest.mark.parametrize("missing", ["unaligned", "nan"])
@pytest.mark.parametrize("reverse", [False, True])
def test_tracking_error_tolerance_ignores_unpaired_observations(metric, dtype, missing, reverse):
    returns = pd.Series([1e-17, 2e-17, 4e-17, 3e-17], dtype=dtype)
    benchmark = pd.Series(0.0, index=returns.index, dtype=dtype)
    if reverse:
        returns, benchmark = benchmark, returns
    function = getattr(ffn, metric)
    expected = function(returns, benchmark)
    returns.loc[4] = 1.0
    if missing == "nan":
        benchmark.loc[4] = np.nan

    assert np.isclose(function(returns, benchmark), expected)
    frame = pd.DataFrame({"asset": returns})
    assert np.isclose(function(frame, benchmark).iloc[0], expected)


@pytest.mark.parametrize("metric", ["calc_information_ratio", "calc_prob_mom"])
def test_tracking_error_tolerance_preserves_each_columns_precision(metric):
    benchmark = pd.Series(np.random.default_rng(0).normal(0.0, 0.02, 250))
    quiet = pd.Series(np.resize([1e-10, 2e-10, 4e-10, 3e-10], len(benchmark)))
    returns = pd.DataFrame({"single": (benchmark + 0.0005).astype("float32"), "double": benchmark + quiet})
    benchmarks = pd.DataFrame({"single": benchmark.astype("float32"), "double": benchmark})
    function = getattr(ffn, metric)
    expected = pd.Series(
        [0.0 if metric == "calc_information_ratio" else 0.5, function(returns["double"], benchmarks["double"])],
        index=returns.columns,
    )

    pd.testing.assert_series_equal(function(returns, benchmarks), expected)
