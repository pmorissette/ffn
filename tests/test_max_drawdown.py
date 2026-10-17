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


@pytest.mark.parametrize(
    "values,expected",
    [
        ([[100, 200], [120, 180], [90, 210], [135, 205]], [-0.25, -0.1]),
        ([[np.nan, 200], [100, 180], [120, 210], [90, 205]], [-0.25, -0.1]),
        ([[0, 0], [2, 0], [1, 0], [3, 0]], [-0.5, np.nan]),
        # With negative running peaks, lower prices give ratios above one.
        ([[-100, -200], [-120, -180], [-90, -210], [-135, -205]], [0, 0]),
        ([[np.nan, np.nan]] * 4, [np.nan, np.nan]),
        ([[100, 200]], [0, 0]),
    ],
)
def test_max_drawdown_column_reductions(values, expected):
    """Running-peak ratios define each asset independently, including missing and zero prices."""
    prices = pd.DataFrame(values, dtype="float64", columns=pd.Index(["a", "b"], name="assets"))
    original = prices.copy(deep=True)
    expected = pd.Series(expected, index=prices.columns, dtype="float64")

    for result in [ffn.calc_max_drawdown(prices), prices.calc_max_drawdown()]:
        pd.testing.assert_series_equal(result, expected, rtol=0, atol=1e-15)
    pd.testing.assert_frame_equal(prices, original)


@pytest.mark.parametrize(
    "columns",
    [
        pd.Index(["same", "same"], name="assets"),
        pd.CategoricalIndex(["a", "b"], categories=["b", "a", "unused"], ordered=True, name="assets"),
        pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["asset", "leg"]),
        pd.Index([["a"], ["b"]]),
        pd.Index([{"a": 1}, {"b": 2}]),
    ],
    ids=["duplicate", "categorical", "multiindex", "list", "dict"],
)
def test_max_drawdown_preserves_native_column_labels(columns):
    """Column-wise dispatch must retain native label acceptance, errors and reduction metadata."""
    prices = pd.DataFrame([[100.0, 200.0], [120.0, 180.0], [90.0, 210.0], [135.0, 205.0]], columns=columns)
    try:
        expected = (prices / prices.cummax()).min() - 1
    except TypeError as error:
        # Older pandas rejects list labels here; newer pandas accepts them.
        for calculate in [ffn.calc_max_drawdown, lambda frame: frame.calc_max_drawdown()]:
            with pytest.raises(TypeError) as actual:
                calculate(prices)
            assert str(actual.value) == str(error)
    else:
        for result in [ffn.calc_max_drawdown(prices), prices.calc_max_drawdown()]:
            pd.testing.assert_series_equal(result, expected, check_exact=True)
        np.testing.assert_allclose(expected.to_numpy(), [-0.25, -0.1], rtol=0, atol=1e-15)


@pytest.mark.parametrize("dated_columns,allows_duplicate_labels", [(False, True), (True, True), (True, False)], ids=["ordinary", "datetime", "unique-datetime"])
@pytest.mark.parametrize("mutate", ["result", "input"])
@pytest.mark.parametrize("surface", ["values", "axis", "attrs"])
def test_max_drawdown_preserves_native_metadata_ownership(dated_columns, allows_duplicate_labels, mutate, surface):
    """Preserve native ownership, which differs from equal-looking apply output across pandas versions."""

    def make_prices():
        columns = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC", name="assets") if dated_columns else pd.Index(["a", "b"], name="assets")
        frame = pd.DataFrame([[100.0, 200.0], [120.0, 180.0], [90.0, 210.0], [135.0, 205.0]], columns=columns)
        frame.attrs = {"nested": {"labels": ["input"]}}
        return frame.set_flags(allows_duplicate_labels=allows_duplicate_labels)

    native_prices, prices = make_prices(), make_prices()
    expected = (native_prices / native_prices.cummax()).min() - 1
    result = prices.calc_max_drawdown()
    pd.testing.assert_series_equal(result, expected, check_exact=True)
    assert result.attrs == expected.attrs

    # Mutate each surface on fresh objects: value assignment can detach a shared axis.
    for frame, reduced in [(native_prices, expected), (prices, result)]:
        target = reduced if mutate == "result" else frame
        if surface == "values":
            target.iloc[0] = -99.0
        elif surface == "axis":
            axis = reduced.index if mutate == "result" else frame.columns
            axis.name = "changed"
            if dated_columns:
                axis.freq = None
        elif target.attrs:
            target.attrs["nested"]["labels"].append("changed")

    pd.testing.assert_series_equal(result, expected, check_exact=True)
    pd.testing.assert_frame_equal(prices, native_prices, check_exact=True)
    assert result.attrs == expected.attrs
    assert prices.attrs == native_prices.attrs


def test_max_drawdown_preserves_subclass_dispatch():
    class PriceFrame(pd.DataFrame):
        def apply(self, *args, **kwargs):
            raise AssertionError("The native panel calculation does not call a subclass's apply override")

    prices = PriceFrame({"asset": [100.0, 120.0, 90.0, 135.0]})

    pd.testing.assert_series_equal(ffn.calc_max_drawdown(prices), pd.Series([-0.25], index=["asset"]))
