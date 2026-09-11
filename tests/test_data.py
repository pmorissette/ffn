import numpy as np
import pandas as pd
import pytest

import ffn


def sample_provider(ticker, field):
    prices = {"ABC": [np.nan, 10.0, np.nan, 12.0], "DEF": [20.0, np.nan, 22.0, 23.0]}
    return pd.Series(prices[ticker], index=pd.date_range("2024-01-01", periods=4))


def test_get_isolates_cached_data_from_caller_mutation():
    """Keep caller assignments from changing data cached by ``ffn.get``."""
    expected = pd.DataFrame(
        {"abc": [10.0, 12.0]},
        index=pd.date_range("2024-01-02", periods=2, freq="2D"),
    )
    ffn.get.mcache.clear()
    try:
        first = ffn.get("ABC", provider=sample_provider)
        assert isinstance(first, pd.DataFrame)
        first.iloc[0, 0] = 999.0
        second = ffn.get("ABC", provider=sample_provider)

        assert isinstance(second, pd.DataFrame)
        assert second is not first
        pd.testing.assert_frame_equal(second, expected)
    finally:
        ffn.get.mcache.clear()


@pytest.mark.parametrize("forward_fill", [False, True])
@pytest.mark.parametrize("common_dates", [False, True])
def test_get_forward_fill(forward_fill, common_dates):
    result = ffn.get(
        "ABC,DEF",
        provider=sample_provider,
        common_dates=common_dates,
        forward_fill=forward_fill,
        mrefresh=True,
    )

    if common_dates:
        expected = pd.DataFrame({"abc": [12.0], "def": [23.0]}, index=pd.date_range("2024-01-04", periods=1))
    else:
        expected = pd.DataFrame(
            {"abc": [np.nan, 10.0, 10.0 if forward_fill else np.nan, 12.0], "def": [20.0, 20.0 if forward_fill else np.nan, 22.0, 23.0]},
            index=pd.date_range("2024-01-01", periods=4),
        )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)
