import pandas as pd
import pytest

import ffn


@pytest.mark.parametrize("index", [pd.RangeIndex(0), pd.DatetimeIndex([], name="date")])
@pytest.mark.parametrize("as_frame", [False, True])
def test_to_price_index_empty(index, as_frame):
    returns = pd.Series([], index=index, dtype=float, name="strategy")
    if as_frame:
        returns = returns.to_frame()
    result = ffn.to_price_index(returns, start=50)
    if as_frame:
        pd.testing.assert_frame_equal(result, returns)
    else:
        pd.testing.assert_series_equal(result, returns)
    assert result is not returns


@pytest.mark.parametrize("index", [pd.RangeIndex(0), pd.DatetimeIndex([], name="date")])
@pytest.mark.parametrize("as_frame", [False, True])
def test_rebase_empty(index, as_frame):
    prices = pd.Series([], index=index, dtype=float, name="strategy")
    if as_frame:
        prices = prices.to_frame()
    result = ffn.rebase(prices, value=50)
    if as_frame:
        pd.testing.assert_frame_equal(result, prices)
    else:
        pd.testing.assert_series_equal(result, prices)
    assert result is not prices
