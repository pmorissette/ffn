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
