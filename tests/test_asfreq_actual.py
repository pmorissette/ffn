import pandas as pd
import pytest

import ffn


@pytest.mark.parametrize("freq", ["D", ffn.core._MonthEnd])
@pytest.mark.parametrize("periods", [0, 2, 8])
@pytest.mark.parametrize("timezone", [None, "UTC", "America/New_York"])
@pytest.mark.parametrize("as_frame", [False, True])
def test_asfreq_actual_preserves_regular_frequency(freq, periods, timezone, as_frame):
    index = pd.date_range("2020-01-31", periods=periods, freq=freq, tz=timezone, name="date")
    prices = pd.Series(range(periods), index=index, dtype="Float64", name=0)
    if as_frame:
        prices = pd.concat([prices, prices.rename("dt")], axis=1)
        prices.columns.name = "assets"
    original = prices.copy()
    assert_equal = pd.testing.assert_frame_equal if as_frame else pd.testing.assert_series_equal

    results = [ffn.asfreq_actual(prices, freq), prices.asfreq_actual(freq)]
    if freq == ffn.core._MonthEnd:
        results.append(prices.to_monthly())

    for result in results:
        assert_equal(result, original, check_exact=True)
        if periods == 2:
            assert_equal(result.shift(freq="infer"), original.shift(freq=freq), check_exact=True)
    assert_equal(prices, original, check_exact=True)
