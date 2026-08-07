import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.fixture(
    scope="module",
    params=[(252, 3), (2520, 10)],
    ids=["1y-3-assets", "10y-10-assets"],
)
def prices(request):
    periods, assets = request.param
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, size=(periods, assets))
    return pd.DataFrame(
        100.0 * np.exp(returns.cumsum(axis=0)),
        index=pd.date_range("2010-01-01", periods=periods, freq="B"),
        columns=[f"asset_{index}" for index in range(assets)],
    )


@pytest.mark.benchmark(group="returns")
def test_to_returns(benchmark, prices):
    result = benchmark(ffn.to_returns, prices)

    assert result.shape == prices.shape


@pytest.mark.benchmark(group="drawdown")
def test_to_drawdown_series(benchmark, prices):
    result = benchmark(ffn.to_drawdown_series, prices)

    assert result.shape == prices.shape
    assert result.max().max() <= 0.0


@pytest.mark.benchmark(group="statistics")
def test_calc_stats(benchmark, prices):
    result = benchmark(ffn.calc_stats, prices)

    assert result.stats.shape[1] == prices.shape[1]
