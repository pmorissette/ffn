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


@pytest.fixture(scope="module")
def returns(prices):
    return ffn.to_returns(prices).dropna()


@pytest.fixture(scope="module")
def drawdown(prices):
    return ffn.to_drawdown_series(prices.iloc[:, 0])


@pytest.mark.benchmark(group="returns")
def test_to_returns(benchmark, prices):
    result = benchmark(ffn.to_returns, prices)

    assert result.shape == prices.shape


@pytest.mark.benchmark(group="returns")
def test_to_log_returns(benchmark, prices):
    result = benchmark(ffn.to_log_returns, prices)

    assert result.shape == prices.shape


@pytest.mark.benchmark(group="resampling")
@pytest.mark.parametrize("minutes", [1, 5])
def test_asfreq_actual_intraday(benchmark, minutes):
    prices = pd.Series(np.arange(100_000, dtype=float), index=pd.date_range("2020-01-01", periods=100_000, freq="min"), name="asset")

    result = benchmark(ffn.asfreq_actual, prices, f"{minutes}min")

    pd.testing.assert_series_equal(result, prices.iloc[::minutes], check_freq=False)


@pytest.mark.benchmark(group="drawdown")
def test_to_drawdown_series(benchmark, prices):
    result = benchmark(ffn.to_drawdown_series, prices)

    assert result.shape == prices.shape
    assert result.max().max() <= 0.0


@pytest.mark.benchmark(group="drawdown")
def test_calc_max_drawdown(benchmark, prices):
    result = benchmark(ffn.calc_max_drawdown, prices)

    assert result.index.equals(prices.columns)
    assert (result <= 0.0).all()


@pytest.mark.benchmark(group="drawdown")
def test_drawdown_details(benchmark, drawdown):
    result = benchmark(ffn.drawdown_details, drawdown)

    assert list(result.columns) == ["Start", "End", "Length", "drawdown"]


@pytest.mark.benchmark(group="statistics")
def test_calc_perf_stats(benchmark, prices):
    result = benchmark(ffn.calc_perf_stats, prices.iloc[:, 0])

    assert result.name == prices.columns[0]


@pytest.mark.benchmark(group="statistics")
def test_calc_stats(benchmark, prices):
    result = benchmark(ffn.calc_stats, prices)

    assert result.stats.shape[1] == prices.shape[1]


@pytest.mark.benchmark(group="statistics")
def test_calc_information_ratio(benchmark, returns):
    result = benchmark(ffn.calc_information_ratio, returns, returns.iloc[:, 0])

    assert result.index.equals(returns.columns)
    assert result.iloc[0] == 0.0


@pytest.mark.benchmark(group="statistics")
def test_calc_sortino_ratio(benchmark, returns):
    result = benchmark(ffn.calc_sortino_ratio, returns.iloc[:, 0], nperiods=252)

    assert np.isfinite(result)


@pytest.mark.benchmark(group="statistics")
def test_calc_prob_mom(benchmark, returns):
    result = benchmark(ffn.calc_prob_mom, returns, returns.iloc[:, 0])

    assert result.index.equals(returns.columns)


@pytest.mark.benchmark(group="weights")
def test_calc_erc_weights(benchmark, returns):
    result = benchmark(ffn.calc_erc_weights, returns)

    assert result.index.equals(returns.columns)
    assert result.sum() == pytest.approx(1.0)
