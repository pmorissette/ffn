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


@pytest.mark.benchmark(group="resampling")
@pytest.mark.parametrize("periods, assets, trials", [(252, 3, 1000), (2520, 20, 5000)])
def test_resample_returns(benchmark, periods, assets, trials):
    returns = pd.DataFrame(
        np.random.default_rng(42).normal(0.0002, 0.01, size=(periods, assets)),
        index=pd.bdate_range("2010-01-01", periods=periods),
        columns=[f"asset_{i}" for i in range(assets)],
    )

    result = benchmark(ffn.resample_returns, returns, pd.DataFrame.mean, seed=0, num_trials=trials)

    assert result.shape == (trials, assets)
    # Check a seeded sample independently of pandas' row selection and reduction.
    positions = np.random.RandomState(0).choice(periods, size=periods, replace=True)
    expected = returns.to_numpy()[positions].mean(axis=0)
    np.testing.assert_allclose(result.iloc[0].to_numpy(dtype=float), expected)
    pd.testing.assert_index_equal(result.columns, returns.columns)


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
@pytest.mark.parametrize("calculate", [ffn.calc_cagr, ffn.calc_total_return], ids=["cagr", "total-return"])
@pytest.mark.parametrize("as_series", [False, True], ids=["frame", "series"])
def test_return_helpers(benchmark, prices, calculate, as_series):
    data = prices.iloc[:, 0] if as_series else prices

    result = benchmark(calculate, data)

    assert np.isfinite(result).all()


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


@pytest.mark.benchmark(group="pbo")
@pytest.mark.parametrize("n_blocks", [8, 12, 16])
def test_calc_prob_backtest_overfitting(benchmark, n_blocks):
    returns = pd.DataFrame(np.random.default_rng(42).normal(0.0002, 0.01, (640, 40)))

    result = benchmark(ffn.calc_prob_backtest_overfitting, returns, n_blocks=n_blocks, full_output=True)

    assert 0.0 <= result["pbo"] <= 1.0
    assert result["logits"].notna().all()
    assert 0.0 < result["mean_oos_rank"] < 1.0


@pytest.mark.benchmark(group="weights")
@pytest.mark.parametrize("covar_method", ["ledoit-wolf", "standard"])
def test_calc_mean_var_weights(benchmark, covar_method):
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(1_000, 40)),
        columns=[f"asset_{index}" for index in range(40)],
    )

    result = benchmark(ffn.calc_mean_var_weights, returns, covar_method=covar_method)

    assert result.index.equals(returns.columns)
    assert result.sum() == pytest.approx(1.0)


@pytest.mark.benchmark(group="weights")
def test_calc_erc_weights(benchmark, returns):
    result = benchmark(ffn.calc_erc_weights, returns)

    assert result.index.equals(returns.columns)
    assert result.sum() == pytest.approx(1.0)
