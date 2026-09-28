from functools import partial

import numpy as np
import pandas as pd
import pytest

import ffn


@pytest.fixture(params=[ffn.calc_sharpe, ffn.calc_sortino_ratio, ffn.calc_risk_return_ratio])
def metric(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["series", "frame"])
def returns(request):
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    data = pd.Series([0.01, -0.02, 0.03, -0.01], index=index, name="asset")
    return pd.concat([data, data * 2], axis=1) if request.param else data


@pytest.mark.parametrize("index_kind", ["irregular", "shuffled", "duplicates", "integer", "object_integer", "mixed_numeric", "short", "empty"])
def test_annualization_requires_known_frequency(metric, returns, index_kind):
    if index_kind == "shuffled":
        returns.index = pd.bdate_range("2024-01-02", periods=4).take([2, 0, 3, 1])
    elif index_kind == "duplicates":
        returns.index = returns.index.take([0, 1, 1, 3])
    elif index_kind == "integer":
        returns.index = pd.RangeIndex(4)
    elif index_kind == "object_integer":
        returns.index = pd.Index([0, 1, 2, 3], dtype=object)
    elif index_kind == "mixed_numeric":
        returns.index = pd.Index([0, 1.0, 2, 3], dtype=object)
    elif index_kind in ("short", "empty"):
        returns = returns.iloc[: 2 if index_kind == "short" else 0]

    with pytest.raises(ValueError, match="nperiods.*annualize=False"):
        metric(returns)


@pytest.mark.parametrize("nperiods", [1, 252])
def test_explicit_annualization_and_per_period_results(metric, returns, nperiods):
    denominator = np.sqrt((returns.clip(upper=0.0) ** 2).mean()) if metric is ffn.calc_sortino_ratio else returns.std(ddof=1)
    expected = returns.mean() / denominator

    assert np.allclose(metric(returns, annualize=False), expected)
    assert np.allclose(metric(returns, nperiods=nperiods), expected * np.sqrt(nperiods))
    assert np.allclose(metric(returns, nperiods=nperiods, annualize=False), expected)


@pytest.mark.parametrize("freq,periods", [(pd.offsets.BDay(), 252), (pd.offsets.Week(), 52), (pd.offsets.MonthEnd(), 12)])
def test_regular_frequency_still_annualizes(metric, returns, freq, periods):
    returns.index = pd.date_range("2024-01-01", periods=4, freq=freq)

    assert np.allclose(metric(returns), metric(returns, nperiods=periods))


@pytest.mark.parametrize("name", ["calc_sharpe", "calc_sharpe_ratio", "calc_sortino", "calc_sortino_ratio", "calc_risk_return", "calc_risk_return_ratio"])
def test_pandas_ratio_aliases_require_explicit_annualization(returns, name):
    metric = getattr(returns, name)

    with pytest.raises(ValueError, match="nperiods.*annualize=False"):
        metric()
    assert np.allclose(metric(nperiods=252), metric(annualize=False) * np.sqrt(252))


def test_bootstrap_ratios_keep_explicit_annualization(metric, returns):
    returns.index = pd.bdate_range("2024-01-01", periods=4)
    original = returns.copy()

    with pytest.raises(ValueError, match="nperiods.*annualize=False"):
        ffn.resample_returns(returns, metric, num_trials=1)
    per_period = ffn.resample_returns(returns, partial(metric, annualize=False), num_trials=5)
    annualized = ffn.resample_returns(returns, partial(metric, nperiods=252), num_trials=5)

    assert np.allclose(annualized.astype(float), per_period.astype(float) * np.sqrt(252), equal_nan=True)
    if isinstance(returns, pd.DataFrame):
        pd.testing.assert_frame_equal(returns, original)
    else:
        pd.testing.assert_series_equal(returns, original)


@pytest.mark.parametrize("metric", [ffn.calc_sharpe, ffn.calc_sortino_ratio])
def test_per_period_scalar_risk_free_rate_still_requires_frequency(metric, returns):
    with pytest.raises(ValueError, match="nperiods"):
        metric(returns, rf=0.05, annualize=False)

    excess = returns - ffn.deannualize(0.05, 252)
    assert np.allclose(metric(returns, rf=0.05, nperiods=252, annualize=False), metric(excess, annualize=False))


@pytest.mark.parametrize("metric", [ffn.calc_sharpe, ffn.calc_sortino_ratio])
def test_series_risk_free_rate_does_not_supply_frequency(metric, returns):
    risk_free = pd.Series(0.0001, index=returns.index)

    with pytest.raises(ValueError, match="nperiods.*annualize=False"):
        metric(returns, rf=risk_free)
    assert np.allclose(metric(returns, rf=risk_free, annualize=False), metric(returns - 0.0001, annualize=False))


@pytest.mark.parametrize("dtype", [int, float, object])
def test_numeric_labels_do_not_imply_a_return_frequency(dtype):
    returns = pd.Series([0.01, -0.02, 0.03, -0.01], index=pd.Index([0, 1, 2, 3], dtype=dtype))

    assert ffn.core.infer_freq(returns) is None
    assert ffn.core.infer_nperiods(returns) is None
