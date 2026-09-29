"""
Reference values for ffn's performance metrics.

Each metric is checked against the convention ffn implements, on five fixed
return series: the 24-month portfolio from Bacon (2008), a seven-point sample,
a Gaussian series, a fat-tailed Student t(3) series and a left-skewed one.
Where R PerformanceAnalytics implements the convention, the expected value is
its output (the call is recorded in tests/data/reference_values.json); the
rest were checked against numpy or a direct calendar-time definition. The
values come from vetted (https://github.com/WatchTree-19/vetted).

The calendar-time conventions (CAGR and Calmar) depend on the dates, so the
index is fixed here: monthly returns are dated at month ends from 2000-01-31,
other returns on business days from 2000-01-03, and the price path starts at
1.0 one period before the first return (1999-12-31 in both cases).

A failure here means an estimator changed. If the change was intended, the
convention recorded for that metric should change with it.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ffn

DATA = json.loads((Path(__file__).parent / "data" / "reference_values.json").read_text())


def _returns(name):
    fixture = DATA["fixtures"][name]
    returns = fixture["returns"]
    if fixture["periods"] == 12:
        index = pd.date_range("2000-01-31", periods=len(returns), freq=pd.offsets.MonthEnd())
    else:
        index = pd.bdate_range("2000-01-03", periods=len(returns))
    return pd.Series(returns, index=index, dtype=float), fixture["periods"]


def _prices(returns, periods):
    step = pd.offsets.MonthEnd() if periods == 12 else pd.offsets.BDay()
    start = pd.Series([1.0], index=[returns.index[0] - step])
    return pd.concat([start, (1 + returns).cumprod()])


METRICS = {
    "return_annual": lambda r, p: ffn.calc_cagr(_prices(r, p)),
    "sharpe_annual": lambda r, p: ffn.calc_sharpe(r, rf=0.0, nperiods=p, annualize=True),
    "sortino_annual": lambda r, p: ffn.calc_sortino_ratio(r, rf=0.0, nperiods=p, annualize=True),
    "max_drawdown": lambda r, p: ffn.calc_max_drawdown(_prices(r, p)),
    "calmar": lambda r, p: ffn.calc_calmar_ratio(_prices(r, p)),
}

CASES = [pytest.param(metric, fixture, expected, id=f"{metric}-{fixture}") for metric, spec in DATA["metrics"].items() for fixture, expected in spec["values"].items()]


def test_every_metric_has_a_check():
    assert set(METRICS) == set(DATA["metrics"])
    for metric, spec in DATA["metrics"].items():
        assert set(spec["values"]) == set(DATA["fixtures"]), metric


@pytest.mark.parametrize("metric, fixture, expected", CASES)
def test_reference_value(metric, fixture, expected):
    returns, periods = _returns(fixture)
    value = float(np.asarray(METRICS[metric](returns, periods)).item())
    convention = DATA["metrics"][metric]["convention"]
    assert value == pytest.approx(expected, rel=1e-9, abs=1e-12), f"{metric} on {fixture} no longer matches the {convention} convention"
