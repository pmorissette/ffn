import ffn
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal


df = pd.read_csv('tests/data/test_data.csv', index_col=0, parse_dates=True)
ts = df['AAPL'][0:10]


def test_to_returns_ts():
    data = ts
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual[0])
    assert_almost_equal(actual[1], -0.019, 3)
    assert_almost_equal(actual[9], -0.022, 3)


def test_to_returns_df():
    data = df
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.ix[0]))
    assert_almost_equal(actual['AAPL'][1], -0.019, 3)
    assert_almost_equal(actual['AAPL'][9], -0.022, 3)
    assert_almost_equal(actual['MSFT'][1], -0.011, 3)
    assert_almost_equal(actual['MSFT'][9], -0.014, 3)
    assert_almost_equal(actual['C'][1], -0.012, 3)
    assert_almost_equal(actual['C'][9], 0.004, 3)


def test_to_log_returns_ts():
    data = ts
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual[0])
    assert_almost_equal(actual[1], -0.019, 3)
    assert_almost_equal(actual[9], -0.022, 3)


def test_to_log_returns_df():
    data = df
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.ix[0]))
    assert_almost_equal(actual['AAPL'][1], -0.019, 3)
    assert_almost_equal(actual['AAPL'][9], -0.022, 3)
    assert_almost_equal(actual['MSFT'][1], -0.011, 3)
    assert_almost_equal(actual['MSFT'][9], -0.014, 3)
    assert_almost_equal(actual['C'][1], -0.012, 3)
    assert_almost_equal(actual['C'][9], 0.004, 3)


def test_to_price_index():
    data = df
    rets = data.to_returns()
    actual = rets.to_price_index()

    assert len(actual) == len(data)
    assert_almost_equal(actual['AAPL'][0], 100, 3)
    assert_almost_equal(actual['MSFT'][0], 100, 3)
    assert_almost_equal(actual['C'][0], 100, 3)
    assert_almost_equal(actual['AAPL'][9], 91.366, 3)
    assert_almost_equal(actual['MSFT'][9], 95.191, 3)
    assert_almost_equal(actual['C'][9], 101.199, 3)


def test_rebase():
    data = df
    actual = data.rebase()

    assert len(actual) == len(data)
    assert_almost_equal(actual['AAPL'][0], 100, 3)
    assert_almost_equal(actual['MSFT'][0], 100, 3)
    assert_almost_equal(actual['C'][0], 100, 3)
    assert_almost_equal(actual['AAPL'][9], 91.366, 3)
    assert_almost_equal(actual['MSFT'][9], 95.191, 3)
    assert_almost_equal(actual['C'][9], 101.199, 3)


def test_to_drawdown_series_ts():
    data = ts
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    assert_almost_equal(actual[0], 0, 3)
    assert_almost_equal(actual[1], -0.019, 3)
    assert_almost_equal(actual[9], -0.086, 3)


def test_to_drawdown_series_df():
    data = df
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    assert_almost_equal(actual['AAPL'][0], 0, 3)
    assert_almost_equal(actual['MSFT'][0], 0, 3)
    assert_almost_equal(actual['C'][0], 0, 3)

    assert_almost_equal(actual['AAPL'][1], -0.019, 3)
    assert_almost_equal(actual['MSFT'][1], -0.011, 3)
    assert_almost_equal(actual['C'][1], -0.012, 3)

    assert_almost_equal(actual['AAPL'][9], -0.086, 3)
    assert_almost_equal(actual['MSFT'][9], -0.048, 3)
    assert_almost_equal(actual['C'][9], -0.029, 3)


def test_max_drawdown_ts():
    data = ts
    actual = data.calc_max_drawdown()

    assert_almost_equal(actual, -0.086, 3)


def test_max_drawdown_df():
    data = df
    data = data[0:10]
    actual = data.calc_max_drawdown()

    assert_almost_equal(actual['AAPL'], -0.086, 3)
    assert_almost_equal(actual['MSFT'], -0.048, 3)
    assert_almost_equal(actual['C'], -0.033, 3)


def test_year_frac():
    actual = ffn.year_frac(pd.to_datetime('2004-03-10'),
                           pd.to_datetime('2004-03-29'))
    # not exactly the same as excel but close enough
    assert_almost_equal(actual, 0.0520, 4)


def test_cagr_ts():
    data = ts
    actual = data.calc_cagr()
    assert_almost_equal(actual, -0.921, 3)


def test_cagr_df():
    data = df
    actual = data.calc_cagr()
    assert_almost_equal(actual['AAPL'], 0.440, 3)
    assert_almost_equal(actual['MSFT'], 0.041, 3)
    assert_almost_equal(actual['C'], -0.205, 3)
