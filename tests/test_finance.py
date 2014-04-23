import ffn
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal as aae


df = pd.read_csv('tests/data/test_data.csv', index_col=0, parse_dates=True)
ts = df['AAPL'][0:10]


def test_to_returns_ts():
    data = ts
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual[0])
    aae(actual[1], -0.019, 3)
    aae(actual[9], -0.022, 3)


def test_to_returns_df():
    data = df
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.ix[0]))
    aae(actual['AAPL'][1], -0.019, 3)
    aae(actual['AAPL'][9], -0.022, 3)
    aae(actual['MSFT'][1], -0.011, 3)
    aae(actual['MSFT'][9], -0.014, 3)
    aae(actual['C'][1], -0.012, 3)
    aae(actual['C'][9], 0.004, 3)


def test_to_log_returns_ts():
    data = ts
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual[0])
    aae(actual[1], -0.019, 3)
    aae(actual[9], -0.022, 3)


def test_to_log_returns_df():
    data = df
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.ix[0]))
    aae(actual['AAPL'][1], -0.019, 3)
    aae(actual['AAPL'][9], -0.022, 3)
    aae(actual['MSFT'][1], -0.011, 3)
    aae(actual['MSFT'][9], -0.014, 3)
    aae(actual['C'][1], -0.012, 3)
    aae(actual['C'][9], 0.004, 3)


def test_to_price_index():
    data = df
    rets = data.to_returns()
    actual = rets.to_price_index()

    assert len(actual) == len(data)
    aae(actual['AAPL'][0], 100, 3)
    aae(actual['MSFT'][0], 100, 3)
    aae(actual['C'][0], 100, 3)
    aae(actual['AAPL'][9], 91.366, 3)
    aae(actual['MSFT'][9], 95.191, 3)
    aae(actual['C'][9], 101.199, 3)


def test_rebase():
    data = df
    actual = data.rebase()

    assert len(actual) == len(data)
    aae(actual['AAPL'][0], 100, 3)
    aae(actual['MSFT'][0], 100, 3)
    aae(actual['C'][0], 100, 3)
    aae(actual['AAPL'][9], 91.366, 3)
    aae(actual['MSFT'][9], 95.191, 3)
    aae(actual['C'][9], 101.199, 3)


def test_to_drawdown_series_ts():
    data = ts
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual[0], 0, 3)
    aae(actual[1], -0.019, 3)
    aae(actual[9], -0.086, 3)


def test_to_drawdown_series_df():
    data = df
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual['AAPL'][0], 0, 3)
    aae(actual['MSFT'][0], 0, 3)
    aae(actual['C'][0], 0, 3)

    aae(actual['AAPL'][1], -0.019, 3)
    aae(actual['MSFT'][1], -0.011, 3)
    aae(actual['C'][1], -0.012, 3)

    aae(actual['AAPL'][9], -0.086, 3)
    aae(actual['MSFT'][9], -0.048, 3)
    aae(actual['C'][9], -0.029, 3)


def test_max_drawdown_ts():
    data = ts
    actual = data.calc_max_drawdown()

    aae(actual, -0.086, 3)


def test_max_drawdown_df():
    data = df
    data = data[0:10]
    actual = data.calc_max_drawdown()

    aae(actual['AAPL'], -0.086, 3)
    aae(actual['MSFT'], -0.048, 3)
    aae(actual['C'], -0.033, 3)


def test_year_frac():
    actual = ffn.year_frac(pd.to_datetime('2004-03-10'),
                           pd.to_datetime('2004-03-29'))
    # not exactly the same as excel but close enough
    aae(actual, 0.0520, 4)


def test_cagr_ts():
    data = ts
    actual = data.calc_cagr()
    aae(actual, -0.921, 3)


def test_cagr_df():
    data = df
    actual = data.calc_cagr()
    aae(actual['AAPL'], 0.440, 3)
    aae(actual['MSFT'], 0.041, 3)
    aae(actual['C'], -0.205, 3)


def test_merge():
    a = pd.TimeSeries(index=pd.date_range('2010-01-01', periods=5),
                      data=100, name='a')
    b = pd.TimeSeries(index=pd.date_range('2010-01-02', periods=5),
                      data=200, name='b')
    actual = ffn.merge(a, b)

    assert 'a' in actual
    assert 'b' in actual
    assert len(actual) == 6
    assert len(actual.columns) == 2
    assert np.isnan(actual['a'][-1])
    assert np.isnan(actual['b'][0])
    assert actual['a'][0] == 100
    assert actual['a'][1] == 100
    assert actual['b'][-1] == 200
    assert actual['b'][1] == 200

    old = actual
    old.columns = ['c', 'd']

    actual = ffn.merge(old, a, b)

    assert 'a' in actual
    assert 'b' in actual
    assert 'c' in actual
    assert 'd' in actual
    assert len(actual) == 6
    assert len(actual.columns) == 4
    assert np.isnan(actual['a'][-1])
    assert np.isnan(actual['b'][0])
    assert actual['a'][0] == 100
    assert actual['a'][1] == 100
    assert actual['b'][-1] == 200
    assert actual['b'][1] == 200


def test_calc_inv_vol_weights():
    prc = df.ix[0:11]
    rets = prc.to_returns().dropna()
    actual = ffn.finance.calc_inv_vol_weights(rets)

    assert len(actual) == 3
    assert 'AAPL' in actual
    assert 'MSFT' in actual
    assert 'C' in actual

    aae(actual['AAPL'], 0.218, 3)
    aae(actual['MSFT'], 0.464, 3)
    aae(actual['C'], 0.318, 3)


def test_calc_mean_var_weights():
    prc = df.ix[0:11]
    rets = prc.to_returns().dropna()
    actual = ffn.finance.calc_mean_var_weights(rets)

    assert len(actual) == 3
    assert 'AAPL' in actual
    assert 'MSFT' in actual
    assert 'C' in actual

    aae(actual['AAPL'], 0.000, 3)
    aae(actual['MSFT'], 0.000, 3)
    aae(actual['C'], 1.000, 3)


def test_calc_total_return():
    prc = df.ix[0:11]
    actual = prc.calc_total_return()

    assert len(actual) == 3
    aae(actual['AAPL'], -0.079, 3)
    aae(actual['MSFT'], -0.038, 3)
    aae(actual['C'], 0.012, 3)


def test_get_num_days_required():
    actual = ffn.finance.get_num_days_required(pd.DateOffset(months=3),
                                               perc_required=1.)
    assert actual >= 60

    actual = ffn.finance.get_num_days_required(pd.DateOffset(months=3),
                                               perc_required=1.,
                                               period='m')
    assert actual >= 3


def test_as_percent():
    ser = pd.Series([0.01, -0.02, 0.0532])
    actual = ser.as_percent()

    assert actual[0] == 1.00
    assert actual[1] == -2.00
    assert actual[2] == 5.32


def test_asfreq_actual():
    a = pd.TimeSeries({pd.to_datetime('2010-02-27'): 100,
                       pd.to_datetime('2010-03-25'): 200})
    actual = a.asfreq_actual(freq='M', method='ffill')

    assert len(actual) == 1
    assert '2010-02-27' in actual


def test_to_monthly():
    a = pd.TimeSeries(range(100), index=pd.date_range('2010-01-01',
                                                      periods=100))
    # to test for actual dates
    a['2010-01-31'] = np.nan
    a = a.dropna()

    actual = a.to_monthly()

    assert len(actual) == 3
    assert '2010-01-30' in actual
    assert actual['2010-01-30'] == 29
