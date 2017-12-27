import ffn
import pandas as pd
import numpy as np
import os.path
from numpy.testing import assert_almost_equal as aae

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
data_path = os.path.join(data_dir, 'test_data.csv')
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
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

    actual = rets.to_price_index(start=1)

    assert len(actual) == len(data)
    aae(actual['AAPL'][0], 1, 3)
    aae(actual['MSFT'][0], 1, 3)
    aae(actual['C'][0], 1, 3)
    aae(actual['AAPL'][9], 0.914, 3)
    aae(actual['MSFT'][9], 0.952, 3)
    aae(actual['C'][9], 1.012, 3)


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
    a = pd.Series(index=pd.date_range('2010-01-01', periods=5),
                  data=100, name='a')
    b = pd.Series(index=pd.date_range('2010-01-02', periods=5),
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
    actual = ffn.core.calc_inv_vol_weights(rets)

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
    actual = ffn.core.calc_mean_var_weights(rets)

    assert len(actual) == 3
    assert 'AAPL' in actual
    assert 'MSFT' in actual
    assert 'C' in actual

    aae(actual['AAPL'], 0.000, 3)
    aae(actual['MSFT'], 0.000, 3)
    aae(actual['C'], 1.000, 3)


def test_calc_erc_weights():
    prc = df.ix[0:11]
    rets = prc.to_returns().dropna()
    actual = ffn.core.calc_erc_weights(rets)

    assert len(actual) == 3
    assert 'AAPL' in actual
    assert 'MSFT' in actual
    assert 'C' in actual

    aae(actual['AAPL'], 0.270, 3)
    aae(actual['MSFT'], 0.374, 3)
    aae(actual['C'], 0.356, 3)


def test_calc_total_return():
    prc = df.ix[0:11]
    actual = prc.calc_total_return()

    assert len(actual) == 3
    aae(actual['AAPL'], -0.079, 3)
    aae(actual['MSFT'], -0.038, 3)
    aae(actual['C'], 0.012, 3)


def test_get_num_days_required():
    actual = ffn.core.get_num_days_required(pd.DateOffset(months=3),
                                            perc_required=1.)
    assert actual >= 60

    actual = ffn.core.get_num_days_required(pd.DateOffset(months=3),
                                            perc_required=1.,
                                            period='m')
    assert actual >= 3


def test_asfreq_actual():
    a = pd.Series({pd.to_datetime('2010-02-27'): 100,
                   pd.to_datetime('2010-03-25'): 200})
    actual = a.asfreq_actual(freq='M', method='ffill')

    assert len(actual) == 1
    assert '2010-02-27' in actual


def test_to_monthly():
    a = pd.Series(range(100), index=pd.date_range(
        '2010-01-01', periods=100))
    # to test for actual dates
    a['2010-01-31'] = np.nan
    a = a.dropna()

    actual = a.to_monthly()

    assert len(actual) == 3
    assert '2010-01-30' in actual
    assert actual['2010-01-30'] == 29


def test_drop_duplicate_cols():
    a = pd.Series(index=pd.date_range('2010-01-01', periods=5),
                  data=100, name='a')
    # second version of a w/ less data
    a2 = pd.Series(index=pd.date_range('2010-01-02', periods=4),
                   data=900, name='a')
    b = pd.Series(index=pd.date_range('2010-01-02', periods=5),
                  data=200, name='b')
    actual = ffn.merge(a, a2, b)

    assert actual['a'].shape[1] == 2
    assert len(actual.columns) == 3

    actual = actual.drop_duplicate_cols()

    assert len(actual.columns) == 2
    assert 'a' in actual
    assert 'b' in actual
    assert len(actual['a'].dropna()) == 5


def test_limit_weights():
    w = {'a': 0.3, 'b': 0.1,
         'c': 0.05, 'd': 0.05, 'e': 0.5}

    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0

    assert actual['a'] == 0.3
    assert actual['b'] == 0.2
    assert actual['c'] == 0.1
    assert actual['d'] == 0.1
    assert actual['e'] == 0.3

    w = pd.Series({'a': 0.3, 'b': 0.1,
                   'c': 0.05, 'd': 0.05, 'e': 0.5})

    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0

    assert actual['a'] == 0.3
    assert actual['b'] == 0.2
    assert actual['c'] == 0.1
    assert actual['d'] == 0.1
    assert actual['e'] == 0.3

    w = pd.Series({'a': 0.29, 'b': 0.1,
                   'c': 0.06, 'd': 0.05, 'e': 0.5})

    assert w.sum() == 1.0

    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0

    assert np.all([x <= 0.3 for x in actual])

    aae(actual['a'], 0.300, 3)
    aae(actual['b'], 0.190, 3)
    aae(actual['c'], 0.114, 3)
    aae(actual['d'], 0.095, 3)
    aae(actual['e'], 0.300, 3)


def test_random_weights():
    n = 10
    bounds = (0., 1.)
    tot = 1.0000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.ix[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert df.applymap(lambda x: (x >= low and x <= high)).all().all()

    n = 4
    bounds = (0., 0.25)
    tot = 1.0000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.ix[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert df.applymap(
        lambda x: (np.round(x, 2) >= low and
                   np.round(x, 2) <= high)).all().all()

    n = 7
    bounds = (0., 0.25)
    tot = 0.8000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.ix[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert df.applymap(
        lambda x: (np.round(x, 2) >= low and
                   np.round(x, 2) <= high)).all().all()

    n = 10
    bounds = (-.25, 0.25)
    tot = 0.0
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.ix[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert df.applymap(
        lambda x: (np.round(x, 2) >= low and
                   np.round(x, 2) <= high)).all().all()


def test_random_weights_throws_error():
    try:
        ffn.random_weights(2, (0., 0.25), 1.0)
        assert False
    except ValueError:
        assert True

    try:
        ffn.random_weights(10, (0.5, 0.25), 1.0)
        assert False
    except ValueError:
        assert True

    try:
        ffn.random_weights(10, (0.5, 0.75), 0.2)
        assert False
    except ValueError:
        assert True


def test_rollapply():
    a = pd.Series([1, 2, 3, 4, 5])

    actual = a.rollapply(3, np.mean)

    assert np.isnan(actual[0])
    assert np.isnan(actual[1])
    assert actual[2] == 2
    assert actual[3] == 3
    assert actual[4] == 4

    b = pd.DataFrame({'a': a, 'b': a})

    actual = b.rollapply(3, np.mean)

    assert all(np.isnan(actual.iloc[0]))
    assert all(np.isnan(actual.iloc[1]))
    assert all(actual.iloc[2] == 2)
    assert all(actual.iloc[3] == 3)
    assert all(actual.iloc[4] == 4)


def test_winsorize():
    x = pd.Series(range(20), dtype='float')
    res = x.winsorize(limits=0.05)
    assert res.iloc[0] == 1
    assert res.iloc[-1] == 18

    # make sure initial values still intact
    assert x.iloc[0] == 0
    assert x.iloc[-1] == 19

    x = pd.DataFrame({
        'a': pd.Series(range(20), dtype='float'),
        'b': pd.Series(range(20), dtype='float')
    })
    res = x.winsorize(axis=0, limits=0.05)

    assert res['a'].iloc[0] == 1
    assert res['b'].iloc[0] == 1
    assert res['a'].iloc[-1] == 18
    assert res['b'].iloc[-1] == 18

    assert x['a'].iloc[0] == 0
    assert x['b'].iloc[0] == 0
    assert x['a'].iloc[-1] == 19
    assert x['b'].iloc[-1] == 19


def test_rescale():
    x = pd.Series(range(10), dtype='float')
    res = x.rescale()

    assert res.iloc[0] == 0
    assert res.iloc[4] == (4. - 0.) / (9. - 0.)
    assert res.iloc[-1] == 1

    assert x.iloc[0] == 0
    assert x.iloc[4] == 4
    assert x.iloc[-1] == 9

    x = pd.DataFrame({
        'a': pd.Series(range(10), dtype='float'),
        'b': pd.Series(range(10), dtype='float')
    })
    res = x.rescale(axis=0)

    assert res['a'].iloc[0] == 0
    assert res['a'].iloc[4] == (4. - 0.) / (9. - 0.)
    assert res['a'].iloc[-1] == 1
    assert res['b'].iloc[0] == 0
    assert res['b'].iloc[4] == (4. - 0.) / (9. - 0.)
    assert res['b'].iloc[-1] == 1

    assert x['a'].iloc[0] == 0
    assert x['a'].iloc[4] == 4
    assert x['a'].iloc[-1] == 9
    assert x['b'].iloc[0] == 0
    assert x['b'].iloc[4] == 4
    assert x['b'].iloc[-1] == 9


def test_annualize():
    assert ffn.annualize(0.1, 60) == (1.1 ** (1. / (60. / 365)) - 1)


def test_calc_sortino_ratio():
    rf = 0
    p = 1
    r = df.to_returns()
    a = r.calc_sortino_ratio(rf=rf, nperiods=p)
    assert np.allclose(a, (r.mean() - rf) / r[r < rf].std() * np.sqrt(p))

    a = r.calc_sortino_ratio()
    assert np.allclose(a, (r.mean() - rf) / r[r < rf].std() * np.sqrt(p))

    rf = 0.02
    p = 252
    r = df.to_returns()
    er = r.to_excess_returns(rf, nperiods=p)

    a = r.calc_sortino_ratio(rf=rf, nperiods=p)
    assert np.allclose(a, er.mean() / er[er < 0].std() * np.sqrt(p))


def test_calmar_ratio():
    cagr = df.calc_cagr()
    mdd = df.calc_max_drawdown()

    a = df.calc_calmar_ratio()
    assert np.allclose(a, cagr / abs(mdd))


def test_calc_stats():
    # test twelve_month_win_perc divide by zero
    prices = df.C['2010-10-01':'2011-08-01']
    stats = ffn.calc_stats(prices).stats
    assert 'twelve_month_win_perc' not in stats.index
    prices = df.C['2009-10-01':'2011-08-01']
    stats = ffn.calc_stats(prices).stats
    assert 'twelve_month_win_perc' in stats.index

    # test yearly_sharpe divide by zero
    prices = df.C['2009-01-01':'2012-01-01']
    stats = ffn.calc_stats(prices).stats
    assert 'yearly_sharpe' in stats.index
    prices[prices > 0.0] = 1.0
    stats = ffn.calc_stats(prices).stats
    assert 'yearly_sharpe' not in stats.index


def test_calc_sharpe():
    x = pd.Series()
    assert np.isnan(x.calc_sharpe())

    r = df.to_returns()

    res = r.calc_sharpe()
    assert np.allclose(res, r.mean() / r.std())

    res = r.calc_sharpe(rf=0.05, nperiods=252)
    drf = ffn.deannualize(0.05, 252)
    ar = r - drf
    assert np.allclose(res, ar.mean() / ar.std() * np.sqrt(252))


def test_deannualize():
    res = ffn.deannualize(0.05, 252)
    assert np.allclose(res, np.power(1.05, 1 / 252.) - 1)


def test_to_excess_returns():
    rf = 0.05
    r = df.to_returns()

    np.allclose(r.to_excess_returns(0), r)

    np.allclose(r.to_excess_returns(rf, nperiods=252),
                r.to_excess_returns(ffn.deannualize(rf, 252)))

    np.allclose(r.to_excess_returns(rf), r - rf)
