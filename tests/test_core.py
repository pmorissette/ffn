import ffn
import pandas as pd
import numpy as np
from pytest import fixture
from numpy.testing import assert_almost_equal as aae


@fixture
def df():
    try:
        df = pd.read_csv("tests/data/test_data.csv", index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        try:
            df = pd.read_csv("data/test_data.csv", index_col=0, parse_dates=True)
        except FileNotFoundError as e2:
            raise (str(e2))
    return df

@fixture
def ts(df):
    return df["AAPL"][0:10]


def test_mtd_ytd(df):
    data = df["AAPL"]

    # Intramonth
    prices = data[pd.to_datetime("2004-12-10") : pd.to_datetime("2004-12-25")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample("M").last().dropna()
    yp = prices.resample("A").last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, -0.0175, 4)
    assert mtd_actual == ytd_actual

    # Year change - first month
    prices = data[pd.to_datetime("2004-12-10") : pd.to_datetime("2005-01-15")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample("M").last().dropna()
    yp = prices.resample("A").last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, 0.0901, 4)
    assert mtd_actual == ytd_actual

    # Year change - second month
    prices = data[pd.to_datetime("2004-12-10") : pd.to_datetime("2005-02-15")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample("M").last().dropna()
    yp = prices.resample("A").last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, 0.1497, 4)
    aae(ytd_actual, 0.3728, 4)

    # Single day
    prices = data[[pd.to_datetime("2004-12-10")]]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample("M").last().dropna()
    yp = prices.resample("A").last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    assert mtd_actual == ytd_actual == 0


def test_to_returns_ts(ts):
    data = ts
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual[0])
    aae(actual[1], -0.019, 3)
    aae(actual[9], -0.022, 3)


def test_to_returns_df(df):
    data = df
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.iloc[0]))
    aae(actual["AAPL"][1], -0.019, 3)
    aae(actual["AAPL"][9], -0.022, 3)
    aae(actual["MSFT"][1], -0.011, 3)
    aae(actual["MSFT"][9], -0.014, 3)
    aae(actual["C"][1], -0.012, 3)
    aae(actual["C"][9], 0.004, 3)


def test_to_log_returns_ts(ts):
    data = ts
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual[0])
    aae(actual[1], -0.019, 3)
    aae(actual[9], -0.022, 3)


def test_to_log_returns_df(df):
    data = df
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.iloc[0]))
    aae(actual["AAPL"][1], -0.019, 3)
    aae(actual["AAPL"][9], -0.022, 3)
    aae(actual["MSFT"][1], -0.011, 3)
    aae(actual["MSFT"][9], -0.014, 3)
    aae(actual["C"][1], -0.012, 3)
    aae(actual["C"][9], 0.004, 3)


def test_to_price_index(df):
    data = df
    rets = data.to_returns()
    actual = rets.to_price_index()

    assert len(actual) == len(data)
    aae(actual["AAPL"][0], 100, 3)
    aae(actual["MSFT"][0], 100, 3)
    aae(actual["C"][0], 100, 3)
    aae(actual["AAPL"][9], 91.366, 3)
    aae(actual["MSFT"][9], 95.191, 3)
    aae(actual["C"][9], 101.199, 3)

    actual = rets.to_price_index(start=1)

    assert len(actual) == len(data)
    aae(actual["AAPL"][0], 1, 3)
    aae(actual["MSFT"][0], 1, 3)
    aae(actual["C"][0], 1, 3)
    aae(actual["AAPL"][9], 0.914, 3)
    aae(actual["MSFT"][9], 0.952, 3)
    aae(actual["C"][9], 1.012, 3)


def test_rebase(df):
    data = df
    actual = data.rebase()

    assert len(actual) == len(data)
    aae(actual["AAPL"][0], 100, 3)
    aae(actual["MSFT"][0], 100, 3)
    aae(actual["C"][0], 100, 3)
    aae(actual["AAPL"][9], 91.366, 3)
    aae(actual["MSFT"][9], 95.191, 3)
    aae(actual["C"][9], 101.199, 3)


def test_to_drawdown_series_ts(ts):
    data = ts
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual[0], 0, 3)
    aae(actual[1], -0.019, 3)
    aae(actual[9], -0.086, 3)


def test_to_drawdown_series_df(df):
    data = df
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual["AAPL"][0], 0, 3)
    aae(actual["MSFT"][0], 0, 3)
    aae(actual["C"][0], 0, 3)

    aae(actual["AAPL"][1], -0.019, 3)
    aae(actual["MSFT"][1], -0.011, 3)
    aae(actual["C"][1], -0.012, 3)

    aae(actual["AAPL"][9], -0.086, 3)
    aae(actual["MSFT"][9], -0.048, 3)
    aae(actual["C"][9], -0.029, 3)


def test_max_drawdown_ts(ts):
    data = ts
    actual = data.calc_max_drawdown()

    aae(actual, -0.086, 3)


def test_max_drawdown_df(df):
    data = df
    data = data[0:10]
    actual = data.calc_max_drawdown()

    aae(actual["AAPL"], -0.086, 3)
    aae(actual["MSFT"], -0.048, 3)
    aae(actual["C"], -0.033, 3)


def test_year_frac():
    actual = ffn.year_frac(pd.to_datetime("2004-03-10"), pd.to_datetime("2004-03-29"))
    # not exactly the same as excel but close enough
    aae(actual, 0.0520, 4)


def test_cagr_ts(ts):
    data = ts
    actual = data.calc_cagr()
    aae(actual, -0.921, 3)


def test_cagr_df(df):
    data = df
    actual = data.calc_cagr()
    aae(actual["AAPL"], 0.440, 3)
    aae(actual["MSFT"], 0.041, 3)
    aae(actual["C"], -0.205, 3)


def test_merge():
    a = pd.Series(index=pd.date_range("2010-01-01", periods=5), data=100, name="a")
    b = pd.Series(index=pd.date_range("2010-01-02", periods=5), data=200, name="b")
    actual = ffn.merge(a, b)

    assert "a" in actual
    assert "b" in actual
    assert len(actual) == 6
    assert len(actual.columns) == 2
    assert np.isnan(actual["a"][-1])
    assert np.isnan(actual["b"][0])
    assert actual["a"][0] == 100
    assert actual["a"][1] == 100
    assert actual["b"][-1] == 200
    assert actual["b"][1] == 200

    old = actual
    old.columns = ["c", "d"]

    actual = ffn.merge(old, a, b)

    assert "a" in actual
    assert "b" in actual
    assert "c" in actual
    assert "d" in actual
    assert len(actual) == 6
    assert len(actual.columns) == 4
    assert np.isnan(actual["a"][-1])
    assert np.isnan(actual["b"][0])
    assert actual["a"][0] == 100
    assert actual["a"][1] == 100
    assert actual["b"][-1] == 200
    assert actual["b"][1] == 200


def test_calc_inv_vol_weights(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna()
    actual = ffn.core.calc_inv_vol_weights(rets)

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.218, 3)
    aae(actual["MSFT"], 0.464, 3)
    aae(actual["C"], 0.318, 3)


def test_calc_mean_var_weights(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna()
    actual = ffn.core.calc_mean_var_weights(rets)

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.000, 3)
    aae(actual["MSFT"], 0.000, 3)
    aae(actual["C"], 1.000, 3)


def test_calc_erc_weights(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna()

    actual = ffn.core.calc_erc_weights(rets)

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.270, 3)
    aae(actual["MSFT"], 0.374, 3)
    aae(actual["C"], 0.356, 3)

    actual = ffn.core.calc_erc_weights(
        rets, covar_method="ledoit-wolf", risk_parity_method="slsqp", tolerance=1e-9
    )

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.270, 3)
    aae(actual["MSFT"], 0.374, 3)
    aae(actual["C"], 0.356, 3)

    actual = ffn.core.calc_erc_weights(
        rets, covar_method="standard", risk_parity_method="ccd", tolerance=1e-9
    )

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.234, 3)
    aae(actual["MSFT"], 0.409, 3)
    aae(actual["C"], 0.356, 3)

    actual = ffn.core.calc_erc_weights(
        rets, covar_method="standard", risk_parity_method="slsqp", tolerance=1e-9
    )

    assert len(actual) == 3
    assert "AAPL" in actual
    assert "MSFT" in actual
    assert "C" in actual

    aae(actual["AAPL"], 0.234, 3)
    aae(actual["MSFT"], 0.409, 3)
    aae(actual["C"], 0.356, 3)


def test_calc_total_return(df):
    prc = df.iloc[0:11]
    actual = prc.calc_total_return()

    assert len(actual) == 3
    aae(actual["AAPL"], -0.079, 3)
    aae(actual["MSFT"], -0.038, 3)
    aae(actual["C"], 0.012, 3)


def test_get_num_days_required():
    actual = ffn.core.get_num_days_required(pd.DateOffset(months=3), perc_required=1.0)
    assert actual >= 60

    actual = ffn.core.get_num_days_required(
        pd.DateOffset(months=3), perc_required=1.0, period="m"
    )
    assert actual >= 3


def test_asfreq_actual():
    a = pd.Series(
        {pd.to_datetime("2010-02-27"): 100, pd.to_datetime("2010-03-25"): 200}
    )
    actual = a.asfreq_actual(freq="M", method="ffill")

    assert len(actual) == 1
    assert "2010-02-27" in actual


def test_to_monthly():
    a = pd.Series(range(100), index=pd.date_range("2010-01-01", periods=100))
    # to test for actual dates
    a["2010-01-31"] = np.nan
    a = a.dropna()

    actual = a.to_monthly()

    assert len(actual) == 3
    assert "2010-01-30" in actual
    assert actual["2010-01-30"] == 29


def test_drop_duplicate_cols():
    a = pd.Series(index=pd.date_range("2010-01-01", periods=5), data=100, name="a")
    # second version of a w/ less data
    a2 = pd.Series(index=pd.date_range("2010-01-02", periods=4), data=900, name="a")
    b = pd.Series(index=pd.date_range("2010-01-02", periods=5), data=200, name="b")
    actual = ffn.merge(a, a2, b)

    assert actual["a"].shape[1] == 2
    assert len(actual.columns) == 3

    actual = actual.drop_duplicate_cols()

    assert len(actual.columns) == 2
    assert "a" in actual
    assert "b" in actual
    assert len(actual["a"].dropna()) == 5


def test_limit_weights():
    w = {"a": 0.3, "b": 0.1, "c": 0.05, "d": 0.05, "e": 0.5}
    actual_exp = {"a": 0.3, "b": 0.2, "c": 0.1, "d": 0.1, "e": 0.3}
    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0
    for k in actual_exp:
        assert actual[k] == actual_exp[k]

    w = pd.Series(w)
    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0
    for k in actual_exp:
        assert actual[k] == actual_exp[k]

    w = pd.Series({"a": 0.29, "b": 0.1, "c": 0.06, "d": 0.05, "e": 0.5})

    assert w.sum() == 1.0

    actual = ffn.core.limit_weights(w, 0.3)

    assert actual.sum() == 1.0

    assert all(x <= 0.3 for x in actual)

    aae(actual["a"], 0.300, 3)
    aae(actual["b"], 0.190, 3)
    aae(actual["c"], 0.114, 3)
    aae(actual["d"], 0.095, 3)
    aae(actual["e"], 0.300, 3)


def test_random_weights():
    n = 10
    bounds = (0.0, 1.0)
    tot = 1.0000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert df.applymap(lambda x: (x >= low and x <= high)).all().all()

    n = 4
    bounds = (0.0, 0.25)
    tot = 1.0000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (
        df.applymap(lambda x: (np.round(x, 2) >= low and np.round(x, 2) <= high))
        .all()
        .all()
    )

    n = 7
    bounds = (0.0, 0.25)
    tot = 0.8000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (
        df.applymap(lambda x: (np.round(x, 2) >= low and np.round(x, 2) <= high))
        .all()
        .all()
    )

    n = 10
    bounds = (-0.25, 0.25)
    tot = 0.0
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (
        df.applymap(lambda x: (np.round(x, 2) >= low and np.round(x, 2) <= high))
        .all()
        .all()
    )


def test_random_weights_throws_error():
    try:
        ffn.random_weights(2, (0.0, 0.25), 1.0)
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

    b = pd.DataFrame({"a": a, "b": a})

    actual = b.rollapply(3, np.mean)

    assert all(np.isnan(actual.iloc[0]))
    assert all(np.isnan(actual.iloc[1]))
    assert all(actual.iloc[2] == 2)
    assert all(actual.iloc[3] == 3)
    assert all(actual.iloc[4] == 4)


def test_winsorize():
    x = pd.Series(range(20), dtype="float")
    res = x.winsorize(limits=0.05)
    assert res.iloc[0] == 1
    assert res.iloc[-1] == 18

    # make sure initial values still intact
    assert x.iloc[0] == 0
    assert x.iloc[-1] == 19

    x = pd.DataFrame(
        {
            "a": pd.Series(range(20), dtype="float"),
            "b": pd.Series(range(20), dtype="float"),
        }
    )
    res = x.winsorize(axis=0, limits=0.05)

    assert res["a"].iloc[0] == 1
    assert res["b"].iloc[0] == 1
    assert res["a"].iloc[-1] == 18
    assert res["b"].iloc[-1] == 18

    assert x["a"].iloc[0] == 0
    assert x["b"].iloc[0] == 0
    assert x["a"].iloc[-1] == 19
    assert x["b"].iloc[-1] == 19


def test_rescale():
    x = pd.Series(range(10), dtype="float")
    res = x.rescale()

    assert res.iloc[0] == 0
    assert res.iloc[4] == (4.0 - 0.0) / (9.0 - 0.0)
    assert res.iloc[-1] == 1

    assert x.iloc[0] == 0
    assert x.iloc[4] == 4
    assert x.iloc[-1] == 9

    x = pd.DataFrame(
        {
            "a": pd.Series(range(10), dtype="float"),
            "b": pd.Series(range(10), dtype="float"),
        }
    )
    res = x.rescale(axis=0)

    assert res["a"].iloc[0] == 0
    assert res["a"].iloc[4] == (4.0 - 0.0) / (9.0 - 0.0)
    assert res["a"].iloc[-1] == 1
    assert res["b"].iloc[0] == 0
    assert res["b"].iloc[4] == (4.0 - 0.0) / (9.0 - 0.0)
    assert res["b"].iloc[-1] == 1

    assert x["a"].iloc[0] == 0
    assert x["a"].iloc[4] == 4
    assert x["a"].iloc[-1] == 9
    assert x["b"].iloc[0] == 0
    assert x["b"].iloc[4] == 4
    assert x["b"].iloc[-1] == 9


def test_annualize():
    assert ffn.annualize(0.1, 60) == (1.1 ** (1.0 / (60.0 / 365)) - 1)


def test_calc_sortino_ratio(df):
    rf = 0
    p = 1
    r = df.to_returns()
    a = r.calc_sortino_ratio(rf=rf, nperiods=p)
    er = r.to_excess_returns(rf, p)
    negative_returns = np.minimum(er[1:], 0)
    assert np.allclose(
        a, np.divide((er.mean() - rf), np.std(negative_returns, ddof=1)) * np.sqrt(p)
    )


def test_calmar_ratio(df):
    cagr = df.calc_cagr()
    mdd = df.calc_max_drawdown()

    a = df.calc_calmar_ratio()
    assert np.allclose(a, cagr / abs(mdd))


def test_calc_stats(df):
    # test twelve_month_win_perc divide by zero
    prices = df.C["2010-10-01":"2011-08-01"]
    stats = ffn.calc_stats(prices).stats
    assert pd.isnull(stats["twelve_month_win_perc"])
    prices = df.C["2009-10-01":"2011-08-01"]
    stats = ffn.calc_stats(prices).stats
    assert not pd.isnull(stats["twelve_month_win_perc"])

    # test yearly_sharpe divide by zero
    prices = df.C["2009-01-01":"2012-01-01"]
    stats = ffn.calc_stats(prices).stats
    assert "yearly_sharpe" in stats.index

    prices[prices > 0.0] = 1.0
    # throws warnings
    stats = ffn.calc_stats(prices).stats
    assert pd.isnull(stats["yearly_sharpe"])


def test_calc_sharpe(df):
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
    assert np.allclose(res, np.power(1.05, 1 / 252.0) - 1)


def test_to_excess_returns(df):
    rf = 0.05
    r = df.to_returns()

    np.allclose(r.to_excess_returns(0), r)

    np.allclose(
        r.to_excess_returns(rf, nperiods=252),
        r.to_excess_returns(ffn.deannualize(rf, 252)),
    )

    np.allclose(r.to_excess_returns(rf), r - rf)


def test_set_riskfree_rate(df):
    r = df.to_returns()

    performanceStats = ffn.PerformanceStats(df["MSFT"])
    groupStats = ffn.GroupStats(df)
    daily_returns = df["MSFT"].resample("D").last().dropna().pct_change()

    aae(
        performanceStats.daily_sharpe,
        daily_returns.dropna().mean() / (daily_returns.dropna().std()) * (np.sqrt(252)),
        3,
    )

    aae(performanceStats.daily_sharpe, groupStats["MSFT"].daily_sharpe, 3)

    monthly_returns = df["MSFT"].resample("M").last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        monthly_returns.dropna().mean()
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample("A").last().pct_change()
    aae(
        performanceStats.yearly_sharpe,
        yearly_returns.dropna().mean() / (yearly_returns.dropna().std()) * (np.sqrt(1)),
        3,
    )
    aae(performanceStats.yearly_sharpe, groupStats["MSFT"].yearly_sharpe, 3)

    performanceStats.set_riskfree_rate(0.02)
    groupStats.set_riskfree_rate(0.02)

    daily_returns = df["MSFT"].pct_change()
    aae(
        performanceStats.daily_sharpe,
        np.mean(daily_returns.dropna() - 0.02 / 252)
        / (daily_returns.dropna().std())
        * (np.sqrt(252)),
        3,
    )
    aae(performanceStats.daily_sharpe, groupStats["MSFT"].daily_sharpe, 3)

    monthly_returns = df["MSFT"].resample("M").last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        np.mean(monthly_returns.dropna() - 0.02 / 12)
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample("A").last().pct_change()
    aae(
        performanceStats.yearly_sharpe,
        np.mean(yearly_returns.dropna() - 0.02 / 1)
        / (yearly_returns.dropna().std())
        * (np.sqrt(1)),
        3,
    )
    aae(performanceStats.yearly_sharpe, groupStats["MSFT"].yearly_sharpe, 3)

    rf = np.zeros(df.shape[0])
    # annual rf is 2%
    rf[1:] = 0.02 / 252
    rf[0] = 0.0
    # convert to price series
    rf = 100 * np.cumprod(1 + pd.Series(data=rf, index=df.index, name="rf"))

    performanceStats.set_riskfree_rate(rf)
    groupStats.set_riskfree_rate(rf)

    daily_returns = df["MSFT"].pct_change()
    rf_daily_returns = rf.pct_change()
    aae(
        performanceStats.daily_sharpe,
        np.mean(daily_returns - rf_daily_returns)
        / (daily_returns.dropna().std())
        * (np.sqrt(252)),
        3,
    )
    aae(performanceStats.daily_sharpe, groupStats["MSFT"].daily_sharpe, 3)

    monthly_returns = df["MSFT"].resample("M").last().pct_change()
    rf_monthly_returns = rf.resample("M").last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        np.mean(monthly_returns - rf_monthly_returns)
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample("A").last().pct_change()
    rf_yearly_returns = rf.resample("A").last().pct_change()
    aae(
        performanceStats.yearly_sharpe,
        np.mean(yearly_returns - rf_yearly_returns)
        / (yearly_returns.dropna().std())
        * (np.sqrt(1)),
        3,
    )
    aae(performanceStats.yearly_sharpe, groupStats["MSFT"].yearly_sharpe, 3)


def test_performance_stats(df):
    ps = ffn.PerformanceStats(df["AAPL"])

    num_stats = len(ps.stats.keys())
    num_unique_stats = len(ps.stats.keys().drop_duplicates())
    assert num_stats == num_unique_stats


def test_group_stats_calc_stats(df):
    gs = df.calc_stats()

    num_stats = len(gs.stats.index)
    num_unique_stats = len(gs.stats.index.drop_duplicates())
    assert num_stats == num_unique_stats


def test_resample_returns(df):
    num_years = 30
    num_months = num_years * 12
    np.random.seed(0)
    returns = np.random.normal(loc=0.06 / 12, scale=0.20 / np.sqrt(12), size=num_months)
    returns = pd.Series(returns)

    sample_mean = np.mean(returns)

    sample_stats = ffn.resample_returns(returns, np.mean, seed=0, num_trials=100)

    resampled_mean = np.mean(sample_stats)
    std_resampled_means = np.std(sample_stats, ddof=1)

    # resampled statistics should be within 3 std devs of actual
    assert np.abs((sample_mean - resampled_mean) / std_resampled_means) < 3

    np.random.seed(0)
    returns = np.random.normal(
        loc=0.06 / 12, scale=0.20 / np.sqrt(12), size=num_months * 3
    ).reshape(num_months, 3)
    returns = pd.DataFrame(returns)

    sample_mean = np.mean(returns, axis=0)

    sample_stats = ffn.resample_returns(
        returns, lambda x: np.mean(x, axis=0), seed=0, num_trials=100
    )

    resampled_mean = np.mean(sample_stats)
    std_resampled_means = np.std(sample_stats, ddof=1)

    # resampled statistics should be within 3 std devs of actual
    assert np.all(np.abs((sample_mean - resampled_mean) / std_resampled_means) < 3)

    returns = df.to_returns().dropna()
    sample_mean = np.mean(returns, axis=0)

    sample_stats = ffn.resample_returns(
        returns, lambda x: np.mean(x, axis=0), seed=0, num_trials=100
    )

    resampled_mean = np.mean(sample_stats)
    std_resampled_means = np.std(sample_stats, ddof=1)

    assert np.all(np.abs((sample_mean - resampled_mean) / std_resampled_means) < 3)


def test_monthly_returns():

    dates = [
        "31/12/2017",
        "5/1/2018",
        "9/1/2018",
        "13/1/2018",
        "17/1/2018",
        "21/1/2018",
        "25/1/2018",
        "29/1/2018",
        "2/2/2018",
        "6/2/2018",
        "10/2/2018",
        "14/2/2018",
        "18/2/2018",
        "22/2/2018",
        "26/2/2018",
        "1/5/2018",
        "5/5/2018",
        "9/5/2018",
        "13/5/2018",
        "17/5/2018",
        "21/5/2018",
        "25/5/2018",
        "29/5/2018",
        "2/6/2018",
        "6/6/2018",
        "10/6/2018",
        "14/6/2018",
        "18/6/2018",
        "22/6/2018",
        "26/6/2018",
    ]

    prices = [
        100,
        98,
        100,
        103,
        106,
        106,
        107,
        111,
        115,
        115,
        118,
        122,
        120,
        119,
        118,
        119,
        118,
        120,
        122,
        126,
        130,
        131,
        131,
        134,
        138,
        139,
        139,
        138,
        140,
        140,
    ]

    df1 = pd.DataFrame(
        prices, index=pd.to_datetime(dates, format="%d/%m/%Y"), columns=["Price"]
    )

    obj1 = ffn.PerformanceStats(df1["Price"])

    obj1.monthly_returns == df1["Price"].resample("M").last().pct_change()


def test_drawdown_details(df):
    drawdown = ffn.to_drawdown_series(df["MSFT"])
    drawdown_details = ffn.drawdown_details(drawdown)

    assert drawdown_details.loc[drawdown_details.index[1], "Length"] == 18

    num_years = 30
    num_months = num_years * 12
    np.random.seed(0)
    returns = np.random.normal(loc=0.06 / 12, scale=0.20 / np.sqrt(12), size=num_months)
    returns = pd.Series(np.cumprod(1 + returns))

    drawdown = ffn.to_drawdown_series(returns)
    drawdown_details = ffn.drawdown_details(drawdown, index_type=drawdown.index)


def test_infer_nperiods():
    daily = pd.DataFrame(np.random.randn(10),
            index = pd.date_range(start='2018-01-01', periods = 10, freq = 'D'))
    hourly = pd.DataFrame(np.random.randn(10),
            index = pd.date_range(start='2018-01-01', periods = 10, freq = 'H'))
    yearly = pd.DataFrame(np.random.randn(10),
            index = pd.date_range(start='2018-01-01', periods = 10, freq = 'Y'))
    monthly = pd.DataFrame(np.random.randn(10),
            index = pd.date_range(start='2018-01-01', periods = 10, freq = 'M'))
    minutely = pd.DataFrame(np.random.randn(10),
            index = pd.date_range(start='2018-01-01', periods = 10, freq = 'T'))
    secondly = pd.DataFrame(np.random.randn(10),
            index = pd.date_range(start='2018-01-01', periods = 10, freq = 'S'))
    
    minutely_30 = pd.DataFrame(np.random.randn(10),
            index = pd.date_range(start='2018-01-01', periods = 10, freq = '30T'))
    
    
    not_known_vals = np.concatenate((pd.date_range(start='2018-01-01', periods = 5, freq = '1H').values,
    pd.date_range(start='2018-01-02', periods = 5, freq = '5H').values))
        
    not_known = pd.DataFrame(np.random.randn(10),
            index = pd.DatetimeIndex(not_known_vals))
    
    assert ffn.core.infer_nperiods(daily) == ffn.core.TRADING_DAYS_PER_YEAR
    assert ffn.core.infer_nperiods(hourly) == ffn.core.TRADING_DAYS_PER_YEAR * 24
    assert ffn.core.infer_nperiods(minutely) == ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60
    assert ffn.core.infer_nperiods(secondly) == ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60 * 60
    assert ffn.core.infer_nperiods(monthly) == 12
    assert ffn.core.infer_nperiods(yearly) == 1
    assert ffn.core.infer_nperiods(minutely_30) == ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60 * 30
    assert ffn.core.infer_nperiods(not_known) is None
    