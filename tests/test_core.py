import ffn
import pandas as pd
import numpy as np
from pytest import fixture
from numpy.testing import assert_almost_equal as aae
from packaging.version import Version


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
    return df["AAPL"].iloc[0:10]


def test_mtd_ytd(df):
    data = df["AAPL"]

    # Intramonth
    prices = data[pd.to_datetime("2004-12-10"): pd.to_datetime("2004-12-25")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, -0.0175, 4)
    assert mtd_actual == ytd_actual

    # Year change - first month
    prices = data[pd.to_datetime("2004-12-10"): pd.to_datetime("2005-01-15")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, 0.0901, 4)
    assert mtd_actual == ytd_actual

    # Year change - second month
    prices = data[pd.to_datetime("2004-12-10"): pd.to_datetime("2005-02-15")]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    aae(mtd_actual, 0.1497, 4)
    aae(ytd_actual, 0.3728, 4)

    # Single day
    prices = data[[pd.to_datetime("2004-12-10")]]
    dp = prices.resample("D").last().dropna()
    mp = prices.resample(ffn.core._MonthEnd).last().dropna()
    yp = prices.resample(ffn.core._YearEnd).last().dropna()
    mtd_actual = ffn.calc_mtd(dp, mp)
    ytd_actual = ffn.calc_ytd(dp, yp)

    assert mtd_actual == ytd_actual == 0


def test_to_returns_ts(ts):
    data = ts
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual.iloc[0])
    aae(actual.iloc[1], -0.019, 3)
    aae(actual.iloc[9], -0.022, 3)


def test_to_returns_df(df):
    data = df
    actual = data.to_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.iloc[0]))
    aae(actual["AAPL"].iloc[1], -0.019, 3)
    aae(actual["AAPL"].iloc[9], -0.022, 3)
    aae(actual["MSFT"].iloc[1], -0.011, 3)
    aae(actual["MSFT"].iloc[9], -0.014, 3)
    aae(actual["C"].iloc[1], -0.012, 3)
    aae(actual["C"].iloc[9], 0.004, 3)


def test_to_log_returns_ts(ts):
    data = ts
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert np.isnan(actual.iloc[0])
    aae(actual.iloc[1], -0.019, 3)
    aae(actual.iloc[9], -0.022, 3)


def test_to_log_returns_df(df):
    data = df
    actual = data.to_log_returns()

    assert len(actual) == len(data)
    assert all(np.isnan(actual.iloc[0]))
    aae(actual["AAPL"].iloc[1], -0.019, 3)
    aae(actual["AAPL"].iloc[9], -0.022, 3)
    aae(actual["MSFT"].iloc[1], -0.011, 3)
    aae(actual["MSFT"].iloc[9], -0.014, 3)
    aae(actual["C"].iloc[1], -0.012, 3)
    aae(actual["C"].iloc[9], 0.004, 3)


def test_to_price_index(df):
    data = df
    rets = data.to_returns()
    actual = rets.to_price_index()

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 100, 3)
    aae(actual["MSFT"].iloc[0], 100, 3)
    aae(actual["C"].iloc[0], 100, 3)
    aae(actual["AAPL"].iloc[9], 91.366, 3)
    aae(actual["MSFT"].iloc[9], 95.191, 3)
    aae(actual["C"].iloc[9], 101.199, 3)

    actual = rets.to_price_index(start=1)

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 1, 3)
    aae(actual["MSFT"].iloc[0], 1, 3)
    aae(actual["C"].iloc[0], 1, 3)
    aae(actual["AAPL"].iloc[9], 0.914, 3)
    aae(actual["MSFT"].iloc[9], 0.952, 3)
    aae(actual["C"].iloc[9], 1.012, 3)


def test_rebase(df):
    data = df
    actual = data.rebase()

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 100, 3)
    aae(actual["MSFT"].iloc[0], 100, 3)
    aae(actual["C"].iloc[0], 100, 3)
    aae(actual["AAPL"].iloc[9], 91.366, 3)
    aae(actual["MSFT"].iloc[9], 95.191, 3)
    aae(actual["C"].iloc[9], 101.199, 3)


def test_to_drawdown_series_ts(ts):
    data = ts
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual.iloc[0], 0, 3)
    aae(actual.iloc[1], -0.019, 3)
    aae(actual.iloc[9], -0.086, 3)


def test_to_drawdown_series_df(df):
    data = df
    actual = data.to_drawdown_series()

    assert len(actual) == len(data)
    aae(actual["AAPL"].iloc[0], 0, 3)
    aae(actual["MSFT"].iloc[0], 0, 3)
    aae(actual["C"].iloc[0], 0, 3)

    aae(actual["AAPL"].iloc[1], -0.019, 3)
    aae(actual["MSFT"].iloc[1], -0.011, 3)
    aae(actual["C"].iloc[1], -0.012, 3)

    aae(actual["AAPL"].iloc[9], -0.086, 3)
    aae(actual["MSFT"].iloc[9], -0.048, 3)
    aae(actual["C"].iloc[9], -0.029, 3)


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
    assert np.isnan(actual["a"].iloc[-1])
    assert np.isnan(actual["b"].iloc[0])
    assert actual["a"].iloc[0] == 100
    assert actual["a"].iloc[1] == 100
    assert actual["b"].iloc[-1] == 200
    assert actual["b"].iloc[1] == 200

    old = actual
    old.columns = ["c", "d"]

    actual = ffn.merge(old, a, b)

    assert "a" in actual
    assert "b" in actual
    assert "c" in actual
    assert "d" in actual
    assert len(actual) == 6
    assert len(actual.columns) == 4
    assert np.isnan(actual["a"].iloc[-1])
    assert np.isnan(actual["b"].iloc[0])
    assert actual["a"].iloc[0] == 100
    assert actual["a"].iloc[1] == 100
    assert actual["b"].iloc[-1] == 200
    assert actual["b"].iloc[1] == 200


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


def test_calc_inv_vol_weights_object_regression_204(df):
    prc = df.iloc[0:11]
    rets = prc.to_returns().dropna().astype(object)
    actual = ffn.core.calc_inv_vol_weights(rets)

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
    actual = a.asfreq_actual(freq=ffn.core._MonthEnd, method="ffill")

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
    PANDAS_VERSION = Version(pd.__version__)
    PANDAS_210 = PANDAS_VERSION >= Version("2.1.0")

    select_map = "map"
    if not PANDAS_210:
        select_map = "applymap"

    n = 10
    bounds = (0.0, 1.0)
    tot = 1.0000
    low = bounds[0]
    high = bounds[1]

    df = pd.DataFrame(index=range(1000), columns=range(n))
    for i in df.index:
        df.loc[i] = ffn.random_weights(n, bounds, tot)
    assert df.sum(axis=1).apply(lambda x: np.round(x, 4) == tot).all()
    assert (getattr(df, select_map)(lambda x: (x >= low and x <= high))
            .all().all())

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
        getattr(df, select_map)(lambda x: (np.round(x, 2) >= low
                                           and np.round(x, 2) <= high))
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
        getattr(df, select_map)(lambda x: (np.round(x, 2) >= low
                                           and np.round(x, 2) <= high))
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
        getattr(df, select_map)(lambda x: (np.round(x, 2) >= low
                                           and np.round(x, 2) <= high))
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
    negative_returns = er.clip(upper=0.0)
    downside_deviation = np.sqrt((negative_returns**2).mean())
    assert np.allclose(
        a, (er.mean() - rf) / downside_deviation * np.sqrt(p)
    )


def test_calc_sortino_ratio_is_order_invariant():
    # Both the mean and the downside deviation are symmetric functions of the
    # sample, so reordering the same returns must not change the ratio.
    idx = pd.date_range("2026-01-31", periods=4, freq=ffn.core._MonthEnd)
    negative_first = pd.Series([-0.10, 0.02, 0.01, 0.03], index=idx)
    negative_last = pd.Series([0.02, 0.01, 0.03, -0.10], index=idx)

    assert np.isclose(
        negative_first.calc_sortino_ratio(annualize=False), -0.2
    )
    assert np.isclose(
        negative_last.calc_sortino_ratio(annualize=False), -0.2
    )


def test_calc_sortino_ratio_counts_first_period_downside():
    # A series whose only losing period comes first still has downside risk.
    idx = pd.date_range("2026-01-31", periods=4, freq=ffn.core._MonthEnd)
    returns = pd.Series([-0.10, 0.02, 0.01, 0.03], index=idx)

    assert np.isfinite(returns.calc_sortino_ratio(annualize=False))


def test_calc_sortino_ratio_ignores_leading_nan(df):
    # Returns built from prices carry a leading NaN, which pandas already skips
    # in both the mean and the downside deviation.
    r = df.to_returns()

    assert np.allclose(
        r.calc_sortino_ratio(annualize=False),
        r[1:].calc_sortino_ratio(annualize=False),
    )

def test_to_ulcer_index_is_in_percentage_points():
    # 100 -> 90 -> 100 has drawdowns of 0%, -10%, 0%, so the ulcer index is
    # sqrt(mean([0, 100, 0])) = sqrt(100 / 3)
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 90.0, 100.0], index=idx)

    assert np.isclose(prices.to_ulcer_index(), np.sqrt(100 / 3))


def test_to_ulcer_index_without_drawdown_is_zero():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)

    assert np.isclose(prices.to_ulcer_index(), 0.0)


def test_to_ulcer_performance_index_matches_ulcer_index_scale():
    # The ulcer index is expressed in percentage points, so the excess return
    # must be too. Mean excess return here is (-0.1 + 1/9) / 2 = 0.005555...,
    # i.e. 0.5555...% against an ulcer index of sqrt(100 / 3).
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 90.0, 100.0], index=idx)

    expected = (0.5555555555555556) / np.sqrt(100 / 3)

    assert np.isclose(prices.to_ulcer_performance_index(), expected)


def test_to_ulcer_performance_index_is_dimensionally_consistent():
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    prices = pd.Series([100.0, 110, 105, 120, 90, 95, 130, 125], index=idx)

    upi = prices.to_ulcer_performance_index()
    mean_excess_pct = prices.to_returns().mean() * 100

    assert np.isclose(upi * prices.to_ulcer_index(), mean_excess_pct)


def _diff_series(n, mean=0.001, std=0.01):
    # A return series against a flat benchmark, with the differential's mean and
    # standard deviation pinned so the information ratio is exactly mean / std
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, n)
    raw = (raw - raw.mean()) / raw.std(ddof=1)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return (
        pd.Series(mean + std * raw, index=idx),
        pd.Series(np.zeros(n), index=idx),
    )


def test_calc_prob_mom_increases_with_sample_size():
    """Test that the same per-period edge observed for longer gives more confidence"""
    probs = []
    for n in (10, 50, 250, 1000):
        returns, benchmark = _diff_series(n)
        assert np.isclose(returns.calc_information_ratio(benchmark), 0.1)
        probs.append(returns.calc_prob_mom(benchmark))

    for earlier, later in zip(probs, probs[1:]):
        assert later > earlier

    assert probs[0] < 0.7
    assert probs[-1] > 0.99


def test_calc_prob_mom_matches_one_sample_t_test():
    """Test that the probability is the t CDF of the information ratio scaled by sqrt(n)"""
    # 250 observations at an information ratio of 0.1 give a t statistic of
    # 0.1 * sqrt(250) = 1.5811 on 249 degrees of freedom
    returns, benchmark = _diff_series(250)

    assert np.isclose(returns.calc_prob_mom(benchmark), 0.942442, atol=1e-6)


def test_calc_prob_mom_without_an_edge_is_half():
    """Test that a series compared against itself carries no information"""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(np.linspace(0.001, 0.002, 100), index=idx)

    assert np.isclose(returns.calc_prob_mom(returns), 0.5)


def test_calc_prob_mom_ignores_unaligned_observations():
    """Test that NaNs and non-overlapping dates don't count toward the sample size"""
    returns, benchmark = _diff_series(250)
    padded = returns.copy()
    padded.iloc[0] = np.nan  # e.g. the leading NaN from to_returns()
    short_benchmark = benchmark.iloc[:100]

    # Only the 99 dates that are non-NaN in both series carry information, so
    # the result must match the computation restricted to that window
    expected = returns.iloc[1:100].calc_prob_mom(benchmark.iloc[1:100])

    assert np.isclose(padded.calc_prob_mom(short_benchmark), expected)


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


def test_twelve_month_win_perc_uses_twelve_month_window():
    # 13 month end prices => exactly one full twelve month window,
    # 2020-01-31 -> 2021-01-31, which returns 101 / 100 - 1 = +1%.
    # The eleven month window 2020-01-31 -> 2020-12-31 is 99 / 100 - 1 = -1%,
    # so measuring the wrong window flips the result.
    index = pd.date_range("2020-01-31", periods=13, freq=ffn.core._MonthEnd)
    prices = pd.Series([100.0] * 11 + [99.0, 101.0], index=index)

    stats = ffn.calc_stats(prices).stats

    assert stats["twelve_month_win_perc"] == 1.0

    # 12 month end prices are only eleven monthly returns, which is not
    # enough for a twelve month window
    stats = ffn.calc_stats(prices.iloc[:12]).stats
    assert pd.isnull(stats["twelve_month_win_perc"])

    # Missing endpoint prices do not represent losing windows and are excluded.
    full_index = pd.date_range("2020-01-31", periods=14, freq=ffn.core._MonthEnd)
    prices = pd.Series(range(100, 113), index=full_index.delete(1))
    stats = ffn.calc_stats(prices).stats
    assert stats["twelve_month_win_perc"] == 1.0


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


def test_calc_expected_max_sharpe():
    # No dispersion, or a single trial, means no selection to correct for
    assert ffn.calc_expected_max_sharpe(1, 0.5) == 0.0
    assert ffn.calc_expected_max_sharpe(50, 0.0) == 0.0

    # The hurdle grows with the number of trials and scales with their dispersion
    assert ffn.calc_expected_max_sharpe(100, 0.5) > ffn.calc_expected_max_sharpe(10, 0.5)
    aae(
        ffn.calc_expected_max_sharpe(10, 1.0) * 0.5,
        ffn.calc_expected_max_sharpe(10, 0.5),
    )


def test_calc_deflated_sharpe_ratio():
    np.random.seed(0)
    n_trials, n_periods = 40, 1000
    index = pd.date_range(start="2015-01-01", periods=n_periods, freq="D")
    # A search over trials that have no skill whatsoever
    trials = pd.DataFrame(np.random.normal(0, 0.01, (n_periods, n_trials)), index=index)
    sharpes = trials.calc_sharpe()
    winner = trials[sharpes.idxmax()]

    dsr = ffn.calc_deflated_sharpe_ratio(winner, sharpes)
    assert 0 <= dsr <= 1
    # The winner of a skill-less search must not survive deflation ...
    assert dsr < 0.95
    # ... though it looks significant when its selection is ignored
    assert ffn.calc_deflated_sharpe_ratio(winner, [sharpes.max()]) > dsr

    # More trials set a higher hurdle, hence a lower probability
    assert ffn.calc_deflated_sharpe_ratio(winner, sharpes[:10]) > dsr

    # Attached to pandas objects like the other metrics
    aae(winner.calc_deflated_sharpe_ratio(sharpes), dsr)


def test_calc_information_ratio_dataframe():
    returns = pd.DataFrame(
        {
            "varying": [0.03, 0.01, -0.02, 0.04],
            "constant": [0.01, 0.01, 0.01, 0.01],
        }
    )
    benchmark = pd.DataFrame(
        {
            "varying": [0.01, 0.0, -0.01, 0.02],
            "constant": [0.01, 0.01, 0.01, 0.01],
        }
    )

    actual = returns.calc_information_ratio(benchmark)
    difference = returns - benchmark
    expected = difference.mean() / difference.std(ddof=1)
    expected["constant"] = 0.0

    pd.testing.assert_series_equal(actual, expected)


def test_calc_information_ratio_dataframe_with_series_benchmark():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "fund_a": [0.03, 0.01, -0.02, 0.04],
            "fund_b": [0.02, 0.02, -0.01, 0.03],
        },
        index=index,
    )
    benchmark = pd.Series([0.01, 0.0, -0.01, 0.02], index=index)

    actual = returns.calc_information_ratio(benchmark)
    difference = returns.sub(benchmark, axis="index")
    expected = difference.mean() / difference.std(ddof=1)

    pd.testing.assert_series_equal(actual, expected)


def test_calc_deflated_sharpe_ratio_zero_dispersion():
    sharpes = [0.5, 1.0, 1.5, 2.0]

    # A constant series has no dispersion, so no Sharpe ratio and no deflated one.
    # Its standard deviation is floating-point residue rather than an exact zero, so
    # the ratio comes out finite (~4.6e15) and reaches the deflation arithmetic, which
    # answered 1.0 -- certainty of an edge, from the one input that cannot show one.
    # Checked across values and lengths, not at one point: the residue depends on both,
    # so a guard calibrated on a single series passes while still leaking elsewhere.
    for value in (1e-7, 1e-4, 0.001, 0.01, 1.0, 100.0):
        for n in (3, 10, 250, 5000):
            flat = pd.Series([value] * n)
            assert np.isnan(
                ffn.calc_deflated_sharpe_ratio(flat, sharpes)
            ), f"leaked at value={value}, n={n}"
    assert np.isnan(ffn.calc_deflated_sharpe_ratio(pd.Series([0.0] * 250), sharpes))

    # The guard is relative to the scale of the data: a real but very quiet series
    # still gets a number.
    quiet = pd.Series(np.random.default_rng(1).normal(0, 1e-8, 250))
    assert 0 <= ffn.calc_deflated_sharpe_ratio(quiet, sharpes) <= 1

    # A long, quiet series around a nonzero mean still has real dispersion. The
    # zero-dispersion threshold must not grow with the sample size and swallow it.
    quiet_nonzero = pd.Series(1.0 + np.random.default_rng(1).normal(0, 1e-12, 5000))
    assert 0 <= ffn.calc_deflated_sharpe_ratio(quiet_nonzero, sharpes, nperiods=252) <= 1

    # Dispersion is measured after subtracting a series risk-free rate, matching the
    # returns used to calculate Sharpe. Check both directions of that distinction.
    rf = pd.Series(np.linspace(0.0001, 0.0002, 250))
    constant_excess = rf + 0.001
    assert np.isnan(ffn.calc_deflated_sharpe_ratio(constant_excess, sharpes, rf=rf, nperiods=252))

    variable_excess = pd.Series([0.001] * 250)
    assert 0 <= ffn.calc_deflated_sharpe_ratio(variable_excess, sharpes, rf=rf, nperiods=252) <= 1


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

    monthly_returns = df["MSFT"].resample(ffn.core._MonthEnd).last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        monthly_returns.dropna().mean()
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample(ffn.core._YearEnd).last().pct_change()
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

    monthly_returns = df["MSFT"].resample(ffn.core._MonthEnd).last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        np.mean(monthly_returns.dropna() - 0.02 / 12)
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample(ffn.core._YearEnd).last().pct_change()
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

    monthly_returns = df["MSFT"].resample(ffn.core._MonthEnd).last().pct_change()
    rf_monthly_returns = rf.resample(ffn.core._MonthEnd).last().pct_change()
    aae(
        performanceStats.monthly_sharpe,
        np.mean(monthly_returns - rf_monthly_returns)
        / (monthly_returns.dropna().std())
        * (np.sqrt(12)),
        3,
    )
    aae(performanceStats.monthly_sharpe, groupStats["MSFT"].monthly_sharpe, 3)

    yearly_returns = df["MSFT"].resample(ffn.core._YearEnd).last().pct_change()
    rf_yearly_returns = rf.resample(ffn.core._YearEnd).last().pct_change()
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


def test_calc_stats_annualization_factor(df):
    prices = df[["AAPL", "MSFT"]]
    stats = prices.calc_stats(annualization_factor=365)

    assert stats["AAPL"].annualization_factor == 365
    assert stats["MSFT"].annualization_factor == 365

    stats.set_date_range(start=prices.index[10])
    assert stats["AAPL"].annualization_factor == 365
    assert stats["MSFT"].annualization_factor == 365

    single_stats = prices["AAPL"].calc_stats(annualization_factor=365)
    assert single_stats.annualization_factor == 365


def test_group_stats_uses_each_series_own_calendar():
    # GH #155: a NaN row in one series should not drop that date from the
    # other series' individual stats
    dates = pd.date_range("2020-01-31", periods=24, freq=ffn.core._MonthEnd)
    np.random.seed(1)
    sym1 = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.01, 0.04, 24)), index=dates, name="SYM1"
    )
    sym2 = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.01, 0.04, 24)), index=dates, name="SYM2"
    )
    # SYM2 missing for three months in the middle
    sym2.iloc[9:12] = np.nan

    gs = ffn.GroupStats(sym1, sym2)

    # per-series stats match the single-series result
    aae(gs["SYM1"].monthly_sharpe, ffn.PerformanceStats(sym1).monthly_sharpe, 9)
    aae(gs["SYM1"].total_return, ffn.PerformanceStats(sym1).total_return, 9)
    aae(
        gs["SYM2"].monthly_sharpe, ffn.PerformanceStats(sym2.dropna()).monthly_sharpe, 9
    )

    # SYM1 keeps all of its own dates
    assert len(gs["SYM1"].prices) == 24

    # cross-sectional prices still use the common calendar
    assert len(gs.prices) == 21

    # date range slicing preserves the per-series calendar behaviour
    gs.set_date_range(start=dates[3])
    aae(
        gs["SYM1"].monthly_sharpe,
        ffn.PerformanceStats(sym1[dates[3]:]).monthly_sharpe,
        9,
    )
    gs.set_date_range()
    aae(gs["SYM1"].monthly_sharpe, ffn.PerformanceStats(sym1).monthly_sharpe, 9)


def test_group_stats_date_range_reset_restores_full_calendars():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    early = pd.Series(range(100, 105), index=dates[:5], name="EARLY")
    late = pd.Series(range(200, 206), index=dates[2:], name="LATE")

    gs = ffn.GroupStats(early, late)
    gs.set_date_range()

    assert gs["EARLY"].start == dates[0]
    assert gs["EARLY"].end == dates[4]
    assert gs["LATE"].start == dates[2]
    assert gs["LATE"].end == dates[-1]


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

    resampled_mean = np.mean(sample_stats, axis=0)
    std_resampled_means = np.std(sample_stats, ddof=1, axis=0)

    # resampled statistics should be within 3 std devs of actual
    assert np.all(np.abs((sample_mean - resampled_mean) / std_resampled_means) < 3)

    returns = df.to_returns().dropna()
    sample_mean = np.mean(returns, axis=0)

    sample_stats = ffn.resample_returns(
        returns, lambda x: np.mean(x, axis=0), seed=0, num_trials=100
    )

    resampled_mean = np.mean(sample_stats, axis=0)
    std_resampled_means = np.std(sample_stats, ddof=1, axis=0)

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

    obj1.monthly_returns == df1["Price"].resample(ffn.core._MonthEnd).last().fillna(1.0).pct_change(fill_method=None)


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
                         index=pd.date_range(start='2018-01-01', periods=10, freq='D'))
    hourly = pd.DataFrame(np.random.randn(10),
                          index=pd.date_range(start='2018-01-01', periods=10, freq='h'))
    yearly = pd.DataFrame(np.random.randn(10),
                          index=pd.date_range(start='2018-01-01', periods=10, freq=ffn.core._YearEnd))
    monthly = pd.DataFrame(np.random.randn(10),
                           index=pd.date_range(start='2018-01-01', periods=10, freq=ffn.core._MonthEnd))
    minutely = pd.DataFrame(np.random.randn(10),
                            index=pd.date_range(start='2018-01-01', periods=10, freq='min'))
    secondly = pd.DataFrame(np.random.randn(10),
                            index=pd.date_range(start='2018-01-01', periods=10, freq='s'))

    minutely_30 = pd.DataFrame(np.random.randn(10),
                               index=pd.date_range(start='2018-01-01', periods=10, freq='30min'))

    not_known_vals = np.concatenate((pd.date_range(start='2018-01-01', periods=5, freq='1h').values,
                                     pd.date_range(start='2018-01-02', periods=5, freq='5h').values))

    not_known = pd.DataFrame(np.random.randn(10),
                             index=pd.DatetimeIndex(not_known_vals))

    assert ffn.core.infer_nperiods(daily) == ffn.core.TRADING_DAYS_PER_YEAR
    assert ffn.core.infer_nperiods(hourly) == ffn.core.TRADING_DAYS_PER_YEAR * 24
    assert ffn.core.infer_nperiods(minutely) == ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60
    assert ffn.core.infer_nperiods(secondly) == ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60 * 60
    assert ffn.core.infer_nperiods(monthly) == 12
    assert ffn.core.infer_nperiods(yearly) == 1
    expected_30min_periods = ffn.core.TRADING_DAYS_PER_YEAR * 24 * 60 / 30
    assert ffn.core.infer_nperiods(minutely_30) == expected_30min_periods

    returns_30min = minutely_30.squeeze()
    expected_sharpe = returns_30min.mean() / returns_30min.std(ddof=1) * np.sqrt(expected_30min_periods)
    assert np.allclose(returns_30min.calc_sharpe(), expected_sharpe)

    descending_30min = minutely_30.sort_index(ascending=False).squeeze()
    assert ffn.core.infer_nperiods(descending_30min) == expected_30min_periods
    assert np.allclose(descending_30min.calc_sharpe(), expected_sharpe)
    assert ffn.core.infer_nperiods(not_known) is None
