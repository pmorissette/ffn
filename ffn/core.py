import numpy as np
import pandas as pd
from pandas.core.base import PandasObject


def to_returns(self):
    """
    Calculates the simple arithmetic returns of a price series.

    Formula is: (t1 / t0) - 1

    Args:
        self: Expects a price series
    """
    return self / self.shift(1) - 1


def to_log_returns(self):
    """
    Calculates the log returns of a price series.

    Formula is: ln(p1/p0)

    Args:
        self: Expects a price series
    """
    return np.log(self / self.shift(1))


def to_price_index(self, start=100):
    """
    Returns a price index given a series of returns.

    Args:
        * self: Expects a return series
        * start (number): Starting level

    Assumes arithmetic returns.

    Formula is: cumprod (1+r)
    """
    return (self.replace(to_replace=np.nan, value=0) + 1).cumprod() * 100


def rebase(self, value=100):
    """
    Rebase all series to a given intial value.

    This makes comparing/plotting different series
    together easier.

    Args:
        * self: Expects a price series
        * value (number): starting value for all series.
    """
    return self / self.ix[0] * value


def calc_perf_stats(obj):
    """
    Calculates the performance statistics given an object.
    The object should be a TimeSeries of prices.

    A dictionary will be returned containing all the stats.

    Args:
        * obj: A Pandas TimeSeries representing a series of prices.
    Returns:
        * object -- Returns an object (Bunch) containing
            all relevant statistics.
    """
    stats = {}

    if len(obj) is 0:
        return stats

    stats['start'] = obj.index[0]
    stats['end'] = obj.index[-1]

    # default values
    stats['daily_mean'] = np.nan
    stats['daily_vol'] = np.nan
    stats['daily_sharpe'] = np.nan
    stats['best_day'] = np.nan
    stats['worst_day'] = np.nan
    stats['total_return'] = np.nan
    stats['cagr'] = np.nan
    stats['incep'] = np.nan
    stats['drawdown'] = np.nan
    stats['max_drawdown'] = np.nan
    stats['drawdown_details'] = np.nan
    stats['daily_skew'] = np.nan
    stats['daily_kurt'] = np.nan
    stats['monthly_returns'] = np.nan
    stats['avg_drawdown'] = np.nan
    stats['avg_drawdown_days'] = np.nan
    stats['monthly_mean'] = np.nan
    stats['monthly_vol'] = np.nan
    stats['monthly_sharpe'] = np.nan
    stats['best_month'] = np.nan
    stats['worst_month'] = np.nan
    stats['mtd'] = np.nan
    stats['three_month'] = np.nan
    stats['pos_month_perc'] = np.nan
    stats['avg_up_month'] = np.nan
    stats['avg_down_month'] = np.nan
    stats['monthly_skew'] = np.nan
    stats['monthly_kurt'] = np.nan
    stats['six_month'] = np.nan
    stats['yearly_returns'] = np.nan
    stats['ytd'] = np.nan
    stats['one_year'] = np.nan
    stats['yearly_mean'] = np.nan
    stats['yearly_vol'] = np.nan
    stats['yearly_sharpe'] = np.nan
    stats['best_year'] = np.nan
    stats['worst_year'] = np.nan
    stats['three_year'] = np.nan
    stats['win_year_perc'] = np.nan
    stats['twelve_month_win_perc'] = np.nan
    stats['yearly_skew'] = np.nan
    stats['yearly_kurt'] = np.nan
    stats['five_year'] = np.nan
    stats['ten_year'] = np.nan
    stats['return_table'] = {}
    # end default values

    # save daily prices for future use
    stats['daily_prices'] = obj
    # M = month end frequency
    stats['monthly_prices'] = obj.resample('M', 'last')
    # A == year end frequency
    stats['yearly_prices'] = obj.resample('A', 'last')

    # let's save some typing
    p = obj
    mp = stats['monthly_prices']
    yp = stats['yearly_prices']

    if len(p) is 1:
        return stats

    # stats using daily data
    stats['returns'] = p.to_returns()
    stats['log_returns'] = p.to_log_returns()
    r = stats['returns']

    if len(r) < 2:
        return stats

    stats['daily_mean'] = r.mean() * 252
    stats['daily_vol'] = r.std() * np.sqrt(252)
    stats['daily_sharpe'] = stats['daily_mean'] / stats['daily_vol']
    stats['best_day'] = r.max()
    stats['worst_day'] = r.min()

    stats['total_return'] = obj[-1] / obj[0] - 1
    # save ytd as total_return for now - if we get to real ytd
    # then it will get updated
    stats['ytd'] = stats['total_return']
    stats['cagr'] = calc_cagr(p)
    stats['incep'] = stats['cagr']

    stats['drawdown'] = p.to_drawdown_series()
    stats['max_drawdown'] = stats['drawdown'].min()
    stats['drawdown_details'] = drawdown_details(stats['drawdown'])
    if stats['drawdown_details']:
        stats['avg_drawdown'] = stats['drawdown_details']['drawdown'].mean()
        stats['avg_drawdown_days'] = stats['drawdown_details']['days'].mean()

    if len(r) < 4:
        return stats

    stats['daily_skew'] = r.skew()

    # if all zero/nan kurt fails division by zero
    if len(r[(~np.isnan(r)) & (r != 0)]) > 0:
        stats['daily_kurt'] = r.kurt()

    # stats using monthly data
    stats['monthly_returns'] = stats['monthly_prices'].to_returns()
    mr = stats['monthly_returns']

    if len(mr) < 2:
        return stats

    stats['monthly_mean'] = mr.mean() * 12
    stats['monthly_vol'] = mr.std() * np.sqrt(12)
    stats['monthly_sharpe'] = stats['monthly_mean'] / stats['monthly_vol']
    stats['best_month'] = mr.max()
    stats['worst_month'] = mr.min()

    # -2 because p[-1] will be mp[-1]
    stats['mtd'] = p[-1] / mp[-2] - 1

    # -1 here to account for first return that will be nan
    stats['pos_month_perc'] = len(mr[mr > 0]) / float(len(mr) - 1)
    stats['avg_up_month'] = mr[mr > 0].mean()
    stats['avg_down_month'] = mr[mr <= 0].mean()

    # return_table
    for idx in mr.index:
        if idx.year not in stats['return_table']:
            stats['return_table'][idx.year] = {1: 0, 2: 0, 3: 0,
                                               4: 0, 5: 0, 6: 0,
                                               7: 0, 8: 0, 9: 0,
                                               10: 0, 11: 0, 12: 0}
        if not np.isnan(mr[idx]):
            stats['return_table'][idx.year][idx.month] = mr[idx]
    # add first month
    fidx = mr.index[0]
    stats['return_table'][fidx.year][fidx.month] = float(mp[0]) / p[0] - 1
    # calculate the YTD values
    for idx in stats['return_table']:
        arr = np.array(stats['return_table'][idx].values())
        stats['return_table'][idx][13] = np.prod(arr + 1) - 1

    if len(mr) < 3:
        return stats

    stats['three_month'] = p[-1] / \
        p[:p.index[-1] - pd.DateOffset(months=3)][-1] - 1

    if len(mr) < 4:
        return stats

    stats['monthly_skew'] = mr.skew()

    # if all zero/nan kurt fails division by zero
    if len(mr[(~np.isnan(mr)) & (mr != 0)]) > 0:
        stats['monthly_kurt'] = mr.kurt()

    stats['six_month'] = p[-1] / \
        p[:p.index[-1] - pd.DateOffset(months=6)][-1] - 1
    # -2 because p[-1] == yp[-1]

    stats['yearly_returns'] = stats['yearly_prices'].to_returns()
    yr = stats['yearly_returns']

    if len(yr) < 2:
        return stats

    stats['ytd'] = p[-1] / yp[-2] - 1
    stats['one_year'] = p[-1] / \
        p[:p.index[-1] - pd.DateOffset(years=1)][-1] - 1

    stats['yearly_mean'] = yr.mean()
    stats['yearly_vol'] = yr.std()
    stats['yearly_sharpe'] = stats['yearly_mean'] / stats['yearly_vol']
    stats['best_year'] = yr.max()
    stats['worst_year'] = yr.min()

    # annualize stat for over 1 year
    stats['three_year'] = calc_cagr(p[p.index[-1] - pd.DateOffset(years=3):])

    # -1 here to account for first return that will be nan
    stats['win_year_perc'] = len(yr[yr > 0]) / float(len(yr) - 1)

    tot = 0
    win = 0
    for i in range(11, len(mr)):
        tot = tot + 1
        if mp[i] / mp[i - 11] > 1:
            win = win + 1
    stats['twelve_month_win_perc'] = float(win) / tot

    if len(yr) < 4:
        return stats

    stats['yearly_skew'] = yr.skew()

    # if all zero/nan kurt fails division by zero
    if len(yr[(~np.isnan(yr)) & (yr != 0)]) > 0:
        stats['yearly_kurt'] = yr.kurt()

    stats['five_year'] = calc_cagr(p[p.index[-1] - pd.DateOffset(years=5):])
    stats['ten_year'] = calc_cagr(p[p.index[-1] - pd.DateOffset(years=10):])

    return stats


def to_drawdown_series(prices):
    """
    Calculates the drawdown series.

    This returns a series representing a drawdown.
    When the price is at all time highs, the drawdown
    is 0. However, when prices are below high water marks,
    the drawdown series = current / hwm - 1

    The max drawdown can be obtained by simply calling .min()
    on the result (since the drawdown series is negative)

    Args:
        * prices (TimeSeries or DataFrame): Series of prices.
    """
    # make a copy so that we don't modify original data
    drawdown = prices.copy()

    # set initial hwm (copy to avoid issues w/ overwriting)
    hwm = drawdown.ix[0].copy()
    isdf = isinstance(drawdown, pd.DataFrame)

    for idx in drawdown.index:
        tmp = drawdown.ix[idx]
        if isdf:
            hwm[tmp > hwm] = tmp
        else:
            hwm = max(tmp, hwm)

        drawdown.ix[idx] = tmp / hwm - 1

    # first row is 0 by definition
    drawdown.ix[0] = 0
    return drawdown


def calc_max_drawdown(prices):
    """
    Calculates the max drawdown of a price series. If you want the
    actual drawdown series, please use to_drawdown_series.
    """
    return prices.to_drawdown_series().min()


def drawdown_details(drawdown):
    """
    Returns a data frame with start, end, days (duration) and
    drawdown for each drawdown in a drawdown series.

    .. note::

        days are actual calendar days, not trading days

    Args:
        * drawdown (pandas.TimeSeries): A drawdown TimeSeries
            (can be obtained w/ drawdown(prices).
    Returns:
        * pandas.DataFrame -- A data frame with the following
            columns: start, end, days, drawdown.
    """
    is_zero = drawdown == 0
    # find start dates (first day where dd is non-zero after a zero)
    start = ~is_zero & is_zero.shift(1)
    start = list(start[start == True].index)  # NOQA

    # find end dates (first day where dd is 0 after non-zero)
    end = is_zero & (~is_zero).shift(1)
    end = list(end[end == True].index)  # NOQA

    if len(start) is 0:
        return None

    # drawdown has no end (end period in dd)
    if len(end) is 0:
        end.append(drawdown.index[-1])

    # if the first drawdown start is larger than the first drawdown end it
    # means the drawdown series begins in a drawdown and therefore we must add
    # the first index to the start series
    if start[0] > end[0]:
        start.insert(0, drawdown.index[0])

    # if the last start is greater than the end then we must add the last index
    # to the end series since the drawdown series must finish with a drawdown
    if start[-1] > end[-1]:
        end.append(drawdown.index[-1])

    result = pd.DataFrame(columns=('start', 'end', 'days', 'drawdown'),
                          index=range(0, len(start)))

    for i in range(0, len(start)):
        dd = drawdown[start[i]:end[i]].min()
        result.ix[i] = (start[i], end[i], (end[i] - start[i]).days, dd)

    return result


def calc_cagr(prices):
    """
    Calculates the CAGR (compound annual growth rate) for a given price series.

    Args:
        * prices (pandas.TimeSeries): A TimeSeries of prices.
    Returns:
        * float -- cagr.
    """
    start = prices.index[0]
    end = prices.index[-1]
    return (prices.ix[-1] / prices.ix[0]) ** (1 / year_frac(start, end)) - 1


def year_frac(start, end):
    """
    Similar to excel's yearfrac function. Returns
    a year fraction between two dates (i.e. 1.53 years).

    Approximation using the average number of seconds
    in a year.

    Args:
        * start (datetime): start date
        * end (datetime): end date
    """
    if start > end:
        raise ValueError('start cannot be larger than end')

    # obviously not perfect but good enough
    return (end - start).total_seconds() / (31557600)


def extend_pandas():
    PandasObject.to_returns = to_returns
    PandasObject.to_log_returns = to_log_returns
    PandasObject.to_price_index = to_price_index
    PandasObject.rebase = rebase
    PandasObject.calc_perf_stats = calc_perf_stats
    PandasObject.to_drawdown_series = to_drawdown_series
    PandasObject.calc_max_drawdown = calc_max_drawdown
    PandasObject.calc_cagr = calc_cagr
