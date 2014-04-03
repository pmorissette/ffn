from ffn.utils import fmtp, fmtn, fmtpn, get_period_name
import numpy as np
import pandas as pd
from pandas.core.base import PandasObject
from tabulate import tabulate
from matplotlib import pyplot as plt
try:
    import prettyplotlib
except ImportError:
    pass


class PerformanceStats(object):

    def __init__(self, prices):
        self.prices = prices
        self.name = self.prices.name
        self._start = self.prices.index[0]
        self._end = self.prices.index[-1]
        self._calculate(self.prices)

    def _calculate(self, obj):
        # default values
        self.daily_mean = np.nan
        self.daily_vol = np.nan
        self.daily_sharpe = np.nan
        self.best_day = np.nan
        self.worst_day = np.nan
        self.total_return = np.nan
        self.cagr = np.nan
        self.incep = np.nan
        self.drawdown = np.nan
        self.max_drawdown = np.nan
        self.drawdown_details = np.nan
        self.daily_skew = np.nan
        self.daily_kurt = np.nan
        self.monthly_returns = np.nan
        self.avg_drawdown = np.nan
        self.avg_drawdown_days = np.nan
        self.monthly_mean = np.nan
        self.monthly_vol = np.nan
        self.monthly_sharpe = np.nan
        self.best_month = np.nan
        self.worst_month = np.nan
        self.mtd = np.nan
        self.three_month = np.nan
        self.pos_month_perc = np.nan
        self.avg_up_month = np.nan
        self.avg_down_month = np.nan
        self.monthly_skew = np.nan
        self.monthly_kurt = np.nan
        self.six_month = np.nan
        self.yearly_returns = np.nan
        self.ytd = np.nan
        self.one_year = np.nan
        self.yearly_mean = np.nan
        self.yearly_vol = np.nan
        self.yearly_sharpe = np.nan
        self.best_year = np.nan
        self.worst_year = np.nan
        self.three_year = np.nan
        self.win_year_perc = np.nan
        self.twelve_month_win_perc = np.nan
        self.yearly_skew = np.nan
        self.yearly_kurt = np.nan
        self.five_year = np.nan
        self.ten_year = np.nan
        self.return_table = {}
        # end default values

        if len(obj) is 0:
            return

        self.start = obj.index[0]
        self.end = obj.index[-1]

        # save daily prices for future use
        self.daily_prices = obj
        # M = month end frequency
        self.monthly_prices = obj.resample('M', 'last')
        # A == year end frequency
        self.yearly_prices = obj.resample('A', 'last')

        # let's save some typing
        p = obj
        mp = self.monthly_prices
        yp = self.yearly_prices

        if len(p) is 1:
            return

        # stats using daily data
        self.returns = p.to_returns()
        self.log_returns = p.to_log_returns()
        r = self.returns

        if len(r) < 2:
            return

        self.daily_mean = r.mean() * 252
        self.daily_vol = r.std() * np.sqrt(252)
        self.daily_sharpe = self.daily_mean / self.daily_vol
        self.best_day = r.max()
        self.worst_day = r.min()

        self.total_return = obj[-1] / obj[0] - 1
        # save ytd as total_return for now - if we get to real ytd
        # then it will get updated
        self.ytd = self.total_return
        self.cagr = calc_cagr(p)
        self.incep = self.cagr

        self.drawdown = p.to_drawdown_series()
        self.max_drawdown = self.drawdown.min()
        self.drawdown_details = drawdown_details(self.drawdown)
        if self.drawdown_details:
            self.avg_drawdown = self.drawdown_details['drawdown'].mean()
            self.avg_drawdown_days = self.drawdown_details['days'].mean()

        if len(r) < 4:
            return

        self.daily_skew = r.skew()

        # if all zero/nan kurt fails division by zero
        if len(r[(~np.isnan(r)) & (r != 0)]) > 0:
            self.daily_kurt = r.kurt()

        # stats using monthly data
        self.monthly_returns = self.monthly_prices.to_returns()
        mr = self.monthly_returns

        if len(mr) < 2:
            return

        self.monthly_mean = mr.mean() * 12
        self.monthly_vol = mr.std() * np.sqrt(12)
        self.monthly_sharpe = self.monthly_mean / self.monthly_vol
        self.best_month = mr.max()
        self.worst_month = mr.min()

        # -2 because p[-1] will be mp[-1]
        self.mtd = p[-1] / mp[-2] - 1

        # -1 here to account for first return that will be nan
        self.pos_month_perc = len(mr[mr > 0]) / float(len(mr) - 1)
        self.avg_up_month = mr[mr > 0].mean()
        self.avg_down_month = mr[mr <= 0].mean()

        # return_table
        for idx in mr.index:
            if idx.year not in self.return_table:
                self.return_table[idx.year] = {1: 0, 2: 0, 3: 0,
                                               4: 0, 5: 0, 6: 0,
                                               7: 0, 8: 0, 9: 0,
                                               10: 0, 11: 0, 12: 0}
            if not np.isnan(mr[idx]):
                self.return_table[idx.year][idx.month] = mr[idx]
        # add first month
        fidx = mr.index[0]
        self.return_table[fidx.year][fidx.month] = float(mp[0]) / p[0] - 1
        # calculate the YTD values
        for idx in self.return_table:
            arr = np.array(self.return_table[idx].values())
            self.return_table[idx][13] = np.prod(arr + 1) - 1

        if len(mr) < 3:
            return

        denom = p[:p.index[-1] - pd.DateOffset(months=3)]
        if len(denom) > 0:
            self.three_month = p[-1] / denom[-1] - 1

        if len(mr) < 4:
            return

        self.monthly_skew = mr.skew()

        # if all zero/nan kurt fails division by zero
        if len(mr[(~np.isnan(mr)) & (mr != 0)]) > 0:
            self.monthly_kurt = mr.kurt()

        denom = p[:p.index[-1] - pd.DateOffset(months=6)]
        if len(denom) > 0:
            self.six_month = p[-1] / denom[-1] - 1

        self.yearly_returns = self.yearly_prices.to_returns()
        yr = self.yearly_returns

        if len(yr) < 2:
            return

        self.ytd = p[-1] / yp[-2] - 1

        denom = p[:p.index[-1] - pd.DateOffset(years=1)]
        if len(denom) > 0:
            self.one_year = p[-1] / denom[-1] - 1

        self.yearly_mean = yr.mean()
        self.yearly_vol = yr.std()
        self.yearly_sharpe = self.yearly_mean / self.yearly_vol
        self.best_year = yr.max()
        self.worst_year = yr.min()

        # annualize stat for over 1 year
        self.three_year = calc_cagr(p[p.index[-1] - pd.DateOffset(years=3):])

        # -1 here to account for first return that will be nan
        self.win_year_perc = len(yr[yr > 0]) / float(len(yr) - 1)

        tot = 0
        win = 0
        for i in range(11, len(mr)):
            tot = tot + 1
            if mp[i] / mp[i - 11] > 1:
                win = win + 1
        self.twelve_month_win_perc = float(win) / tot

        if len(yr) < 4:
            return

        self.yearly_skew = yr.skew()

        # if all zero/nan kurt fails division by zero
        if len(yr[(~np.isnan(yr)) & (yr != 0)]) > 0:
            self.yearly_kurt = yr.kurt()

        self.five_year = calc_cagr(p[p.index[-1] - pd.DateOffset(years=5):])
        self.ten_year = calc_cagr(p[p.index[-1] - pd.DateOffset(years=10):])

        return

    def set_date_range(self, start=None, end=None):
        if start is None:
            start = self._start
        else:
            start = pd.to_datetime(start)

        if end is None:
            end = self._end
        else:
            end = pd.to_datetime(end)

        self._calculate(self.prices.ix[start:end])

    def display(self):
        print 'Stats for %s from %s - %s' % (self.name, self.start, self.end)
        print 'Summary:'
        data = [[fmtn(self.daily_sharpe), fmtp(self.cagr),
                 fmtp(self.max_drawdown)]]
        print tabulate(data, headers=['Sharpe', 'CAGR', 'Max Drawdown'])

        print '\nAnnualized Returns:'
        data = [[fmtp(self.mtd), fmtp(self.three_month), fmtp(self.six_month),
                 fmtp(self.ytd), fmtp(self.one_year), fmtp(self.three_year),
                 fmtp(self.five_year), fmtp(self.ten_year),
                 fmtp(self.incep)]]
        print tabulate(data,
                       headers=['mtd', '3m', '6m', 'ytd', '1y',
                                '3y', '5y', '10y', 'incep.'])

        print '\nPeriodic:'
        data = [
            ['sharpe', fmtn(self.daily_sharpe), fmtn(self.monthly_sharpe),
             fmtn(self.yearly_sharpe)],
            ['mean', fmtp(self.daily_mean), fmtp(self.monthly_mean),
             fmtp(self.yearly_mean)],
            ['vol', fmtp(self.daily_vol), fmtp(self.monthly_vol),
             fmtp(self.yearly_vol)],
            ['skew', fmtn(self.daily_skew), fmtn(self.monthly_skew),
             fmtn(self.yearly_skew)],
            ['kurt', fmtn(self.daily_kurt), fmtn(self.monthly_kurt),
             fmtn(self.yearly_kurt)],
            ['best', fmtp(self.best_day), fmtp(self.best_month),
             fmtp(self.best_year)],
            ['worst', fmtp(self.worst_day), fmtp(self.worst_month),
             fmtp(self.worst_year)]]
        print tabulate(data, headers=['daily', 'monthly', 'yearly'])

        print '\nDrawdowns:'
        data = [
            [fmtp(self.max_drawdown), fmtp(self.avg_drawdown),
             fmtn(self.avg_drawdown_days)]]
        print tabulate(data, headers=['max', 'avg', '# days'])

        print '\nMisc:'
        data = [['avg. up month', fmtp(self.avg_up_month)],
                ['avg. down month', fmtp(self.avg_down_month)],
                ['up year %', fmtp(self.win_year_perc)],
                ['12m up %', fmtp(self.twelve_month_win_perc)]]
        print tabulate(data)

    def display_monthly_returns(self):
        data = [['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'YTD']]
        for k in self.return_table.keys():
            r = self.return_table[k]
            data.append([k, fmtpn(r[1]), fmtpn(r[2]), fmtpn(r[3]), fmtpn(r[4]),
                         fmtpn(r[5]), fmtpn(r[6]), fmtpn(r[7]), fmtpn(r[8]),
                         fmtpn(r[9]), fmtpn(r[10]), fmtpn(r[11]), fmtpn(r[12]),
                         fmtpn(r[13])])
        print tabulate(data, headers='firstrow')

    def plot(self, period='d', figsize=(15, 5), title=None, logy=False, **kwargs):
        if title is None:
            title = '%s %s price series' % (self.name, get_period_name(period))

        ser = self._get_series(period)
        ser.plot(figsize=figsize, title=title, logy=logy, **kwargs)

    def plot_histogram(self, period='d', figsize=(15, 5), title=None, bins=20, **kwargs):
        if title is None:
            title = '%s %s return histogram' % (self.name, get_period_name(period))

        ser = self._get_series(period).to_returns().dropna()

        plt.figure(figsize=figsize)
        ax = ser.hist(bins=bins, figsize=figsize, normed=True, **kwargs)
        ax.set_title(title)
        plt.axvline(0, linewidth=4)
        ax2 = ser.plot(kind='kde')

    def _get_series(self, per):
        if per is 'd':
            return self.daily_prices
        elif per is 'm':
            return self.monthly_prices
        elif per is 'y':
            return self.yearly_prices


class GroupStats(dict):

    def __init__(self, *prices):
        # store original prices
        self.prices = merge(*prices).dropna()
        # duplicate columns
        if len(self.prices.columns) != len(set(self.prices.columns)):
            raise ValueError('One or more data series provided',
                             'have same name! Please provide unique names')

        # calculate stats for entire series
        self._calculate(self.prices)

    def _calculate(self, data):
        for c in data.columns:
            prc = data[c]
            self[c] = prc.calc_perf_stats()

    def set_date_range(self, start=None, end=None):
        for k in self:
            self[k].set_date_range(start, end)

    def display(self):
        data = []
        first_row = ['Stat']
        first_row.extend(self.keys())
        data.append(first_row)

        stats = [('start', 'Start', 'dt'),
                 ('end', 'End', 'dt'),
                 (None, None, None),
                 ('daily_sharpe', 'Daily Sharpe', 'n'),
                 ('cagr', 'CAGR', 'p'),
                 ('max_drawdown', 'Max Drawdown', 'p'),
                 (None, None, None),
                 ('mtd', 'MTD', 'p'),
                 ('three_month', '3m', 'p'),
                 ('six_month', '6m', 'p'),
                 ('ytd', 'YTD', 'p'),
                 ('one_year', '1Y', 'p'),
                 ('three_year', '3Y (ann.)', 'p'),
                 ('five_year', '5Y (ann.)', 'p'),
                 ('ten_year', '10Y (ann.)', 'p'),
                 ('incep', 'Since Incep. (ann.)', 'p'),
                 (None, None, None),
                 ('daily_sharpe', 'Daily Sharpe', 'n'),
                 ('daily_mean', 'Daily Mean (ann.)', 'p'),
                 ('daily_vol', 'Daily Vol (ann.)', 'p'),
                 ('daily_skew', 'Daily Skew', 'n'),
                 ('daily_kurt', 'Daily Kurt', 'n'),
                 ('best_day', 'Best Day', 'p'),
                 ('worst_day', 'Worst Day', 'p'),
                 (None, None, None),
                 ('monthly_sharpe', 'Monthly Sharpe', 'p'),
                 ('monthly_mean', 'Monthly Mean (ann.)', 'p'),
                 ('monthly_vol', 'Monthly Vol (ann.)', 'p'),
                 ('monthly_skew', 'Monthly Skew', 'n'),
                 ('monthly_kurt', 'Monthly Kurt', 'n'),
                 ('best_month', 'Best Month', 'p'),
                 ('worst_month', 'Worst Month', 'p'),
                 (None, None, None),
                 ('yearly_sharpe', 'Yearly Sharpe', 'n'),
                 ('yearly_mean', 'Yearly Mean', 'p'),
                 ('yearly_vol', 'Yearly Vol', 'p'),
                 ('yearly_skew', 'Yearly Skew', 'n'),
                 ('yearly_kurt', 'Yearly Kurt', 'n'),
                 ('best_year', 'Best Year', 'p'),
                 ('worst_year', 'Worst Year', 'p'),
                 (None, None, None),
                 ('avg_drawdown', 'Avg. Drawdown', 'p'),
                 ('avg_drawdown_days', 'Avg. Drawdown Days', 'n'),
                 ('avg_up_month', 'Avg. Up Month', 'p'),
                 ('avg_down_month', 'Avg. Down Month', 'p'),
                 ('win_year_perc', 'Win Year %', 'p'),
                 ('twelve_month_win_perc', 'Win 12m %', 'p')]

        for stat in stats:
            k, n, f = stat
            # blank row
            if k is None:
                row = [''] * len(data[0])
                data.append(row)
                continue

            row = [n]
            for key in self.keys():
                raw = getattr(self[key], k)
                if f is None:
                    row.append(raw)
                elif f == 'p':
                    row.append(fmtp(raw))
                elif f == 'n':
                    row.append(fmtn(raw))
                elif f == 'dt':
                    row.append(raw.strftime('%Y-%m-%d'))
                else:
                    raise NotImplementedError('unsupported format %s' % f)
            data.append(row)

        print tabulate(data, headers='firstrow')

    def plot(self, period='d', figsize=(15,5), title=None, logy=False, **kwargs):
        if title is None:
            title = '%s equity progression' % get_period_name(period)
        ser = self._get_series(period).rebase()
        ser.plot(figsize=figsize, logy=logy,
                 title=title, **kwargs)

    def plot_scatter_matrix(self, period='d', title=None, figsize=(10, 10), **kwargs):
        if title is None:
            title = '%s return scatter matrix' % get_period_name(period)

        plt.figure()
        ser = self._get_series(period).to_returns().dropna()
        ax = pd.scatter_matrix(ser, figsize=figsize, **kwargs)
        plt.suptitle(title)

    def _get_series(self, per):
        if per is 'd':
            return self.prices
        elif per is 'm':
            return self.prices.resample('M', 'last')
        elif per is 'y':
            return self.prices.resample('A', 'last')


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

    A PerformanceStats object will be returned containing all the stats.

    Args:
        * obj: A Pandas TimeSeries representing a series of prices.
    """
    return PerformanceStats(obj)


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


def merge(*series):
    dfs = []
    for s in series:
        if isinstance(s, pd.DataFrame):
            dfs.append(s)
        elif isinstance(s, pd.Series):
            tmpdf = pd.DataFrame({s.name: s})
            dfs.append(tmpdf)
        else:
            raise NotImplementedError('Unsupported merge type')

    return pd.concat(dfs, axis=1)


def extend_pandas():
    PandasObject.to_returns = to_returns
    PandasObject.to_log_returns = to_log_returns
    PandasObject.to_price_index = to_price_index
    PandasObject.rebase = rebase
    PandasObject.calc_perf_stats = calc_perf_stats
    PandasObject.to_drawdown_series = to_drawdown_series
    PandasObject.calc_max_drawdown = calc_max_drawdown
    PandasObject.calc_cagr = calc_cagr
