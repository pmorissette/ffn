import re

import decorator
import numpy as np
import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle


def _memoize(func, *args, **kw):
    # should we refresh the cache?
    refresh = False
    refresh_kw = func.mrefresh_keyword

    # kw is not always set - check args
    if refresh_kw in func.__code__.co_varnames:
        if args[func.__code__.co_varnames.index(refresh_kw)]:
            refresh = True

    # check in kw if not already set above
    if not refresh and refresh_kw in kw:
        if kw[refresh_kw]:
            refresh = True

    key = pickle.dumps(args, 1) + pickle.dumps(kw, 1)

    cache = func.mcache
    if not refresh and key in cache:
        return cache[key]
    else:
        cache[key] = result = func(*args, **kw)
        return result


def memoize(f, refresh_keyword="mrefresh"):
    """
    Memoize decorator. The refresh keyword is the keyword
    used to bypass the cache (in the function call).
    """
    f.mcache = {}
    f.mrefresh_keyword = refresh_keyword
    return decorator.decorator(_memoize, f)


def parse_arg(arg):
    """
    Parses arguments for convenience. Argument can be a
    csv list ('a,b,c'), a string, a list, a tuple.

    Returns a list.
    """
    # handle string input
    if type(arg) == str:
        arg = arg.strip()
        # parse csv as tickers and create children
        if "," in arg:
            arg = arg.split(",")
            arg = [x.strip() for x in arg]
        # assume single string - create single item list
        else:
            arg = [arg]

    return arg


def clean_ticker(ticker):
    """
    Cleans a ticker for easier use throughout MoneyTree

    Splits by space and only keeps first bit. Also removes
    any characters that are not letters. Returns as lowercase.

    >>> clean_ticker('^VIX')
    'vix'
    >>> clean_ticker('SPX Index')
    'spx'
    """
    pattern = re.compile("[\\W_]+")
    res = pattern.sub("", ticker.split(" ")[0])
    return res.lower()


def clean_tickers(tickers):
    """
    Maps clean_ticker over tickers.
    """
    return [clean_ticker(x) for x in tickers]


def fmtp(number):
    """
    Formatting helper - percent
    """
    if np.isnan(number):
        return "-"
    return format(number, ".2%")


def fmtpn(number):
    """
    Formatting helper - percent no % sign
    """
    if np.isnan(number):
        return "-"
    return format(number * 100, ".2f")


def fmtn(number):
    """
    Formatting helper - float
    """
    if np.isnan(number):
        return "-"
    return format(number, ".2f")


def get_freq_name(period):
    period = period.upper()
    periods = {
        "B": "business day",
        "C": "custom business day",
        "D": "daily",
        "W": "weekly",
        "M": "monthly",
        "BM": "business month end",
        "CBM": "custom business month end",
        "MS": "month start",
        "BMS": "business month start",
        "CBMS": "custom business month start",
        "Q": "quarterly",
        "BQ": "business quarter end",
        "QS": "quarter start",
        "BQS": "business quarter start",
        "Y": "yearly",
        "A": "yearly",
        "BA": "business year end",
        "AS": "year start",
        "BAS": "business year start",
        "H": "hourly",
        "T": "minutely",
        "S": "secondly",
        "L": "milliseonds",
        "U": "microseconds",
    }

    if period in periods:
        return periods[period]
    else:
        return None


def scale(val, src, dst):
    """
    Scale value from src range to dst range.
    If value outside bounds, it is clipped and set to
    the low or high bound of dst.

    Ex:
        scale(0, (0.0, 99.0), (-1.0, 1.0)) == -1.0
        scale(-5, (0.0, 99.0), (-1.0, 1.0)) == -1.0

    """
    if val < src[0]:
        return dst[0]
    if val > src[1]:
        return dst[1]

    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


def as_percent(self, digits=2):
    return as_format(self, ".%s%%" % digits)


def as_format(item, format_str=".2f"):
    """
    Map a format string over a pandas object.
    """
    if isinstance(item, pd.Series):
        return item.map(lambda x: format(x, format_str))
    elif isinstance(item, pd.DataFrame):
        return item.applymap(lambda x: format(x, format_str))
