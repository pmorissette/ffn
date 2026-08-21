from __future__ import annotations

import pickle
import re
from collections.abc import Sequence

import decorator
import pandas as pd
from packaging.version import Version


def _memoize(func, *args, **kw):
    # should we refresh the cache?
    refresh = False
    refresh_kw = func.mrefresh_keyword

    # kw is not always set - check args
    positional_vars = func.__code__.co_varnames[: func.__code__.co_argcount]
    if refresh_kw in positional_vars:
        refresh_idx = positional_vars.index(refresh_kw)
        if refresh_idx < len(args) and args[refresh_idx]:
            refresh = True

    # check in kw if not already set above
    if not refresh and refresh_kw in kw and kw[refresh_kw]:
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


def parse_arg(arg: str | list[str] | tuple[str]):
    """
    Parses arguments for convenience. Argument can be a
    csv list ('a,b,c'), a string, a list, a tuple.

    Returns a list.
    """
    # handle string input
    if isinstance(arg, str):
        arg = arg.strip()
        # parse csv as tickers and create children
        if "," in arg:
            arg = arg.split(",")
            arg = [x.strip() for x in arg]
        # assume single string - create single item list
        else:
            arg = [arg]

    return arg


def clean_ticker(ticker: str) -> str:
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


def clean_tickers(tickers: Sequence[str]) -> list[str]:
    """
    Maps clean_ticker over tickers.
    """
    return [clean_ticker(x) for x in tickers]


def fmtp(number: float) -> str:
    """
    Formatting helper - percent
    """
    if pd.isna(number):
        return "-"
    return format(number, ".2%")


def fmtpn(number: float) -> str:
    """
    Formatting helper - percent no % sign
    """
    if pd.isna(number):
        return "-"
    return format(number * 100, ".2f")


def fmtn(number: float) -> str:
    """
    Formatting helper - float
    """
    if pd.isna(number):
        return "-"
    return format(number, ".2f")


def get_freq_name(period: str) -> str | None:
    # pandas anchors some aliases to a month or weekday, e.g. "YE-DEC" or "W-SUN"
    base = period.split("-", 1)[0]

    # These aliases are case sensitive in pandas: "ms" is milliseconds while
    # "MS" is month start, so they have to be matched before upper casing
    case_sensitive = {
        "min": "minutely",
        "ms": "milliseconds",
        "us": "microseconds",
        "ns": "nanoseconds",
    }
    if base in case_sensitive:
        return case_sensitive[base]

    periods = {
        "B": "business day",
        "C": "custom business day",
        "D": "daily",
        "W": "weekly",
        "M": "monthly",
        "ME": "monthly",
        "BM": "business month end",
        "BME": "business month end",
        "CBM": "custom business month end",
        "CBME": "custom business month end",
        "MS": "month start",
        "BMS": "business month start",
        "CBMS": "custom business month start",
        "Q": "quarterly",
        "QE": "quarterly",
        "BQ": "business quarter end",
        "BQE": "business quarter end",
        "QS": "quarter start",
        "BQS": "business quarter start",
        "Y": "yearly",
        "YE": "yearly",
        "A": "yearly",
        "BA": "business year end",
        "BY": "business year end",
        "BYE": "business year end",
        "AS": "year start",
        "YS": "year start",
        "BAS": "business year start",
        "BYS": "business year start",
        "H": "hourly",
        "T": "minutely",
        "S": "secondly",
        "L": "milliseconds",
        "U": "microseconds",
        "N": "nanoseconds",
    }

    return periods.get(base.upper())


def scale(val: float, src: Sequence[float], dst: Sequence[float]) -> float:
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
    return as_format(self, f".{digits}%")


def as_format(item: pd.DataFrame | pd.Series, format_str=".2f") -> pd.DataFrame | pd.Series:
    """
    Map a format string over a pandas object.
    """
    PANDAS_VERSION = Version(pd.__version__)
    PANDAS_210 = PANDAS_VERSION >= Version("2.1.0")

    select_map = "map"
    if not PANDAS_210:
        select_map = "applymap"

    if isinstance(item, pd.Series):
        return item.map(lambda x: format(x, format_str))
    elif isinstance(item, pd.DataFrame):
        return getattr(item, select_map)(lambda x: format(x, format_str))
