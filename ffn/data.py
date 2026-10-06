from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence
from importlib import metadata
from typing import Protocol

import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

import ffn

from . import utils


class DataProvider(Protocol):
    """Callable accepting ticker/field keywords and returning one price Series.

    Additional request options, including start/end, are passed unchanged.
    The response must have a unique, increasing DatetimeIndex without NaT,
    and real numeric values. Missing values and timezone-aware indexes are
    supported. Providers own authentication, transport, and optional caching.
    """

    def __call__(self, *, ticker: str, field: str | None = None, **kwargs) -> pd.Series: ...


def get_provider(name: str) -> DataProvider:
    """Load one callable registered in the ``ffn.providers`` entry-point group.

    Raises ValueError for unknown or duplicate names, and TypeError if the
    selected entry point is not callable. Unselected providers are not loaded.
    """
    entries = metadata.entry_points()
    if hasattr(entries, "select"):
        matches = list(entries.select(group="ffn.providers", name=name))
    else:  # Python 3.9's importlib.metadata returns a mapping.
        matches = [entry for entry in entries.get("ffn.providers", ()) if entry.name == name]
    if not matches:
        raise ValueError(f"Unknown data provider {name!r}; install its package or pass a callable")
    if len(matches) != 1:
        raise ValueError(f"Multiple data providers registered as {name!r}")
    provider = matches[0].load()
    if not callable(provider):
        raise TypeError(f"Data provider {name!r} must be callable")
    return provider


def _validate_response(series, ticker):
    if not isinstance(series, pd.Series):
        raise TypeError(f"Provider response for {ticker!r} must be a Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"Provider response for {ticker!r} must have a DatetimeIndex")
    if series.index.hasnans:
        raise ValueError(f"Provider response for {ticker!r} must not contain NaT dates")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError(f"Provider response for {ticker!r} must have unique, increasing dates")
    if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype) or is_complex_dtype(series.dtype):
        raise TypeError(f"Provider response for {ticker!r} must contain real numeric values")


def get(
    tickers: Sequence[str],
    provider: DataProvider | str | None = None,
    common_dates=True,
    forward_fill=False,
    clean_tickers=True,
    column_names=None,
    ticker_field_sep=":",
    mrefresh=False,
    existing=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Helper function for retrieving data as a DataFrame.

    Args:
        * tickers (list, string, csv string): Tickers to download.
        * provider (callable, str): A DataProvider or an ``ffn.providers``
            entry-point name. Defaults to ffn.data.DEFAULT_PROVIDER.
            An unset default falls back to Yahoo with a deprecation warning.
        * common_dates (bool): Keep common dates only? Drop na's.
        * forward_fill (bool): forward fill values if missing. Only works
            if common_dates is False, since common_dates will remove
            all nan's, so no filling forward necessary.
        * clean_tickers (bool): Should the tickers be 'cleaned' using
            ffn.utils.clean_tickers? Basically remove non-standard
            characters (^VIX -> vix) and standardize to lower case.
            Raises ValueError if distinct columns would clean to the
            same name; disable cleaning or provide column_names to
            preserve their identities.
        * column_names (list): List of column names if clean_tickers
            is not satisfactory.
        * ticker_field_sep (char): separator used to determine the
            ticker and field. This is in case we want to specify
            particular, non-default fields. For example, we might
            want: AAPL:Low,AAPL:High,AAPL:Close. ':' is the separator.
        * mrefresh (bool): Request a refresh from providers that explicitly
            declare an mrefresh keyword. Legacy providers without it are
            called normally. This function does not cache requests or results.
        * existing (DataFrame): Existing DataFrame to append returns
            to - used when we download from multiple sources
        * kwargs: passed to provider

    """

    if provider is None:
        provider = DEFAULT_PROVIDER
    if provider is None:
        warnings.warn(
            "Implicit Yahoo data access is deprecated; pass provider='yahoo' or configure ffn.data.DEFAULT_PROVIDER",
            DeprecationWarning,
            stacklevel=2,
        )
        provider = "yahoo"
    if isinstance(provider, str):
        provider = get_provider(provider)
    if not callable(provider):
        raise TypeError("provider must be callable or a registered provider name")

    if mrefresh:
        try:
            refresh_parameter = inspect.signature(provider).parameters.get("mrefresh")
        except (TypeError, ValueError):
            refresh_parameter = None
        if refresh_parameter is not None and refresh_parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            kwargs["mrefresh"] = True

    tickers = utils.parse_arg(tickers)

    data = {}
    for ticker in tickers:
        t = ticker
        f = None

        # check for field
        bits = ticker.split(ticker_field_sep, 1)
        if len(bits) == 2:
            t = bits[0]
            f = bits[1]

        data[ticker] = provider(ticker=t, field=f, **kwargs)
        _validate_response(data[ticker], ticker)

    df = pd.DataFrame(data).copy(deep=True)
    df.index = df.index.copy(deep=True)

    # ensure same order as provided
    df = df[tickers]

    if existing is not None:
        df = ffn.merge(existing, df)

    if common_dates:
        df = df.dropna()

    if forward_fill:
        df = df.ffill()

    if column_names:
        cnames = utils.parse_arg(column_names)
        if len(cnames) != len(df.columns):
            raise ValueError("column_names must be of same length as tickers")
        df.columns = cnames
    elif clean_tickers:
        cleaned_columns = [utils.clean_ticker(column) for column in df.columns]
        if len(set(cleaned_columns)) < len(set(df.columns)):
            raise ValueError("cleaned ticker names are not unique; set clean_tickers=False or provide unique column_names")
        df.columns = cleaned_columns

    return df


def web(ticker: str, field=None, start=None, end=None, mrefresh=False, source="yahoo"):
    """
    Data provider wrapper around pandas.io.data provider. Provides
    memoization.
    """
    if source == "yahoo":
        warnings.warn("web function is deprecated, as , use yf() instead")
        return yf(ticker, field, start, end, mrefresh)
    raise ValueError("""pandas_datareader data readers are unmaintained and mostly broken, If you
                    still want them, go import the datareader directly from that library.
                    https://github.com/pydata/pandas-datareader/issues/977
                    """)


@utils.memoize
def yf(ticker: str, field=None, start=None, end=None, mrefresh=False) -> pd.Series:
    """Memoized compatibility wrapper for :func:`ffn.yahoo.download`."""
    from .yahoo import download

    return download(ticker, field=field, start=start, end=end)


@utils.memoize
def csv(ticker: str, path="data.csv", field="", mrefresh=False, start=None, end=None, **kwargs) -> pd.Series:
    """
    Data provider wrapper around pandas' read_csv. Provides memoization.
    The date window includes start and excludes end. Duplicate dates keep
    the last observation, and the result is sorted by date.
    """
    # set defaults if not specified
    if "index_col" not in kwargs:
        kwargs["index_col"] = 0
    if "parse_dates" not in kwargs:
        kwargs["parse_dates"] = True

    # read in dataframe from csv file
    df = pd.read_csv(path, **kwargs)

    tf = ticker
    if field != "" and field is not None:
        tf = f"{tf}:{field}"

    # check that required column exists
    if tf not in df:
        raise ValueError("Ticker(field) not present in csv file!")

    series = df[tf]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if start is not None:
        series = series.loc[series.index >= pd.Timestamp(start)]
    if end is not None:
        series = series.loc[series.index < pd.Timestamp(end)]
    return series


DEFAULT_PROVIDER: DataProvider | str | None = None
