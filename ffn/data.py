import ffn
import pandas as pd
import yfinance
from packaging.version import Version
from pandas_datareader import data as pdata

# import ffn.utils as utils
from . import utils

# This is a temporary fix until pandas_datareader 0.7 is released.
# pandas 0.23 has moved is_list_like from common to api.types, hence the monkey patch
if Version(pd.__version__) > Version("0.23.0"):
    pd.core.common.is_list_like = pd.api.types.is_list_like


@utils.memoize
def get(
    tickers,
    provider=None,
    common_dates=True,
    forward_fill=False,
    clean_tickers=True,
    column_names=None,
    ticker_field_sep=":",
    mrefresh=False,
    existing=None,
    **kwargs
):
    """
    Helper function for retrieving data as a DataFrame.

    Args:
        * tickers (list, string, csv string): Tickers to download.
        * provider (function): Provider to use for downloading data.
            By default it will be ffn.DEFAULT_PROVIDER if not provided.
        * common_dates (bool): Keep common dates only? Drop na's.
        * forward_fill (bool): forward fill values if missing. Only works
            if common_dates is False, since common_dates will remove
            all nan's, so no filling forward necessary.
        * clean_tickers (bool): Should the tickers be 'cleaned' using
            ffn.utils.clean_tickers? Basically remove non-standard
            characters (^VIX -> vix) and standardize to lower case.
        * column_names (list): List of column names if clean_tickers
            is not satisfactory.
        * ticker_field_sep (char): separator used to determine the
            ticker and field. This is in case we want to specify
            particular, non-default fields. For example, we might
            want: AAPL:Low,AAPL:High,AAPL:Close. ':' is the separator.
        * mrefresh (bool): Ignore memoization.
        * existing (DataFrame): Existing DataFrame to append returns
            to - used when we download from multiple sources
        * kwargs: passed to provider

    """

    if provider is None:
        provider = DEFAULT_PROVIDER

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

        # call provider - check if supports memoization
        if hasattr(provider, "mcache"):
            data[ticker] = provider(ticker=t, field=f, mrefresh=mrefresh, **kwargs)
        else:
            data[ticker] = provider(ticker=t, field=f, **kwargs)

        data[ticker] = data[ticker][~data[ticker].index.duplicated(keep="last")]

    df = pd.DataFrame(data)

    # ensure same order as provided
    df = df[tickers]

    if existing is not None:
        df = ffn.merge(existing, df)

    if common_dates:
        df = df.dropna()

    if forward_fill:
        df = df.fillna(method="ffill")

    if column_names:
        cnames = utils.parse_arg(column_names)
        if len(cnames) != len(df.columns):
            raise ValueError("column_names must be of same length as tickers")
        df.columns = cnames
    elif clean_tickers:
        df.columns = map(utils.clean_ticker, df.columns)

    return df


@utils.memoize
def web(ticker, field=None, start=None, end=None, mrefresh=False, source="yahoo"):
    """
    Data provider wrapper around pandas.io.data provider. Provides
    memoization.
    """
    if source == "yahoo" and field is None:
        field = "Adj Close"

    tmp = _download_web(ticker, data_source=source, start=start, end=end)

    if tmp is None:
        raise ValueError("failed to retrieve data for %s:%s" % (ticker, field))

    if field:
        return tmp[field]
    else:
        return tmp


@utils.memoize
def _download_web(name, **kwargs):
    """
    Thin wrapper to enable memoization
    """
    return pdata.DataReader(name, **kwargs)


@utils.memoize
def yf(ticker, field, start=None, end=None, mrefresh=False):
    if field is None:
        field = "Adj Close"

    yfinance.pdr_override()

    tmp = pdata.get_data_yahoo(ticker, start=start, end=end)

    if tmp is None:
        raise ValueError("failed to retrieve data for %s:%s" % (ticker, field))

    if field:
        return tmp[field]
    else:
        return tmp


@utils.memoize
def csv(ticker, path="data.csv", field="", mrefresh=False, **kwargs):
    """
    Data provider wrapper around pandas' read_csv. Provides memoization.
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
        tf = "%s:%s" % (tf, field)

    # check that required column exists
    if tf not in df:
        raise ValueError("Ticker(field) not present in csv file!")

    return df[tf]


DEFAULT_PROVIDER = yf
