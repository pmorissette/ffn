import json
import os
import warnings
from typing import Sequence, Union
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance

import ffn

from . import utils


@utils.memoize
def get(
    tickers: Sequence[str],
    provider=None,
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
        if isinstance(data[ticker], pd.DataFrame):
            # newer yfinance returns as dataframe,
            # convert to series
            data[ticker] = data[ticker][data[ticker].columns[0]]

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


def web(ticker: str, field=None, start=None, end=None, mrefresh=False, source="yahoo"):
    """
    Data provider wrapper around pandas.io.data provider. Provides
    memoization.
    """
    if source == "yahoo":
        warnings.warn("web function is deprecated, as , use yf() instead")
        return yf(ticker, field, start, end, mrefresh)
    raise Exception("""pandas_datareader data readers are unmaintained and mostly broken, If you
                    still want them, go import the datareader directly from that library.
                    https://github.com/pydata/pandas-datareader/issues/977
                    """)


@utils.memoize
def yf(ticker: str, field, start=None, end=None, mrefresh=False) -> Union[pd.Series, pd.DataFrame]:
    if field is None:
        field = "Adj Close"

    tmp = yfinance.download(ticker, auto_adjust=False, start=start, end=end)

    if tmp is None:
        raise ValueError("failed to retrieve data for %s:%s" % (ticker, field))

    if field:
        return tmp[field]
    else:
        return tmp


@utils.memoize
def csv(ticker: str, path="data.csv", field="", mrefresh=False, **kwargs) -> pd.Series:
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


FXMACRODATA_API_BASE_URL = "https://fxmacrodata.com/api/v1"

_FXMACRODATA_INDICATOR_FIELDS = {
    "sma_20": "sma_20",
    "sma_50": "sma_50",
    "sma_200": "sma_200",
    "ema_12": "ema_12",
    "ema_20": "ema_20",
    "ema_26": "ema_26",
    "ema_50": "ema_50",
    "ema_200": "ema_200",
    "rsi_14": "rsi_14",
    "atr_14": "atr_14",
    "adx_14": "adx_14",
    "macd": "macd",
    "macd_signal": "macd",
    "macd_histogram": "macd",
    "bb_upper": "bollinger_bands",
    "bb_middle": "bollinger_bands",
    "bb_lower": "bollinger_bands",
}


def _format_fxmacrodata_date(value):
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def _split_fxmacrodata_pair(ticker: str) -> tuple[str, str]:
    normalized = "".join(char for char in ticker.upper() if char.isalpha())
    if len(normalized) != 6:
        raise ValueError("FXMacroData tickers must look like 'EURUSD' or 'EUR/USD'")
    return normalized[:3], normalized[3:]


def _read_fxmacrodata_error(error: HTTPError) -> str:
    try:
        body = error.read().decode("utf-8").strip()
    except Exception:
        body = ""
    message = f"FXMacroData API error {error.code}"
    if body:
        message = f"{message}: {body}"
    return message


@utils.memoize
def fxmacrodata(
    ticker: str,
    field=None,
    start=None,
    end=None,
    api_key=None,
    base_url: str = FXMACRODATA_API_BASE_URL,
    timeout: float = 30,
    mrefresh=False,
) -> pd.Series:
    """
    Data provider for FXMacroData daily FX spot rates.

    Use with :func:`ffn.get` by passing this function as the provider:

    >>> prices = ffn.get("EURUSD", provider=ffn.data.fxmacrodata, start="2024-01-01", api_key="...")

    ``ticker`` accepts six-character FX pairs such as ``EURUSD`` or separated
    forms such as ``EUR/USD``. By default the returned series contains the
    ``val`` field from the FXMacroData response. Technical fields such as
    ``sma_20`` or ``rsi_14`` can be requested via ffn's ticker field syntax,
    for example ``EURUSD:sma_20``. Pass ``api_key`` or set
    ``FXMACRODATA_API_KEY`` or ``FXMD_API_KEY`` for protected pairs.
    """
    base_currency, quote_currency = _split_fxmacrodata_pair(ticker)
    output_field = field or "val"

    params = {}
    start_date = _format_fxmacrodata_date(start)
    end_date = _format_fxmacrodata_date(end)
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    indicator = _FXMACRODATA_INDICATOR_FIELDS.get(output_field)
    if indicator:
        params["indicators"] = indicator

    api_key = api_key or os.getenv("FXMACRODATA_API_KEY") or os.getenv("FXMD_API_KEY")
    query = urlencode(params)
    url = f"{base_url.rstrip('/')}/forex/{base_currency.lower()}/{quote_currency.lower()}"
    if query:
        url = f"{url}?{query}"

    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as error:
        if error.code == 401:
            raise PermissionError(_read_fxmacrodata_error(error)) from error
        raise ValueError(_read_fxmacrodata_error(error)) from error

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("FXMacroData response did not include a data list")

    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_value = row.get("date")
        series_value = row.get(output_field)
        if date_value is None or series_value is None:
            continue
        records.append((pd.Timestamp(date_value), series_value))

    if not records:
        raise ValueError(f"FXMacroData response did not include dated '{output_field}' rows for {base_currency}/{quote_currency}")

    series = pd.Series(
        data=[value for _, value in records],
        index=pd.DatetimeIndex([date for date, _ in records]),
        name=ticker,
    )
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"FXMacroData response did not include numeric '{output_field}' rows for {base_currency}/{quote_currency}")

    return series[~series.index.duplicated(keep="last")].sort_index()


DEFAULT_PROVIDER = yf
