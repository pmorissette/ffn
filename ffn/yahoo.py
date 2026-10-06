"""Optional Yahoo Finance provider, registered through ``ffn.providers``."""

from __future__ import annotations

import pandas as pd


def download(ticker: str, field=None, start=None, end=None, **kwargs) -> pd.Series:
    """Fetch one Yahoo series using the optional yfinance package.

    Defaults to Adj Close with auto_adjust=False. Start is inclusive and end
    exclusive. Extra options are forwarded to yfinance.download. Responses
    are sorted by date, with the last observation kept for duplicate dates.
    This adapter does not cache results or require ffn-specific utilities.
    """
    field = "Adj Close" if field is None else field
    data = _download(ticker, start=start, end=end, **kwargs)
    if isinstance(data.columns, pd.MultiIndex) and kwargs.get("group_by") == "ticker":
        data = data.swaplevel(axis=1)
    if field not in data:
        raise ValueError(f"Yahoo response does not contain field {field!r} for {ticker!r}")
    series = data[field]
    if isinstance(series, pd.DataFrame):
        if len(series.columns) != 1:
            raise ValueError("Yahoo provider requires one series per ticker/field request")
        series = series.iloc[:, 0]
    return series[~series.index.duplicated(keep="last")].sort_index().rename(ticker)


def _legacy_download(ticker, field=None, start=None, end=None):
    field = "Adj Close" if field is None else field
    data = _download(ticker, start=start, end=end)
    return data[field] if field else data


def _download(ticker, start=None, end=None, **kwargs):
    try:
        import yfinance
    except ModuleNotFoundError as error:
        if error.name != "yfinance":
            raise
        raise ImportError("Yahoo data requires yfinance; install 'ffn[yahoo]' or choose another provider") from error

    kwargs.setdefault("auto_adjust", False)
    data = yfinance.download(ticker, start=start, end=end, **kwargs)
    if data is None:
        raise ValueError(f"failed to retrieve data for {ticker}")
    return data
