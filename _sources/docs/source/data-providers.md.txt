# How to use a data provider

## Fetch Yahoo data

Install the optional Yahoo dependency:

```bash
pip install 'ffn[yahoo]'
```

Select the registered provider:

```python
import ffn

prices = ffn.get("AAPL,MSFT", provider="yahoo", start="2024-01-01", end="2025-01-01")
```

To set a process-wide default instead, assign `ffn.data.DEFAULT_PROVIDER = "yahoo"`.
The same setting accepts a callable or another registered name.

## Supply your own data

Wrap your data source in a callable accepting `ticker` and `field` keywords.
Return one numeric Series with unique, increasing dates:

```python
import pandas as pd
import ffn

history = pd.DataFrame(
    {"ABC": [100.0, 101.0, 102.0]},
    index=pd.date_range("2024-01-01", periods=3),
)

def local_prices(ticker, field=None, start=None, end=None):
    if field is not None:
        raise ValueError("This source has only a default price field")
    series = history[ticker]
    if start is not None:
        series = series.loc[series.index >= pd.Timestamp(start)]
    if end is not None:
        series = series.loc[series.index < pd.Timestamp(end)]
    return series

prices = ffn.get("ABC", provider=local_prices, start="2024-01-02")
assert prices["abc"].tolist() == [101.0, 102.0]
```

Keep authentication, request construction, pagination, and response parsing in
your provider. See the [provider contract](provider-contract.md) for validation
rules and cache-refresh behavior.

## Register a standalone provider

Put the callable in an installable package, for example `my_feed.provider:download`.
Register it in that package's `pyproject.toml`:

```toml
[project.entry-points."ffn.providers"]
my_feed = "my_feed.provider:download"
```

Install the package in the same environment as ffn, then select it by name:

```python
prices = ffn.get("ABC", provider="my_feed", start="2024-01-01")
```

Use `ffn.data.get_provider("my_feed")` to verify that the registration resolves.
No registration call or change to ffn's source is needed.

For a working adapter to copy, see
[`ffn/yahoo.py`](https://github.com/pmorissette/ffn/blob/master/ffn/yahoo.py).
It imports yfinance only when called and has no dependency on ffn's dispatcher
or caching utilities. ffn registers it in its own `pyproject.toml` using the same
entry-point group:

```toml
[project.entry-points."ffn.providers"]
yahoo = "ffn.yahoo:download"
```
