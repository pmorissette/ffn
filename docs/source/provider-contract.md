# Provider contract

## Requests

`ffn.data.DataProvider` describes the callable interface:

```python
def provider(ticker: str, field: str | None = None, **options) -> pandas.Series:
    ...
```

`ffn.get` calls the provider once for each requested ticker/field pair, using
keyword arguments. `ABC:Close` becomes `ticker="ABC", field="Close"`; `ABC`
uses `field=None`. The default field belongs to the provider.

Other keyword arguments are passed unchanged. Providers accepting `start` and
`end` use an inclusive start and exclusive end. Credentials and client objects
can be configured on a callable instance or closure. Providers may reject
options they do not support; their exceptions propagate unchanged.

## Responses

Each response is one pandas Series with:

- A real numeric dtype, including nullable numeric dtypes; booleans, complex
  values, and numeric strings are rejected.
- A unique, increasing DatetimeIndex without NaT dates. Timezones are preserved.
- Missing values where observations are unavailable. An empty numeric Series
  with an empty DatetimeIndex is valid.

The Series name is not used to select the output column. ffn uses the requested
ticker/field string, then applies `column_names` or `clean_tickers`.
DataFrames are rejected, including single-column frames. Each adapter is
responsible for extracting the requested series and resolving duplicate dates.
Responses combined in one call must use compatible timezone conventions.

`common_dates=True` drops rows containing missing values. Otherwise
`forward_fill=True` fills gaps after alignment. Neither option changes the
provider's data. ffn does not convert timezones or resample observations.
Each response is copied before the next provider call, so providers may reuse
their data buffers.

## Provider selection

`provider` accepts a callable or an entry-point name in the `ffn.providers`
group. If omitted, `ffn.data.DEFAULT_PROVIDER` supplies either value. If that
setting is also `None`, implicit Yahoo selection emits a DeprecationWarning.

Named providers are resolved once per `get` call. Only the selected entry point
is loaded. Missing or duplicate names raise ValueError; a non-callable entry
point raises TypeError. Provider import errors propagate.

ffn's own registrations are `yahoo = "ffn.yahoo:download"` and
`csv = "ffn.data:csv"`. The Yahoo provider requires the optional `ffn[yahoo]`
extra. Importing ffn and using other providers does not require yfinance.

## Caching and refresh

`ffn.get` does not memoize calls, serialize request options, or retain credentials.
Providers own caching, expiration, and cache isolation between accounts.

For compatibility, `mrefresh=True` is forwarded if the callable explicitly
declares an `mrefresh` keyword parameter. A bare `**kwargs` does not opt in.
Providers without that parameter are called normally. Providers with an
uninspectable signature can be wrapped in a function declaring `mrefresh`.

The registered Yahoo provider makes a fresh request on every call. The CSV
provider and legacy `ffn.data.yf` wrapper retain their provider-local caches and
support `mrefresh=True`.

Direct calls to `ffn.data.yf` and the deprecated `ffn.data.web` preserve Yahoo's
original Series or DataFrame shape. An empty field string returns the full
frame. These legacy helpers do not normalize dates or columns; the registered
`yahoo` provider returns the single, normalized Series required by `ffn.get`.

## Compatibility changes

- `ffn.get.mcache` is removed. Cache control belongs to the provider.
- Returning a DataFrame, unsorted dates, or duplicate dates now raises an error
  instead of selecting the first column or dropping duplicates in `get`.
- Yahoo-specific response normalization is in `ffn.yahoo`, not in the dispatcher.
- Calls without an explicit or configured provider are deprecated. They still
  select Yahoo while that optional dependency is available.

See [how to use a data provider](data-providers.md) for configuration and packaging examples.
