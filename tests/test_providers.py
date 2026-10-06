import subprocess
import sys
from importlib.metadata import EntryPoint
from types import SimpleNamespace

import pandas as pd
import pytest

import ffn


class EntryPoints(tuple):
    def select(self, **filters):
        return EntryPoints(entry for entry in self if all(getattr(entry, key) == value for key, value in filters.items()))


def prices(ticker, field=None, **kwargs):
    return pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2), name=ticker)


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_provider_entry_points_load_only_selected_provider(monkeypatch, legacy_metadata):
    entries = EntryPoints(
        [
            EntryPoint(name="external", value="vendor:prices", group="ffn.providers"),
            EntryPoint(name="unused", value="missing_vendor:prices", group="ffn.providers"),
            EntryPoint(name="external", value="unrelated:prices", group="other.plugins"),
        ]
    )
    loaded = []

    def load(entry):
        loaded.append(entry.value)
        assert entry.value == "vendor:prices"
        return prices

    discovered = {"ffn.providers": entries.select(group="ffn.providers")} if legacy_metadata else entries
    monkeypatch.setattr(ffn.data.metadata, "entry_points", lambda: discovered)
    monkeypatch.setattr(EntryPoint, "load", load)

    result = ffn.get("ABC,DEF", provider="external")

    assert result.columns.tolist() == ["abc", "def"]
    assert result.to_numpy().tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert loaded == ["vendor:prices"]


@pytest.mark.parametrize("count", [0, 2])
def test_provider_entry_points_reject_unknown_or_ambiguous_names(monkeypatch, count):
    entries = EntryPoints([EntryPoint(name="external", value=f"vendor{index}:prices", group="ffn.providers") for index in range(count)])
    monkeypatch.setattr(ffn.data.metadata, "entry_points", lambda: entries)
    monkeypatch.setattr(EntryPoint, "load", lambda entry: pytest.fail("must not load an ambiguous provider"))

    with pytest.raises(ValueError, match="Unknown|Multiple"):
        ffn.get("ABC", provider="external")


def test_provider_entry_points_require_callable(monkeypatch):
    entries = EntryPoints([EntryPoint(name="external", value="vendor:prices", group="ffn.providers")])
    monkeypatch.setattr(ffn.data.metadata, "entry_points", lambda: entries)
    monkeypatch.setattr(EntryPoint, "load", lambda entry: object())

    with pytest.raises(TypeError, match="callable"):
        ffn.get("ABC", provider="external")


def test_installed_provider_entry_points():
    from ffn.yahoo import download

    assert ffn.data.get_provider("yahoo") is download
    assert ffn.data.get_provider("csv") is ffn.data.csv


def test_external_package_provider_registration(tmp_path, monkeypatch):
    distribution = tmp_path / "external_feed-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Metadata-Version: 2.1\nName: external-feed\nVersion: 1.0\n")
    (distribution / "entry_points.txt").write_text("[ffn.providers]\nexternal_test_feed = external_test_feed:download\n")
    (tmp_path / "external_test_feed.py").write_text(
        "import pandas as pd\ndef download(ticker, field=None, **kwargs):\n    return pd.Series([42.0], index=pd.date_range('2024-01-01', periods=1))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        result = ffn.get("ABC", provider="external_test_feed")
        assert result["abc"].tolist() == [42.0]
    finally:
        sys.modules.pop("external_test_feed", None)


@pytest.mark.parametrize("provider", [1, object(), False])
def test_get_rejects_noncallable_provider(provider):
    with pytest.raises(TypeError, match="callable"):
        ffn.get("ABC", provider=provider)


@pytest.mark.parametrize(
    "result, error, message",
    [
        (None, TypeError, "Series"),
        (prices("ABC").to_frame(), TypeError, "Series"),
        (pd.concat([prices("ABC"), prices("DEF")], axis=1), TypeError, "Series"),
        (pd.Series([1.0, 2.0]), TypeError, "DatetimeIndex"),
        (pd.Series(["1", "2"], index=prices("ABC").index), TypeError, "real numeric"),
        (prices("ABC").astype(bool), TypeError, "real numeric"),
        (prices("ABC").astype(complex), TypeError, "real numeric"),
        (prices("ABC").iloc[::-1], ValueError, "increasing"),
        (pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"])), ValueError, "unique"),
        (pd.Series([1.0], index=pd.DatetimeIndex([pd.NaT])), ValueError, "NaT"),
    ],
)
def test_get_validates_provider_response(result, error, message):
    with pytest.raises(error, match=message):
        ffn.get("ABC", provider=lambda ticker, field: result)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64"])
def test_get_preserves_valid_provider_response(monkeypatch, dtype):
    index = pd.date_range("2024-01-01", periods=3, tz="America/New_York", name="date")
    provided = pd.Series([1, None, 3], index=index, dtype=dtype, name="source")
    before = provided.copy(deep=True)
    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", lambda ticker, field: provided)

    result = ffn.get("ABC:Close", common_dates=False, clean_tickers=False)

    pd.testing.assert_series_equal(result["ABC:Close"], before.rename("ABC:Close"))
    result.iloc[0, 0] = -1
    result.index.freq = None
    pd.testing.assert_series_equal(provided, before)


def test_get_does_not_cache_credentials_or_provider_state(monkeypatch):
    calls = []

    def provider(ticker, field, api_key):
        calls.append(api_key)
        return prices(ticker) * len(calls)

    first = ffn.get("ABC", provider=provider, api_key="synthetic-key")
    second = ffn.get("ABC", provider=provider, api_key="synthetic-key")
    assert first.iloc[0, 0] == 1.0
    assert second.iloc[0, 0] == 2.0
    assert calls == ["synthetic-key", "synthetic-key"]
    assert not hasattr(ffn.get, "mcache")

    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", lambda ticker, field: prices(ticker))
    assert ffn.get("ABC").iloc[0, 0] == 1.0
    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", lambda ticker, field: prices(ticker) * 3)
    assert ffn.get("ABC").iloc[0, 0] == 3.0


def test_get_forwards_declared_refresh_without_memoize():
    calls = []

    def provider(ticker, field, mrefresh=False):
        calls.append(mrefresh)
        return prices(ticker)

    ffn.get("ABC", provider=provider)
    ffn.get("ABC", provider=provider, mrefresh=True)
    assert calls == [False, True]


def test_get_does_not_serialize_provider_options():
    class Client:
        def __reduce__(self):
            raise RuntimeError("must not serialize a client")

    client = Client()

    def provider(ticker, field, session):
        assert session is client
        return prices(ticker)

    assert ffn.get("ABC", provider=provider, session=client).iloc[0, 0] == 1.0


def test_get_observes_changed_provider_credentials(monkeypatch):
    import os

    calls = []

    def provider(ticker, field):
        calls.append(os.environ["EXAMPLE_PROVIDER_KEY"])
        return prices(ticker) * len(calls)

    monkeypatch.setenv("EXAMPLE_PROVIDER_KEY", "first-synthetic-key")
    assert ffn.get("ABC", provider=provider).iloc[0, 0] == 1.0
    monkeypatch.setenv("EXAMPLE_PROVIDER_KEY", "second-synthetic-key")
    assert ffn.get("ABC", provider=provider).iloc[0, 0] == 2.0
    assert calls == ["first-synthetic-key", "second-synthetic-key"]


def test_get_passes_request_options_and_exceptions_unchanged():
    token = object()
    failure = RuntimeError("provider failure")
    seen = []

    def provider(**kwargs):
        seen.append(kwargs)
        raise failure

    with pytest.raises(RuntimeError) as raised:
        ffn.get("ABC:Close", provider=provider, start="2024-01-01", end="2024-02-01", option=token)
    assert raised.value is failure
    assert seen == [{"ticker": "ABC", "field": "Close", "start": "2024-01-01", "end": "2024-02-01", "option": token}]


def test_get_deprecates_implicit_yahoo_provider(monkeypatch):
    names = []
    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", None)
    monkeypatch.setattr(ffn.data, "get_provider", lambda name: names.append(name) or prices)

    with pytest.warns(DeprecationWarning, match="provider"):
        assert ffn.get("ABC").iloc[0, 0] == 1.0
    assert names == ["yahoo"]
    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", "external")
    assert ffn.get("ABC").iloc[0, 0] == 1.0
    assert names == ["yahoo", "external"]


def test_core_and_custom_provider_work_without_yfinance():
    script = """
import importlib.abc
import sys
class NoYahoo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'yfinance':
            raise ModuleNotFoundError('no yfinance', name='yfinance')
sys.meta_path.insert(0, NoYahoo())
import ffn
import pandas as pd
series = pd.Series([100.0, 110.0], index=pd.date_range('2024-01-01', periods=2))
assert ffn.get('ABC', provider=lambda ticker, field: series).iloc[0, 0] == 100.0
assert abs(series.calc_total_return() - 0.1) < 1e-10
assert 'ffn.yahoo' not in sys.modules
try:
    ffn.get('ABC', provider='yahoo')
except ImportError as error:
    assert 'ffn[yahoo]' in str(error)
else:
    raise AssertionError('expected optional dependency error')
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("multiindex", [False, True])
def test_yahoo_adapter_normalizes_response(monkeypatch, multiindex):
    from ffn.yahoo import download

    dates = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"])
    columns = pd.MultiIndex.from_tuples([("Adj Close", "ABC"), ("Volume", "ABC")]) if multiindex else ["Adj Close", "Volume"]
    raw = pd.DataFrame([[2.0, 20.0], [0.0, 0.0], [1.0, 10.0]], index=dates, columns=columns)
    before = raw.copy(deep=True)
    calls = []

    def fetch(*args, **kwargs):
        calls.append((args, kwargs))
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fetch))
    result = download("ABC", start="2024-01-01", end="2024-01-03", interval="1d")
    pd.testing.assert_series_equal(result, prices("ABC"), check_freq=False)
    assert calls == [(("ABC",), {"auto_adjust": False, "start": "2024-01-01", "end": "2024-01-03", "interval": "1d"})]
    pd.testing.assert_frame_equal(raw, before)


def test_yahoo_adapter_rejects_ambiguous_response(monkeypatch):
    from ffn.yahoo import download

    raw = pd.DataFrame([[1.0, 2.0]], index=pd.date_range("2024-01-01", periods=1), columns=pd.MultiIndex.from_tuples([("Adj Close", "ABC"), ("Adj Close", "DEF")]))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *args, **kwargs: raw))
    with pytest.raises(ValueError, match="one series"):
        download("ABC DEF")


@pytest.mark.parametrize("raw", [None, pd.DataFrame(), prices("ABC").rename("Close").to_frame()])
def test_yahoo_adapter_rejects_missing_response_or_field(monkeypatch, raw):
    from ffn.yahoo import download

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *args, **kwargs: raw))
    with pytest.raises(ValueError, match="failed to retrieve|does not contain field"):
        download("ABC")


def test_yahoo_adapter_forwards_adjustment_and_field(monkeypatch):
    from ffn.yahoo import download

    calls = []

    def fetch(ticker, **kwargs):
        calls.append(kwargs)
        return prices(ticker).rename("Close").to_frame()

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fetch))
    pd.testing.assert_series_equal(download("ABC", field="Close", auto_adjust=True), prices("ABC"))
    assert calls == [{"start": None, "end": None, "auto_adjust": True}]


def test_yahoo_compatibility_wrapper(monkeypatch):
    from ffn import yahoo

    calls = []
    monkeypatch.setattr(yahoo, "download", lambda *args, **kwargs: calls.append((args, kwargs)) or prices("ABC"))
    ffn.data.yf("ABC", "Close", start="2024-01-01", mrefresh=True)
    assert calls == [(("ABC",), {"field": "Close", "start": "2024-01-01", "end": None})]


def test_csv_provider_respects_date_window_and_refresh(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"ABC": [0.0, 1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=3)).to_csv(path)
    result = ffn.get("ABC", provider="csv", path=path, start="2024-01-02", end="2024-01-03")
    assert result["abc"].tolist() == [1.0]
    pd.DataFrame({"ABC": [0.0, 9.0, 2.0]}, index=pd.date_range("2024-01-01", periods=3)).to_csv(path)
    result = ffn.get("ABC", provider="csv", path=path, start="2024-01-02", end="2024-01-03", mrefresh=True)
    assert result["abc"].tolist() == [9.0]
