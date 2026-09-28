import io
import json
from unittest import mock
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest

import ffn


def sample_provider(ticker, field):
    prices = {"ABC": [np.nan, 10.0, np.nan, 12.0], "DEF": [20.0, np.nan, 22.0, 23.0]}
    return pd.Series(prices[ticker], index=pd.date_range("2024-01-01", periods=4))


def test_get_isolates_cached_data_from_caller_mutation(monkeypatch):
    """Keep caller assignments from changing data cached by ``ffn.get``."""
    expected = pd.DataFrame(
        {"abc": [10.0, 12.0]},
        index=pd.date_range("2024-01-02", periods=2, freq="2D"),
    )
    calls = []

    def provider(ticker, field):
        calls.append((ticker, field))
        return sample_provider(ticker, field)

    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", provider)
    ffn.get.mcache.clear()
    try:
        first = ffn.get("ABC")
        assert isinstance(first, pd.DataFrame)
        first.iloc[0, 0] = 999.0
        first.index.freq = None
        second = ffn.get("ABC")

        assert isinstance(second, pd.DataFrame)
        assert second is not first
        pd.testing.assert_frame_equal(second, expected)
        second.iloc[-1, 0] = -1.0
        second.index.freq = None
        pd.testing.assert_frame_equal(ffn.get("ABC"), expected)
        assert calls == [("ABC", None)]
    finally:
        ffn.get.mcache.clear()


def test_csv_isolates_cached_data_from_caller_mutation(monkeypatch):
    expected = pd.Series([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2), name="ABC")
    calls = []

    def read_csv(path, **kwargs):
        calls.append(path)
        return pd.DataFrame({"ABC": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2))

    monkeypatch.setattr(pd, "read_csv", read_csv)
    ffn.data.csv.mcache.clear()
    try:
        first = ffn.data.csv("ABC", path="fixture.csv")
        first.iloc[0] = 999.0
        first.index.freq = None
        second = ffn.data.csv("ABC", path="fixture.csv")
        pd.testing.assert_series_equal(second, expected)
        second.iloc[-1] = -1.0
        second.index.freq = None
        pd.testing.assert_series_equal(ffn.data.csv("ABC", path="fixture.csv"), expected)
        assert calls == ["fixture.csv"]
    finally:
        ffn.data.csv.mcache.clear()


@pytest.mark.parametrize("forward_fill", [False, True])
@pytest.mark.parametrize("common_dates", [False, True])
def test_get_forward_fill(forward_fill, common_dates):
    result = ffn.get(
        "ABC,DEF",
        provider=sample_provider,
        common_dates=common_dates,
        forward_fill=forward_fill,
        mrefresh=True,
    )

    if common_dates:
        expected = pd.DataFrame({"abc": [12.0], "def": [23.0]}, index=pd.date_range("2024-01-04", periods=1))
    else:
        expected = pd.DataFrame(
            {"abc": [np.nan, 10.0, 10.0 if forward_fill else np.nan, 12.0], "def": [20.0, 20.0 if forward_fill else np.nan, 22.0, 23.0]},
            index=pd.date_range("2024-01-01", periods=4),
        )
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


@pytest.mark.parametrize(
    ("tickers", "existing"),
    [
        (["A-B", "AB"], None),
        (["A-B:Close", "AB:Close"], None),
        (["AB"], pd.DataFrame({"A-B": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))),
    ],
    ids=["requested-tickers", "ticker-fields", "existing-data"],
)
def test_get_rejects_clean_ticker_collisions(monkeypatch, tickers, existing):
    index = pd.date_range("2024-01-01", periods=2)
    prices = {"A-B": [1.0, 2.0], "AB": [10.0, 20.0]}
    original = None if existing is None else existing.copy(deep=True)

    def provider(ticker, field):
        return pd.Series(prices[ticker], index=index)

    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", provider)
    with pytest.raises(
        ValueError,
        match="cleaned ticker names are not unique.*clean_tickers=False.*column_names",
    ):
        ffn.get(
            tickers,
            common_dates=False,
            existing=existing,
            mrefresh=True,
        )

    if existing is not None:
        pd.testing.assert_frame_equal(existing, original)


def test_get_clean_ticker_collision_controls(monkeypatch):
    index = pd.date_range("2024-01-01", periods=2)
    prices = {"A-B": [1.0, 2.0], "AB": [10.0, 20.0], "C_D": [100.0, 200.0]}
    calls = []

    def provider(ticker, field):
        calls.append((ticker, field))
        return pd.Series(prices[ticker], index=index)

    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", provider)
    ordinary = ffn.get(
        ["A-B:Close", "C_D:Open"],
        common_dates=False,
        mrefresh=True,
    )
    uncleaned = ffn.get(
        ["A-B", "AB"],
        common_dates=False,
        clean_tickers=False,
        mrefresh=True,
    )
    renamed = ffn.get(
        ["A-B", "AB"],
        common_dates=False,
        column_names=["hyphenated", "plain"],
        mrefresh=True,
    )

    pd.testing.assert_frame_equal(
        ordinary,
        pd.DataFrame({"abclose": [1.0, 2.0], "cdopen": [100.0, 200.0]}, index=index),
    )
    assert calls[:2] == [("A-B", "Close"), ("C_D", "Open")]
    assert uncleaned.columns.tolist() == ["A-B", "AB"]
    assert renamed.columns.tolist() == ["hyphenated", "plain"]
    assert uncleaned.to_numpy().tolist() == renamed.to_numpy().tolist()


class FakeResponse(io.StringIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_fxmacrodata_fetches_spot_series():
    payload = {
        "data": [
            {"date": "2024-01-03", "val": 1.0920},
            {"date": "2024-01-01", "val": "1.1038"},
            {"date": "2024-01-02", "val": 1.0943},
        ]
    }
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["accept"] = request.get_header("Accept")
        captured["api_key"] = request.get_header("X-api-key")
        captured["timeout"] = timeout
        return FakeResponse(json.dumps(payload))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        actual = ffn.data.fxmacrodata(
            "eur/usd",
            start="2024-01-01",
            end=pd.Timestamp("2024-01-31"),
            api_key="placeholder-key",
            timeout=12,
            mrefresh=True,
        )

    expected = pd.Series(
        [1.1038, 1.0943, 1.0920],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        name="eur/usd",
    )
    pd.testing.assert_series_equal(actual, expected)
    assert captured == {
        "url": "https://api.fxmacrodata.com/v1/forex/eur/usd?start_date=2024-01-01&end_date=2024-01-31",
        "accept": "application/json",
        "api_key": "placeholder-key",
        "timeout": 12,
    }


def test_fxmacrodata_requests_indicator_for_technical_field():
    payload = {"data": [{"date": "2024-01-03", "rsi_14": 54.25}]}
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse(json.dumps(payload))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        actual = ffn.data.fxmacrodata("EURUSD", field="rsi_14", start="2024-01-01", mrefresh=True)

    assert captured["url"] == "https://api.fxmacrodata.com/v1/forex/eur/usd?start_date=2024-01-01&indicators=rsi_14"
    assert actual.loc[pd.Timestamp("2024-01-03")] == 54.25


def test_fxmacrodata_fetches_public_usd_indicator_without_api_key(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    monkeypatch.delenv("FXMD_API_KEY", raising=False)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-api-key")
        return FakeResponse(json.dumps({"data": [{"date": "2024-01-31", "val": 3.1}]}))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        actual = ffn.data.fxmacrodata("USD", field="inflation", start="2024-01-01", mrefresh=True)

    assert captured == {
        "url": "https://api.fxmacrodata.com/v1/announcements/usd/inflation?start_date=2024-01-01",
        "api_key": None,
    }
    assert actual.loc[pd.Timestamp("2024-01-31")] == 3.1


def test_fxmacrodata_omits_api_key_header_when_not_configured(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    monkeypatch.delenv("FXMD_API_KEY", raising=False)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-api-key")
        return FakeResponse(json.dumps({"data": [{"date": "2024-01-31", "val": 3.1}]}))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        ffn.data.fxmacrodata("USD", field="inflation", mrefresh=True)

    assert "api_key" not in captured["url"]
    assert captured["api_key"] is None


def test_fxmacrodata_does_not_serialize_api_keys_into_cache_keys():
    cache = ffn.data._fxmacrodata_cached.mcache
    cache.clear()
    request_keys = []

    def fake_urlopen(request, timeout):
        request_keys.append(request.get_header("X-api-key"))
        return FakeResponse(json.dumps({"data": [{"date": "2024-01-03", "val": 1.092}]}))

    try:
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            ffn.data.fxmacrodata("EURUSD", api_key="first-placeholder")
            ffn.data.fxmacrodata("EURUSD", api_key="second-placeholder")

        assert request_keys == ["first-placeholder", "second-placeholder"]
        assert not cache
    finally:
        cache.clear()


def test_fxmacrodata_integrates_with_get():
    payload = {"data": [{"date": "2024-01-31", "val": 3.1}]}

    def fake_urlopen(request, timeout):
        return FakeResponse(json.dumps(payload))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        actual = ffn.get(
            "USD:inflation",
            provider=ffn.data.fxmacrodata,
            start="2024-01-01",
            mrefresh=True,
        )

    expected = pd.DataFrame({"usdinflation": [3.1]}, index=pd.to_datetime(["2024-01-31"]))
    pd.testing.assert_frame_equal(actual, expected)


def test_fxmacrodata_rejects_unknown_pair_shape():
    with pytest.raises(ValueError, match="EURUSD"):
        ffn.data.fxmacrodata("EUR", mrefresh=True)


def test_fxmacrodata_rejects_missing_rows():
    payload = {"data": [{"date": "2024-01-01"}]}

    def fake_urlopen(request, timeout):
        return FakeResponse(json.dumps(payload))

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen), pytest.raises(ValueError, match="dated 'val' rows"):
        ffn.data.fxmacrodata("EURUSD", mrefresh=True)


def test_fxmacrodata_redacts_http_error_details():
    error = HTTPError(
        "https://api.fxmacrodata.com/v1/forex/eur/usd",
        500,
        "server error",
        {},
        io.BytesIO(b"sensitive upstream details"),
    )

    with mock.patch("urllib.request.urlopen", side_effect=error), pytest.raises(ffn.data.FXMacroDataError) as raised:
        ffn.data.fxmacrodata("EURUSD", api_key="placeholder-key")

    assert str(raised.value) == "FXMacroData API request failed with status 500"
    assert raised.value.__cause__ is None


def test_fxmacrodata_wraps_network_errors():
    with mock.patch("urllib.request.urlopen", side_effect=URLError("unavailable")), pytest.raises(ffn.data.FXMacroDataError, match="API request failed$"):
        ffn.data.fxmacrodata("EURUSD", mrefresh=True)


def test_fxmacrodata_rejects_invalid_json():
    with mock.patch("urllib.request.urlopen", return_value=FakeResponse("not-json")), pytest.raises(ffn.data.FXMacroDataError, match="invalid JSON"):
        ffn.data.fxmacrodata("EURUSD", mrefresh=True)
