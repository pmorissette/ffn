import io
import json
from unittest import mock
from urllib.error import HTTPError, URLError

import ffn
import pandas as pd
import pytest


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

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(ValueError, match="dated 'val' rows"):
            ffn.data.fxmacrodata("EURUSD", mrefresh=True)


def test_fxmacrodata_redacts_http_error_details():
    error = HTTPError(
        "https://api.fxmacrodata.com/v1/forex/eur/usd",
        500,
        "server error",
        {},
        io.BytesIO(b"sensitive upstream details"),
    )

    with mock.patch("urllib.request.urlopen", side_effect=error):
        with pytest.raises(ffn.data.FXMacroDataError) as raised:
            ffn.data.fxmacrodata("EURUSD", api_key="placeholder-key")

    assert str(raised.value) == "FXMacroData API request failed with status 500"
    assert raised.value.__cause__ is None


def test_fxmacrodata_wraps_network_errors():
    with mock.patch("urllib.request.urlopen", side_effect=URLError("unavailable")):
        with pytest.raises(ffn.data.FXMacroDataError, match="API request failed$"):
            ffn.data.fxmacrodata("EURUSD", mrefresh=True)


def test_fxmacrodata_rejects_invalid_json():
    with mock.patch("urllib.request.urlopen", return_value=FakeResponse("not-json")):
        with pytest.raises(ffn.data.FXMacroDataError, match="invalid JSON"):
            ffn.data.fxmacrodata("EURUSD", mrefresh=True)
