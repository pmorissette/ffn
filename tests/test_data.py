import io
import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import ffn


def sample_provider(ticker, field):
    prices = {"ABC": [np.nan, 10.0, np.nan, 12.0], "DEF": [20.0, np.nan, 22.0, 23.0]}
    return pd.Series(prices[ticker], index=pd.date_range("2024-01-01", periods=4))


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

    with mock.patch.object(ffn.data, "urlopen", side_effect=fake_urlopen):
        actual = ffn.data.fxmacrodata(
            "eur/usd",
            start="2024-01-01",
            end=pd.Timestamp("2024-01-31"),
            api_key="test-key",
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
        "url": "https://fxmacrodata.com/api/v1/forex/eur/usd?start_date=2024-01-01&end_date=2024-01-31",
        "accept": "application/json",
        "api_key": "test-key",
        "timeout": 12,
    }


def test_fxmacrodata_requests_indicator_for_technical_field():
    payload = {"data": [{"date": "2024-01-03", "rsi_14": 54.25}]}
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return FakeResponse(json.dumps(payload))

    with mock.patch.object(ffn.data, "urlopen", side_effect=fake_urlopen):
        actual = ffn.data.fxmacrodata("EURUSD", field="rsi_14", start="2024-01-01", mrefresh=True)

    assert captured["url"] == "https://fxmacrodata.com/api/v1/forex/eur/usd?start_date=2024-01-01&indicators=rsi_14"
    assert actual.loc[pd.Timestamp("2024-01-03")] == 54.25


def test_fxmacrodata_omits_api_key_header_when_not_configured(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    monkeypatch.delenv("FXMD_API_KEY", raising=False)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-api-key")
        return FakeResponse(json.dumps({"data": [{"date": "2024-01-03", "val": 1.092}]}))

    with mock.patch.object(ffn.data, "urlopen", side_effect=fake_urlopen):
        ffn.data.fxmacrodata("EURUSD", mrefresh=True)

    assert "api_key" not in captured["url"]
    assert captured["api_key"] is None


def test_fxmacrodata_integrates_with_get():
    payload = {"data": [{"date": "2024-01-01", "val": 1.1038}]}

    def fake_urlopen(request, timeout):
        return FakeResponse(json.dumps(payload))

    with mock.patch.object(ffn.data, "urlopen", side_effect=fake_urlopen):
        actual = ffn.get("EURUSD", provider=ffn.data.fxmacrodata, start="2024-01-01", mrefresh=True)

    expected = pd.DataFrame({"eurusd": [1.1038]}, index=pd.to_datetime(["2024-01-01"]))
    pd.testing.assert_frame_equal(actual, expected)


def test_fxmacrodata_rejects_unknown_pair_shape():
    with pytest.raises(ValueError, match="EURUSD"):
        ffn.data.fxmacrodata("EUR", mrefresh=True)


def test_fxmacrodata_rejects_missing_rows():
    payload = {"data": [{"date": "2024-01-01"}]}

    def fake_urlopen(request, timeout):
        return FakeResponse(json.dumps(payload))

    with mock.patch.object(ffn.data, "urlopen", side_effect=fake_urlopen):
        with pytest.raises(ValueError, match="dated 'val' rows"):
            ffn.data.fxmacrodata("EURUSD", mrefresh=True)
