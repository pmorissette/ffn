import numpy as np
import pandas as pd
import pytest

import ffn


def sample_provider(ticker, field):
    prices = {"ABC": [np.nan, 10.0, np.nan, 12.0], "DEF": [20.0, np.nan, 22.0, 23.0]}
    return pd.Series(prices[ticker], index=pd.date_range("2024-01-01", periods=4))


@pytest.mark.parametrize("provider_kind", ["lambda", "closure", "memoized"])
@pytest.mark.parametrize("refresh", [False, True])
def test_get_supports_uncacheable_providers(provider_kind, refresh):
    """Provider dispatch must preserve fields, kwargs, and the inner provider's refresh policy."""
    index = pd.date_range("2024-01-01", periods=2, name="date")
    calls = []

    def provide(ticker, field, offset, mrefresh=False):
        calls.append((ticker, field, offset, mrefresh))
        return pd.Series([1.0 + offset, 2.0 + offset], index=index)

    provider = (lambda **kwargs: provide(**kwargs)) if provider_kind == "lambda" else provide
    if provider_kind == "memoized":
        provider = ffn.utils.memoize(provide)

    expected = pd.DataFrame({"abcclose": [11.0, 12.0]}, index=index)
    first = ffn.get("ABC:Close", provider=provider, offset=10.0, mrefresh=refresh)
    pd.testing.assert_frame_equal(first, expected)
    first.iloc[0, 0] = -1.0
    second = ffn.get("ABC:Close", provider=provider, offset=10.0, mrefresh=refresh)
    pd.testing.assert_frame_equal(second, expected)
    # Only the provider owns a cache; refresh is forwarded when explicitly declared.
    count = 1 if provider_kind == "memoized" and not refresh else 2
    forwarded_refresh = refresh if provider_kind != "lambda" else False
    assert calls == [("ABC", "Close", 10.0, forwarded_refresh)] * count
    assert not hasattr(ffn.get, "mcache")


def test_get_isolates_provider_data_from_caller_mutation(monkeypatch):
    """Keep caller assignments from changing data cached by a provider."""
    expected = pd.DataFrame(
        {"abc": [10.0, 12.0]},
        index=pd.date_range("2024-01-02", periods=2, freq="2D"),
    )
    calls = []

    @ffn.utils.memoize
    def provider(ticker, field):
        calls.append((ticker, field))
        return sample_provider(ticker, field)

    monkeypatch.setattr(ffn.data, "DEFAULT_PROVIDER", provider)
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
