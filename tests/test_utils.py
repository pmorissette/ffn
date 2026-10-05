import functools
import pickle

import pandas as pd
import pytest

import ffn
from ffn import utils

# A module-level lambda raises PicklingError; local closures raise AttributeError.
_UNCACHEABLE_PROVIDERS = {"lambda": lambda: 10, "partial": functools.partial(bool, memoryview(b""))}


def test_memoize_isolates_series_results():
    """Keep caller assignments from changing a cached Series snapshot."""
    index = pd.date_range("2024-01-01", periods=2, name="date")
    expected = pd.Series([100.0, 101.0], index=index, name="price")
    expected.attrs["provider"] = "fixture"
    calls = []

    @utils.memoize
    def cached(value):
        """Return fresh data when the decorated function runs."""
        calls.append(value)
        return expected.copy(deep=True)

    first = cached("value")
    assert isinstance(first, pd.Series)
    first.iloc[0] = 999.0
    second = cached("value")
    assert isinstance(second, pd.Series)
    assert second is not first
    pd.testing.assert_series_equal(second, expected)
    assert second.attrs == expected.attrs

    second.iloc[-1] = -1.0
    third = cached("value")
    assert isinstance(third, pd.Series)
    assert third is not second
    pd.testing.assert_series_equal(third, expected)
    assert third.attrs == expected.attrs
    assert calls == ["value"]


def test_memoize_isolates_dataframe_results():
    """Keep caller assignments from changing a cached DataFrame snapshot."""
    index = pd.date_range("2024-01-01", periods=2, name="date")
    expected = pd.DataFrame({"price": [100.0, 101.0]}, index=index)
    expected.attrs["provider"] = "fixture"
    calls = []

    @utils.memoize
    def cached(value):
        """Return fresh data when the decorated function runs."""
        calls.append(value)
        return expected.copy(deep=True)

    first = cached("value")
    assert isinstance(first, pd.DataFrame)
    first.iloc[0, 0] = 999.0
    second = cached("value")
    assert isinstance(second, pd.DataFrame)
    assert second is not first
    pd.testing.assert_frame_equal(second, expected)
    assert second.attrs == expected.attrs

    second.iloc[-1, 0] = -1.0
    third = cached("value")
    assert isinstance(third, pd.DataFrame)
    assert third is not second
    pd.testing.assert_frame_equal(third, expected)
    assert third.attrs == expected.attrs
    assert calls == ["value"]


@pytest.mark.parametrize("constructor,axis", [(pd.Series, "index"), (pd.DataFrame, "index"), (pd.DataFrame, "columns")])
@pytest.mark.parametrize("cache_state", ["miss", "hit", "refresh"])
def test_memoize_isolates_axis_metadata(constructor, axis, cache_state):
    calls = []

    @utils.memoize
    def cached(mrefresh=False):
        calls.append(mrefresh)
        result = constructor([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2))
        if isinstance(result, pd.DataFrame):
            result.columns = pd.date_range("2024-02-01", periods=1)
        return result

    result = cached()
    if cache_state != "miss":
        result = cached(mrefresh=cache_state == "refresh")
    getattr(result, axis).freq = None

    for snapshot in cached.mcache.values():
        assert getattr(snapshot, axis).freq == pd.offsets.Day()
    assert getattr(cached(), axis).freq == pd.offsets.Day()
    assert calls == ([False, True] if cache_state == "refresh" else [False])


@pytest.mark.parametrize("constructor", [pd.Series, pd.DataFrame])
@pytest.mark.parametrize("cache_state", ["miss", "hit", "refresh"])
def test_memoize_isolates_nested_attrs(constructor, cache_state):
    calls = []

    @utils.memoize
    def cached(mrefresh=False):
        calls.append(mrefresh)
        result = constructor([100.0, 101.0])
        result.attrs["provider"] = {"fields": ["price"]}
        return result

    result = cached()
    if cache_state != "miss":
        result = cached(mrefresh=cache_state == "refresh")
    result.attrs["provider"]["fields"].append("volume")

    for snapshot in cached.mcache.values():
        assert snapshot.attrs == {"provider": {"fields": ["price"]}}
    assert cached().attrs == {"provider": {"fields": ["price"]}}
    assert calls == ([False, True] if cache_state == "refresh" else [False])


def test_memoize_preserves_non_pandas_result_identity():
    calls = []

    @utils.memoize
    def cached():
        calls.append(True)
        return {"prices": [100.0, 101.0]}

    first = cached()
    assert cached() is first
    assert calls == [True]


def test_memoize_handles_keyword_only_refresh():
    calls = []

    @utils.memoize
    def cached(value, *, mrefresh=False):
        calls.append((value, mrefresh))
        return len(calls)

    assert cached("value") == 1
    assert cached("value") == 1
    assert cached("value", mrefresh=True) == 2
    assert cached("value", mrefresh=True) == 3
    assert cached("value") == 1


def test_memoize_does_not_treat_varargs_as_keyword_only_refresh():
    calls = []

    @utils.memoize
    def cached(value, *items, mrefresh=False):
        calls.append((value, items, mrefresh))
        return len(calls)

    assert cached("value", True) == 1
    assert cached("value", True) == 1
    assert cached("value", True, mrefresh=True) == 2


@pytest.mark.parametrize("provider_kind", ["lambda", "closure", "partial"])
@pytest.mark.parametrize("refresh", [False, True])
def test_memoize_executes_uncacheable_callables(provider_kind, refresh):
    """A reusable pickle key is optional; an uncached call must leave existing entries intact."""
    calls = []

    def closure():
        return 10

    # Captured memoryview state adds the TypeError path to lambda/local-function pickle failures.
    provider = closure if provider_kind == "closure" else _UNCACHEABLE_PROVIDERS[provider_kind]
    expected = False if provider_kind == "partial" else 10
    error = {"lambda": pickle.PicklingError, "closure": AttributeError, "partial": TypeError}[provider_kind]
    with pytest.raises(error):
        pickle.dumps((provider,), 1)

    @utils.memoize
    def cached(*args, mrefresh=False, **kwargs):
        calls.append(mrefresh)
        callback = args[0] if args else kwargs["callback"]
        return callback()

    sentinel = object()
    cached.mcache[b"existing"] = sentinel
    # Exercise both serialized containers; a failure in kwargs follows a successful args dump.
    assert cached(provider, mrefresh=refresh) == expected
    assert cached(callback=provider, mrefresh=refresh) == expected
    assert calls == [refresh, refresh]
    assert cached.mcache == {b"existing": sentinel}


def test_memoize_does_not_alias_distinct_closures():
    """Identical apparent names must not become an alternate cache identity."""
    calls = []

    def make_provider(value):
        def provider():
            calls.append(value)
            return value

        return provider

    @utils.memoize
    def cached(provider):
        return provider()

    first, second = make_provider(1), make_provider(2)
    assert first.__qualname__ == second.__qualname__
    assert [cached(provider) for provider in (first, second, first, second)] == [1, 2, 1, 2]
    assert calls == [1, 2, 1, 2]
    assert cached.mcache == {}


@pytest.mark.parametrize("error", [TypeError, AttributeError, pickle.PicklingError])
def test_memoize_preserves_uncached_provider_exceptions(error):
    """Pickle fallback must not swallow the same exception class from the provider body."""
    failure = error("provider failed")

    def provider():
        raise failure

    @utils.memoize
    def cached(callback):
        return callback()

    sentinel = object()
    cached.mcache[b"existing"] = sentinel
    with pytest.raises(error) as raised:
        cached(provider)
    assert raised.value is failure
    assert cached.mcache == {b"existing": sentinel}


def test_memoize_preserves_unexpected_pickle_errors():
    """A failing user reducer must not be mistaken for a known unsupported pickle key."""
    failure = RuntimeError("reducer failed")
    calls = []

    class BrokenReducer:
        def __reduce__(self):
            raise failure

    @utils.memoize
    def cached(value):
        calls.append(value)

    with pytest.raises(RuntimeError) as raised:
        cached(BrokenReducer())
    assert raised.value is failure
    assert calls == []
    assert cached.mcache == {}


def test_parse_args():
    actual = utils.parse_arg('a,b,c')
    assert actual == ['a', 'b', 'c']

    # should ignore spaces
    actual = utils.parse_arg(' a ,b ,c ')
    assert actual == ['a', 'b', 'c']

    actual = utils.parse_arg('a')
    assert actual == ['a']

    # should stay same for list
    actual = utils.parse_arg(['a', 'b'])
    assert actual == ['a', 'b']

    # should stay same for dict
    actual = utils.parse_arg({'a': 1})
    assert actual == {'a': 1}


def test_clean_ticker():
    actual = utils.clean_ticker('aapl us equity')
    assert actual == 'aapl'

    actual = utils.clean_ticker('^vix')
    assert actual == 'vix'

    actual = utils.clean_ticker('^vix index')
    assert actual == 'vix'

    actual = utils.clean_ticker('Aapl us Equity')
    assert actual == 'aapl'

    actual = utils.clean_ticker('C')
    assert actual == 'c'


def test_fmtp():
    actual = utils.fmtp(0.2364)
    assert actual == '23.64%'

    actual = utils.fmtp(0.2364222)
    assert actual == '23.64%'

    actual = utils.fmtp(0.2364922)
    assert actual == '23.65%'

    actual = utils.fmtp(0.236)
    assert actual == '23.60%'


def test_fmtn():
    actual = utils.fmtn(0.2364)
    assert actual == '0.24'

    actual = utils.fmtn(1000.2364)
    assert actual == '1000.24'

    actual = utils.fmtn(1000.2)
    assert actual == '1000.20'


def test_fmtpn():
    actual = utils.fmtpn(0.2364)
    assert actual == '23.64'

    actual = utils.fmtpn(0.2364222)
    assert actual == '23.64'

    actual = utils.fmtpn(0.2364922)
    assert actual == '23.65'

    actual = utils.fmtpn(0.236)
    assert actual == '23.60'


def test_scale():
    assert utils.scale(0, (0.0, 99.0), (-1.0, 1.0)) == -1.0
    assert utils.scale(-5, (0.0, 99.0), (-1.0, 1.0)) == -1.0
    assert utils.scale(105, (0.0, 99.0), (-1.0, 1.0)) == 1.0
    assert utils.scale(50, (0.0, 100.0), (-1.0, 1.0)) == 0.0


def test_get_freq_name():
    assert utils.get_freq_name('D') == 'daily'
    assert utils.get_freq_name('M') == 'monthly'
    assert utils.get_freq_name('L') == 'milliseconds'
    assert utils.get_freq_name('zzz') is None


def test_get_freq_name_period_end_aliases():
    # pandas 2.2 renamed the period end aliases
    assert utils.get_freq_name('ME') == 'monthly'
    assert utils.get_freq_name('QE') == 'quarterly'
    assert utils.get_freq_name('YE') == 'yearly'
    assert utils.get_freq_name('BME') == 'business month end'
    assert utils.get_freq_name('BQE') == 'business quarter end'
    assert utils.get_freq_name('BYE') == 'business year end'


def test_get_freq_name_anchored_aliases():
    # pd.infer_freq anchors these to a month or weekday
    assert utils.get_freq_name('YE-DEC') == 'yearly'
    assert utils.get_freq_name('QE-DEC') == 'quarterly'
    assert utils.get_freq_name('A-DEC') == 'yearly'
    assert utils.get_freq_name('W-SUN') == 'weekly'


def test_get_freq_name_rejects_invalid_anchors():
    assert utils.get_freq_name('D-NOTREAL') is None
    assert utils.get_freq_name('ME-NOTREAL') is None
    assert utils.get_freq_name('W-DEC') is None
    assert utils.get_freq_name('YE-MON') is None


def test_get_freq_name_is_case_sensitive_where_pandas_is():
    # 'ms' is milliseconds in pandas while 'MS' is month start
    assert utils.get_freq_name('ms') == 'milliseconds'
    assert utils.get_freq_name('MS') == 'month start'
    assert utils.get_freq_name('min') == 'minutely'
    assert utils.get_freq_name('us') == 'microseconds'


def test_get_freq_name_accepts_what_infer_freq_returns():
    # the round trip a default plot title actually takes
    for freq, expected in (
        (ffn.core._MonthEnd, 'monthly'),
        (f'2{ffn.core._MonthEnd}', 'monthly'),
        (ffn.core._YearEnd, 'yearly'),
        (f'2{ffn.core._YearEnd}', 'yearly'),
        ('D', 'daily'),
        ('-1D', 'daily'),
        ('2h', 'hourly'),
        ('2ms', 'milliseconds'),
    ):
        idx = pd.date_range('2020-01-31', periods=8, freq=freq)
        assert utils.get_freq_name(pd.infer_freq(idx)) == expected


def test_as_format():
    ser = pd.Series(
        data=[5.672083e-01, 4.327917e-01, 0.000000e+00, 3.469447e-18,
              8.673617e-19], index=['aapl', 'msft', 'c', 'gs', 'ge'])

    actual = ser.as_format('.2f')
    assert actual.loc['aapl'] == '0.57'
    assert actual.loc['msft'] == '0.43'
    assert actual.loc['c'] == '0.00'
    assert actual.loc['gs'] == '0.00'
    assert actual.loc['ge'] == '0.00'

    df = pd.DataFrame({'aapl': [217.960007, 218.240005],
                       'msft': [23.389397, 23.396961],
                       }, index=['aapl', 'msft'])

    actual = df.as_format('.2f')
    assert actual.loc['aapl', 'aapl'] == '217.96'
    assert actual.loc['msft', 'aapl'] == '218.24'
    assert actual.loc['aapl', 'msft'] == '23.39'
    assert actual.loc['msft', 'msft'] == '23.40'
