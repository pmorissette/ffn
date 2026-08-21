import ffn
import ffn.utils as utils
import pandas as pd


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
        (ffn.core._YearEnd, 'yearly'),
        ('D', 'daily'),
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
