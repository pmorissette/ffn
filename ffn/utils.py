import re
import decorator
import numpy as np
import cPickle


def _memoize(func, *args, **kw):
    # should we refresh the cache?
    refresh = False
    refresh_kw = func.mrefresh_keyword

    # kw is not always set - check args
    if refresh_kw in func.func_code.co_varnames:
        if args[func.func_code.co_varnames.index(refresh_kw)]:
            refresh = True

    # check in kw if not already set above
    if not refresh and refresh_kw in kw:
        if kw[refresh_kw]:
            refresh = True

    key = cPickle.dumps(args, 1) + cPickle.dumps(kw, 1)

    cache = func.mcache
    if not refresh and key in cache:
        return cache[key]
    else:
        cache[key] = result = func(*args, **kw)
        return result


def memoize(f, refresh_keyword='mrefresh'):
    f.mcache = {}
    f.mrefresh_keyword = refresh_keyword
    return decorator.decorator(_memoize, f)


def parse_arg(arg):
    # handle string input
    if type(arg) == str:
        arg = arg.strip()
        # parse csv as tickers and create children
        if ',' in arg:
            arg = arg.split(',')
            arg = [x.strip() for x in arg]
        # assume single string - create single item list
        else:
            arg = [arg]

    return arg


def clean_ticker(ticker):
    """
    Cleans a ticker for easier use throughout MoneyTree

    Splits by space and only keeps first bit. Also removes
    any characters that are not letters. Returns as lowercase.

    >>> clean_ticker('^VIX')
    'vix'
    >>> clean_ticker('SPX Index')
    'spx'
    """
    pattern = re.compile('[\W_]+')
    res = pattern.sub('', ticker.split(' ')[0])
    return res.lower()


def fmtp(number):
    if np.isnan(number):
        return '-'
    return format(number, '.2%')


def fmtpn(number):
    if np.isnan(number):
        return '-'
    return format(number * 100, '.2f')


def fmtn(number):
    if np.isnan(number):
        return '-'
    return format(number, '.2f')


def get_period_name(period):
    if period is 'd':
        return 'daily'
    elif period is 'm':
        return 'monthly'
    elif period is 'y':
        return 'yearly'
