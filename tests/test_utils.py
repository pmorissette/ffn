import ffn.utils as utils


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
