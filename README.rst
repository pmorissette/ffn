.. image:: http://pmorissette.github.io/ffn/_static/logo.png

.. image:: https://github.com/pmorissette/ffn/workflows/Build%20Status/badge.svg
    :target: https://github.com/pmorissette/ffn/actions/

.. image:: https://codecov.io/gh/pmorissette/ffn/branch/master/graph/badge.svg
    :target: https://codecov.io/pmorissette/ffn

.. image:: https://img.shields.io/pypi/v/ffn
    :alt: PyPI
    :target: https://pypi.org/project/ffn/

.. image:: https://img.shields.io/pypi/l/ffn
    :alt: PyPI - License
    :target: https://pypi.org/project/ffn/

ffn - Financial Functions for Python
====================================

Alpha release - please let me know if you find any bugs!

If you are looking for a full backtesting framework, please check out `bt
<https://github.com/pmorissette/bt>`_. bt is built atop ffn and makes it easy
and fast to backtest quantitative strategies.

Overview
--------

ffn is a library that contains many useful functions for those who work in **quantitative
finance**. It stands on the shoulders of giants (Pandas, Numpy, Scipy, etc.) and provides
a vast array of utilities, from performance measurement and evaluation to
graphing and common data transformations.

.. code:: python

    >> import ffn
    >> returns = ffn.get('aapl,msft,c,gs,ge', start='2010-01-01').to_returns().dropna()
    >> returns.calc_mean_var_weights().as_format('.2%')
    aapl    62.54%
    c       -0.00%
    ge      36.19%
    gs      -0.00%
    msft     1.26%
    dtype: object


Installation
------------

The easiest way to install ``ffn`` is from the `Python Package Index <https://pypi.python.org/pypi/ffn/>`_
using ``pip`` or ``easy_install``:

.. code-block:: bash

    $ pip install ffn

Since ffn has many dependencies, we strongly recommend installing the `Anaconda Scientific Python Distribution <https://store.continuum.io/cshop/anaconda/>`_. This distribution comes with many of the required packages pre-installed, including pip. Once Anaconda is installed, the above command should complete the installation. 

ffn should be compatible with Python 2.7 and Python 3.

Documentation
-------------

Read the docs at http://pmorissette.github.io/ffn

- `Quickstart <http://pmorissette.github.io/ffn/quick.html>`__
- `Full API <http://pmorissette.github.io/ffn/ffn.html>`__
