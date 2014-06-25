ffn
===

ffn - a financial function library for Python.

Alpha release - please let me know if you find any bugs!

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

To install ffn, simply run:

.. code-block:: bash
    
    $ pip install ffn

If pip is not installed, I recommend installing the `Anaconda Scientific Python
Distribution <https://store.continuum.io/cshop/anaconda/>`_. This distribution comes with many of the required packages pre-installed, including pip. Once Anaconda is installed, the above command should complete the installation. 

ffn should be compatible with Python 2.7. 

Documentation
-------------

Read the docs at http://pmorissette.github.io/ffn

- `Quickstart <http://pmorissette.github.io/ffn/quickstart.html>`__
- `Full API <http://pmorissette.github.io/ffn/ffn.html>`__
