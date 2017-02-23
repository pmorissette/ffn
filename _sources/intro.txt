
.. code:: python

    import ffn
    #%pylab inline

.. code:: python

    # download price data from Yahoo! Finance. By default, 
    # the Adj. Close will be used. 
    prices = ffn.get('aapl,msft', start='2010-01-01')

.. code:: python

    # let's compare the relative performance of each stock 
    # we will rebase here to get a common starting point for both securities
    ax = prices.rebase().plot()



.. image:: _static/intro_2_0.png
    :class: pynb


.. code:: python

    # now what do the return distributions look like?
    returns = prices.to_returns().dropna()
    ax = returns.hist(figsize(10, 5))



.. image:: _static/intro_3_0.png
    :class: pynb


.. code:: python

    # ok now what about some performance metrics?
    stats = prices.calc_stats()
    stats.display()


.. parsed-literal::
    :class: pynb-result

    Stat                 aapl        msft
    -------------------  ----------  ----------
    Start                2010-01-04  2010-01-04
    End                  2014-06-19  2014-06-19
    
    Total Return         214.37%     51.06%
    Daily Sharpe         1.08        0.52
    CAGR                 29.32%      9.70%
    Max Drawdown         -43.80%     -26.36%
    
    MTD                  1.58%       1.39%
    3m                   21.72%      6.46%
    6m                   19.47%      16.18%
    YTD                  15.94%      12.58%
    1Y                   55.69%      23.54%
    3Y (ann.)            28.75%      22.76%
    5Y (ann.)            29.32%      9.70%
    10Y (ann.)           29.32%      9.70%
    Since Incep. (ann.)  29.32%      9.70%
    
    Daily Sharpe         1.08        0.52
    Daily Mean (ann.)    29.48%      11.81%
    Daily Vol (ann.)     27.35%      22.54%
    Daily Skew           -0.12       -0.21
    Daily Kurt           4.76        5.21
    Best Day             8.88%       7.29%
    Worst Day            -12.36%     -11.39%
    
    Monthly Sharpe       1.24        0.64
    Monthly Mean (ann.)  31.91%      13.76%
    Monthly Vol (ann.)   25.72%      21.41%
    Monthly Skew         -0.00       -0.08
    Monthly Kurt         -0.06       0.24
    Best Month           18.84%      15.69%
    Worst Month          -14.40%     -15.12%
    
    Yearly Sharpe        1.91        0.69
    Yearly Mean          20.54%      14.54%
    Yearly Vol           10.75%      21.05%
    Yearly Skew          -0.09       1.34
    Yearly Kurt          -1.97       2.16
    Best Year            32.57%      44.31%
    Worst Year           8.08%       -4.51%
    
    Avg. Drawdown        -4.47%      -4.00%
    Avg. Drawdown Days   31.46       52.13
    Avg. Up Month        6.76%       5.23%
    Avg. Down Month      -4.69%      -4.61%
    Win Year %           100.00%     75.00%
    Win 12m %            76.74%      76.74%


.. code:: python

    # what about the drawdowns?
    ax = stats.prices.to_drawdown_series().plot()



.. image:: _static/intro_5_0.png
    :class: pynb

