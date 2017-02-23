
.. code:: python

    import ffn
    #%pylab inline


Data Retrieval
--------------

The main method for data retrieval is the :func:`get <ffn.get>` function. The get function uses a data provider to download data from an external service and packs that data into a pandas DataFrame for further manipulation.

.. code:: python

    data = ffn.get('agg,hyg,spy,eem,efa', start='2010-01-01', end='2014-01-01')
    print data.head()


.. parsed-literal::
    :class: pynb-result

                  agg    hyg     spy    eem    efa
    Date                                          
    2010-01-04  90.45  64.87  103.44  39.32  49.08
    2010-01-05  90.86  65.18  103.71  39.61  49.13
    2010-01-06  90.81  65.35  103.79  39.69  49.33
    2010-01-07  90.70  65.61  104.23  39.46  49.14
    2010-01-08  90.75  65.72  104.57  39.77  49.53
    
    [5 rows x 5 columns]


By default, the data is downloaded from Yahoo! Finance and the Adjusted
Close is used as the security's price. Other data sources are also
available and you may select other fields as well. Fields are specified
by using the following format: {ticker}:{field}. So, if we want to get
the Open, High, Low, Close for aapl, we would do the following:

.. code:: python

    print ffn.get('aapl:Open,aapl:High,aapl:Low,aapl:Close', start='2010-01-01', end='2014-01-01').head()


.. parsed-literal::
    :class: pynb-result

                aaplopen  aaplhigh  aapllow  aaplclose
    Date                                              
    2010-01-04    213.43    214.50   212.38     214.01
    2010-01-05    214.60    215.59   213.25     214.38
    2010-01-06    214.38    215.23   210.75     210.97
    2010-01-07    211.75    212.00   209.05     210.58
    2010-01-08    210.30    212.00   209.06     211.98
    
    [5 rows x 4 columns]



The default data provider is :func:`ffn.data.web`. This is basically just a thin wrapper around pandas' pandas.io.data provider. Please refer to the appropriate docs for more info (data sources, etc.). The :func:`ffn.data.csv` provider is also available when we want to load data from a local file. In this case, we can tell :func:`ffn.data.get` to use the csv provider. In this case, we also want to merge this new data with the existing data we downloaded earlier. Therefore, we will provide the **data** object as the *existing* argument, and the new data will be merged into the existing DataFrame.

.. code:: python

    data = ffn.get('dbc', provider=ffn.data.csv, path='test_data.csv', existing=data)
    print data.head()


.. parsed-literal::
    :class: pynb-result

                  agg    hyg     spy    eem    efa    dbc
    Date                                                 
    2010-01-04  90.45  64.87  103.44  39.32  49.08  25.24
    2010-01-05  90.86  65.18  103.71  39.61  49.13  25.27
    2010-01-06  90.81  65.35  103.79  39.69  49.33  25.72
    2010-01-07  90.70  65.61  104.23  39.46  49.14  25.40
    2010-01-08  90.75  65.72  104.57  39.77  49.53  25.38
    
    [5 rows x 6 columns]



As we can see above, the dbc column was added to the DataFrame. Internally, get is using the function ffn.merge, which is useful when you want to merge TimeSeries and DataFrames together. We plan on adding many more data sources over time. If you know your way with Python and would like to contribute a data provider, please feel free to submit a pull request - contributions are always welcome!

Data Manipulation
-----------------

Now that we have some data, let's start manipulating it. In quantitative finance, we are often interested in the returns of a given time series. Let's calculate the returns by simply calling the :func:`to_returns <ffn.core.to_returns>` or :func:`to_log_returns <ffn.core.to_log_returns>` extension methods.

.. code:: python

    returns = data.to_log_returns().dropna()
    print returns.head()


.. parsed-literal::
    :class: pynb-result

                     agg       hyg       spy       eem       efa       dbc
    Date                                                                  
    2010-01-05  0.004523  0.004767  0.002607  0.007348  0.001018  0.001188
    2010-01-06 -0.000550  0.002605  0.000771  0.002018  0.004063  0.017651
    2010-01-07 -0.001212  0.003971  0.004230 -0.005812 -0.003859 -0.012520
    2010-01-08  0.000551  0.001675  0.003257  0.007825  0.007905 -0.000788
    2010-01-11 -0.000772 -0.000913  0.001433 -0.002014  0.008244 -0.003157
    
    [5 rows x 6 columns]


Let's look at the different distributions to see how they look.

.. code:: python

    ax = returns.hist(figsize=(12, 5))



.. image:: _static/quickstart_10_0.png
    :class: pynb


We can also use the numerous functions packed into numpy, pandas and the
like to further analyze the returns. For example, we can use the corr
function to get the pairwise correlations between assets.

.. code:: python

    returns.corr().as_format('.2f')




.. raw:: html

    <div class="pynb-result" style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>agg</th>
          <th>hyg</th>
          <th>spy</th>
          <th>eem</th>
          <th>efa</th>
          <th>dbc</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>agg</th>
          <td>  1.00</td>
          <td> -0.11</td>
          <td> -0.33</td>
          <td> -0.23</td>
          <td> -0.29</td>
          <td> -0.18</td>
        </tr>
        <tr>
          <th>hyg</th>
          <td> -0.11</td>
          <td>  1.00</td>
          <td>  0.77</td>
          <td>  0.75</td>
          <td>  0.76</td>
          <td>  0.49</td>
        </tr>
        <tr>
          <th>spy</th>
          <td> -0.33</td>
          <td>  0.77</td>
          <td>  1.00</td>
          <td>  0.88</td>
          <td>  0.92</td>
          <td>  0.59</td>
        </tr>
        <tr>
          <th>eem</th>
          <td> -0.23</td>
          <td>  0.75</td>
          <td>  0.88</td>
          <td>  1.00</td>
          <td>  0.90</td>
          <td>  0.62</td>
        </tr>
        <tr>
          <th>efa</th>
          <td> -0.29</td>
          <td>  0.76</td>
          <td>  0.92</td>
          <td>  0.90</td>
          <td>  1.00</td>
          <td>  0.61</td>
        </tr>
        <tr>
          <th>dbc</th>
          <td> -0.18</td>
          <td>  0.49</td>
          <td>  0.59</td>
          <td>  0.62</td>
          <td>  0.61</td>
          <td>  1.00</td>
        </tr>
      </tbody>
    </table>
    <p>6 rows × 6 columns</p>
    </div>



Here we used the convenience method as\_format to have a prettier
output. We could also plot a heatmap to better visualize the results.

.. code:: python

    returns.plot_corr_heatmap()



.. image:: _static/quickstart_14_0.png
    :class: pynb



We used the :func:`ffn.core.plot_corr_heatmap`, which is a convenience method that simply calls ffn's :func:`ffn.core.plot_heatmap` with sane arguments.

Let's start looking at how all these securities performed over the period. To achieve this, we will plot rebased time series so that we can see how they each performed relative to eachother.

.. code:: python

    ax = data.rebase().plot(figsize=(12,5))



.. image:: _static/quickstart_16_0.png
    :class: pynb



Performance Measurement
-----------------------

For a more complete view of each asset's performance over the period, we can use the :func:`ffn.core.calc_stats` method which will create a :class:`ffn.core.GroupStats` object. A GroupStats object wraps a bunch of :class:`ffn.core.PerformanceStats` objects in a dict with some added convenience methods.

.. code:: python

    perf = data.calc_stats()

Now that we have our GroupStats object, we can analyze the performance
in greater detail. For example, the **plot** method yields a graph
similar to the one above.

.. code:: python

    perf.plot()



.. image:: _static/quickstart_20_0.png
    :class: pynb


We can also display a wide array of statistics that are all contained in
the PerformanceStats object. This will probably look crappy in the docs,
but do try it out in a Notebook. We are also actively trying to improve
the way we display this wide array of stats.

.. code:: python

    print perf.display()


.. parsed-literal::
    :class: pynb-result

    Stat                 agg         hyg         spy         eem         efa         dbc
    -------------------  ----------  ----------  ----------  ----------  ----------  ----------
    Start                2010-01-04  2010-01-04  2010-01-04  2010-01-04  2010-01-04  2010-01-04
    End                  2013-12-31  2013-12-31  2013-12-31  2013-12-31  2013-12-31  2013-12-31
    
    Total Return         16.36%      39.23%      76.91%      5.47%       33.44%      1.66%
    Daily Sharpe         1.11        0.97        0.93        0.18        0.44        0.11
    CAGR                 3.87%       8.65%       15.37%      1.34%       7.50%       0.41%
    Max Drawdown         -5.14%      -10.07%     -18.61%     -30.87%     -25.86%     -24.34%
    
    MTD                  -0.56%      0.41%       2.59%       -0.41%      2.17%       0.59%
    3m                   0.02%       3.42%       10.52%      3.47%       6.07%       -0.39%
    6m                   0.57%       5.85%       16.32%      9.54%       18.11%      2.11%
    YTD                  -1.97%      5.76%       32.30%      -3.65%      21.44%      -7.63%
    1Y                   -1.97%      5.76%       32.30%      -3.65%      21.44%      -7.63%
    3Y (ann.)            3.08%       7.83%       16.07%      -2.34%      8.17%       -2.34%
    5Y (ann.)            3.87%       8.65%       15.37%      1.34%       7.50%       0.41%
    10Y (ann.)           3.87%       8.65%       15.37%      1.34%       7.50%       0.41%
    Since Incep. (ann.)  3.87%       8.65%       15.37%      1.34%       7.50%       0.41%
    
    Daily Sharpe         1.11        0.97        0.93        0.18        0.44        0.11
    Daily Mean (ann.)    3.86%       8.70%       15.73%      4.35%       9.73%       1.83%
    Daily Vol (ann.)     3.47%       8.98%       16.83%      24.56%      22.32%      16.84%
    Daily Skew           -0.40       -0.55       -0.39       -0.12       -0.26       -0.47
    Daily Kurt           2.28        7.52        4.02        3.06        3.64        2.90
    Best Day             0.84%       3.06%       4.65%       7.20%       6.75%       4.34%
    Worst Day            -1.24%      -4.27%      -6.51%      -8.34%      -7.46%      -6.70%
    
    Monthly Sharpe       1.23        1.11        1.22        0.30        0.60        0.27
    Monthly Mean (ann.)  3.59%       9.51%       16.99%      6.43%       11.06%      4.61%
    Monthly Vol (ann.)   2.93%       8.56%       13.92%      21.45%      18.41%      17.10%
    Monthly Skew         -0.34       0.14        -0.32       -0.10       -0.37       -0.74
    Monthly Kurt         0.02        1.74        0.24        1.28        0.17        1.16
    Best Month           1.77%       8.49%       10.92%      16.27%      11.63%      9.89%
    Worst Month          -2.00%      -5.30%      -7.94%      -17.89%     -11.19%     -14.62%
    
    Yearly Sharpe        0.65        2.79        1.10        -0.06       0.50        -0.40
    Yearly Mean          3.16%       7.85%       16.73%      -1.13%      9.32%       -2.24%
    Yearly Vol           4.86%       2.81%       15.22%      19.06%      18.72%      5.57%
    Yearly Skew          -0.54       1.50        0.22        0.58        -1.69       0.27
    Yearly Kurt          -           -           -           -           -           -
    Best Year            7.70%       11.05%      32.30%      19.06%      21.44%      3.50%
    Worst Year           -1.97%      5.76%       1.90%       -18.80%     -12.23%     -7.63%
    
    Avg. Drawdown        -0.48%      -1.18%      -1.79%      -5.16%      -4.96%      -5.09%
    Avg. Drawdown Days   17.16       15.69       17.55       78.22       60.04       107.85
    Avg. Up Month        0.83%       1.86%       3.58%       5.87%       4.37%       4.28%
    Avg. Down Month      -0.49%      -2.31%      -3.21%      -3.41%      -4.15%      -3.35%
    Win Year %           66.67%      100.00%     100.00%     33.33%      66.67%      33.33%
    Win 12m %            81.08%      97.30%      94.59%      59.46%      70.27%      45.95%
    None


Lots to look at here. We can also access the underlying PerformanceStats
for each series, either by index or name.

.. code:: python

    # we can also use perf[2] in this case
    perf['spy'].display_monthly_returns()


.. parsed-literal::
    :class: pynb-result

      Year    Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec    YTD
    ------  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  -----
      2010  -5.24   3.12   6.08   1.55  -7.94  -5.18   6.83  -4.5    8.96   3.82   0      6.68  13.14
      2011   2.33   3.47   0.01   2.9   -1.12  -1.69  -2     -5.5   -6.94  10.92  -0.41   1.05   1.9
      2012   4.64   4.34   3.22  -0.67  -6      4.06   1.18   2.51   2.54  -1.82   0.56   0.9   15.99
      2013   5.12   1.27   3.8    1.92   2.36  -1.34   5.17  -3      3.17   4.63   2.97   2.59  32.3


.. code:: python

    perf[2].plot_histogram()



.. image:: _static/quickstart_25_0.png
    :class: pynb


Most of the stats are also available as pandas objects - see the
**stats, return\_table, lookback\_returns** attributes.

.. code:: python

    perf['spy'].stats




.. parsed-literal::
    :class: pynb-result

    start                    2010-01-04 00:00:00
    end                      2013-12-31 00:00:00
    total_return                       0.7691415
    daily_sharpe                       0.9343834
    cagr                               0.1537473
    max_drawdown                      -0.1860885
    mtd                               0.02589976
    three_month                        0.1052059
    six_month                          0.1631602
    ytd                                0.3230191
    three_year                         0.1606551
    daily_mean                          0.157277
    daily_vol                          0.1683217
    daily_skew                        -0.3877139
    daily_kurt                          4.021851
    best_day                          0.04646752
    worst_day                        -0.06507669
    monthly_sharpe                      1.220851
    monthly_mean                       0.1699025
    monthly_vol                        0.1391673
    monthly_skew                      -0.3188291
    monthly_kurt                       0.2366642
    best_month                          0.109239
    worst_month                      -0.07943796
    yearly_sharpe                       1.099516
    yearly_mean                        0.1673016
    yearly_vol                         0.1521593
    yearly_skew                        0.2179042
    yearly_kurt                              NaN
    worst_year                         0.0189695
    avg_drawdown                     -0.01785071
    avg_drawdown_days                   17.55072
    avg_up_month                      0.03583112
    avg_down_month                   -0.03207629
    win_year_perc                              1
    twelve_month_win_perc              0.9459459
    dtype: object




Numerical Routines and Financial Functions
------------------------------------------

ffn also provides commonly used numerical routines and plans to add many more in the future. One can easily determine the proper weights using a mean-variance approach using the :func:`ffn.core.calc_mean_var_weights` function.

.. code:: python

    returns.calc_mean_var_weights().as_format('.2%')




.. parsed-literal::
    :class: pynb-result

    agg    79.57%
    dbc     0.00%
    eem     0.00%
    efa     0.00%
    hyg     6.39%
    spy    14.03%
    dtype: object



Some other interesting functions are the clustering routines, such as a
Python implementation of David Varadi's Fast Threshold Clustering
Algorithm (FTCA)

.. code:: python

    returns.calc_ftca(threshold=0.8)




.. parsed-literal::
    :class: pynb-result

    {1: ['eem', 'spy', 'efa'], 2: ['agg'], 3: ['dbc'], 4: ['hyg']}


