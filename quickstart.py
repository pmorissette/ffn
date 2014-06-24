
# coding: utf-8

# In[1]:

import ffn
get_ipython().magic(u'pylab inline')


# ## Data Retrieval
# The main method for data retrieval is the **get** function. The get function uses a data provider to download data from an external service and packs that data into a pandas DataFrame for further manipulation.

# In[2]:

data = ffn.get('agg,hyg,spy,eem,efa', start='2010-01-01', end='2014-01-01')
print data.head()


# By default, the data is downloaded from Yahoo! Finance and the Adjusted Close is used as the security's price. Other data sources are also available and you may select other fields as well. Fields are specified by using the following format: {ticker}:{field}. So, if we want to get the Open, High, Low, Close for aapl, we would do the following:

# In[3]:

print ffn.get('aapl:Open,aapl:High,aapl:Low,aapl:Close', start='2010-01-01', end='2014-01-01').head()


# 
# The default data provider is ffn.data.web. This is basically just a thin wrapper around pandas' pandas.io.data provider. Please refer to the appropriate docs for more info (data sources, etc.). The ffn.data.csv provider is also available when we want to load data from a local file. In this case, we can tell **get** to use the csv provider. In this case, we also want to merge this new data with the existing data we downloaded earlier. Therefore, we will provide the **data** object as the *existing* argument, and the new data will be merged into the existing DataFrame.

# In[4]:

data = ffn.get('dbc', provider=ffn.data.csv, path='source/test_data.csv', existing=data)
print data.head()


# As we can see above, the dbc column was added to the DataFrame. Internally, get is using the function ffn.merge, which is useful when you want to merge TimeSeries and DataFrames together. We plan on adding many more data sources over time. If you know your way with Python and would like to contribute a data provider, please feel free to submit a pull request - contributions are always welcome!
# 
# ## Data Manipulation
# 
# Now that we have some data, let's start manipulating it. In quantitative finance, we are often interested in the returns of a given time series. Let's calculate the returns by simply calling the to_returns or to_log_returns extension methods.

# In[5]:

returns = data.to_log_returns().dropna()
print returns.head()


# Let's look at the different distributions to see how they look.

# In[11]:

ax = returns.hist(figsize=(12, 5))


# We can also use the numerous functions packed into numpy, pandas and the like to further analyze the returns. For example, we can use the corr function to get the pairwise correlations between assets.

# In[7]:

returns.corr().as_format('.2f')


# Here we used the convenience method as_format to have a prettier output. We could also plot a heatmap to better visualize the results.

# In[8]:

returns.plot_corr_heatmap()


# We used the plot_corr_heatmap, which is a convenience method that simply calls ffn's plot_heatmap with sane arguments.
# 
# Let's start looking at how all these securities performed over the period. To achieve this, we will plot rebased time series so that we can see how they each performed relative to eachother.

# In[13]:

ax = data.rebase().plot(figsize=(12,5))


# ## Performance Measurement
# 
# For a more complete view of each asset's performance over the period, we can use the **calc_stats** method which will create a GroupStats object. A GroupStats object wraps a bunch of PerformanceStats objects in a dict with some added convenience methods.

# In[14]:

perf = data.calc_stats()


# Now that we have our GroupStats object, we can analyze the performance in greater detail. For example, the **plot** method yields a graph similar to the one above.

# In[15]:

perf.plot()


# We can also display a wide array of statistics that are all contained in the PerformanceStats object. This will probably look crappy in the docs, but do try it out in a Notebook. We are also actively trying to improve the way we display this wide array of stats.

# In[40]:

print perf.display()


# Lots to look at here. We can also access the underlying PerformanceStats for each series, either by index or name.

# In[24]:

# we can also use perf[2] in this case
perf['spy'].display_monthly_returns()


# In[39]:

perf[2].plot_histogram()


# Most of the stats are also available as pandas objects - see the **stats, return_table, lookback_returns** attributes.

# In[28]:

perf['spy'].stats


# ## Numerical Routines and Financial Functions
# 
# ffn also provides commonly used numerical routines and plans to add many more in the future. One can easily determine the proper weights using a mean-variance approach using the **calc_mean_var_weights** method.

# In[31]:

returns.calc_mean_var_weights().as_format('.2%')


# Some other interesting functions are the clustering routines, such as a Python implementation of David Varadi's Fast Threshold Clustering Algorithm (FTCA)

# In[34]:

returns.calc_ftca(threshold=0.8)

