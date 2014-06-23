
# coding: utf-8

# In[1]:

import ffn
get_ipython().magic(u'pylab inline')


# In[2]:

# download price data from Yahoo! Finance. By default, 
# the Adj. Close will be used. 
prices = ffn.get('aapl,msft', start='2010-01-01')


# In[3]:

# let's compare the relative performance of each stock 
# we will rebase here to get a common starting point for both securities
ax = prices.rebase().plot()


# In[4]:

# now what do the return distributions look like?
returns = prices.to_returns().dropna()
ax = returns.hist(figsize(10, 5))


# In[5]:

# ok now what about some performance metrics?
stats = prices.calc_stats()
stats.display()


# In[10]:

# what about the drawdowns?
ax = stats.prices.to_drawdown_series().plot()

