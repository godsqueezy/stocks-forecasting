#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the yfinance to get stock's data https://pypi.org/project/yfinance/
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
#importing Facebook Prophet for forecasting https://facebook.github.io/prophet/docs/installation.html#python
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import numpy as np
#importing sklearn to see the perfomance of the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# In[2]:


#getting and printing data of input stock by user
while True:
    ticker = input('''Please input stock's ticker like "AAPL" > ''')
    ticker = ticker.upper()
    picked_stock = yf.Ticker(ticker)
    #checking if a ticker that input by user existing
    try:
        if (picked_stock.info['regularMarketPrice'] != None):
            break
    except:
        continue

#all the data of a stock, we only need Adj Close price and Date
data = yf.download(ticker, period="max")
print(data)


# In[3]:


#showing the price of a stock with Adj Close price which is more accurate than Close price
plt.figure(figsize=(15, 5))
data['Adj Close'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.suptitle(ticker)
plt.show()


# In[4]:


#resetting index so Date will become a column and making table with only needed data for analysis
data = data.reset_index()
data = data[['Date','Adj Close']]
#renaming columns for Prophet because the input to Prophet is always a dataframe with two columns: ds and y
data = data.rename(columns = {'Date':'ds','Adj Close':'y'}) 
print(data)


# In[5]:


#building a model using this guide https://medium.com/analytics-vidhya/predicting-stock-prices-using-facebooks-prophet-model-b1716c733ea6
#https://facebook.github.io/prophet/docs/quick_start.html#python-api
model = Prophet(daily_seasonality=True)
#Fiting the model 
model.fit(data)
#predicting the price for the next 365 days
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)


# In[6]:


#showing predicted data, using fbprophet.plot module
plot_plotly(model, forecast)


# In[7]:


#creating table with original price and predicted price to see the performance of the model
# y - original price
# yhat - forecasted price
cmp_data = forecast.set_index('ds')[['yhat']].join(data.set_index('ds'))
cmp_data = cmp_data.dropna()
print(cmp_data)


# In[8]:


#creating array of orginalPrice and predictedPrice from the table and calculating the r2 score using sklearn on historical data
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
orginalPrice = cmp_data['y'].to_numpy()
predictedPrice = cmp_data['yhat'].to_numpy()
print('R2 score is', r2_score(orginalPrice, predictedPrice))


# In[ ]:




