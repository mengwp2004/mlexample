# Moving Averages Code

# Load the necessary packages and modules
import pandas as pd
#import pandas.io.data as web
#from pandas_datareader import data,wb  
import tushare as ts
import matplotlib.pyplot as plt

# Simple Moving Average 
def SMA(data, ndays): 
 SMA = pd.Series(pd.rolling_mean(data['close'], ndays), name = 'SMA') 
 #SMA = pd.Series.rolling(window=50,center=False).mean()
 data = data.join(SMA) 
 return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
 EMA = pd.Series(pd.ewma(data['close'], span = ndays, min_periods = ndays - 1), 
 name = 'EWMA_' + str(ndays)) 
 data = data.join(EMA) 
 return data

# Retrieve the Nifty data from Yahoo finance:
#data1 = data.DataReader('^NSEI',data_source='yahoo',start='1/1/2013', end='1/1/2016')

start_date='1/1/2013'

end_date='1/1/2016'

data = ts.get_hist_data('600000',start='2015-01-01',end='2016-08-01')

#data1 = data.DataReader(
#    	symbol, "yahoo", 
#    	start_date, 
#    	end_date
#    )


data = pd.DataFrame(data) 
print(data)
close = data['close']

print('close:')
print(close)
# Compute the 50-day SMA for NIFTY
n = 50
SMA_NIFTY = SMA(data,n)
SMA_NIFTY = SMA_NIFTY.dropna()
SMA = SMA_NIFTY['SMA']

# Compute the 200-day EWMA for NIFTY
ew = 200
EWMA_NIFTY = EWMA(data,ew)
EWMA_NIFTY = EWMA_NIFTY.dropna()
EWMA = EWMA_NIFTY['EWMA_200']

# Plotting the NIFTY Price Series chart and Moving Averages below
plt.figure(figsize=(9,5))
#plt.plot(data['close'],lw=1, label='NSE Prices')

print(close[1])
print(close.values)
print(type(close))
print(close.shape)
#plt.plot(close.values,close.index,lw=1, label='NSE Prices')
values = close.values

print(values)
print(type(values))
plt.plot(values, label='NSE Prices')


plt.plot(SMA.values,'g',lw=1, label='50-day SMA (green)')
plt.plot(EWMA.values,'r', lw=1, label='200-day EWMA (red)')
plt.legend(loc=2,prop={'size':11})
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()
