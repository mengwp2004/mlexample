#encoding=utf-8
# Load the necessary packages and modules
import pandas as pd
#import pandas.io.data as web
import matplotlib.pyplot as plt

import tushare as ts

# Commodity Channel Index 
def CCI(data, ndays): 
  TP = (data['high'] + data['low'] + data['close']) / 3 
  CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),name = 'CCI') 
  data = data.join(CCI) 
  return data
# Retrieve the Nifty data from Yahoo finance:
#data = web.DataReader('^NSEI',data_source='yahoo',start='1/1/2014', end='1/1/2016')
#data = pd.DataFrame(data)
data = ts.get_hist_data('300104',start='2015-01-01',end='2016-08-01')


# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
n = 20
NIFTY_CCI = CCI(data, n)
CCI = NIFTY_CCI['CCI']
# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
print(type(data['close']))
plt.plot(data['close'].values,lw=1)
plt.title('NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)
plt.plot(CCI.values,'k',lw=0.75,linestyle='-',label='CCI')
plt.legend(loc=2,prop={'size':9.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()
