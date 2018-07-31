#!/usr/bin/python
# -*- coding: utf-8 -*-

# forecast.py

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import sklearn
import tushare as ts
import update_data_to_db  

#from pandas.io.data import DataReader
from pandas_datareader import data, wb  
import matplotlib.pyplot as plt  


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix
#from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC


def get_data(label):
    
       return ts.get_hist_data(label)

def get_data_list(datalist):
   
    datas =[] 
    for data in datalist: 
       datas.append(ts.get_hist_data(data))
       
    return datas

def get_code(filename):
    datalist =[]
    try:
      f = open(filename, 'r')
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if(len(line) >6):
          line = line[:6]
        datalist.append(line)
    finally:
      if f:
        f.close()  
     
    return datalist 

def get_data_from_db():
    filename = './code.txt'
    return update_data_to_db.get_data_list_from_db(filename)

def create_lagged_series(datalist, start_date, end_date, lags=5):
    """
    This creates a pandas DataFrame that stores the 
    percentage returns of the adjusted closing value of 
    a stock obtained from Yahoo Finance, along with a 
    number of lagged returns from the prior trading days 
    (lags defaults to 5 days). Trading volume, as well as 
    the Direction from the previous day, are also included.
    """
    
    tsrets =[]   
    # Obtain stock information from Yahoo Finance
    #tss = get_data_list(datalist)
    tss = get_data_from_db()
    for ts in tss:
      ts =ts.sort_index(axis = 0,ascending = True) 
      print(ts)
      print(type(ts))
      print(type(ts['close']))
      # Create the new lagged DataFrame
      tslag = pd.DataFrame(index=ts.index)
      tslag["Today"] = ts["close"]
      tslag["Volume"] = ts["volume"]
    
      # Create the shifted lag series of prior trading period close values
      for i in range(0, lags):
        tslag["Lag%s" % str(i+1)] = ts["close"].shift(i+1)
      print(tslag)
      # Create the returns DataFrame
      tsret = pd.DataFrame(index=tslag.index)
      tsret["Volume"] = tslag["Volume"]
      tsret["Today"] = tslag["Today"].pct_change()*100.0

      # If any of the values of percentage returns equal zero, set them to
      # a small number (stops issues with QDA model in scikit-learn)
      for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

      # Create the lagged percentage returns columns
      for i in range(0, lags):
        tsret["Lag%s" % str(i+1)] = \
        tslag["Lag%s" % str(i+1)].pct_change()*100.0
      print(tsret)
      # Create the "Direction" column (+1 or -1) indicating an up/down day
      tsret["Direction"] = np.sign(tsret["Today"])
      print(tsret.index)
      #tsret = tsret[tsret.index >= start_date]
      tsrets.append(tsret)
    return tsrets

def get_data_set_from_db(snprets,start_train,start_test):

    #X_trains = pd.DataFrame()
    #y_trains = pd.Serias()
    #X_tests = pd.DataFrame()
    #y_tests = pd.DataFrame()
    i =0 
    for snpret in snprets:
      X = snpret[["Lag1","Lag2"]]
      y = snpret["Direction"]

      # The test data is split into two parts: Before and after 1st Jan 2005.
      # Create training and test sets
      X_train = X[X[date] < start_test]
      X_train = X_train[X_train[date] >= start_train]
      X_test = X[X[date] >= start_test ]
      y_train = y[y.date < start_test]
      y_train = y_train[y_train.date >= start_train]
      y_test = y[y.date >= start_test]
      if i==0:
         X_trains = X_train
         y_trains = y_train
         X_tests = X_test
         y_tests = y_test
         i = i+1
      else:
         X_trains =pd.concat([X_trains,X_train])
         y_trains =pd.concat([y_trains,y_train])
         X_tests =pd.concat([X_tests,X_test])
         y_tests =pd.concat([y_tests,y_test])

      print("type(y_train) %s " % (type(y_train)))
       
    #X_trains = X_trains.reset_index(drop = True)
    nanList = np.where(np.isnan(X_trains))[0]
    print(len(X_trains))
    print(nanList)

    return X_trains,X_tests,y_trains,y_tests 
 
def get_data_set(snprets,start_train,start_test):

    #X_trains = pd.DataFrame()
    #y_trains = pd.Serias()
    #X_tests = pd.DataFrame()
    #y_tests = pd.DataFrame()
    i =0 
    for snpret in snprets:
      X = snpret[["Lag1","Lag2"]]
      y = snpret["Direction"]

      # The test data is split into two parts: Before and after 1st Jan 2005.
      # Create training and test sets
      X_train = X[X.index < start_test]
      X_train = X_train[X_train.index >= start_train]
      X_test = X[X.index >= start_test ]
      y_train = y[y.index < start_test]
      y_train = y_train[y_train.index >= start_train]
      y_test = y[y.index >= start_test]
      if i==0:
         X_trains = X_train
         y_trains = y_train
         X_tests = X_test
         y_tests = y_test
         i = i+1
      else:
         X_trains =pd.concat([X_trains,X_train])
         y_trains =pd.concat([y_trains,y_train])
         X_tests =pd.concat([X_tests,X_test])
         y_tests =pd.concat([y_tests,y_test])

      print("type(y_train) %s " % (type(y_train)))
       
    #X_trains = X_trains.reset_index(drop = True)
    nanList = np.where(np.isnan(X_trains))[0]
    print(len(X_trains))
    print(nanList)

    return X_trains,X_tests,y_trains,y_tests 
     

if __name__ == "__main__":

    datalist =['300104','000001']
    name = "./code.txt"
    #codelist = get_code(name) 
    #print(codelist)
    
    # Create a lagged series of the S&P500 US stock market index
    snprets = create_lagged_series(
    	datalist, datetime.datetime(2015,9,10), 
    	datetime.datetime(2017,12,31), lags=5
    )
    label = '300104'
    # Use the prior two days of returns as predictor 
    # values, with direction as the response
    start_test = datetime.datetime(2016,1,1)
    start_train = u'2015-08-06'
    print(start_test)
    start_test = u'2017-08-29'    
    end_test = u'2018-07-24'

    X_train,X_test,y_train,y_test = get_data_set_from_db(snprets,start_train,start_test)

    # Create the (parametrised) models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()), 
            #  ("LDA", LDA()), 
            #  ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
              	C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
              ),
              ("RF", RandomForestClassifier(
              	n_estimators=1000, criterion='gini', 
                max_depth=None, min_samples_split=2, 
                min_samples_leaf=1, max_features='auto', 
                bootstrap=True, oob_score=False, n_jobs=1, 
                random_state=None, verbose=0)
              )]

    # Iterate through the models
    for m in models:
        
        # Train each of the models on the training set
        m[1].fit(X_train, y_train)

        # Make an array of predictions on the test set
        pred = m[1].predict(X_test)

        # Output the hit-rate and the confusion matrix for each model
        print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
        print("%s\n" % confusion_matrix(pred, y_test))


    #start_date = datetime.datetime(2001,1,10)
    #end_date = datetime.datetime(2005,12,31)
    #symbol =  "^GSPC"
    #ts = data.DataReader(
    #	symbol, "yahoo", 
    #	start_date-datetime.timedelta(days=365), 
    #	end_date
    #)
    #ts = get_data(label)
    #ts =ts.sort_index(axis = 0,ascending = True) 
    #ts['close'].plot(legend=True, figsize=(10,4))
    #plt.show()
 
