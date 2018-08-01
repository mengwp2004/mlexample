#!/usr/bin/python
# -*- coding: utf-8 -*-


from sqlalchemy import create_engine 
import tushare as ts 
import pandas as pd

def init():
  engine = create_engine('mysql://root:mengwp_2004@127.0.0.1/china_stock_07_31?charset=utf8') 
  return engine

def save_tick_data(engine):
  df = ts.get_tick_data('600848', date='2014-12-22') 
  print(df)
  #存入数据库 
  df.to_sql('tick_data',engine)

def save_code_data(code,engine):
  df = get_data(code) 
  #print(df)
  #存入数据库
  try:
     if df is None:
       print("get %s none" %(code))
     else:
       df.to_sql('tick_data_' + code,engine,if_exists='append')
    
  except:
     print("insert %s except" %(code))
     if df is None:
        print("get %s none" %(code))
     else:
       df.to_sql('tick_data_' + code,engine,if_exists='append')

def save_code_datas(codes,engine):
   for code in codes:
     save_code_data(code,engine)

def update_db_to_db(filename):
   engine = init()
   datalist = get_code(filename)
   save_code_datas(datalist,engine)

def get_data_from_db(filename):
   engine = init()
   codelist = get_code(filename)
   dfs = read_codes_data(codelist,engine)
   return dfs

def read_codes_data(codelist,engine):
    
    i = 0
    for code in codelist:
       df = read_code_data(code,engine)
       if df is None:
          continue
       else:
          
          if i == 0:
            dfs = df
            i = i+1
          else:
            pd.concat([dfs,df])
    print(dfs)
    print("len %d" %(len(dfs)))
    return dfs   
             

def read_code_data(code,engine):
   df = pd.read_sql('tick_data_' + code,engine)
   print("code %s len=%d" %(code,len(df)))
   return df   

def get_data(label):    
   return ts.get_hist_data(label)

def get_data_list_from_db(filename):
    print('get data list from db')    
    engine = init()
    codelist = get_code(filename)
    dfs =[] 
    i = 0
    for code in codelist: 
       df = read_code_data(code,engine)
       if df is None:
         print('code %s table none' %(code))
         continue
       dfs.append(df)
       i = i+1
       #print(i)
       if (i >= 10):
         break
    #print(dfs)
    print("len %d" %(len(dfs)))
      
    return dfs

def get_code(filename):
    datalist =[]
    try:
      f = open(filename, 'r')
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if(len(line) >6):
          line = line[:6]
        if line != None:
          datalist.append(line)
    finally:
      if f:
        f.close()  
     
    return datalist 

def test():
  code = '000001'
  save_code_data(code,engine)
  read_code_data(code,engine)

if __name__ == "__main__":
   filename = "./code.txt"
#  #update_data_to_db(filename)
   get_data_list_from_db(filename)
