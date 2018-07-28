#!/usr/bin/python  
# coding: UTF-8  
      
"""This script parse stock info"""  


import matplotlib.pyplot as plt  

     
import tushare as ts  
      
def get_all_price(code_list):  
        '''''process all stock'''  
        df = ts.get_realtime_quotes(STOCK)  
        print df  

        df = ts.get_latest_news()   
        print df     

        df = ts.get_cpi()
        print df


        df =ts.get_stock_basics()  
        print df

        df = ts.get_sz50s() 
        print df

        df = ts.get_hist_data('600848')  
        print df

if __name__ == '__main__':  
        STOCK = ['600219',       ##南山铝业  
                 '000002',       ##万  科Ａ  
                 '000623',       ##吉林敖东  
                 '000725',       ##京东方Ａ  
                 '600036',       ##招商银行  
                 '601166',       ##兴业银行  
                 '600298',       ##安琪酵母  
                 '600881',       ##亚泰集团  
                 '002582',       ##好想你  
                 '600750',       ##江中药业  
                 '601088',       ##中国神华  
                 '000338',       ##潍柴动力  
                 '000895',       ##双汇发展  
                 '000792']       ##盐湖股份  
      
        #get_all_price(STOCK) 
        data_frame = ts.get_hist_data('300104',start='2015-01-01',end='2016-08-01')
        for index, row in data_frame.iterrows():
           # 这里修改为组装SQL并执行的语句
           print index,row 

        
        data_frame['close'].plot(legend=True, figsize=(10,4))  
        plt.show()  
