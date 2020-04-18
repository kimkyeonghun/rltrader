import os
import settings
import pandas as pd
import numpy as np
from data_loader import DataLoader


if __name__ == '__main__':

    stockList = pd.read_csv(os.path.join(settings.BASE_DIR,'data/kospi100.csv'))
    for s in range(len(stockList)):
        print(stockList.loc[s,'종목명'])
        if len(str(stockList.loc[s,'종목코드']))<6:
            stock_code= '0'*(6-len(str(stockList.loc[s,'종목코드'])))+str(stockList.loc[s,'종목코드'])
        else:
            stock_code = str(stockList.loc[s,'종목코드'])

        #stock_code = input("Enter Stock Code : ")

        stock = DataLoader(stock_code)

        if stock_code+".csv" not in [file for file in os.listdir('./data/chart_data') if file.endswith(".csv")]:
            stock.makeNewFile()
        else:
            stock.updateFile()


        chart_data = pd.read_csv(os.path.join(settings.BASE_DIR,
                            'data/chart_data/{}.csv'.format(stock_code)), thousands=',', header=None)
        chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        chart_data = chart_data[(chart_data['date']>= '2017-01-01') & 
                                (chart_data['date'] <= '2018-12-31')]
        chart_data = chart_data.dropna().reset_index()
        chart_data["plusDI"] = 0
        chart_data["minusDI"] = 0
        for i in range(1,len(chart_data)):
            chart_data.loc[i,'plusDI']=chart_data.loc[i,'high']-chart_data.loc[i-1,'high']
            chart_data.loc[i,'minusDI']=chart_data.loc[i-1,'low']-chart_data.loc[i,'low']
            chart_data.loc[i,'TR'] = max(chart_data.loc[i,'high']-chart_data.loc[i,'low'],chart_data.loc[i,'high']-chart_data.loc[i-1,'close'],chart_data.loc[i,'low']-chart_data.loc[i-1,'close'])
            
        chart_data["PDI{}".format(14)]=chart_data["plusDI"].rolling(14).mean()/chart_data["TR"].rolling(14).mean()
        chart_data["MDI{}".format(14)]=chart_data["minusDI"].rolling(14).mean()/chart_data["TR"].rolling(14).mean()

        PC=0
        MC=0

        for i in range(len(chart_data)):
            if not chart_data.loc[i].isnull()['PDI14']:
                if chart_data.loc[i,'PDI14'] > chart_data.loc[i,'MDI14']:
                    PC+=1
                else:
                    MC+=1
        stockList.loc[s,'PC']=PC
        stockList.loc[s,'MC']=MC

        stockList.to_csv("./kospi100_DMI.csv",columns=['name','code','PC','MC'],encoding='euc-kr')
