import re
import pandas as pd
from pykrx import stock
from datetime import datetime,timedelta

class DataLoader:
    def __init__(self,stock_code):
        self.stock_code=stock_code

    #make csv when you don't have stock_code.csv
    def makeNewFile(self):
        df = stock.get_market_ohlcv_by_date("19990101","20201231",self.stock_code)
        df = df.reset_index()
        df.to_csv('./data/chart_data/{}.csv'.format(self.stock_code),sep=',',index=False,header=False)
        print("\n\n{}.csv File Generation Complete!".format(self.stock_code))
    
    #update csv when you  have stock_code.csv
    def updateFile(self):
        df = pd.read_csv("data/chart_data/"+self.stock_code+".csv",names=["date","open","high","low","close","volume"],header=None)
        try:
            dt = datetime.strptime(df.date.iloc[-1],'%Y-%m-%d %H:%M:%S')
        except:
            dt = datetime.strptime(df.date.iloc[-1],'%Y-%m-%d')
        Ndt = dt + timedelta(days=1)
        lastday = Ndt.strftime("%Y%m%d")
        today = datetime.today().strftime("%Y%m%d")
        if int(lastday) < int(datetime.today().strftime("%Y%m%d")):
            df2 = stock.get_market_ohlcv_by_date(lastday, today, self.stock_code)
            df2 = df2.reset_index()
            df2.rename(columns={"날짜":"date","시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"},inplace=True)
            df_new = pd.concat([df,df2],ignore_index=True)
            df_new.to_csv('./data/chart_data/{}.csv'.format(self.stock_code),sep=',',index=False,header=False)
            print("\n\n{}.csv File Update Complete!".format(self.stock_code))
        