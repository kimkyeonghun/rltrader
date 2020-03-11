import re
import pandas as pd
from pykrx import stock
from datetime import datetime

class DataLoader:
    def __init__(self,stock_code):
        self.stock_code=stock_code

    def makeNewFile(self):
        df = stock.get_market_ohlcv_by_date("19990101","20201231",self.stock_code)
        df = df.reset_index()
        df.to_csv('./data/chart_data/{}.csv'.format(self.stock_code),sep=',',index=False,header=False)
        print("\n\n{}.csv File Generation Complete!".format(self.stock_code))
    
    def updateFile(self):
        df = pd.read_csv("data/chart_data/"+self.stock_code+".csv",names=["date","open","high","low","close","volume"],header=None)
        p = re.compile('[0-9]+')
        m = p.findall(df.date.iloc[-1])
        lastday = m[0]+m[1]+str(int(m[2])+1)
        yesterday = str(int(datetime.today().strftime("%Y%m%d"))-1)
        if m[0]+m[1]+m[2]!= datetime.today().strftime("%Y%m%d"):
            df2 = stock.get_market_ohlcv_by_date(lastday, yesterday, self.stock_code)
            df2 = df2.reset_index()
            df2.rename(columns={"날짜":"date","시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"},inplace=True)
            df_new = pd.concat([df,df2],ignore_index=True)
            df_new.to_csv('./data/chart_data/{}.csv'.format(self.stock_code),sep=',',index=False,header=False)
            print("\n\n{}.csv File Update Complete!".format(self.stock_code))
        