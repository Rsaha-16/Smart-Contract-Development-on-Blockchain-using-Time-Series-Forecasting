import yfinance as yf
import pandas as pd
import talib

class DataMapper:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def fetch_historical_data(self):
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        return data

    def calculate_rsi(self, data, timeperiod=14):
        rsi = talib.RSI(data, timeperiod=timeperiod)
        return rsi

    def calculate_stochastic(self, high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
        print(len(high))
        print(len(low))
        print(len(close))
        slowk, slowd = talib.STOCH(high, low, close, fastk_period, slowk_period, slowd_period)
        return slowk, slowd

    def calculate_macd(self, data, fastperiod=12, slowperiod=26, signalperiod=9):
        macd, signal, _ = talib.MACD(data, fastperiod, slowperiod, signalperiod)
        return macd, signal

    def calculate_bollinger_bands(self, data, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        upperband, middleband, lowerband = talib.BBANDS(data, timeperiod, nbdevup, nbdevdn, matype)
        return upperband, middleband, lowerband

    def calculate_adx(self, high, low, close, timeperiod=14):
        adx = talib.ADX(high, low, close, timeperiod=timeperiod)
        return adx

    def add_statistics_to_dataframe(self, close, open, high, low, volume):
        rsi = self.calculate_rsi(close)
        #slowk, slowd = self.calculate_stochastic(high, low, close)
        macd, signal = self.calculate_macd(close)
        upperband, middleband, lowerband = self.calculate_bollinger_bands(close)
        adx = self.calculate_adx(high, low, close)

        data = pd.DataFrame({
            'Close': close,
            'Open': open,
            'High': high,
            'Low': low,
            'Volume': volume,
            'RSI': rsi,
            'MACD': macd,
            'MACD Signal': signal,
            'Upper Bollinger Band': upperband,
            'Middle Bollinger Band': middleband,
            'Lower Bollinger Band': lowerband,
            'ADX': adx
            
            #'Stochastic SlowK': slowk,
            #'Stochastic SlowD': slowd
        })
        return data

    def process_and_save_data(self, type, csv_file_path):
        data = self.fetch_historical_data()
        rat = data['Adj Close'] / data['Close']
        rat = 1
        if type == "week":
            #close = data['Close'].resample('W').last()
            close = data['Adj Close'].resample('W').last()
            high = data['High'].resample('W').last() *rat
            low = data['Low'].resample('W').last() * rat
            open = data['Open'].resample('W').last() *rat
            vol = data['Volume'].resample('W').last()
        elif type == "month":
            close = data['Adj Close'].resample('M').last()
            high = data['High'].resample('M').last() * rat
            low = data['Low'].resample('M').last() * rat
            open = data['Open'].resample('M').last() * rat
            vol = data['Volume'].resample('M').last()
        elif type == "quarter":
            close = data['Adj Close'].resample('Q').last()
            high = data['High'].resample('Q').last() * rat
            low = data['Low'].resample('Q').last() * rat
            open = data['Open'].resample('Q').last() * rat
            vol = data['Volume'].resample('Q').last()
        else:
            close = data['Adj Close']
            high = data['High'] * rat
            low = data['Low'] * rat
            open = data['Open'] * rat
            vol = data['Volume'] 


        data = self.add_statistics_to_dataframe(close, open, high, low, vol)

        data.to_csv(csv_file_path)
        print(f"Data saved to {csv_file_path}")

'''
# Example usage:
if __name__ == '__main__':
    symbol = '^NSEI'
    start_date = '1995-01-01'
    end_date = '2023-08-01'
    type = "day"
    csv_file_path = './data/try/nifty50_daily_data.csv'

    nifty_data_processor = DataMapper(symbol, start_date, end_date, type, csv_file_path)
    nifty_data_processor.process_and_save_data()
'''