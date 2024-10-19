import yfinance as yf
import pandas as pd
import talib
from map_data import DataMapper

def add_new_indicator(indicator_function, indicator_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            data = func(self, *args, **kwargs)
            data[indicator_name] = indicator_function(data)
            return data
        return wrapper
    return decorator


class DataMapperExt(DataMapper):
    @add_new_indicator(talib.SMA, 'SMA_10')
    def add_statistics_to_dataframe(self, close, open, high, low, volume):
        data = super().add_statistics_to_dataframe(data)
        return data

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