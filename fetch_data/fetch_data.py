import yfinance as yf
import pandas as pd
import talib
from map_data_new import DataMapper
from map_data_ext import DataMapperExt
from add_long_stats import AddLongStats
import os
class DataFetcher:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.types = ["day", "week", "month", "quarter"]
        self.data_processor = DataMapper(self.symbol, self.start_date, 
                                              self.end_date)
        self.add_long_stats = AddLongStats(self.symbol)
    def fetch_historical_data(self):
        for type in self.types:
            csv_file_path = "/home/raj/Rajarshi/Term Project/notebook_files/data" + self.symbol + "_" + type + ".csv"
            self.data_processor.process_and_save_data(type, csv_file_path)        

        self.add_long_stats.merge_data("week", "wk")
        self.add_long_stats.merge_data("month", "mon")

symbol = '^NSEI'
# symbol = "AXISBANK.NS"
# symbol = "TATAMOTORS.NS"
# symbol = "SBIN.NS"
start_date = '1995-01-01'
end_date = '2022-12-30'
fetcher = DataFetcher(symbol, start_date, end_date)
fetcher.fetch_historical_data()