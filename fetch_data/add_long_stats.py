import yfinance as yf
import pandas as pd
import os

class AddLongStats:
    def __init__(self, symbol):
        self.symbol = symbol

    def merge_data(self, type, long_prefix):
        short_file = "../final/data/" + self.symbol + "_" + "day" + ".csv" 
        long_file = "../final/data/" + self.symbol + "_" + type + ".csv"
        
        short_data = pd.read_csv(short_file, parse_dates=['Date'])
        # Get the column names
        column_names = short_data.columns

        # Print the column names
        print(column_names)
        long_data = pd.read_csv(long_file, parse_dates=['Date'])

        # Initialize empty lists for weekly values
        long_close = []
        long_rsi = []
        long_macd = []

        # Process date column in each row in day.csv
        for index, row in short_data.iterrows():
            date = row['Date']

            # Find the first corresponding weekly row in the weekly data
            weekly_rows = long_data[long_data['Date'] <= date]

            if not weekly_rows.empty:
                # Choose the most recent (first) row from the selected weekly rows
                last_weekly_row = weekly_rows.iloc[-1]
                long_close.append(last_weekly_row['Close'])
                long_rsi.append(last_weekly_row['RSI'])
                long_macd.append(last_weekly_row['MACD'])
            else:
                # If there's no corresponding weekly data, use NaN or any default value
                long_close.append(None)
                long_rsi.append(None)
                long_macd.append(None)

        # Add weekly values to day.csv
        col_close = long_prefix + "_" + "close"
        col_rsi = long_prefix + "_" + "rsi"
        col_macd = long_prefix + "_" + "macd"
        short_data[col_close] = long_close
        short_data[col_rsi] = long_rsi
        short_data[col_macd] = long_macd

        # Save the updated data to a new CSV file
        filename = "../final/data/" + self.symbol + "_" + "day" + ".csv" 
        short_data.to_csv(filename, index=False)
