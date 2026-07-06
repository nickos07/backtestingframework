"""
Stock Data Acquisition Module
Downloads historical OHLCV data for multiple stocks using yfinance.
This module serves as the data pipeline for the ML stock prediction project.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


def download_stock_data(tickers, period_years=5, interval='1d'):
    """
    Download historical OHLCV data for specified stock tickers.
    
    Parameters:
    -----------
    tickers : list or str
        Stock ticker symbols (e.g., ['AAPL', 'MSFT'])
    period_years : int, default=5
        Number of years of historical data to download
    interval : str, default='1d'
        Data interval ('1d' for daily, '1h' for hourly, etc.)
    
    Returns:
    --------
    dict
        Dictionary with ticker symbols as keys and DataFrames as values
        Each DataFrame contains OHLCV data with datetime index
    """
    
    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years*365)
    
    print(f"Downloading {interval} stock data...")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Tickers: {', '.join(tickers)}\n")
    
    # Download data for each ticker
    stock_data = {}
    
    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...", end=" ")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                print(f"⚠️  WARNING: No data retrieved for {ticker}")
                continue
            
            stock_data[ticker] = data
            print(f"✓ Success ({len(data)} records)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    return stock_data


def display_data_summary(stock_data):
    """
    Display summary statistics for downloaded stock data.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with stock OHLCV data
    """
    
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    for ticker, df in stock_data.items():
        print(f"\n{ticker}")
        print("-" * 40)
        print(f"Records: {len(df)}")
        print(f"Date Range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Trading Days: {len(df)}")
        print(f"\nOHLCV Statistics:")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().round(2))
        print(f"\nFirst 5 rows:")
        print(df.head())


