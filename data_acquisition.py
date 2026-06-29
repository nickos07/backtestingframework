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


def save_data_to_csv(stock_data, output_dir='data'):
    """
    Save downloaded stock data to CSV files.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with stock OHLCV data
    output_dir : str, default='data'
        Directory to save CSV files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n" + "="*80)
    print("SAVING DATA")
    print("="*80)
    
    for ticker, df in stock_data.items():
        filepath = os.path.join(output_dir, f"{ticker}_5years.csv")
        df.to_csv(filepath)
        print(f"✓ Saved {ticker} data to: {filepath}")
    
    print(f"\nAll data saved to: {os.path.abspath(output_dir)}")


def merge_stock_data(stock_data, use_close_only=False):
    """
    Merge multiple stock DataFrames into a single combined DataFrame.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with stock OHLCV data
    use_close_only : bool, default=False
        If True, only merge 'Close' prices; otherwise merge all columns
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with multi-level columns (ticker, OHLCV)
    """
    
    if use_close_only:
        # Merge only Close prices
        close_prices = pd.DataFrame()
        for ticker, df in stock_data.items():
            close_prices[ticker] = df['Close']
        return close_prices
    else:
        # Merge all OHLCV data with multi-level columns
        merged = pd.concat(stock_data, axis=1)
        return merged


def main():
    """Main execution function"""
    
    # Stock tickers to download
    tickers = ['AAPL', 'MSFT']
    
    # Download data
    stock_data = download_stock_data(
        tickers=tickers,
        period_years=5,
        interval='1d'
    )
    
    # Check if data was successfully downloaded
    if not stock_data:
        print("❌ No data was downloaded. Exiting.")
        return
    
    # Display summary
    display_data_summary(stock_data)
    
    # Save to CSV
    save_data_to_csv(stock_data, output_dir='data')
    
    # Create merged Close prices for easy access
    close_prices = merge_stock_data(stock_data, use_close_only=True)
    print(f"\n" + "="*80)
    print("CLOSE PRICES SUMMARY")
    print("="*80)
    print(f"\nShape: {close_prices.shape}")
    print(f"Date Range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print("\nFirst 5 rows:")
    print(close_prices.head())
    
    # Save merged data
    close_prices.to_csv('data/close_prices_5years.csv')
    print(f"\n✓ Saved merged close prices to: data/close_prices_5years.csv")
    
    return stock_data, close_prices


if __name__ == "__main__":
    stock_data, close_prices = main()
