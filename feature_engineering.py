"""
Feature Engineering Module
Generates technical indicators for machine learning stock prediction models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


class TechnicalIndicators:
    """
    A class to compute technical indicators for stock market analysis and ML feature engineering.
    """
    
    @staticmethod
    def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Parameters:
        -----------
        prices : pd.Series
            Series of prices (typically Close prices)
        window : int
            Number of periods for the moving average window
        
        Returns:
        --------
        pd.Series
            Simple Moving Average values
        """
        return prices.rolling(window=window).mean()
    
    
    @staticmethod
    def relative_strength_index(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures the magnitude of recent price changes to evaluate overbought 
        or oversold conditions. Range: 0-100 (typically >70 = overbought, <30 = oversold)
        
        Parameters:
        -----------
        prices : pd.Series
            Series of prices (typically Close prices)
        period : int, default=14
            Number of periods for RSI calculation
        
        Returns:
        --------
        pd.Series
            RSI values (0-100)
        """
        # Calculate daily changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate exponential moving averages
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    
    @staticmethod
    def daily_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate daily percentage returns.
        
        Parameters:
        -----------
        prices : pd.Series
            Series of prices (typically Close prices)
        
        Returns:
        --------
        pd.Series
            Daily percentage returns (as decimals, e.g., 0.02 = 2%)
        """
        return prices.pct_change()
    
    
    @staticmethod
    def add_technical_features(
        df: pd.DataFrame,
        price_column: str = 'Close',
        sma_windows: list = None,
        rsi_period: int = 14,
        add_returns: bool = True
    ) -> pd.DataFrame:
        """
        Add technical indicator features to a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data (must include price_column)
        price_column : str, default='Close'
            Name of the price column to use for calculations
        sma_windows : list, default=None
            Windows for SMA calculation (default: [20, 50])
        rsi_period : int, default=14
            Period for RSI calculation
        add_returns : bool, default=True
            Whether to add daily returns feature
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with original data plus technical indicator features
        """
        
        if sma_windows is None:
            sma_windows = [20, 50]
        
        df_features = df.copy()
        
        # Add Simple Moving Averages
        for window in sma_windows:
            col_name = f'SMA_{window}'
            df_features[col_name] = TechnicalIndicators.simple_moving_average(
                df_features[price_column],
                window=window
            )
        
        # Add RSI
        df_features['RSI_14'] = TechnicalIndicators.relative_strength_index(
            df_features[price_column],
            period=rsi_period
        )
        
        # Add Daily Returns
        if add_returns:
            df_features['Daily_Return'] = TechnicalIndicators.daily_returns(
                df_features[price_column]
            )

        # Add MACD and Signal line
        df_features['EMA_12'] = df_features[price_column].ewm(span=12, adjust=False).mean()
        df_features['EMA_26'] = df_features[price_column].ewm(span=26, adjust=False).mean()
        df_features['MACD'] = df_features['EMA_12'] - df_features['EMA_26']
        df_features['MACD_Signal'] = df_features['MACD'].ewm(span=9, adjust=False).mean()

        # Add Bollinger Bands (20-day SMA with 2 standard deviations)
        bb_window = 20
        bb_middle = df_features[price_column].rolling(window=bb_window).mean()
        bb_std = df_features[price_column].rolling(window=bb_window).std()
        df_features['BB_Upper'] = bb_middle + (2 * bb_std)
        df_features['BB_Lower'] = bb_middle - (2 * bb_std)

        # Add volume percentage change
        if 'Volume' in df_features.columns:
            df_features['Volume_Change'] = df_features['Volume'].pct_change()
        else:
            df_features['Volume_Change'] = np.nan
        
        return df_features


def engineer_features(stock_data: dict, price_column: str = 'Close') -> dict:
    """
    Apply feature engineering to multiple stock DataFrames.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary with ticker symbols as keys and DataFrames as values
    price_column : str, default='Close'
        Name of the price column to use for calculations
    
    Returns:
    --------
    dict
        Dictionary with same keys but DataFrames containing engineered features
    """
    
    engineered_data = {}
    indicator = TechnicalIndicators()
    
    for ticker, df in stock_data.items():
        print(f"Engineering features for {ticker}...", end=" ")
        engineered_data[ticker] = indicator.add_technical_features(
            df,
            price_column=price_column,
            sma_windows=[20, 50],
            rsi_period=14,
            add_returns=True
        )
        print("✓ Complete")
    
    return engineered_data


def display_feature_statistics(stock_data: dict):
    """
    Display statistics of engineered features.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with technical features
    """
    
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    
    for ticker, df in stock_data.items():
        print(f"\n{ticker}")
        print("-" * 40)
        
        features = [
            'SMA_20', 'SMA_50', 'RSI_14', 'Daily_Return',
            'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volume_Change'
        ]
        feature_df = df[features].describe().round(4)
        print(feature_df)
        
        # Check for NaN values
        nan_counts = df[features].isna().sum()
        print(f"\nMissing values (NaN):")
        for feature, count in nan_counts.items():
            print(f"  {feature}: {count} rows")


def save_engineered_data(stock_data: dict, output_dir: str = 'data'):
    """
    Save engineered features to CSV files.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with technical features
    output_dir : str, default='data'
        Directory to save the files
    """
    
    print("\n" + "="*80)
    print("SAVING ENGINEERED DATA")
    print("="*80)
    
    for ticker, df in stock_data.items():
        filepath = f"{output_dir}/{ticker}_features.csv"
        df.to_csv(filepath)
        print(f"✓ Saved {ticker} engineered features to: {filepath}")


def main():
    """Main execution function"""
    
    import sys
    sys.path.insert(0, r'c:\Users\Hp\Desktop\myapp\backtestingframework')
    from data_acquisition import download_stock_data
    
    # Download raw data
    print("STEP 1: Downloading stock data...")
    print("-" * 80)
    stock_data = download_stock_data(tickers=['AAPL', 'MSFT'], period_years=5)
    
    if not stock_data:
        print("❌ No data downloaded. Exiting.")
        return
    
    # Engineer features
    print("\n\nSTEP 2: Engineering technical features...")
    print("-" * 80)
    engineered_data = engineer_features(stock_data, price_column='Close')
    
    # Display statistics
    display_feature_statistics(engineered_data)
    
    # Save engineered data
    save_engineered_data(engineered_data, output_dir='data')
    
    # Display sample data
    print("\n" + "="*80)
    print("SAMPLE ENGINEERED DATA")
    print("="*80)
    
    for ticker, df in engineered_data.items():
        print(f"\n{ticker} - Last 5 rows with features:")
        print("-" * 40)
        sample_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'Daily_Return']
        print(df[sample_cols].tail().round(4))
    
    return engineered_data


if __name__ == "__main__":
    engineered_data = main()
