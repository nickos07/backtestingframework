"""
Target Variable Creation Module
Generates binary classification targets for stock price prediction without look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TargetVariableGenerator:
    """
    A class to create target variables for supervised machine learning models.
    Handles proper data alignment to prevent look-ahead bias.
    """
    
    @staticmethod
    def create_binary_target(
        df: pd.DataFrame,
        price_column: str = 'Close'
    ) -> pd.Series:
        """
        Create binary classification target: 1 if next day's close > today's close, 0 otherwise.
        
        This method properly shifts data to prevent look-ahead bias:
        - Uses forward-looking data (tomorrow's close)
        - Aligns it with today's features
        - Removes the final row which has no tomorrow
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data and price_column
        price_column : str, default='Close'
            Name of the price column to use for target calculation
        
        Returns:
        --------
        pd.Series
            Binary target (1 if next day up, 0 if down/flat)
        """
        
        # Shift prices forward by -1 to get tomorrow's price
        # This aligns tomorrow's close with today's row
        tomorrow_close = df[price_column].shift(-1)
        
        # Create binary target: 1 if tomorrow > today, 0 otherwise
        target = (tomorrow_close > df[price_column]).astype(int)
        
        return target
    
    
    @staticmethod
    def add_target_variable(
        df: pd.DataFrame,
        price_column: str = 'Close',
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Add target variable to DataFrame and handle NaN values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with engineered features
        price_column : str, default='Close'
            Name of the price column
        drop_na : bool, default=True
            Whether to drop rows with NaN values
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with Target column and cleaned data
        """
        
        df_with_target = df.copy()
        
        # Create target variable
        df_with_target['Target'] = TargetVariableGenerator.create_binary_target(
            df_with_target,
            price_column=price_column
        )
        
        # Drop NaN values if requested
        # This removes:
        # - The last row (no tomorrow's price for target)
        # - Initial rows needed for SMA and RSI calculation
        if drop_na:
            df_with_target = df_with_target.dropna()
        
        return df_with_target
    
    
    @staticmethod
    def create_train_test_features_targets(
        df: pd.DataFrame,
        price_column: str = 'Close',
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable for ML model training.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with engineered features
        price_column : str, default='Close'
            Name of the price column
        drop_na : bool, default=True
            Whether to drop rows with NaN values
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Tuple of (features_df, target_series)
        """
        
        df_with_target = TargetVariableGenerator.add_target_variable(
            df,
            price_column=price_column,
            drop_na=drop_na
        )
        
        # Separate features and target
        # Exclude OHLCV prices from features (keep only derived indicators)
        feature_columns = [col for col in df_with_target.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Dividends', 'Stock Splits']]
        
        X = df_with_target[feature_columns]
        y = df_with_target['Target']
        
        return X, y, df_with_target


def prepare_ml_dataset(
    stock_data: dict,
    price_column: str = 'Close'
) -> dict:
    """
    Prepare complete ML datasets with features and targets.
    
    Parameters:
    -----------
    stock_data : dict
        Dictionary of DataFrames with engineered features
    price_column : str, default='Close'
        Name of the price column
    
    Returns:
    --------
    dict
        Dictionary containing for each ticker:
        - 'features': Feature DataFrame (X)
        - 'target': Target Series (y)
        - 'full_data': Complete DataFrame with all columns
    """
    
    ml_datasets = {}
    generator = TargetVariableGenerator()
    
    for ticker, df in stock_data.items():
        print(f"Preparing ML dataset for {ticker}...", end=" ")
        
        X, y, df_full = generator.create_train_test_features_targets(
            df,
            price_column=price_column,
            drop_na=True
        )
        
        ml_datasets[ticker] = {
            'features': X,
            'target': y,
            'full_data': df_full
        }
        
        print(f"✓ Complete ({len(X)} samples)")
    
    return ml_datasets


def display_target_statistics(ml_datasets: dict):
    """
    Display statistics about target variable distribution.
    
    Parameters:
    -----------
    ml_datasets : dict
        Dictionary with ML datasets for each ticker
    """
    
    print("\n" + "="*80)
    print("TARGET VARIABLE STATISTICS")
    print("="*80)
    
    for ticker, data in ml_datasets.items():
        y = data['target']
        X = data['features']
        
        print(f"\n{ticker}")
        print("-" * 40)
        print(f"Total samples: {len(y)}")
        print(f"Features: {len(X.columns)}")
        
        # Target distribution
        class_counts = y.value_counts().sort_index()
        print(f"\nTarget Distribution:")
        print(f"  Class 0 (Price Down/Flat): {class_counts[0]} ({class_counts[0]/len(y)*100:.1f}%)")
        print(f"  Class 1 (Price Up):        {class_counts[1]} ({class_counts[1]/len(y)*100:.1f}%)")
        print(f"  Balance ratio: {class_counts[1]/class_counts[0]:.2f}")
        
        # Feature info
        print(f"\nFeature Summary:")
        print(f"  Features: {list(X.columns)}")
        print(f"\n  Feature Statistics (first 5 rows):")
        print(X.head().round(4))


def display_sample_data(ml_datasets: dict, n_samples: int = 10):
    """
    Display sample data with features and targets.
    
    Parameters:
    -----------
    ml_datasets : dict
        Dictionary with ML datasets for each ticker
    n_samples : int, default=10
        Number of samples to display
    """
    
    print("\n" + "="*80)
    print(f"SAMPLE DATA (Last {n_samples} rows)")
    print("="*80)
    
    for ticker, data in ml_datasets.items():
        df_full = data['full_data']
        
        print(f"\n{ticker}")
        print("-" * 40)
        
        # Show price, indicators, and target
        display_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'Daily_Return', 'Target']
        sample_df = df_full[display_cols].tail(n_samples)
        
        print(sample_df.round(4))


def save_ml_datasets(ml_datasets: dict, output_dir: str = 'data'):
    """
    Save ML-ready datasets to CSV files.
    
    Parameters:
    -----------
    ml_datasets : dict
        Dictionary with ML datasets for each ticker
    output_dir : str, default='data'
        Directory to save files
    """
    
    print("\n" + "="*80)
    print("SAVING ML DATASETS")
    print("="*80)
    
    for ticker, data in ml_datasets.items():
        # Save full dataset
        full_filepath = f"{output_dir}/{ticker}_ml_ready.csv"
        data['full_data'].to_csv(full_filepath)
        print(f"✓ Saved {ticker} full dataset to: {full_filepath}")
        
        # Save features only
        features_filepath = f"{output_dir}/{ticker}_X.csv"
        data['features'].to_csv(features_filepath)
        print(f"✓ Saved {ticker} features (X) to: {features_filepath}")
        
        # Save target only
        target_filepath = f"{output_dir}/{ticker}_y.csv"
        data['target'].to_csv(target_filepath)
        print(f"✓ Saved {ticker} target (y) to: {target_filepath}")


def main():
    """Main execution function"""
    
    import sys
    sys.path.insert(0, r'c:\Users\Hp\Desktop\myapp\backtestingframework')
    from feature_engineering import engineer_features
    from data_acquisition import download_stock_data
    
    # Step 1: Download data
    print("STEP 1: Downloading stock data...")
    print("-" * 80)
    stock_data = download_stock_data(tickers=['AAPL', 'MSFT'], period_years=5)
    
    if not stock_data:
        print("❌ No data downloaded. Exiting.")
        return
    
    # Step 2: Engineer features
    print("\n\nSTEP 2: Engineering technical features...")
    print("-" * 80)
    engineered_data = engineer_features(stock_data, price_column='Close')
    
    # Step 3: Create target variable
    print("\n\nSTEP 3: Creating target variable...")
    print("-" * 80)
    ml_datasets = prepare_ml_dataset(engineered_data, price_column='Close')
    
    # Display statistics
    display_target_statistics(ml_datasets)
    
    # Display sample data
    display_sample_data(ml_datasets, n_samples=10)
    
    # Save datasets
    save_ml_datasets(ml_datasets, output_dir='data')
    
    print("\n" + "="*80)
    print("✓ ML PIPELINE COMPLETE")
    print("="*80)
    print("\nYour data is ready for model training!")
    print("Next steps:")
    print("  1. Train/test split")
    print("  2. Feature scaling/normalization")
    print("  3. Model selection and training")
    print("  4. Model evaluation and validation")
    
    return ml_datasets


if __name__ == "__main__":
    ml_datasets = main()
