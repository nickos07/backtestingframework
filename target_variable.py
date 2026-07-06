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