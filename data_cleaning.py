"""
Data Cleaning Module
Handles data quality checks and NaN removal for ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class DataCleaner:
    """
    A class to clean and validate stock market data for machine learning.
    Handles NaN values created by rolling windows and technical indicators.
    """
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, verbose: bool = True) -> pd.Series:
        """
        Check for missing values in the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to check for missing values
        verbose : bool, default=True
            Whether to print missing value statistics
        
        Returns:
        --------
        pd.Series
            Count of missing values per column
        """
        
        missing_counts = df.isna().sum()
        missing_percentage = (df.isna().sum() / len(df)) * 100
        
        if verbose and missing_counts.sum() > 0:
            print("Missing Values Report:")
            print("-" * 50)
            for col in df.columns:
                if missing_counts[col] > 0:
                    print(f"  {col}: {missing_counts[col]} ({missing_percentage[col]:.2f}%)")
        
        return missing_counts
    
    
    @staticmethod
    def remove_nan_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Remove rows containing any NaN values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to clean
        verbose : bool, default=True
            Whether to print cleaning statistics
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with NaN rows removed
        """
        
        rows_before = len(df)
        df_cleaned = df.dropna()
        rows_after = len(df_cleaned)
        rows_removed = rows_before - rows_after
        
        if verbose:
            print(f"\nNaN Removal Summary:")
            print("-" * 50)
            print(f"  Rows before cleaning: {rows_before}")
            print(f"  Rows after cleaning:  {rows_after}")
            print(f"  Rows removed:         {rows_removed}")
            if rows_before > 0:
                print(f"  Percentage removed:   {(rows_removed/rows_before)*100:.2f}%")
        
        return df_cleaned
    
    
    @staticmethod
    def check_data_integrity(df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data integrity checks.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to validate
        
        Returns:
        --------
        Dict
            Dictionary with integrity check results
        """
        
        integrity_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'has_duplicates': df.index.duplicated().sum(),
            'has_nan': df.isna().sum().sum(),
            'is_sorted': df.index.is_monotonic_increasing if hasattr(df.index, 'is_monotonic_increasing') else None,
            'date_range': f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "Empty DataFrame"
        }
        
        return integrity_report
    
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame) -> Dict:
        """
        Validate numeric columns for reasonable values.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to validate
        
        Returns:
        --------
        Dict
            Dictionary with validation results
        """
        
        validation_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col]
            validation_report[col] = {
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'negative_values': (col_data < 0).sum()
            }
        
        return validation_report
    
    
    @staticmethod
    def clean_dataset(
        df: pd.DataFrame,
        remove_nan: bool = True,
        validate: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Complete data cleaning pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw DataFrame to clean
        remove_nan : bool, default=True
            Whether to remove NaN rows
        validate : bool, default=True
            Whether to perform validation checks
        verbose : bool, default=True
            Whether to print detailed reports
        
        Returns:
        --------
        pd.DataFrame
            Cleaned and validated DataFrame
        """
        
        df_clean = df.copy()
        
        if verbose:
            print("\n" + "="*80)
            print("DATA CLEANING PIPELINE")
            print("="*80)
        
        # Check initial missing values
        if verbose:
            print("\nStep 1: Checking for missing values...")
            print("-" * 50)
        DataCleaner.check_missing_values(df_clean, verbose=verbose)
        
        # Remove NaN rows
        if remove_nan:
            if verbose:
                print("\nStep 2: Removing rows with NaN values...")
            df_clean = DataCleaner.remove_nan_rows(df_clean, verbose=verbose)
        
        # Validate integrity
        if validate:
            if verbose:
                print("\nStep 3: Validating data integrity...")
                print("-" * 50)
            integrity = DataCleaner.check_data_integrity(df_clean)
            print(f"  Total rows:      {integrity['total_rows']}")
            print(f"  Total columns:   {integrity['total_columns']}")
            print(f"  Duplicate index: {integrity['has_duplicates']}")
            print(f"  Remaining NaNs:  {integrity['has_nan']}")
            print(f"  Date range:      {integrity['date_range']}")
        
        if verbose:
            print("\n✓ Cleaning complete!")
        
        return df_clean
