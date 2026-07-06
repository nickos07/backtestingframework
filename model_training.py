"""
Model Training Module - Phase 3: Model Selection and Training
Builds and evaluates an XGBClassifier for stock price direction prediction.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

from data_acquisition import download_stock_data
from feature_engineering import TechnicalIndicators
from target_variable import TargetVariableGenerator
from data_cleaning import DataCleaner
from visualization import plot_strategy_comparison


def separate_features_and_target(
    df: pd.DataFrame,
    feature_columns: list = None,
    target_column: str = 'Target'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from the DataFrame.
    Creates target from next day's close price if not present in DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and optionally target
    feature_columns : list, default=None
        List of feature column names. If None, uses default features:
        ['SMA_20', 'SMA_50', 'RSI_14', 'Daily_Return']
    target_column : str, default='Target'
        Name of the target column
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Tuple of (features_X, target_y)
    """
    
    if feature_columns is None:
        feature_columns = ['SMA_20', 'SMA_50', 'RSI_14', 'Daily_Return']
    
    print(f"\nFeature Columns: {feature_columns}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is required in the DataFrame.")
    print(f"Target Column: {target_column}")
    y = df[target_column].copy()
    
    # Extract features
    X = df[feature_columns].copy()
    
    # Remove NaN rows
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform chronological train/test split WITHOUT shuffling.
    Uses the first (1 - test_size) of data for training,
    and the last test_size for testing.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features DataFrame
    y : pd.Series
        Target Series
    test_size : float, default=0.2
        Fraction of data to use for testing (20% by default)
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Tuple of (X_train, X_test, y_train, y_test)
    """
    
    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))
    
    # Chronological split (NO SHUFFLE)
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    # Get date ranges
    train_start = X_train.index[0]
    train_end = X_train.index[-1]
    test_start = X_test.index[0]
    test_end = X_test.index[-1]
    
    print(f"\nChronological Train/Test Split (No Shuffle):")
    print("-" * 60)
    print(f"Total samples: {len(X)}")
    print(f"Training split: {(1-test_size)*100:.0f}% ({len(X_train)} samples)")
    print(f"  Date range: {train_start.date()} to {train_end.date()}")
    print(f"Testing split: {test_size*100:.0f}% ({len(X_test)} samples)")
    print(f"  Date range: {test_start.date()} to {test_end.date()}")
    
    return X_train, X_test, y_train, y_test


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.8,
    random_state: int = 42
) -> XGBClassifier:
    """
    Initialize and train an XGBoost classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int, default=100
        Number of boosting rounds
    learning_rate : float, default=0.05
        Step size shrinkage
    max_depth : int, default=3
        Maximum tree depth
    subsample : float, default=0.8
        Fraction of samples used per tree
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    XGBClassifier
        Trained model
    """
    
    print(f"\nModel Initialization:")
    print("-" * 60)
    print(f"Algorithm: XGBClassifier")
    print(f"Parameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - max_depth: {max_depth}")
    print(f"  - subsample: {subsample}")
    print(f"  - random_state: {random_state}")
    
    # Initialize model
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model
    print(f"\nTraining model...")
    model.fit(X_train, y_train)
    print(f"[OK] Model training complete!")
    
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list = None
) -> dict:
    """
    Evaluate model performance on test set.
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    feature_names : list, default=None
        Feature names for importance analysis
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get classification report
    class_report = classification_report(y_test, y_pred, 
                                         target_names=['Down/Flat', 'Up'])
    
    # Get confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    metrics = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance
    }
    
    return metrics


def calculate_strategy_returns(
    probabilities: pd.Series,
    daily_returns: pd.Series,
    threshold: float
) -> Tuple[pd.Series, pd.Series]:
    """Convert model probabilities into a lagged trading signal and cumulative returns."""
    signal = (probabilities >= threshold).astype(int).shift(1).fillna(0)
    strategy_returns = daily_returns * signal
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    return strategy_returns, cumulative_returns





def print_evaluation_report(metrics: dict, y_test: pd.Series):
    """
    Print comprehensive evaluation report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of evaluation metrics
    y_test : pd.Series
        Test target values
    """
    
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Accuracy
    print(f"\n1. OUT-OF-SAMPLE ACCURACY SCORE")
    print("-" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    # Classification Report
    print(f"\n2. CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
    print("-" * 80)
    print(metrics['classification_report'])
    
    # Confusion Matrix
    print(f"\n3. CONFUSION MATRIX")
    print("-" * 80)
    conf_matrix = metrics['confusion_matrix']
    print(f"\n                Predicted")
    print(f"              Down/Flat    Up")
    print(f"Actual Down/Flat {conf_matrix[0,0]:4d}      {conf_matrix[0,1]:4d}")
    print(f"       Up        {conf_matrix[1,0]:4d}      {conf_matrix[1,1]:4d}")
    
    # Calculate and print additional metrics from confusion matrix
    tn, fp, fn, tp = conf_matrix[0,0], conf_matrix[0,1], conf_matrix[1,0], conf_matrix[1,1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSensitivity (Recall for 'Up'): {sensitivity:.4f}")
    print(f"Specificity (Recall for 'Down/Flat'): {specificity:.4f}")
    
    # Feature Importance
    print(f"\n4. FEATURE IMPORTANCE")
    print("-" * 80)
    print(metrics['feature_importance'].to_string(index=False))
    
    # Class distribution in test set
    print(f"\n5. TEST SET CLASS DISTRIBUTION")
    print("-" * 80)
    class_counts = y_test.value_counts().sort_index()
    print(f"Class 0 (Down/Flat): {class_counts[0]} ({class_counts[0]/len(y_test)*100:.1f}%)")
    print(f"Class 1 (Up):        {class_counts[1]} ({class_counts[1]/len(y_test)*100:.1f}%)")


def save_model_results(
    model: XGBClassifier,
    metrics: dict,
    y_test: pd.Series,
    output_dir: str = 'models'
) -> None:
    """
    Save trained model and results.
    
    Parameters:
    -----------
    model : XGBClassifier
        Trained model
    metrics : dict
        Evaluation metrics
    y_test : pd.Series
        Test target values
    output_dir : str, default='models'
        Directory to save files
    """
    
    import os
    import joblib
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/aapl_xgboost_model.pkl"
    joblib.dump(model, model_path)
    print(f"[OK] Model saved to: {model_path}")
    
    # Save feature importance
    importance_path = f"{output_dir}/feature_importance.csv"
    metrics['feature_importance'].to_csv(importance_path, index=False)
    print(f"[OK] Feature importance saved to: {importance_path}")


def main():
    print("\n" + "="*80)
    print("UNIFIED ML PIPELINE: Stock Price Direction Prediction")
    print("="*80)

    # 1. ACQUIRE DATA
    print("\n--- Phase 1: Data Acquisition ---")
    stock_data = download_stock_data(tickers=['AAPL'], period_years=5)
    df = stock_data['AAPL']

    # 2. ENGINEER FEATURES
    print("\n--- Phase 2: Feature Engineering ---")
    df = TechnicalIndicators.add_technical_features(
        df, price_column='Close', sma_windows=[20, 50], rsi_period=14, add_returns=True
    )

    # 3. CREATE TARGET VARIABLE
    print("\n--- Phase 3: Target Generation ---")
    df = TargetVariableGenerator.add_target_variable(
        df, price_column='Close', drop_na=False
    )

    # 4. CLEAN DATA
    print("\n--- Phase 4: Data Cleaning ---")
    df = DataCleaner.clean_dataset(df, remove_nan=True, validate=True, verbose=True)

    # 5. FEATURE AND TARGET SEPARATION
    print("\n--- Phase 5: Feature and Target Separation ---")
    feature_cols = [
        'SMA_20', 'SMA_50', 'RSI_14', 'Daily_Return',
        'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volume_Change'
    ]
    X, y = separate_features_and_target(df, feature_columns=feature_cols)

    # 6. TRAIN/TEST SPLIT
    print("\n--- Phase 6: Chronological Train/Test Split ---")
    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_size=0.2)

    # 7. MANUAL MODEL (Aggressive High Return)
    print("\n--- Phase 7: Manual Threshold Model ---")
    manual_model = train_xgboost_model(
        X_train, y_train, n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
    )
    manual_probabilities = pd.Series(
        manual_model.predict_proba(X_test)[:, 1], index=X_test.index, name='manual_probabilities'
    )
    Manual_Strategy_Returns, Cumulative_Returns_Manual = calculate_strategy_returns(
        manual_probabilities, X_test['Daily_Return'], threshold=0.47
    )

    # 8. WALK-FORWARD MODEL (Conservative Low Risk)
    print("\n--- Phase 8: Walk-Forward Hyperparameter Tuning ---")
    search_feature_cols = [col for col in feature_cols if col != 'Volume_Change']
    X_train_search = X_train[search_feature_cols].copy()
    X_test_search = X_test[search_feature_cols].copy()

    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
        param_distributions={
            'n_estimators': [50, 100, 200], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1]
        },
        n_iter=10, cv=tscv, scoring='accuracy', random_state=42, n_jobs=-1, verbose=0
    )
    random_search.fit(X_train_search, y_train)
    print(f"Best parameters: {random_search.best_params_}")

    walkforward_model = random_search.best_estimator_
    walkforward_model.fit(X_train_search, y_train)

    walkforward_probabilities = pd.Series(
        walkforward_model.predict_proba(X_test_search)[:, 1], index=X_test.index, name='walkforward_probabilities'
    )
    WalkForward_Strategy_Returns, Cumulative_Returns_WalkForward = calculate_strategy_returns(
        walkforward_probabilities, X_test['Daily_Return'], threshold=0.50
    )

    # 9. EVALUATION & PLOTTING
    print("\n--- Phase 9: Evaluation & Comparison ---")
    metrics = evaluate_model(manual_model, X_test, y_test, feature_names=feature_cols)
    print_evaluation_report(metrics, y_test)

    plot_strategy_comparison(
        X_test['Daily_Return'],
        Manual_Strategy_Returns, Cumulative_Returns_Manual,
        WalkForward_Strategy_Returns, Cumulative_Returns_WalkForward,
        output_path='models/strategy_comparison.png'
    )

    print("\n--- Phase 10: Saving Model ---")
    save_model_results(walkforward_model, metrics, y_test, output_dir='models')

    print("\n✓ Pipeline Execution Complete!")


if __name__ == "__main__":
    main()
