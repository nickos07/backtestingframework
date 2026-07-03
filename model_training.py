"""
Model Training Module - Phase 3: Model Selection and Training
Builds and evaluates a RandomForestClassifier for stock price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple


def load_cleaned_data(ticker: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Load cleaned stock data from CSV.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data_dir : str, default='data'
        Directory containing cleaned data files
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with technical indicators
    """
    
    filepath = f"{data_dir}/{ticker}_cleaned.csv"
    
    try:
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
        print(f"[OK] Loaded cleaned data for {ticker}")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Cleaned data file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


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
    
    # Create target if not present: 1 if next day's close > today's close, 0 otherwise
    if target_column not in df.columns:
        print(f"Target Column: {target_column} (created from price direction)")
        tomorrow_close = df['Close'].shift(-1)
        y = (tomorrow_close > df['Close']).astype(int)
    else:
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


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Initialize and train a RandomForestClassifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int, default=100
        Number of decision trees in the forest
    max_depth : int, default=5
        Maximum depth of each tree
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    
    print(f"\nModel Initialization:")
    print("-" * 60)
    print(f"Algorithm: RandomForestClassifier")
    print(f"Parameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_depth: {max_depth}")
    print(f"  - random_state: {random_state}")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available processors
    )
    
    # Train model
    print(f"\nTraining model...")
    model.fit(X_train, y_train)
    print(f"[OK] Model training complete!")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list = None
) -> dict:
    """
    Evaluate model performance on test set.
    
    Parameters:
    -----------
    model : RandomForestClassifier
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
    model: RandomForestClassifier,
    metrics: dict,
    y_test: pd.Series,
    output_dir: str = 'models'
) -> None:
    """
    Save trained model and results.
    
    Parameters:
    -----------
    model : RandomForestClassifier
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
    model_path = f"{output_dir}/aapl_random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"[OK] Model saved to: {model_path}")
    
    # Save feature importance
    importance_path = f"{output_dir}/feature_importance.csv"
    metrics['feature_importance'].to_csv(importance_path, index=False)
    print(f"[OK] Feature importance saved to: {importance_path}")


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("PHASE 3: MODEL SELECTION AND TRAINING")
    print("Stock Price Direction Prediction - RandomForest Classifier")
    print("="*80)
    
    # Step 1: Load data
    print("\n\nSTEP 1: Loading Data")
    print("-" * 80)
    df = load_cleaned_data(ticker='AAPL', data_dir='data')
    
    # Step 2: Separate features and target
    print("\n\nSTEP 2: Feature and Target Separation")
    print("-" * 80)
    feature_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'Daily_Return']
    X, y = separate_features_and_target(df, feature_columns=feature_cols)
    
    # Step 3: Chronological train/test split
    print("\n\nSTEP 3: Chronological Train/Test Split")
    print("-" * 80)
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        X, y, test_size=0.2
    )
    
    # Step 4: Train model
    print("\n\nSTEP 4: Model Initialization & Training")
    print("-" * 80)
    model = train_random_forest_model(
        X_train, y_train,
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    # Step 5: Evaluate model
    print("\n\nSTEP 5: Model Evaluation")
    print("-" * 80)
    metrics = evaluate_model(model, X_test, y_test, feature_names=feature_cols)
    
    # Step 6: Print evaluation report
    print_evaluation_report(metrics, y_test)
    
    # Step 7: Save model
    print("\n\nSTEP 6: Saving Model")
    print("-" * 80)
    save_model_results(model, metrics, y_test, output_dir='models')
    
    print("\n" + "="*80)
    print("[OK] MODEL TRAINING PIPELINE COMPLETE")
    print("="*80)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metrics': metrics
    }


if __name__ == "__main__":
    results = main()
