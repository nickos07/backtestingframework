"""
Data Visualization Module
Creates professional visualizations of stock price data and technical indicators.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Tuple
import os


# Set style for professional appearance
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def create_price_trends_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    ticker: str
) -> None:
    """
    Plot Close price with 20-day and 50-day SMAs.
    
    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axis to plot on
    df : pd.DataFrame
        DataFrame with Close, SMA_20, SMA_50 columns
    ticker : str
        Stock ticker symbol for title
    """
    
    # Plot Close price
    ax.plot(df.index, df['Close'], label='Close Price', 
            color='#1f77b4', linewidth=2, alpha=0.9)
    
    # Plot SMAs
    ax.plot(df.index, df['SMA_20'], label='20-day SMA', 
            color='#ff7f0e', linewidth=1.5, alpha=0.8, linestyle='--')
    ax.plot(df.index, df['SMA_50'], label='50-day SMA', 
            color='#2ca02c', linewidth=1.5, alpha=0.8, linestyle='--')
    
    # Styling
    ax.set_title(f'{ticker} - Price & Trend Analysis', fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))


def create_rsi_plot(
    ax: plt.Axes,
    df: pd.DataFrame
) -> None:
    """
    Plot RSI with overbought/oversold levels.
    
    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axis to plot on
    df : pd.DataFrame
        DataFrame with RSI_14 column
    """
    
    # Plot RSI
    ax.plot(df.index, df['RSI_14'], label='RSI (14)', 
            color='#9467bd', linewidth=2, alpha=0.9)
    
    # Add overbought/oversold levels
    ax.axhline(y=70, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='Overbought (70)')
    ax.axhline(y=30, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='Oversold (30)')
    
    # Fill between for visualization
    ax.fill_between(df.index, 70, 100, alpha=0.1, color='red', label='Overbought Zone')
    ax.fill_between(df.index, 0, 30, alpha=0.1, color='red', label='Oversold Zone')
    
    # Styling
    ax.set_title('Relative Strength Index (RSI) - Momentum Indicator', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('RSI Value', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))


def create_return_distribution_plot(
    ax: plt.Axes,
    df: pd.DataFrame
) -> None:
    """
    Plot histogram of daily returns.
    
    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axis to plot on
    df : pd.DataFrame
        DataFrame with Daily_Return column
    """
    
    returns = df['Daily_Return'].dropna()
    
    # Create histogram
    n, bins, patches = ax.hist(returns, bins=50, color='#d62728', 
                                alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add normal distribution overlay
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, len(returns) * (bins[1] - bins[0]) * (1/(sigma * np.sqrt(2*np.pi)) * 
            np.exp(-0.5*((x-mu)/sigma)**2)), 'b-', linewidth=2, label='Normal Distribution')
    
    # Add mean and median lines
    ax.axvline(mu, color='green', linestyle='-', linewidth=2, 
               alpha=0.8, label=f'Mean: {mu:.4f}')
    ax.axvline(returns.median(), color='orange', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Median: {returns.median():.4f}')
    
    # Styling
    ax.set_title('Daily Return Distribution - Volatility Profile', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Daily Return (Decimal)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
    
    # Add statistics text
    stats_text = f'Mean: {mu:.4f}\nStd Dev: {sigma:.4f}\nMin: {returns.min():.4f}\nMax: {returns.max():.4f}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def format_dates_on_axis(ax: plt.Axes) -> None:
    """
    Format date labels on x-axis for readability.
    
    Parameters:
    -----------
    ax : plt.Axes
        Matplotlib axis to format
    """
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate date labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def create_visualization(df: pd.DataFrame, ticker: str = 'AAPL'):
    """
    Create comprehensive 3-subplot visualization of stock data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Close, SMA_20, SMA_50, RSI_14, Daily_Return, etc.
    ticker : str, default='AAPL'
        Stock ticker symbol for titles
    """

    print("\n" + "="*80)
    print("STOCK DATA VISUALIZATION")
    print("="*80 + "\n")

    # Create figure with 3 subplots
    print("\nCreating visualization with 3 subplots...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
    
    # Plot 1: Price & Trends (largest)
    print("  1. Plotting price and trend analysis...")
    create_price_trends_plot(ax1, df, ticker)
    format_dates_on_axis(ax1)
    
    # Plot 2: RSI Momentum
    print("  2. Plotting RSI momentum indicator...")
    create_rsi_plot(ax2, df)
    format_dates_on_axis(ax2)
    
    # Plot 3: Return Distribution
    print("  3. Plotting return distribution...")
    create_return_distribution_plot(ax3, df)
    # Note: Return distribution has different x-axis, so no date formatting needed
    
    # Set x-labels
    ax3.set_xlabel('Daily Return', fontsize=11, fontweight='bold')
    ax1.set_xlabel('')  # Remove x-label from top plot
    ax2.set_xlabel('')  # Remove x-label from middle plot
    
    # Adjust layout to prevent label overlap
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle(f'{ticker} Stock Analysis - 5 Year Historical Data', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Add footer with data info
    info_text = f'Data Range: {df.index[0].date()} to {df.index[-1].date()} | Total Trading Days: {len(df)}'
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, style='italic', color='gray')
    
    print("\n✓ Visualization created successfully!")
    print(f"  Ticker: {ticker}")
    print(f"  Data points: {len(df)}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    return fig, (ax1, ax2, ax3), df

def calculate_risk_metrics(strategy_returns: pd.Series) -> Tuple[float, float]:
    """Return annualized Sharpe ratio and maximum drawdown for a return series."""
    if strategy_returns.std() == 0 or np.isnan(strategy_returns.std()):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())

    cumulative_value = (1 + strategy_returns).cumprod()
    drawdown = 1 - (cumulative_value / cumulative_value.cummax())
    max_drawdown = drawdown.max()
    return sharpe_ratio, max_drawdown

def plot_strategy_comparison(
    daily_returns: pd.Series,
    manual_strategy_returns: pd.Series,
    cumulative_manual: pd.Series,
    walkforward_strategy_returns: pd.Series,
    cumulative_walkforward: pd.Series,
    output_path: str = 'models/strategy_comparison.png'
) -> None:
    """Plot cumulative returns for buy-and-hold versus both AI strategies."""
    cumulative_buy_hold = (1 + daily_returns).cumprod() - 1

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        daily_returns.index,
        cumulative_buy_hold,
        color='gray',
        alpha=0.6,
        linewidth=1.8,
        label='Buy & Hold'
    )
    ax.plot(
        cumulative_manual.index,
        cumulative_manual,
        color='green',
        linewidth=1.8,
        label='AI Strategy (Manual Tune - High Return)'
    )
    ax.plot(
        cumulative_walkforward.index,
        cumulative_walkforward,
        color='blue',
        linewidth=1.8,
        label='AI Strategy (Walk-Forward Tune - Low Risk)'
    )

    buy_hold_sharpe, buy_hold_drawdown = calculate_risk_metrics(daily_returns)
    manual_sharpe, manual_drawdown = calculate_risk_metrics(manual_strategy_returns)
    walkforward_sharpe, walkforward_drawdown = calculate_risk_metrics(walkforward_strategy_returns)

    metrics_text = (
        f'Buy & Hold\nSharpe: {buy_hold_sharpe:.2f}\nMax Drawdown: {buy_hold_drawdown:.2%}\n\n'
        f'Manual Tune\nSharpe: {manual_sharpe:.2f}\nMax Drawdown: {manual_drawdown:.2%}\n\n'
        f'Walk-Forward Tune\nSharpe: {walkforward_sharpe:.2f}\nMax Drawdown: {walkforward_drawdown:.2%}'
    )
    ax.text(
        0.02,
        0.02,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_title('Cumulative Returns: Buy & Hold vs AI Strategy Variants')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Strategy comparison plot saved to: {output_path}")
    plt.show()

def plot_backtest(backtest_df: pd.DataFrame, output_path: str = 'models/backtest_returns.png'):
    """
    Plot cumulative returns for buy-and-hold versus the strategy and save to PNG.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        backtest_df.index,
        backtest_df['Cumulative_Buy_and_Hold'],
        color='gray',
        label='Buy & Hold'
    )
    plt.plot(
        backtest_df.index,
        backtest_df['Cumulative_Strategy'],
        color='green',
        label='XGBoost Strategy'
    )
    plt.title('Backtest: Buy & Hold vs XGBoost Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(alpha=0.3)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[OK] Backtest plot saved to: {output_path}")
    plt.show()