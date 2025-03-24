import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize

# Part 2.2.a: Implement Moving Average Crossover Strategy
def implement_ma_crossover_strategy(df, short_window=50, long_window=200):
    """
    Implement a simple moving average crossover strategy
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    short_window (int): Shorter moving average period
    long_window (int): Longer moving average period
    
    Returns:
    pd.DataFrame: DataFrame with strategy signals and positions
    """
    # Make a copy to avoid modifying the original dataframe
    strategy_df = df.copy()
    
    # Calculate moving averages if not already in the dataframe
    if f'MA{short_window}' not in strategy_df.columns:
        strategy_df[f'MA{short_window}'] = strategy_df['Close'].rolling(window=short_window).mean()
    if f'MA{long_window}' not in strategy_df.columns:
        strategy_df[f'MA{long_window}'] = strategy_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals: 1 when short MA crosses above long MA, -1 when short MA crosses below long MA
    strategy_df['Signal'] = 0
    strategy_df['Signal'] = np.where(
        strategy_df[f'MA{short_window}'] > strategy_df[f'MA{long_window}'], 1, 0)
    
    # Generate trading orders: position changes
    strategy_df['Position'] = strategy_df['Signal'].diff()
    
    # Generate actual positions: 1 (long), 0 (neutral), -1 (short)
    strategy_df['Position_Actual'] = strategy_df['Signal']
    
    # Calculate strategy returns
    returns = strategy_df['Close'].pct_change()
    positions = strategy_df['Position_Actual'].shift(1)

    # Drop any rows with NaN values to align them properly
    aligned = pd.concat([positions, returns], axis=1).dropna()
    result_series = aligned.iloc[:, 0] * aligned.iloc[:, 1]
    strategy_df['Strategy_Returns'] = result_series
    
    # Calculate cumulative returns
    strategy_df['Cumulative_Returns'] = (1 + strategy_df['Strategy_Returns']).cumprod()
    strategy_df['Buy_Hold_Returns'] = (1 + strategy_df['Close'].pct_change()).cumprod()
    
    return strategy_df

# Part 2.2.b: Evaluate strategy performance
def evaluate_strategy(strategy_df, risk_free_rate=0.02):
    """
    Evaluate the performance of the trading strategy
    
    Parameters:
    strategy_df (pd.DataFrame): DataFrame with strategy results
    risk_free_rate (float): Annual risk-free rate for Sharpe ratio calculation
    
    Returns:
    dict: Dictionary containing performance metrics
    """
    # Calculate daily risk-free rate
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate metrics
    total_return = strategy_df['Cumulative_Returns'].iloc[-1] - 1
    buy_hold_return = strategy_df['Buy_Hold_Returns'].iloc[-1] - 1
    
    # Daily returns stats
    returns = strategy_df['Strategy_Returns'].dropna()
    daily_returns_mean = returns.mean()
    daily_returns_std = returns.std()
    
    # Calculate annualized Sharpe ratio
    annualized_sharpe = (daily_returns_mean - daily_rf) / daily_returns_std * np.sqrt(252)
    
    # Calculate maximum drawdown
    cumulative_returns = strategy_df['Cumulative_Returns']
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    trade_returns = strategy_df.loc[strategy_df['Position'] != 0, 'Strategy_Returns'].dropna()
    win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns) if len(trade_returns) > 0 else 0
    
    # Calculate profit factor
    profit_factor = abs(trade_returns[trade_returns > 0].sum() / trade_returns[trade_returns < 0].sum()) if trade_returns[trade_returns < 0].sum() != 0 else float('inf')
    
    # Calculate number of trades
    num_trades = len(strategy_df[strategy_df['Position'] != 0])
    
    # Store metrics in a dictionary
    metrics = {
        'Total Return': total_return,
        'Buy & Hold Return': buy_hold_return,
        'Annualized Sharpe Ratio': annualized_sharpe,
        'Maximum Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Number of Trades': int(num_trades)
    }
    
    return metrics

# Plot strategy results
def plot_strategy_results(strategy_df, title="Moving Average Crossover Strategy Results"):
    """
    Plot strategy results including cumulative returns and trade signals
    
    Parameters:
    strategy_df (pd.DataFrame): DataFrame with strategy results
    title (str): The title of the plot
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot prices and moving averages
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax1.plot(strategy_df.index, strategy_df['Close'], label='EURUSD Close', linewidth=2)
    ax1.plot(strategy_df.index, strategy_df['MA50'], label='MA50', linewidth=1.5)
    ax1.plot(strategy_df.index, strategy_df['MA200'], label='MA200', linewidth=1.5)
    
    # Plot buy/sell signals
    buy_signals = strategy_df[strategy_df['Position'] > 0].index
    sell_signals = strategy_df[strategy_df['Position'] < 0].index
    ax1.plot(buy_signals, strategy_df.loc[buy_signals, 'Close'], '^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(sell_signals, strategy_df.loc[sell_signals, 'Close'], 'v', markersize=10, color='r', label='Sell Signal')
    
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title(title)
    
    # Plot strategy returns
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, sharex=ax1)
    ax2.plot(strategy_df.index, strategy_df['Cumulative_Returns'], label='Strategy Returns', linewidth=2, color='g')
    ax2.plot(strategy_df.index, strategy_df['Buy_Hold_Returns'], label='Buy & Hold Returns', linewidth=2, color='b', alpha=0.6)
    ax2.set_ylabel('Cumulative Returns')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Plot drawdown
    ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
    running_max = strategy_df['Cumulative_Returns'].cummax()
    drawdown = (strategy_df['Cumulative_Returns'] / running_max) - 1
    ax3.fill_between(strategy_df.index, drawdown, 0, color='r', alpha=0.3)
    ax3.set_ylabel('Drawdown')
    ax3.grid(True)
    
    # Format the x-axis to show dates nicely
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax3.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig("plots/strategy_results.png")
    plt.show()
    

# Part 2.3.a: Optimize strategy parameters
def objective_function(params, df):
    """
    Objective function for strategy optimization
    Negative Sharpe ratio (we want to minimize, so we're maximizing Sharpe)
    
    Parameters:
    params (tuple): Parameters to optimize (short_window, long_window)
    df (pd.DataFrame): DataFrame with price data
    
    Returns:
    float: Negative Sharpe ratio
    """
    short_window, long_window = int(params[0]), int(params[1])

    print(f"Trying: short={short_window}, long={long_window}")
    
    # Ensure short window is actually shorter than long window
    if short_window >= long_window:
        return -np.inf
    
    try:
        # Implement strategy with these parameters
        strategy_df = implement_ma_crossover_strategy(df, short_window, long_window)
        
        # Evaluate strategy
        metrics = evaluate_strategy(strategy_df)
        
        # Return negative Sharpe ratio (we're minimizing)
        return -metrics['Annualized Sharpe Ratio']
    except Exception as e:
        print(f"Error in objective function: {e}")
        return -np.inf

def optimize_strategy_parameters(df, initial_guess=(70, 300)):
    """
    Optimize the strategy parameters using scipy.optimize
    
    Parameters:
    df (pd.DataFrame): DataFrame with price data
    initial_guess (tuple): Initial guess for parameters (short_window, long_window)
    
    Returns:
    tuple: Optimized parameters (short_window, long_window)
    """
    # Define bounds for parameters
    bounds = [(5, 100), (100, 730)]
    bounds = [(5, 100), (101, 730)]
    df = df.dropna(how='any')

    # Run optimization
    result = minimize(
        objective_function,
        initial_guess,
        args=(df,),
        method='Powell',
        bounds=bounds,
        options={'disp': True}
    )
    
    # Extract optimized parameters, ensuring they're integers
    short_window, long_window = int(round(result['x'][0])), int(round(result['x'][1]))
    print(f"Optimization results: Short window = {short_window}, Long window = {long_window}")
    
    return (short_window, long_window)

# Main function to run Part 2.2 and 2.3.a
def run_trading_strategy(df, optimize=True):
    """
    Run the trading strategy and optionally optimize parameters
    
    Parameters:
    df (pd.DataFrame): DataFrame with price and indicator data
    optimize (bool): Whether to optimize the strategy parameters
    
    Returns:
    tuple: (strategy_df, metrics)
    """
    # Default parameters
    short_window, long_window = 50, 200
    
    # Optimize parameters if requested
    if optimize:
        short_window, long_window = optimize_strategy_parameters(df)
    
    # Implement strategy with optimal parameters
    strategy_df = implement_ma_crossover_strategy(df, short_window, long_window)
    
    # Evaluate strategy
    metrics = evaluate_strategy(strategy_df)
    
    # Print metrics
    print("\nStrategy Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot strategy results
    plot_strategy_results(strategy_df, title=f"MA Crossover Strategy (MA{short_window} / MA{long_window})")
    
    return strategy_df, metrics

if __name__ == "__main__":
    # This would be run after the data analysis part
    df = pd.read_csv("data/eurusd_data_with_indicators.csv", parse_dates=['Date'], index_col='Date')
    strategy_df, metrics = run_trading_strategy(df, optimize=True)
    strategy_df.to_csv("out/strategy_results.csv", index=False)
    print("Trading strategy analysis completed.")
