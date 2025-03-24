import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.dates as mdates
from ta.trend import MACD
from ta.momentum import RSIIndicator
import os

# Part 2.1.a: Download historical EURUSD data
def download_forex_data(ticker='EURUSD=X', period='2y'):
    """
    Download historical forex data using Yahoo Finance
    
    Parameters:
    ticker (str): The forex pair to download (default: 'EURUSD=X')
    period (str): The time period to download (default: '2y' for 2 years)
    
    Returns:
    pd.DataFrame: DataFrame containing the historical data
    """
    print(f"Downloading {ticker} data for the last {period}...")
    data = yf.download(ticker, period=period)
    data.columns = data.columns.droplevel(-1)
    print(f"Downloaded {len(data)} rows of data")
    return data

# Part 2.1.b: Calculate technical indicators
def calculate_indicators(data):
    """
    Calculate technical indicators on the price data
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the price data
    
    Returns:
    pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # Calculate Moving Averages (50 and 200 periods)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI (14 periods)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    # Calculate MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    return df

# Part 2.1.c: Plot price and indicators and save the plots
def plot_price_and_indicators(df, title="EURUSD Price with Technical Indicators", save_path="plots/"):
    """
    Plot the price and calculated indicators and save the plots
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the price and indicator data
    title (str): The title of the plot
    save_path (str): The directory path to save the plots
    """
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create figure and axis objects with subplots()
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the price and moving averages
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax1.plot(df.index, df['Close'], label='EURUSD Close', linewidth=2)
    ax1.plot(df.index, df['MA50'], label='MA50', linewidth=1.5)
    ax1.plot(df.index, df['MA200'], label='MA200', linewidth=1.5)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_title(title)
    
    # Plot RSI
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, sharex=ax1)
    ax2.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.grid(True)
    
    # Plot MACD
    ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
    ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
    ax3.bar(df.index, df['MACD_Histogram'], color='green', alpha=0.5)
    ax3.set_ylabel('MACD')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # Format the x-axis to show dates nicely
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax3.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
    plt.savefig(plot_filename)
    plt.show()
    print(f"Plot saved as {plot_filename}")

# Main function to run Part 2.1
def run_data_analysis():
    # Download data
    data = download_forex_data(period='20y')
    
    # Calculate indicators
    df_with_indicators = calculate_indicators(data)
    
    # Plot price and indicators
    plot_price_and_indicators(df_with_indicators)
    
    # Return the processed dataframe for further use
    return df_with_indicators

if __name__ == "__main__":
    df = run_data_analysis()
    df.to_csv('data/eurusd_data_with_indicators.csv')
    print("Data analysis completed.")
