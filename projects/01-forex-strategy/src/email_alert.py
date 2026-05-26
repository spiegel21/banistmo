import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import timedelta
import io

def setup_email_config():
    """
    Set up email configuration details
    Returns a dictionary with email configuration
    """
    # Replace these with your actual email credentials
    email_config = {
        'smtp_server': 'smtp.gmail.com',  # Change based on your email provider
        'smtp_port': 587,                 # Standard TLS port
        'sender_email': '',
        'password': '',  # Use app password for Gmail
        'recipient_email': ''
    }
    
    return email_config

def create_position_chart(df, position_date, window=10):
    """
    Create a chart showing the recent price action and moving averages
    around the position change
    """
    # Convert position_date to datetime if it's not already
    if isinstance(position_date, str):
        position_date = pd.to_datetime(position_date)
    
    # Get a window of data around the signal date
    start_date = position_date - timedelta(days=window)
    end_date = position_date + timedelta(days=3)
    
    # Convert df.index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Filter dataframe to the window
    mask = (df.index >= start_date) & (df.index <= end_date)
    plot_df = df[mask].copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot price and moving averages
    ax.plot(plot_df.index, plot_df['Close'], label='Price', linewidth=2)
    ax.plot(plot_df.index, plot_df['MA50'], label='MA50', linewidth=1.5)
    ax.plot(plot_df.index, plot_df['MA200'], label='MA200', linewidth=1.5)
    
    # Highlight the signal
    position_value = df.loc[position_date, 'Position']
    if position_value > 0:  # Buy signal
        ax.scatter(position_date, df.loc[position_date, 'Close'], 
                  color='green', s=150, marker='^', label='BUY')
    elif position_value < 0:  # Sell signal
        ax.scatter(position_date, df.loc[position_date, 'Close'], 
                  color='red', s=150, marker='v', label='SELL')
    
    # Format the plot
    ax.set_title(f'Trading Signal on {position_date}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to bytes buffer instead of file
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return buffer

def send_position_email(email_config, df, position_date, position_type):
    """
    Send an email notification about a position change
    """
    # Setup the email
    msg = MIMEMultipart()
    msg['From'] = email_config['sender_email']
    msg['To'] = email_config['recipient_email']
    
    # Create appropriate subject and content based on position type
    if position_type == 'BUY':
        msg['Subject'] = f'🟢 BUY Signal Alert: EURUSD on {position_date}'
        color = 'green'
        emoji = '📈'
    else:  # SELL
        msg['Subject'] = f'🔴 SELL Signal Alert: EURUSD on {position_date}'
        color = 'red'
        emoji = '📉'
    
    # Get the price data
    position_row = df.loc[position_date]
    
    # Format the email content with HTML
    html_content = f"""
    <html>
    <body>
        <h2 style="color:{color};">{emoji} {position_type} Signal Generated {emoji}</h2>
        <p>A {position_type.lower()} signal was generated for EURUSD on <strong>{position_date}</strong>.</p>
        
        <h3>Signal Details:</h3>
        <ul>
            <li><strong>Date:</strong> {position_date}</li>
            <li><strong>Price:</strong> {position_row['Close']:.6f}</li>
            <li><strong>MA50:</strong> {position_row['MA50']:.6f}</li>
            <li><strong>MA200:</strong> {position_row['MA200']:.6f}</li>
            <li><strong>RSI:</strong> {position_row['RSI']:.2f}</li>
            <li><strong>MACD:</strong> {position_row['MACD']:.6f}</li>
        </ul>
        
        <h3>Market Conditions:</h3>
        <p>{get_market_conditions(position_row)}</p>
        
        <p>Please check the attached chart for more details.</p>
        
        <p><em>This is an automated notification from your trading system.</em></p>
    </body>
    </html>
    """
    
    # Attach the HTML content
    msg.attach(MIMEText(html_content, 'html'))
    
    # Create and attach the chart
    chart_buffer = create_position_chart(df, position_date)
    chart_image = MIMEImage(chart_buffer.read())
    chart_image.add_header('Content-Disposition', 'attachment', filename=f'signal_{position_date}.png')
    msg.attach(chart_image)
    
    # Connect to server and send email
    try:
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['password'])
        server.send_message(msg)
        server.quit()
        print(f"Email sent successfully for {position_type} signal on {position_date}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def get_market_conditions(row):
    """
    Generate a summary of market conditions based on indicators
    """
    conditions = []
    
    # RSI conditions
    rsi = row['RSI']
    if rsi < 30:
        conditions.append(f"RSI is oversold at {rsi:.2f}")
    elif rsi > 70:
        conditions.append(f"RSI is overbought at {rsi:.2f}")
    else:
        conditions.append(f"RSI is neutral at {rsi:.2f}")
    
    # MACD conditions
    macd = row['MACD']
    if macd > 0:
        conditions.append(f"MACD is positive at {macd:.6f}, suggesting bullish momentum")
    else:
        conditions.append(f"MACD is negative at {macd:.6f}, suggesting bearish momentum")
    
    # Moving Average conditions
    if row['MA50'] > row['MA200']:
        conditions.append("Short-term trend is above long-term trend (MA50 > MA200)")
    else:
        conditions.append("Long-term trend is above short-term trend (MA200 > MA50)")
    
    return " ".join(conditions)

def process_strategy_data(file_path, send_emails=True):
    """
    Process the strategy results data and send email notifications for position changes
    """
    # Load the strategy results
    df = pd.read_csv(file_path, parse_dates=True)
    
    # Set the date column as index if it exists
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    
    # Get email configuration if sending emails
    email_config = setup_email_config() if send_emails else None
    
    # Find all position changes (non-zero values in Position column)
    position_changes = df[df['Position'] != 0]
    
    print(f"Found {len(position_changes)} position changes in the data")
    
    # Process each position change
    for date, row in position_changes.iterrows():
        position_value = row['Position']
        
        if position_value > 0:  # Buy signal
            position_type = 'BUY'
            print(f"Found BUY signal on {date}")
        elif position_value < 0:  # Sell signal
            position_type = 'SELL'
            print(f"Found SELL signal on {date}")
        else:
            continue  # Skip if not a clear buy/sell (shouldn't happen based on our filter)
        
        if send_emails:
            send_position_email(email_config, df, date, position_type)
        else:
            print(f"Would send {position_type} signal email for {date} (email sending disabled)")
            
    return position_changes

def main():
    """
    Main function to run the email notification system
    """
    # Path to your strategy results file
    file_path = "data/strategy_results.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    
    print(f"Processing trading strategy data from {file_path}")
    
    # Process the data and send emails (set send_emails=False for testing)
    position_changes = process_strategy_data(file_path, send_emails=True)
    
    print(f"Processing complete. Found {len(position_changes)} trading signals.")

if __name__ == "__main__":
    main()