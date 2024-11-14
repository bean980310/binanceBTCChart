import ccxt
import warnings
from matplotlib.pyplot import fill_between
import pandas as pd
import numpy as np
import pandas_ta as ta
import mplfinance as mpf
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from src.update_data import get_chart_data

# def fetch_asset_data(symbol, start_date, interval, exchange):
#     # Convert start_date to milliseconds timestamp
#     start_date_ms = exchange.parse8601(start_date)
#     ohlcv = exchange.fetch_ohlcv(symbol, interval, since=start_date_ms)
#     header = ["date", "Open", "High", "Low", "Close", "Volume"]
#     df = pd.DataFrame(ohlcv, columns=header)
#     df['date'] = pd.to_datetime(df['date'], unit='ms')
#     df.set_index("date", inplace=True)
#     # Drop the last row containing live data
#     df.drop(df.index[-1], inplace=True)
#     return df

def supertrend(df, atr_multiplier=3):
    current_average_high_low = (df['High'] + df['Low']) / 2
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], period=15)
    df.dropna(inplace=True)
    
    # Calculate basic upper and lower bands
    upper_band = current_average_high_low + atr_multiplier * df['atr']
    lower_band = current_average_high_low - atr_multiplier * df['atr']
    
    # Shifted close price for comparisons
    shifted_close = df['Close'].shift(1)
    
    # Align indices by dropping rows with NaN after shifting
    shifted_close, upper_band = shifted_close.align(upper_band, join='inner')
    shifted_close, lower_band = shifted_close.align(lower_band, join='inner')

    # Calculate the final upper and lower bands using vectorized conditions
    df['Upperband'] = np.where((upper_band < shifted_close), upper_band, np.nan)
    df['Lowerband'] = np.where((lower_band > shifted_close), lower_band, np.nan)
    
    df['Upperband'].fillna(method='ffill', inplace=True)
    df['Lowerband'].fillna(method='ffill', inplace=True)
    return df

def generate_signals(df):
    conditions = [
        df['Close'] > df['Upperband'],
        df['Close'] < df['Lowerband']
    ]
    choices = [1, -1]
    df['Signals'] = np.select(conditions, choices, default=0)
    df['Signals'] = df['Signals'].replace(to_replace=0, method='ffill')
    df['Signals'] = df['Signals'].shift(1)
    return df

def create_positions(df):
    buy_positions = np.where((df['Signals'] == 1) & (df['Signals'].shift(1) != 1), df['Close'], np.nan)
    sell_positions = np.where((df['Signals'] == -1) & (df['Signals'].shift(1) != -1), df['Close'], np.nan)
    df['buy_positions'] = buy_positions
    df['sell_positions'] = sell_positions
    df['Upperband'] = np.where(df['Signals'] == 1, np.nan, df['Upperband'])
    df['Lowerband'] = np.where(df['Signals'] == -1, np.nan, df['Lowerband'])
    return df

def plot_data(df, symbol):
    # Define lowerband line plot
    lowerband_line = mpf.make_addplot(df['Lowerband'], label= "Lowerband", color='green')
    # Define upperband line plot
    upperband_line = mpf.make_addplot(df['Upperband'], label= "Upperband", color='red')
    # Define buy and sell markers
    buy_position_makers = mpf.make_addplot(df['buy_positions'], type='scatter', marker='^', label= "Buy", markersize=80, color='#2cf651')
    sell_position_makers = mpf.make_addplot(df['sell_positions'], type='scatter', marker='v', label= "Sell", markersize=80, color='#f50100')
    # A list of all addplots(apd)
    apd = [lowerband_line, upperband_line, buy_position_makers, sell_position_makers]
    # Create fill plots
    lowerband_fill = dict(y1=df['Close'].values, y2=df['Lowerband'].values, panel=0, alpha=0.3, color="#CCFFCC")
    upperband_fill = dict(y1=df['Close'].values, y2=df['Upperband'].values, panel=0, alpha=0.3, color="#FFCCCC")
    fills = [lowerband_fill, upperband_fill]
    # Plot the data 
    mpf.plot(df, addplot=apd, type='candle', volume=True, style='charles', xrotation=20, title=str(symbol + ' Supertrend Plot'), fill_between=fills)

def strategy_performance(strategy_df, capital=100, leverage=1):
    # Initialize the performance variables
    cumulative_balance = capital
    investment = capital
    pl = 0
    max_drawdown = 0
    max_drawdown_percentage = 0

    # Lists to store intermediate values for calculating metrics
    balance_list = [capital]
    pnl_list = [0]
    investment_list = [capital]
    peak_balance = capital

    # Loop from the second row (index 1) of the DataFrame
    for index in range(1, len(strategy_df)):
        row = strategy_df.iloc[index]

        # Calculate P/L for each trade signal
        if row['Signals'] == 1:
            pl = ((row['Close'] - row['Open']) / row['Open']) * \
                investment * leverage
        elif row['Signals'] == -1:
            pl = ((row['Open'] - row['Close']) / row['Close']) * \
                investment * leverage
        else:
            pl = 0

        # Update the investment if there is a signal reversal
        if row['Signals'] != strategy_df.iloc[index - 1]['Signals']:
            investment = cumulative_balance

        # Calculate the new balance based on P/L and leverage
        cumulative_balance += pl

        # Update the investment list
        investment_list.append(investment)

        # Calculate the cumulative balance and add it to the DataFrame
        balance_list.append(cumulative_balance)

        # Calculate the overall P/L and add it to the DataFrame
        pnl_list.append(pl)

        # Calculate max drawdown
        drawdown = cumulative_balance - peak_balance
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            max_drawdown_percentage = (max_drawdown / peak_balance) * 100

        # Update the peak balance
        if cumulative_balance > peak_balance:
            peak_balance = cumulative_balance

    # Add new columns to the DataFrame
    strategy_df['investment'] = investment_list
    strategy_df['cumulative_balance'] = balance_list
    strategy_df['pl'] = pnl_list
    strategy_df['cumPL'] = strategy_df['pl'].cumsum()

    # Calculate other performance metrics (replace with your calculations)
    overall_pl_percentage = (
        strategy_df['cumulative_balance'].iloc[-1] - capital) * 100 / capital
    overall_pl = strategy_df['cumulative_balance'].iloc[-1] - capital
    min_balance = min(strategy_df['cumulative_balance'])
    max_balance = max(strategy_df['cumulative_balance'])

    # Print the performance metrics
    print("Overall P/L: {:.2f}%".format(overall_pl_percentage))
    print("Overall P/L: {:.2f}".format(overall_pl))
    print("Min balance: {:.2f}".format(min_balance))
    print("Max balance: {:.2f}".format(max_balance))
    print("Maximum Drawdown: {:.2f}".format(max_drawdown))
    print("Maximum Drawdown %: {:.2f}%".format(max_drawdown_percentage))

    # Return the Strategy DataFrame
    return strategy_df

# Plot the performance curve
def plot_performance_curve(strategy_df):
    plt.plot(strategy_df['cumulative_balance'], label='Strategy')
    plt.title('Performance Curve')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.xticks(rotation=70)
    plt.legend()
    plt.show()

def get_supertrend(data, multiplier=3, capital=100, leverage=1):
    supertrend_data=supertrend(data, multiplier)
    supertrend_signals=generate_signals(supertrend_data)
    supertrend_positions=create_positions(supertrend_signals)
    supertrend_df=strategy_performance(supertrend_positions, capital, leverage)
    return supertrend_df

if __name__ == '__main__':
    # Initialize data fetch parameters
    symbol = "BTCUSDT"
    start_date = "2019-09-09"
    interval = '4h'
    exchange = ccxt.binance()

    # Fetch historical OHLC data for ETH/USDT
    data = get_chart_data()

    data['time']=(pd.to_datetime(data['Open Time'], unit='ms', utc=True).dt.tz_convert('Asia/Seoul'))
    
    data.set_index('time', inplace=True)

    volatility = 3

    # Apply supertrend formula
    supertrend_data = supertrend(df=data, atr_multiplier=volatility)

    # Generate the Signals
    supertrend_positions = generate_signals(supertrend_data)

    # Generate the Positions
    supertrend_positions = create_positions(supertrend_positions)

    # Calculate performance
    supertrend_df = strategy_performance(supertrend_positions, capital=100, leverage=1)
    print(supertrend_df)

    # Plot data
    plot_data(supertrend_positions, symbol=symbol)
    
    # Plot the performance curve
    plot_performance_curve(supertrend_df)
