from flask import Flask, request, render_template
import threading
import MetaTrader5 as mt5
import joblib
import ta
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

app = Flask(__name__ ,template_folder='templates')

def run_trades(symbol, login, server, password, lot, trades_num):
    if symbol=="BTCUSDm":
        num_bars = 50
        timeframe = mt5.TIMEFRAME_M15
    else:
        num_bars = 200
        timeframe = mt5.TIMEFRAME_M3

    def get_symbol_data(login, server, password, symbol, timeframe, num_bars):

        # Ensure MetaTrader 5 is connected
        if not mt5.initialize(login=login, server=server, password=password):
            print("Failed to initialize MetaTrader 5")
            return None , None
        
        # Get the market data (OHLC) for the symbol
        bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        
        # Convert the market data to a pandas DataFrame
        df = pd.DataFrame(bars)
        
        # Convert time in seconds to a readable datetime format
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Returning the DataFrame
        return df , True

    def calculate_indicators_and_predict(symbol, df):
        min_required_data = 50

        if len(df) >= min_required_data:
            
            if symbol == "BTCUSDm":
                model = joblib.load("model_bitcoin.pkl")
                df['SMA_21'] = ta.trend.sma_indicator(df['close'], window=21)
                df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['RSI'] = ta.momentum.rsi(df['close'], window=9)
                macd = ta.trend.MACD(df['close'], window_slow=21, window_fast=9, window_sign=7)
                df['MACD'] = macd.macd()
                df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=7)

            elif symbol == "EURUSDm":
                model = joblib.load('model_eur.pkl')
                df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
                df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                df['MACD'] = ta.trend.MACD(df['close']).macd()
                df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

            elif symbol == "XAUUSDm":
                model = joblib.load('model_gold.pkl')
                df['SMA_100'] = ta.trend.sma_indicator(df['close'], window=100)
                df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
                df['RSI'] = ta.momentum.rsi(df['close'], window=14)
                macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['MACD'] = macd.macd()
                df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

            elif symbol == "USTECm":
                model = joblib.load('model_nasdaq.pkl')
                df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
                df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['MACD'] = macd.macd()
                df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

            elif symbol == "USOILm":
                model = joblib.load('model_oil.pkl')
                df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
                df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['MACD'] = macd.macd()
                df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

            # Drop missing values before making predictions
            df.dropna(inplace=True)
            
            # Extract features and make predictions
            if 'SMA_21' in df.columns:
                features = df[['SMA_21', 'SMA_50', 'RSI', 'MACD', 'ATR']]
            elif 'SMA_50' in df.columns:
                features = df[['SMA_50', 'SMA_200', 'RSI', 'MACD', 'ATR']]
            else:
                features = df[['SMA_100', 'SMA_200', 'RSI', 'MACD', 'ATR']]
            
            predictions = model.predict(features)
            df['Predictions'] = predictions
            
            print(df['Predictions'].value_counts())
        else:
            print("Not enough data to calculate indicators.")
        
        return df

    def check_entry(login, server, password, symbol, timeframe, num_bars):
        df, _ = get_symbol_data(login, server, password, symbol, timeframe, num_bars)
        calculate_indicators_and_predict(symbol,df)
        rsi = df["RSI"].iloc[-1]

        if rsi < 30 :
            print ("over sold")
            return True
        elif  rsi > 70 :
            print("overbought")
            return True
        else:
            return False

    def execute_trade(signal, symbol, lot, deviation=10):
        symbol_info_tick = mt5.symbol_info_tick(symbol)
        if not symbol_info_tick:
            print(f"Failed to get symbol tick info for {symbol}")
            return
        
        ask_price = symbol_info_tick.ask
        bid_price = symbol_info_tick.bid

        if ask_price is None or bid_price is None:
            print(f"Failed to get ask or bid prices for {symbol}")
            return

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"Failed to get symbol info for {symbol}")
            return
        
        df_sl['range'] = df_sl['high'] - df_sl['low']
        avg_range = df_sl['range'].tail(5).mean()

        if signal == 1:  # Buy order
            order_type = mt5.POSITION_TYPE_BUY
            price = ask_price
            stop_loss = price - avg_range*5
            take_profit = price + avg_range*5
        elif signal == -1:  # Sell order
            order_type = mt5.POSITION_TYPE_SELL
            price = bid_price
            stop_loss = price + avg_range*5
            take_profit = price - avg_range*5
        else:
            # Hold (no trade)
            return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Trading bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: retcode={result.retcode}, error={mt5.last_error()}")
        else:
            print(f"Order placed: {symbol}, signal: {signal}, retcode={result.retcode}")
            print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")

    _ , is_none = get_symbol_data(login, server, password, symbol, timeframe, num_bars)
    if is_none == None:
        return "incorrection"

    check = check_entry(login, server, password, symbol, mt5.TIMEFRAME_M5, num_bars)
    print(check)

    if check == False:
        while check == False:
            print ("waiting for entry point...")
            time.sleep(10)
            check = check_entry(login, server, password, symbol, mt5.TIMEFRAME_M5, num_bars)

    df, _ = get_symbol_data(login, server, password, symbol, timeframe, num_bars)

    df_sl, _ = get_symbol_data(login, server, password, symbol, timeframe=mt5.TIMEFRAME_M3, num_bars=200)

    calculate_indicators_and_predict(symbol,df)

    prediction = df["Predictions"].iloc[-1]
    print(prediction)
    for i in range (trades_num):
        execute_trade(prediction, symbol, lot)

    def check_hanging_man(df):
        # Check if the current candle is a Hanging Man
        if (df['close'].iloc[-1] < df['open'].iloc[-1] and  # bearish candle
            df['low'].iloc[-1] < df['close'].iloc[-1] and  # lower wick is at least 2 times the body
            df['low'].iloc[-1] < df['high'].iloc[-1] * 0.2 and  # lower wick is at least 2 times the body
            df['high'].iloc[-1] - df['low'].iloc[-1] > 2 * (df['open'].iloc[-1] - df['close'].iloc[-1]) and  # upper wick is small
            df['high'].iloc[-1] - df['low'].iloc[-1] > (df['high'].iloc[-2] - df['low'].iloc[-2]) * 0.5):  # wick is larger than previous candle
            # Check the market trend
            print ("There is a hanging man")
            if df['close'].iloc[-2] > df['close'].iloc[-3]:
                # If the market is trending up and a Hanging Man occurs, signal a sell**

                return -1
            elif df['close'].iloc[-2] < df['close'].iloc[-3]:
                # If the market is trending down and a Hanging Man occurs, signal a buy
                return 1  # buy signal
        else:
            print ("There is not a hanging man")
            return None

    def modify_sl(symbol, ticket, adjustment_percentage):
        # Get the current trade details
        trade_info = mt5.positions_get(ticket=ticket)
        if not trade_info:
            print(f"No open positions found for ticket {ticket}")
            return

        # Extract the relevant details from the trade
        current_price = mt5.symbol_info_tick(symbol).ask if trade_info[0].type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        current_sl = trade_info[0].sl
        
        # Calculate the difference between the current price and the current stop-loss
        price_sl_diff = abs(current_price - current_sl)
        
        # Adjust the stop-loss based on the given percentage of the difference
        adjustment_value = price_sl_diff * adjustment_percentage / 100
        
        # Update the stop-loss closer to the current price
        if trade_info[0].type == mt5.ORDER_TYPE_BUY:
            new_sl = current_sl + adjustment_value  # Move the stop-loss closer to the current price for a buy
        else:
            new_sl = current_sl - adjustment_value  # Move the stop-loss closer to the current price for a sell
        
        # Modify the position with the new stop-loss
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": new_sl,
            "tp": trade_info[0].tp,  # Keep the take-profit as it is
            "deviation": 10,
        }
        
        # Send the request to modify the stop-loss
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Stop-loss successfully modified to {new_sl}")
        else:
            print(f"Failed to modify stop-loss. Error code: {result.retcode}")


    def monitor_trades(login, server, password, timeframe, num_bars):
        while True:
            # Get all open positions
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                print(f"{datetime.now()} - No open positions.")
                break
            else:
                for position in positions:
                    symbol = position.symbol
                    ticket = position.ticket
                    
                    # Fetch the most recent market data for the symbol
                    df, _ = get_symbol_data(login, server, password, symbol, timeframe, num_bars)
                    
                    # Predict the future movement using our model
                    signal = calculate_indicators_and_predict(symbol, df)
                    signal = signal['Predictions'].iloc[-1]
                    # Check for Hanging Man pattern
                    hanging_man_signal = check_hanging_man(df)
                    
                    if hanging_man_signal is not None:
                        if signal == hanging_man_signal:
                            # No change in signal
                            pass
                        else:
                            # Change the signal
                            signal = hanging_man_signal
                    else:
                        # Check if the trade is not doing well
                        max_drawdown = 0
                        for i in range(1, len(df)):
                            drawdown = (df['High'].iloc[i] - df['Low'].iloc[i]) / df['Close'].iloc[i-1]
                            if drawdown > max_drawdown:
                                max_drawdown = drawdown
                        if max_drawdown > 2:  # example threshold, adjust as needed
                            # Close the trade
                            close_trade(ticket, symbol, 0.1)
                    # executed_signal = 1 if position.type == mt5.POSITION_TYPE_BUY else -1
                    # # Check if the signal has changed
                    # if signal!= executed_signal:
                    #     # Close the current position
                    #     close_trade(ticket, symbol, 0.1)
                        
                    #     # Open a new position in the opposite direction
                    #     if signal == 1:
                    #         execute_trade(1, symbol)
                    #     elif signal == -1:
                    #         execute_trade(-1, symbol)
                    
                    # Modify the SL if the trade is in profit
                    if position.profit > 20:
                        modify_sl(symbol, ticket, 80)
            
            time.sleep(5)
        
    def close_trade(ticket, symbol, volume):
        print(f"Closing trade: {ticket}, Symbol: {symbol}")
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if mt5.positions_get(ticket=ticket)[0].type == 0 else mt5.ORDER_TYPE_BUY,
            "deviation": 10,
            "magic": 234000,
            "comment": "Closing underperforming trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close trade {ticket}: Error {result.retcode}")
        else:
            print(f"Successfully closed trade {ticket}")

    monitor_trades(login, server, password, timeframe, num_bars)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']
        lot = float(request.form['lot'])
        trades_num = int(request.form['number of trades'])
        login = int(request.form['login'])
        server = request.form['server']
        password = request.form['password']
        thread = threading.Thread(target=run_trades, args=(symbol, login, server, password, lot, trades_num))
        thread.start()
    return render_template('index.html')

# @app.route('/new_trade', methods=['GET', 'POST'])
# def new_trade():
#     if request.method == 'POST':
#         symbol = request.form['symbol']
#         lot = float(request.form['lot'])
#         trades_num = int(request.form['number of trades'])
#         login = int(request.form['login'])
#         server = request.form['server']
#         password = request.form['password']
#         thread = threading.Thread(target=run_trades, args=(symbol, login, server, password, lot, trades_num))
#         thread.start()
#     return "render_template('index.html')"

if __name__ == '__main__':
    app.run(debug=True)