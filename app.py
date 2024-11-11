from flask import Flask, jsonify, render_template
from flask_caching import Cache
from binance.client import Client
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path
from getpass import getpass
import ta
import os
import time
import cachetools.func

app=Flask(__name__)

cache = Cache(config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})  # 5분 캐시 유지
cache.init_app(app)


api_key_file=Path.home()/'.binance'/'api_key.txt'
api_secret_file=Path.home()/'.binance'/'api_secret.txt'
predictions_file = Path.home() / '.binance' / 'predictions.csv'

def get_api(file_path, prompt_func):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        prompt_func()
        
def set_api(file_path, api):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    with open(file_path, 'w') as f:
        f.write(api)

def input_api_key():
    api_key = input("Enter your Binance API key: ")
    set_api(api_key_file, api_key)
    return api_key_file, api_key

def input_api_secret():
    api_secret = getpass("Enter your Binance API secret: ")
    set_api(api_secret_file, api_secret)
    return api_secret_file, api_secret

# API 키 설정 (보안을 위해 환경 변수나 별도 파일에서 불러오는 것을 권장)
api_key = get_api(api_key_file, input_api_key)
api_secret = get_api(api_secret_file, input_api_secret)
client = Client(api_key, api_secret)

def calculate_indicators(data):
    # 이동 평균 계산
    # data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
    # data['SMA60'] = ta.trend.sma_indicator(data['Close'], window=60)
    # data['SMA200'] = ta.trend.sma_indicator(data['Close'], window=200)

    data['EMA9'] = ta.trend.ema_indicator(data['Close'], window=9)
    data['EMA60'] = ta.trend.ema_indicator(data['Close'], window=60)
    data['EMA200'] = ta.trend.ema_indicator(data['Close'], window=200)

    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['RSI_SMA'] = data['RSI'].rolling(window=9).mean()

    stoch_rsi = ta.momentum.StochRSIIndicator(data['Close'], window=14, smooth1=3, smooth2=3)
    data['StochRSI_%K'] = stoch_rsi.stochrsi_k()
    data['StochRSI_%D'] = stoch_rsi.stochrsi_d()

    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()

def calculate_support_resistance_levels(data):
    lookback_period = 60
    highs = data['High'].rolling(window=lookback_period).max().dropna().unique()
    lows = data['Low'].rolling(window=lookback_period).min().dropna().unique()
    levels = ["Level1", "Level2", "Level3"]
    for i, level in enumerate(levels):
        data[f'Resistance_1st_{level}'] = highs[-(i + 1)]
        data[f'Support_1st_{level}'] = lows[i]
        if len(highs) > i + 3:
            data[f'Resistance_2nd_{level}'] = highs[-(i + 4)]
        if len(lows) > i + 3:
            data[f'Support_2nd_{level}'] = lows[i + 3]


# def update_data_cache(symbol, interval, lookback):
#     data = get_historical_klines(symbol, interval, lookback)
#     calculate_indicators(data)
#     calculate_support_resistance_levels(data)
#     # Redis에 데이터 저장 (JSON 형식으로 직렬화)
#     redis_client.set("market_data", data.to_json(orient='records', date_format='iso'))
#     print("Data cache updated.")

def fetch_historical_data(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_4HOUR, lookback='128 days ago UTC+9:00'):
    klines = client.get_historical_klines(symbol, interval, lookback)
    data = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
    data.set_index('Open Time', inplace=True)
    data = data.astype(float)
    return data

def predict_price(data):
    # 결측치 제거
    data.dropna(inplace=True)

    # 특징 변수(X)와 타깃 변수(y) 설정
    features = ['EMA9', 'EMA60', 'EMA200', 'RSI', 'StochRSI_%K', 'StochRSI_%D', 'MACD', 'MACD_Signal', 'Volume']
    X, y = data[features], data['Close']
    y0, y1 = data['High'], data['Low']

    # 데이터 분할 (훈련 세트와 테스트 세트)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    y0_train, y1_train = y0[:train_size], y1[:train_size]

    # 모델 학습
    model, model1, model2 = LinearRegression(), LinearRegression(), LinearRegression()
    
    model1.fit(X_train, y0_train)
    model2.fit(X_train, y1_train)
    model.fit(X_train, y_train)

    # 예측 수행
    data['Predicted_High'] = model1.predict(X)
    data['Predicted_Low'] = model2.predict(X)
    data['Predicted_Close'] = model.predict(X)

    # 마지막 행의 실제 값과 예측값 반환
    return data[['High', 'Low', 'Close', 'Predicted_High', 'Predicted_Low', 'Predicted_Close']].iloc[-1]


# 데이터 저장 함수
def save_prediction_to_csv(actual_high, actual_low, actual_close, pred_high, pred_low, pred_close, timestamp):
    if not predictions_file.exists():
        predictions_file.write_text('timestamp,actual_high,actual_low,actual_close,predicted_high,predicted_low,predicted_close\n')

    with open(predictions_file, 'a') as f:
        f.write(f"{timestamp},{actual_high},{actual_low},{actual_close},{pred_high},{pred_low},{pred_close}\n")

# 1분마다 데이터 예측 및 저장
def update_and_save_prediction():
    while True:
        data = get_data_and_indicators()
        prediction = predict_price(data)
        timestamp = data.index[-1]
        save_prediction_to_csv(
            prediction['High'], prediction['Low'], prediction['Close'],
            prediction['Predicted_High'], prediction['Predicted_Low'], prediction['Predicted_Close'],
            timestamp
        )
        time.sleep(60)  # 1분마다 업데이트

# scheduler.add_job(update_data_cache, 'interval', minutes=1)

@cache.cached(timeout=300, key_prefix='market_data') 
def get_data_and_indicators():
    data = fetch_historical_data()
    calculate_indicators(data)
    calculate_support_resistance_levels(data)
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    data = get_data_and_indicators()
    data_json=data.reset_index().to_json(orient='records', date_format='iso')
    return data_json

if __name__=='__main__':
    from threading import Thread
    prediction_thread = Thread(target=update_and_save_prediction)
    prediction_thread.daemon = True
    prediction_thread.start()
    app.run(debug=True, host='0.0.0.0', port=5001)
