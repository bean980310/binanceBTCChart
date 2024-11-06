from flask import Flask, jsonify, render_template
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

app=Flask(__name__)

api_key_file=Path.home()/'.binance'/'api_key.txt'
api_secret_file=Path.home()/'.binance'/'api_secret.txt'

def get_api_key():
    try:
        with open(api_key_file, 'r') as f:
            return f.read().strip()
    except Exception as e:
        input_api_key()
        
def set_api_key(api_key):
    if not api_key_file.parent.exists():
        api_key_file.parent.mkdir(parents=True)
    with open(api_key_file, 'w') as f:
        f.write(api_key)

def input_api_key():
    api_key = input("Enter your Binance API key: ")
    set_api_key(api_key)
    return api_key

def get_api_secret():
    try:
        with open(api_secret_file, 'r') as f:
            return f.read().strip()
    except Exception as e:
        input_api_secret()

def set_api_secret(api_secret):
    if not api_secret_file.parent.exists():
        api_secret_file.parent.mkdir(parents=True)
    with open(api_secret_file, 'w') as f:
        f.write(api_secret)

def input_api_secret():
    api_secret = getpass("Enter your Binance API secret: ")
    set_api_secret(api_secret)
    return api_secret

# API 키 설정 (보안을 위해 환경 변수나 별도 파일에서 불러오는 것을 권장)
api_key = get_api_key()
api_secret = get_api_secret()
client = Client(api_key, api_secret)

def get_data_and_indicators():
    # 데이터 가져오기
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1HOUR  # 1시간 봉
    lookback = '60 days ago UTC'  # 최근 60일 데이터
    data = get_historical_klines(symbol, interval, lookback)

    # EMA 계산
    data['EMA9'] = ta.trend.ema_indicator(data['Close'], window=9)
    data['EMA60'] = ta.trend.ema_indicator(data['Close'], window=60)
    data['EMA200'] = ta.trend.ema_indicator(data['Close'], window=200)

    # RSI 계산
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

    # 스토캐스틱 RSI 계산
    stoch_rsi = ta.momentum.StochRSIIndicator(data['Close'], window=14, smooth1=3, smooth2=3)
    data['StochRSI_%K'] = stoch_rsi.stochrsi_k()
    data['StochRSI_%D'] = stoch_rsi.stochrsi_d()
    

    # MACD 계산
    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()

    # 볼륨은 이미 데이터에 포함되어 있습니다.
    return data


# 히스토리컬 캔들 데이터 가져오기
def get_historical_klines(symbol, interval, lookback):
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
    X = data[features]
    y = data['Close']

    # 데이터 분할 (훈련 세트와 테스트 세트)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 수행
    data['Predicted_Close'] = model.predict(X)


def show_chart(data):
    # Close 가격과 EMA 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.plot(data.index, data['EMA9'], label='EMA9')
    plt.plot(data.index, data['EMA60'], label='EMA60')
    plt.plot(data.index, data['EMA200'], label='EMA200')
    plt.legend()
    plt.show()

    # 실제 종가와 예측 종가 비교
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Actual Close Price')
    plt.plot(data.index, data['Predicted_Close'], label='Predicted Close Price')
    plt.legend()
    plt.show()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    data = get_data_and_indicators()
    data_json=data.reset_index().to_json(orient='records', date_format='iso')
    return data_json

if __name__=='__main__':
    app.run(debug=True)

# while True:
#     # 데이터 갱신
#     data = get_data_and_indicators()
#     # 지표 재계산 및 예측 수행
#     predict_price(data)
#     # 차트 업데이트 또는 데이터 저장
#     show_chart(data)
#     time.sleep(60)  # 1분마다 업데이트