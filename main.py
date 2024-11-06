from binance.client import Client
import pandas as pd

# API 키 설정 (보안을 위해 환경 변수나 별도 파일에서 불러오는 것을 권장)
api_key = 'your_api_key'
api_secret = 'your_api_secret'
client = Client(api_key, api_secret)

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

# 데이터 가져오기
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR  # 1시간 봉
lookback = '60 days ago UTC'  # 최근 60일 데이터
data = get_historical_klines(symbol, interval, lookback)

import ta

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

from sklearn.linear_model import LinearRegression
import numpy as np

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

import matplotlib.pyplot as plt

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

import time

while True:
    # 데이터 갱신
    data = get_historical_klines(symbol, interval, lookback)
    # 지표 재계산 및 예측 수행
    # ...
    # 차트 업데이트 또는 데이터 저장
    # ...
    time.sleep(60)  # 1분마다 업데이트