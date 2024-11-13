from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import aiohttp
import aiofiles
from binance.client import AsyncClient
from binance.cm_futures import CMFutures
from binance.um_futures import UMFutures
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from getpass import getpass
import ta
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict
import os
import asyncio
import uvicorn
from typing import Dict, Any
from datetime import datetime, timedelta
import json
import lightweight_charts as lwc
import requests

from src.srchannels import SupportResistanceAnalyzer, ChannelAnalyzer
# from src.supertrend import supertrend, generate_signals, create_positions, strategy_performance

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

api_key_file = Path.home() / '.binance' / 'api_key.txt'
api_secret_file = Path.home() / '.binance' / 'api_secret.txt'
predictions_file = Path().parent / 'data' / 'predictions.csv'
predictions_file.parent.mkdir(parents=True, exist_ok=True)

# API 키 관련 함수들은 동일하게 유지
async def get_api(file_path, prompt_func):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        return await prompt_func()

def set_api(file_path, api):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    with open(file_path, 'w') as f:
        f.write(api)

async def input_api_key():
    api_key = input("Enter your Binance API key: ")
    set_api(api_key_file, api_key)
    return api_key

async def input_api_secret():
    api_secret = getpass("Enter your Binance API secret: ")
    set_api(api_secret_file, api_secret)
    return api_secret

async def initialize_client():
    api_key = await get_api(api_key_file, input_api_key)
    api_secret = await get_api(api_secret_file, input_api_secret)
    return await AsyncClient.create(api_key, api_secret)

async def initialize_cm_futures():
    api_key = await get_api(api_key_file, input_api_key)
    api_secret = await get_api(api_secret_file, input_api_secret)
    return CMFutures(key=api_key, secret=api_secret)

async def initialize_um_futures():
    api_key = await get_api(api_key_file, input_api_key)
    api_secret = await get_api(api_secret_file, input_api_secret)
    return UMFutures(key=api_key, secret=api_secret)

# 이동 평균 계산 함수
def calculate_ema(data: pd.DataFrame, period: int) -> pd.DataFrame:
    return ta.trend.ema_indicator(data['Close'], window=period)

def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    data['EMA9'] = calculate_ema(data, 9)
    data['EMA60'] = calculate_ema(data, 60)
    data['EMA200'] = calculate_ema(data, 200)
    return data

# RSI 및 관련 계산 함수
def calculate_rsi(data: pd.DataFrame) -> pd.DataFrame:
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['RSI_SMA'] = data['RSI'].rolling(window=9).mean()
    return data

# Stochastic RSI 계산 함수
def calculate_stochastic_rsi(data: pd.DataFrame) -> pd.DataFrame:
    stoch_rsi = ta.momentum.StochRSIIndicator(data['Close'], window=14, smooth1=3, smooth2=3)
    data['StochRSI_%K'] = stoch_rsi.stochrsi_k()
    data['StochRSI_%D'] = stoch_rsi.stochrsi_d()
    return data

# MACD 계산 함수
def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
    macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()
    return data

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = calculate_moving_averages(data)
    data = calculate_rsi(data)
    data = calculate_stochastic_rsi(data)
    data = calculate_macd(data)
    return data

def calculate_support_resistance_levels(data: pd.DataFrame) -> None:
    analyzer = SupportResistanceAnalyzer()
    channel_analyzer = ChannelAnalyzer()

    levels = analyzer.find_key_levels(data)
    channels = channel_analyzer.find_channels(data)

    channel_lines = []
    
    for i, level in enumerate(levels['resistance_levels'][:3], 1):
        data[f'Resistance_1st_Level{i}'] = level['price']
        if i < len(levels['resistance_levels']):
            data[f'Resistance_2nd_Level{i}'] = levels['resistance_levels'][i+2]['price']
            
    for i, level in enumerate(levels['support_levels'][:3], 1):
        data[f'Support_1st_Level{i}'] = level['price']
        if i < len(levels['support_levels']):
            data[f'Support_2nd_Level{i}'] = levels['support_levels'][i+2]['price']

def calculate_trendlines(data: pd.DataFrame) -> dict:
    analyzer = SupportResistanceAnalyzer()
    channel_analyzer = ChannelAnalyzer()
    
    # 기존 트렌드라인 분석
    levels = analyzer.find_key_levels(data)
    
    # 채널 분석 추가
    channels = channel_analyzer.find_channels(data)
    
    # 채널 라인 생성
    channel_lines = []
    for channel in channels:
        lines = channel_analyzer.get_channel_lines(channel, data)
        
        # 저항선 추가
        channel_lines.append({
            'type': 'resistance_channel',
            'points': lines['resistance'],
            'color': 'rgba(255, 0, 0, 0.3)',
            'lineWidth': 1,
            'strength': channel['strength']
        })
        
        # 지지선 추가
        channel_lines.append({
            'type': 'support_channel',
            'points': lines['support'],
            'color': 'rgba(0, 255, 0, 0.3)',
            'lineWidth': 1,
            'strength': channel['strength']
        })
    
    return {
        'resistance': levels['resistance_lines'],
        'support': levels['support_lines'],
        'channels': channel_lines
    }

# def get_supertrend(data, multiplier=3, capital=100, leverage=1):
#     supertrend_data=supertrend(data, multiplier)
#     supertrend_signals=generate_signals(supertrend_data)
#     supertrend_positions=create_positions(supertrend_signals)
#     supertrend_df=strategy_performance(supertrend_positions, capital, leverage)
#     return supertrend_df

async def get_futures_price(um_futures, symbol: str = "BTCUSDT", interval: str = "4h", limit: int = 1000) -> pd.DataFrame:
    """비트코인 선물 가격 정보를 가져오는 함수"""
    # 비트코인 선물 데이터 가져오기
    klines = um_futures.klines(symbol=symbol, interval=interval, limit=limit)
    
    # DataFrame으로 변환
    data = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
        'Quote Asset Volume', 'Number of Trades', 
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    # 타임스탬프를 날짜 형식으로 변환
    data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
    data.set_index('Open Time', inplace=True)
    return data.astype(float)

async def get_historical_klines(client, symbol, interval, start_date: str = None, end_date: str = None, limit: int = 1000) -> pd.DataFrame:
    # 기본값 설정
    if end_date is None:
        end_time = datetime.now()  # 현재 시간을 기본 종료 시간으로 설정
    else:
        end_time = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    if start_date is None:
        start_str = None
    else:
        start_time = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    # 데이터 수집
    klines = await client.get_historical_klines(symbol, interval, start_str, end_str, limit)

    # 데이터가 없으면 빈 데이터프레임 반환
    if not klines:
        return pd.DataFrame(columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
        ])

    # 데이터프레임으로 변환
    data = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])

    # 타임스탬프 열을 날짜 형식으로 변환하고 인덱스로 설정
    data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
    data.set_index('Open Time', inplace=True)

    return data.astype(float)

def predict_price(data: pd.DataFrame) -> pd.Series:
    data.dropna(inplace=True)

    features = ['EMA9', 'EMA60', 'EMA200', 'RSI', 'StochRSI_%K', 'StochRSI_%D', 'MACD', 'MACD_Signal', 'Volume']
    X, y = data[features], data['Close']
    y0, y1 = data['High'], data['Low']

    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    y0_train, y1_train = y0[:train_size], y1[:train_size]

    model, model1, model2 = LinearRegression(), LinearRegression(), LinearRegression()
    
    model1.fit(X_train, y0_train)
    model2.fit(X_train, y1_train)
    model.fit(X_train, y_train)

    data['Predicted_High'] = model1.predict(X)
    data['Predicted_Low'] = model2.predict(X)
    data['Predicted_Close'] = model.predict(X)

    return data[['High', 'Low', 'Close', 'Predicted_High', 'Predicted_Low', 'Predicted_Close']].iloc[-1]

async def save_prediction_to_csv(prediction: pd.Series, timestamp: datetime) -> None:
    prediction_data = {
        "timestamp": timestamp,
        "actual_high": round(prediction['High'], 1),
        "actual_low": round(prediction['Low'], 1),
        "actual_close": round(prediction['Close'], 1),
        "predicted_high": round(prediction['Predicted_High'], 1),
        "predicted_low": round(prediction['Predicted_Low'], 1),
        "predicted_close": round(prediction['Predicted_Close'], 1)
    }
    
    file_exists = predictions_file.exists()
    async with aiofiles.open(predictions_file, 'a') as f:
        if not file_exists:
            await f.write('timestamp,actual_high,actual_low,actual_close,predicted_high,predicted_low,predicted_close\n')
        await f.write(','.join(map(str, prediction_data.values())) + '\n')

async def update_predictions():
    while True:
        # data = await get_data_and_indicators(app.state.client)
        um_data = await get_futures_price(app.state.um_futures)
        data = calculate_indicators(um_data)
        prediction = predict_price(data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await save_prediction_to_csv(prediction, timestamp)
        print(f"Prediction updated at {timestamp}")
        await asyncio.sleep(60)

async def get_data_and_indicators(client: AsyncClient) -> pd.DataFrame:
    data = await get_historical_klines(client, "BTCUSDT", AsyncClient.KLINE_INTERVAL_4HOUR)
    calculate_indicators(data)
    # calculate_support_resistance_levels(data)
    return data

@app.on_event("startup")
async def startup_event():
    app.state.client = await initialize_client()
    app.state.um_futures = await initialize_um_futures()
    asyncio.create_task(update_predictions())

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/data")
async def get_data():
    data = await get_data_and_indicators(app.state.client)
    um_data = await get_futures_price(app.state.um_futures)
    indicator_data=calculate_indicators(um_data)
    srlines=calculate_support_resistance_levels(um_data)
    trendlines = calculate_trendlines(um_data)
    # supertrend_df = get_supertrend(um_data)

    data['time'] = data.index.astype(int) // 10**9
    data = data.reset_index()

    um_data['time'] = um_data.index.astype(int) // 10**9
    um_data = um_data.reset_index()

    indicator_data['time'] = indicator_data.index.astype(int) // 10**9
    indicator_data = indicator_data.reset_index()

    data=json.loads(data.to_json(orient='records', date_format='iso'))
    price_data=json.loads(um_data.to_json(orient='records', date_format='iso'))
    indicator_json=json.loads(indicator_data.to_json(orient='records', date_format='iso'))
    srlines_json=json.loads(json.dumps(srlines))
    # supertrend_json = json.loads(supertrend_df.to_json(orient='records', date_format='iso'))

    response_data={
        # "data": data,
        "priceData": price_data,
        "indicatorData": indicator_json,
        "supportresistance": srlines_json,
        "trendlines": trendlines,
        # "supertrend": supertrend_json
    }
    return JSONResponse(response_data)

@app.get("/predict")
async def get_prediction():
    # data = await get_data_and_indicators(app.state.client)
    um_data = await get_futures_price(app.state.um_futures)
    data = calculate_indicators(um_data)
    prediction = predict_price(data)
    timestamp=datetime.now()
    await save_prediction_to_csv(prediction, timestamp)
    prediction['time'] = timestamp.isoformat()
    return JSONResponse(prediction.to_dict())

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)