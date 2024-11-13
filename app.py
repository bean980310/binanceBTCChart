from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import aiohttp
import aiofiles
from binance import AsyncClient
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
import ta.momentum
import ta.trend
import uvicorn
from typing import Dict, Any
from datetime import datetime, timedelta
import json
import lightweight_charts as lwc
import requests
from io import StringIO

from src.srchannels import SupportResistanceAnalyzer, ChannelAnalyzer
from src.supertrend import supertrend, generate_signals, create_positions, strategy_performance
from src.fetch import read_existing_csv, save_dataframe_to_csv, fetch_new_klines, get_last_timestamp, calculate_indicators

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

api_key_file = Path.home() / '.binance' / 'api_key.txt'
api_secret_file = Path.home() / '.binance' / 'api_secret.txt'
predictions_file = Path().parent / 'data' / 'predictions.csv'
csv_file_path = Path().parent / 'data' / 'btc_futures_data.csv'
predictions_file.parent.mkdir(parents=True, exist_ok=True)

data_cache = {}
cache_lock = asyncio.Lock()

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

async def read_csv_async(file_path: Path) -> pd.DataFrame:
    """비동기적으로 CSV 파일을 읽어 DataFrame으로 반환하는 함수"""
    if not file_path.exists():
        return pd.DataFrame()
    
    async with aiofiles.open(file_path, mode='r') as f:
        contents = await f.read()
    df = pd.read_csv(StringIO(contents), dtype={'Open Time': str, 'Close Time': str}, parse_dates=['Open Time'], index_col='Open Time')
    
    if not df.empty:
        try:
            df['Close Time'] = pd.to_datetime(df['Close Time'], format='%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"CSV 데이터 변환 오류: {e}")
            return pd.DataFrame()
    return df

async def load_csv_data():
    """애플리케이션 시작 시 CSV 데이터를 로드하여 캐시에 저장"""
    global data_cache
    async with cache_lock:
        data = await read_csv_async(csv_file_path)
        data_cache['btc_futures'] = data
        print("CSV 데이터 로드 완료. 데이터 행 수:", len(data))

async def fetch_and_update_data(client: AsyncClient, symbol, interval, last_timestamp):
    while True:
        try:
            new_data = await fetch_new_klines(client, symbol, interval, start_time=last_timestamp)

            if new_data is not None and not new_data.empty:
                print(f"새로 가져온 데이터 행 수: {len(new_data)}")

                # 기존 CSV 데이터 읽기
                existing_data = await read_existing_csv(csv_file_path)

                if not existing_data.empty:
                    # 기존 데이터와 새 데이터를 병합
                    combined_data = pd.concat([existing_data, new_data])

                    # 'Open Time'과 'Close Time'은 이미 시간대가 지정되어 있음

                    # 숫자 열만 float으로 변환
                    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    combined_data[numeric_columns] = combined_data[numeric_columns].astype(float)

                    # 'Open Time'을 기준으로 중복 제거 (keep='last'로 최근 데이터 유지)
                    combined_data.drop_duplicates(subset=['Open Time'], keep='last', inplace=True)

                    # 지표 계산
                    combined_data = calculate_indicators(combined_data)

                    # 불필요한 컬럼 제거
                    combined_data = combined_data.drop(['Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'], axis=1)

                    # 'Open Time'과 'Close Time'을 문자열로 변환
                    combined_data['Open Time'] = combined_data['Open Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    combined_data['Close Time'] = combined_data['Close Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    # 컬럼 순서 지정
                    column_order = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'EMA9', 'EMA60', 'EMA200', 'RSI', 'RSI_SMA', 'StochRSI_%K', 'StochRSI_%D', 'MACD', 'MACD_Signal', 'MACD_Hist']
                    combined_data = combined_data[column_order]

                    # 데이터 정렬
                    combined_data = combined_data.sort_values(by='Open Time', ascending=True)

                    # CSV 파일에 저장
                    await save_dataframe_to_csv(combined_data, csv_file_path)
                else:
                    # 기존 데이터가 없으면 새 데이터로 CSV 파일 생성
                    combined_data = new_data.copy()

                    # 숫자 열만 float으로 변환
                    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    combined_data[numeric_columns] = combined_data[numeric_columns].astype(float)

                    # 지표 계산
                    combined_data = calculate_indicators(combined_data)

                    # 불필요한 컬럼 제거
                    combined_data = combined_data.drop(['Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'], axis=1)

                    # 'Open Time'과 'Close Time'을 문자열로 변환
                    combined_data['Open Time'] = combined_data['Open Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    combined_data['Close Time'] = combined_data['Close Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    # 컬럼 순서 지정
                    column_order = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'EMA9', 'EMA60', 'EMA200', 'RSI', 'RSI_SMA', 'StochRSI_%K', 'StochRSI_%D', 'MACD', 'MACD_Signal', 'MACD_Hist']
                    combined_data = combined_data[column_order]

                    # 데이터 정렬
                    combined_data = combined_data.sort_values(by='Open Time', ascending=True)

                    # CSV 파일에 저장
                    await save_dataframe_to_csv(combined_data, csv_file_path)
                print("데이터가 성공적으로 업데이트되었습니다.")
            else:
                print("새로운 데이터가 없습니다.")
        except Exception as e:
            print(f"데이터 페칭 중 예외 발생: {e}")

def get_chart_data():
    data = pd.read_csv(csv_file_path, parse_dates=['Open Time'])
    return data

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
        data=get_chart_data()
        prediction = predict_price(data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await save_prediction_to_csv(prediction, timestamp)
        print(f"Prediction updated at {timestamp}")
        await asyncio.sleep(60)

# async def get_support_resistance(um_futures: UMFutures) -> pd.DataFrame:
#     df = await get_chart_data()
#     srlines = calculate_support_resistance_levels(df)
#     return srlines

@app.on_event("startup")
async def startup_event():
    app.state.client = await initialize_client()
    last_timestamp = await get_last_timestamp(csv_file_path)
    asyncio.create_task(fetch_and_update_data(app.state.client, 'BTCUSDT', '4h', last_timestamp))
    asyncio.create_task(update_predictions())
    
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/data")
async def get_data():
    data = get_chart_data()
    data = json.loads(data.to_json(orient="records"))
    return data

@app.get("/predict")
async def get_prediction():
    data = get_chart_data()
    prediction = predict_price(data)
    timestamp=datetime.now()
    await save_prediction_to_csv(prediction, timestamp)
    prediction['time'] = timestamp.isoformat()
    return JSONResponse(prediction.to_dict())

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)