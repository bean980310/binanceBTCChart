from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import aiohttp
from binance import AsyncClient
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
from getpass import getpass
import ta
import os
import asyncio
import uvicorn
from typing import Dict, Any
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

api_key_file = Path.home() / '.binance' / 'api_key.txt'
api_secret_file = Path.home() / '.binance' / 'api_secret.txt'
predictions_file = Path().parent / 'data' / 'predictions.csv'

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

def calculate_indicators(data: pd.DataFrame) -> None:
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

def calculate_support_resistance_levels(data: pd.DataFrame) -> None:
    lookback_period = 60
    highs = data['High'].rolling(window=lookback_period).max().dropna().unique()
    lows = data['Low'].rolling(window=lookback_period).min().dropna().unique()
    levels = ["Level1", "Level2", "Level3"]
    
    for i, level in enumerate(levels):
        data[f'Resistance_1st_{level}'] = highs[-(i + 1)] if len(highs) > i else None
        data[f'Support_1st_{level}'] = lows[i] if len(lows) > i else None
        if len(highs) > i + 3:
            data[f'Resistance_2nd_{level}'] = highs[-(i + 4)]
        if len(lows) > i + 3:
            data[f'Support_2nd_{level}'] = lows[i + 3]

async def get_historical_klines(client: AsyncClient, symbol: str, interval: str, lookback: str) -> pd.DataFrame:
    klines = await client.get_historical_klines(symbol, interval, lookback)
    data = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
    data.set_index('Open Time', inplace=True)
    data = data.astype(float)
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

async def save_prediction_to_csv(prediction: pd.Series, timestamp: pd.Timestamp) -> None:
    if not predictions_file.exists():
        predictions_file.write_text('timestamp,actual_high,actual_low,actual_close,predicted_high,predicted_low,predicted_close\n')

    async with aiohttp.ClientSession() as session:
        async with session.get(predictions_file.as_uri()) as response:
            if response.status == 200:
                content = f"{timestamp},{prediction['High']},{prediction['Low']},{prediction['Close']},"
                content += f"{prediction['Predicted_High']},{prediction['Predicted_Low']},{prediction['Predicted_Close']}\n"
                
                async with aiohttp.ClientSession() as write_session:
                    async with write_session.post(predictions_file.as_uri(), data=content) as write_response:
                        if write_response.status != 200:
                            print(f"Error saving prediction: {write_response.status}")

async def update_and_save_prediction(background_tasks: BackgroundTasks) -> None:
    client = await initialize_client()
    while True:
        data = await get_data_and_indicators(client)
        prediction = predict_price(data)
        timestamp = data.index[-1]
        await save_prediction_to_csv(prediction, timestamp)
        await asyncio.sleep(60)  # 1분 대기

async def get_data_and_indicators(client: AsyncClient) -> pd.DataFrame:
    data = await get_historical_klines(client, 'BTCUSDT', AsyncClient.KLINE_INTERVAL_4HOUR, '180 days ago UTC+9:00')
    calculate_indicators(data)
    calculate_support_resistance_levels(data)
    return data

@app.on_event("startup")
async def startup_event():
    app.state.client = await initialize_client()
    background_tasks = BackgroundTasks()
    background_tasks.add_task(update_and_save_prediction, background_tasks)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/data")
async def get_data():
    data = await get_data_and_indicators(app.state.client)
    return JSONResponse(json.loads(data.reset_index().to_json(orient='records', date_format='iso')))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)