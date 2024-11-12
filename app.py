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
import uvicorn
from typing import Dict, Any
from datetime import datetime, timedelta
import json
import lightweight_charts as lwc
import requests

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

class SupportResistanceAnalyzer:
    def __init__(self, price_sensitivity: float = 0.002, time_sensitivity: int = 5):
        """
        Parameters:
        price_sensitivity: 가격 변동 민감도 (기본값: 0.2%)
        time_sensitivity: 시간 간격 민감도 (기본값: 5 캔들)
        """
        self.price_sensitivity = price_sensitivity
        self.time_sensitivity = time_sensitivity
        
    def find_key_levels(self, data: pd.DataFrame) -> Dict:
        """주요 지지/저항 레벨을 찾는 메인 함수"""
        # 피크와 트로프 찾기
        peaks = self._find_peaks(data)
        troughs = self._find_troughs(data)
        
        # 주요 레벨 클러스터링
        resistance_levels = self._cluster_levels(peaks, data)
        support_levels = self._cluster_levels(troughs, data)
        
        # 강도 계산
        resistance_strength = self._calculate_level_strength(resistance_levels, data)
        support_strength = self._calculate_level_strength(support_levels, data)
        
        # 동적 트렌드라인 계산
        resistance_lines = self._calculate_dynamic_trendlines(peaks, data, is_resistance=True)
        support_lines = self._calculate_dynamic_trendlines(troughs, data, is_resistance=False)
        
        return {
            'resistance_levels': resistance_strength,
            'support_levels': support_strength,
            'resistance_lines': resistance_lines,
            'support_lines': support_lines
        }
    
    def _find_peaks(self, data: pd.DataFrame, order: int = 5) -> np.ndarray:
        """
        개선된 피크 탐지 알고리즘
        프랙탈 이론과 가격 변동성을 고려하여 피크 포인트 탐지
        """
        volatility = self._calculate_volatility(data)
        threshold = volatility * self.price_sensitivity
        
        # 로컬 최대값 찾기
        peaks = argrelextrema(data['High'].values, np.greater, order=order)[0]
        
        # 노이즈 필터링
        filtered_peaks = []
        for peak in peaks:
            if peak < order or peak > len(data) - order:
                continue
                
            window = data['High'].values[peak-order:peak+order+1]
            if (data['High'].values[peak] - window.mean()) > threshold:
                filtered_peaks.append(peak)
                
        return np.array(filtered_peaks)
    
    def _find_troughs(self, data: pd.DataFrame, order: int = 5) -> np.ndarray:
        """개선된 트로프 탐지 알고리즘"""
        volatility = self._calculate_volatility(data)
        threshold = volatility * self.price_sensitivity
        
        troughs = argrelextrema(data['Low'].values, np.less, order=order)[0]
        
        filtered_troughs = []
        for trough in troughs:
            if trough < order or trough > len(data) - order:
                continue
                
            window = data['Low'].values[trough-order:trough+order+1]
            if (window.mean() - data['Low'].values[trough]) > threshold:
                filtered_troughs.append(trough)
                
        return np.array(filtered_troughs)
    
    def _cluster_levels(self, points: np.ndarray, data: pd.DataFrame) -> List[float]:
        """
        DBSCAN 클러스터링을 사용하여 유사한 가격 레벨 그룹화
        """
        if len(points) < 2:
            return []
            
        prices = data['High'].values[points] if len(points) > 0 else np.array([])
        times = points.reshape(-1, 1)
        
        # 가격 정규화
        price_range = data['High'].max() - data['Low'].min()
        eps = price_range * self.price_sensitivity
        
        clustering = DBSCAN(eps=eps, min_samples=2).fit(prices.reshape(-1, 1))
        
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # 노이즈 제외
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(prices[idx])
            
        return [np.mean(cluster) for cluster in clusters.values()]
    
    def _calculate_level_strength(self, levels: List[float], data: pd.DataFrame) -> List[Dict]:
        """
        각 레벨의 강도 계산
        접촉 횟수, 반등/거부 강도, 최근성을 고려
        """
        strength_levels = []
        for level in levels:
            touches = self._count_touches(level, data)
            bounce_strength = self._calculate_bounce_strength(level, data)
            recency = self._calculate_recency_score(level, data)
            
            total_strength = (touches * 0.4 + bounce_strength * 0.4 + recency * 0.2)
            
            strength_levels.append({
                'price': level,
                'strength': total_strength,
                'touches': touches,
                'bounce_strength': bounce_strength,
                'recency': recency
            })
            
        return sorted(strength_levels, key=lambda x: x['strength'], reverse=True)
    
    def _calculate_dynamic_trendlines(self, points: np.ndarray, data: pd.DataFrame, 
                                    is_resistance: bool) -> List[Dict]:
        """
        동적 트렌드라인 계산
        여러 기간의 트렌드를 고려하여 다중 트렌드라인 생성
        """
        if len(points) < 2:
            return []
            
        trendlines = []
        price_col = 'High' if is_resistance else 'Low'
        
        # 다양한 기간의 트렌드라인 계산
        for period in [10, 20, 50]:  # 단기, 중기, 장기
            if len(points) < period:
                continue
                
            recent_points = points[-period:]
            prices = data[price_col].values[recent_points]
            times = recent_points.astype(float)
            
            # 선형 회귀로 트렌드라인 계산
            if len(times) > 1:
                coeffs = np.polyfit(times, prices, deg=1)
                slope = coeffs[0]
                intercept = coeffs[1]
                
                # 트렌드 강도 계산
                strength = self._calculate_trendline_strength(
                    slope, times, prices, data[price_col].values)
                
                # 시작점과 끝점 계산
                start_time = data.index[int(times[0])].timestamp()
                end_time = data.index[int(times[-1])].timestamp()
                start_price = slope * times[0] + intercept
                end_price = slope * times[-1] + intercept
                
                trendlines.append({
                    'start': {'time': start_time, 'value': float(start_price)},
                    'end': {'time': end_time, 'value': float(end_price)},
                    'strength': strength,
                    'color': 'red' if is_resistance else 'green',
                    'lineWidth': 1 + strength
                })
        
        return sorted(trendlines, key=lambda x: x['strength'], reverse=True)
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """변동성 계산 - ATR 사용"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return ranges.max(axis=1).mean()
    
    def _count_touches(self, level: float, data: pd.DataFrame, threshold: float = 0.001) -> int:
        """특정 레벨에 대한 접촉 횟수 계산"""
        upper_band = level * (1 + threshold)
        lower_band = level * (1 - threshold)
        touches = ((data['High'] >= lower_band) & (data['High'] <= upper_band) |
                  (data['Low'] >= lower_band) & (data['Low'] <= upper_band)).sum()
        return touches
    
    def _calculate_bounce_strength(self, level: float, data: pd.DataFrame) -> float:
        """가격 반등/거부 강도 계산"""
        threshold = level * self.price_sensitivity
        bounces = []
        
        for i in range(1, len(data)):
            if abs(data['High'].iloc[i] - level) < threshold:
                price_change = abs(data['Close'].iloc[i] - data['Close'].iloc[i-1])
                bounces.append(price_change)
                
        return np.mean(bounces) if bounces else 0
    
    def _calculate_recency_score(self, level: float, data: pd.DataFrame) -> float:
        """최근성 점수 계산 - 최근 터치에 더 높은 가중치 부여"""
        threshold = level * self.price_sensitivity
        touches = []
        
        for i in range(len(data)):
            if abs(data['High'].iloc[i] - level) < threshold:
                touches.append(i)
                
        if not touches:
            return 0
            
        latest_touch = max(touches)
        return 1 - (latest_touch / len(data))
    
    def _calculate_trendline_strength(self, slope: float, times: np.ndarray, 
                                    prices: np.ndarray, all_prices: np.ndarray) -> float:
        """트렌드라인 강도 계산"""
        # 기울기의 절대값
        slope_strength = min(abs(slope) * 100, 1.0)
        
        # 가격 접촉점들의 R² 값
        predicted = slope * times + np.polyfit(times, prices, deg=1)[1]
        r2 = 1 - np.sum((prices - predicted) ** 2) / np.sum((prices - prices.mean()) ** 2)
        
        # 가격 변동성 대비 트렌드 강도
        price_range = np.ptp(all_prices)
        volatility_ratio = min(abs(slope) / (price_range / len(times)) * 10, 1.0)
        
        return (slope_strength * 0.3 + r2 * 0.4 + volatility_ratio * 0.3)

class ChannelAnalyzer:
    def __init__(self, price_sensitivity: float = 0.002, min_channel_period: int = 20):
        self.price_sensitivity = price_sensitivity
        self.min_channel_period = min_channel_period
        
    def find_channels(self, data: pd.DataFrame) -> List[Dict]:
        """주요 가격 채널을 찾는 함수"""
        channels = []
        
        # 고가와 저가 데이터
        highs = data['High'].values
        lows = data['Low'].values
        
        # 채널 후보군 찾기
        for start_idx in range(len(data) - self.min_channel_period):
            end_idx = len(data)
            
            # 상단 채널 라인 계산
            resistance_points = []
            support_points = []
            
            # 구간 내 고점들 수집
            for i in range(start_idx, end_idx):
                if i > 0 and i < len(highs) - 1:
                    if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                        resistance_points.append((i, highs[i]))
                        
                if i > 0 and i < len(lows) - 1:
                    if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                        support_points.append((i, lows[i]))
            
            if len(resistance_points) >= 2 and len(support_points) >= 2:
                channel = self._validate_channel(
                    data, resistance_points, support_points, start_idx, end_idx)
                if channel:
                    channels.append(channel)
        
        # 채널 필터링 및 병합
        filtered_channels = self._merge_overlapping_channels(channels)
        
        return filtered_channels
    
    def _validate_channel(self, data: pd.DataFrame, 
                         resistance_points: List[Tuple], 
                         support_points: List[Tuple],
                         start_idx: int, end_idx: int) -> Dict:
        """채널의 유효성을 검증하고 채널 정보를 반환"""
        
        # 상단선 회귀분석
        resistance_x = np.array([p[0] for p in resistance_points])
        resistance_y = np.array([p[1] for p in resistance_points])
        resistance_coef = np.polyfit(resistance_x, resistance_y, 1)
        
        # 하단선 회귀분석
        support_x = np.array([p[0] for p in support_points])
        support_y = np.array([p[1] for p in support_points])
        support_coef = np.polyfit(support_x, support_y, 1)
        
        # 채널 기울기 유사성 검증
        slope_diff = abs(resistance_coef[0] - support_coef[0])
        if slope_diff > self.price_sensitivity:
            return None
            
        # 채널 평행성 검증
        channel_height = abs(resistance_coef[1] - support_coef[1])
        avg_price = data['Close'].mean()
        if channel_height / avg_price > 0.1:  # 채널 높이가 너무 크면 제외
            return None
            
        # 가격 포함률 검증
        prices_within_channel = 0
        for i in range(start_idx, end_idx):
            resistance_line = resistance_coef[0] * i + resistance_coef[1]
            support_line = support_coef[0] * i + support_coef[1]
            
            if support_line <= data['High'].iloc[i] <= resistance_line:
                prices_within_channel += 1
                
        containment_ratio = prices_within_channel / (end_idx - start_idx)
        if containment_ratio < 0.7:  # 70% 이상의 가격이 채널 내에 있어야 함
            return None
            
        return {
            'start_time': data.index[start_idx].timestamp(),
            'end_time': data.index[end_idx-1].timestamp(),
            'resistance_coef': resistance_coef.tolist(),
            'support_coef': support_coef.tolist(),
            'strength': containment_ratio,
            'slope': float(resistance_coef[0])
        }
    
    def _merge_overlapping_channels(self, channels: List[Dict]) -> List[Dict]:
        """겹치는 채널들을 병합"""
        if not channels:
            return []
            
        # 채널을 시작 시간 기준으로 정렬
        channels.sort(key=lambda x: x['start_time'])
        
        merged = []
        current = channels[0]
        
        for next_channel in channels[1:]:
            # 채널 겹침 확인
            if current['end_time'] >= next_channel['start_time']:
                # 더 강한 채널 선택
                if next_channel['strength'] > current['strength']:
                    current = next_channel
            else:
                merged.append(current)
                current = next_channel
                
        merged.append(current)
        
        return merged

    def get_channel_lines(self, channel: Dict, data: pd.DataFrame) -> Dict:
        """채널의 상단선과 하단선 좌표 생성"""
        resistance_points = []
        support_points = []
        
        # 채널 기간 동안의 시간 포인트들
        times = np.arange(
            data.index.get_indexer([pd.Timestamp(channel['start_time']*1e9)])[0],
            data.index.get_indexer([pd.Timestamp(channel['end_time']*1e9)])[0] + 1
        )
        
        for t in times:
            resistance_y = channel['resistance_coef'][0] * t + channel['resistance_coef'][1]
            support_y = channel['support_coef'][0] * t + channel['support_coef'][1]
            
            timestamp = data.index[t].timestamp()
            
            resistance_points.append({
                'time': timestamp,
                'value': float(resistance_y)
            })
            
            support_points.append({
                'time': timestamp,
                'value': float(support_y)
            })
            
        return {
            'resistance': resistance_points,
            'support': support_points
        }

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

async def get_historical_klines(client, symbol: str, interval: str, start_date: str = None, end_date: str = None, limit: int = 1000) -> pd.DataFrame:
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
        data = await get_data_and_indicators(app.state.client)
        prediction = predict_price(data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await save_prediction_to_csv(prediction, timestamp)
        print(f"Prediction updated at {timestamp}")
        await asyncio.sleep(60)

async def get_data_and_indicators(client: AsyncClient) -> pd.DataFrame:
    data = await get_historical_klines(client, "BTCUSDT", AsyncClient.KLINE_INTERVAL_4HOUR)
    calculate_indicators(data)
    calculate_support_resistance_levels(data)
    return data

@app.on_event("startup")
async def startup_event():
    app.state.client = await initialize_client()
    asyncio.create_task(update_predictions())

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/data")
async def get_data():
    data = await get_data_and_indicators(app.state.client)
    trendlines = calculate_trendlines(data)

    data['time'] = data.index.astype(int) // 10**9
    data = data.reset_index()
    response_data={
        "data": json.loads(data.to_json(orient='records', date_format='iso')),
        "trendlines": trendlines,
    }
    return JSONResponse(response_data)

@app.get("/predict")
async def get_prediction():
    data = await get_data_and_indicators(app.state.client)
    prediction = predict_price(data)
    timestamp=datetime.now()
    await save_prediction_to_csv(prediction, timestamp)
    prediction['time'] = timestamp.isoformat()
    return JSONResponse(prediction.to_dict())

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)