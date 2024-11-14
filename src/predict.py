import asyncio
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from pathlib import Path
import aiofiles

from src.update_data import get_chart_data

predictions_file = Path().parent / 'data' / 'predictions.csv'

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