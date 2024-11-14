from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pathlib import Path
import asyncio
import uvicorn
from datetime import datetime
import json

from src.fetch_data import get_last_timestamp
from src.api import initialize_client
from src.update_data import get_chart_data, load_csv_data, fetch_and_update_data
from src.predict import update_predictions, save_prediction_to_csv, predict_price

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

predictions_file = Path().parent / 'data' / 'predictions.csv'
csv_file_path = Path().parent / 'data' / 'btc_futures_data.csv'
predictions_file.parent.mkdir(parents=True, exist_ok=True)

data_cache = {}
cache_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    app.state.client = await initialize_client()
    last_timestamp = await get_last_timestamp(csv_file_path)
    load_csv_data()
    asyncio.create_task(fetch_and_update_data(app.state.client, 'BTCUSDT', '4h', last_timestamp))
    asyncio.create_task(update_predictions())
    
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/data")
async def get_data():
    data = get_chart_data()
    data['time'] = data['Open Time'].apply(lambda x: int(x.timestamp()))
    chart_data = data[['time', 'Open', 'High', 'Low', 'Close']].to_dict(orient='records')
    return JSONResponse(chart_data)

@app.get("/ema_data")
async def get_ema_data():
    data = get_chart_data()
    data['time'] = data['Open Time'].apply(lambda x: int(x.timestamp()))
    ema_data = data[['time', 'EMA9', 'EMA60', 'EMA200']].to_dict(orient='records')
    return JSONResponse(ema_data)

@app.get("/volume_data")
async def get_volume_data():
    data = get_chart_data()
    data['time'] = data['Open Time'].apply(lambda x: int(x.timestamp()))
    volume_data = data[['time', 'Volume']].to_dict(orient='records')
    return JSONResponse(volume_data)

@app.get("/rsi_data")
async def get_rsi_data():
    data = get_chart_data()
    data['time'] = data['Open Time'].apply(lambda x: int(x.timestamp()))
    rsi_data = data[['time', 'RSI', 'RSI_SMA']].to_dict(orient='records')
    return JSONResponse(rsi_data)

@app.get("macd_data")
async def get_macd_data():
    data = get_chart_data()
    data['time'] = data['Open Time'].apply(lambda x: int(x.timestamp()))
    macd_data = data[['time', 'MACD', 'MACD_Signal', 'MACD_Hist']].to_dict(orient='records')
    return JSONResponse(macd_data)

@app.get("stoch_rsi")
async def get_stoch_rsi_data():
    data = get_chart_data()
    data['time'] = data['Open Time'].apply(lambda x: int(x.timestamp()))
    stoch_rsi_data = data[['time', 'StochRSI_%K', 'StochRSI_%D']].to_dict(orient='records')
    return JSONResponse(stoch_rsi_data)

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