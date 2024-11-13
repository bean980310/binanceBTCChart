import asyncio
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
import pandas as pd
from pathlib import Path
import aiofiles
from datetime import datetime
from getpass import getpass
from io import StringIO

# API 키 파일 경로 설정
api_key_file = Path.home() / '.binance' / 'api_key.txt'
api_secret_file = Path.home() / '.binance' / 'api_secret.txt'

# API 키 관련 함수들
def get_api(file_path, prompt_func):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        return prompt_func()

def set_api(file_path, api):
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    with open(file_path, 'w') as f:
        f.write(api)

def input_api_key():
    api_key = input("Enter your Binance API key: ")
    set_api(api_key_file, api_key)
    return api_key

def input_api_secret():
    api_secret = getpass("Enter your Binance API secret: ")
    set_api(api_secret_file, api_secret)
    return api_secret

# Binance API 키 설정
API_KEY = get_api(api_key_file, input_api_key)
API_SECRET = get_api(api_secret_file, input_api_secret)

# 데이터 저장 경로 설정
csv_file_path = Path().parent / 'data' / 'btc_futures_data.csv'
csv_file_path.parent.mkdir(parents=True, exist_ok=True)

# CSV 파일에서 가장 최근 'Open Time' 타임스탬프를 반환하는 함수
async def get_last_timestamp(file_path: Path) -> int:
    """CSV 파일에서 가장 최근 'Open Time' 타임스탬프를 반환하는 함수"""
    if not file_path.exists():
        return None

    # aiofiles를 사용하여 비동기적으로 파일 읽기
    async with aiofiles.open(file_path, mode='r') as f:
        contents = await f.read()

    # Pandas로 DataFrame 생성
    df = pd.read_csv(StringIO(contents))

    if df.empty:
        return None

    # 'Open Time'을 datetime으로 변환
    df['Open Time'] = pd.to_datetime(df['Open Time'], format='%Y-%m-%d %H:%M:%S')
    last_open_time = df['Open Time'].max()
    last_timestamp = int(last_open_time.timestamp() * 1000)
    print(f"Last Open Time in CSV: {last_open_time}")
    print(f"Last timestamp (ms): {last_timestamp}")
    return last_timestamp

# 기존 CSV 파일을 읽어 DataFrame 반환
async def read_existing_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    async with aiofiles.open(file_path, mode='r') as f:
        contents = await f.read()

    df = pd.read_csv(StringIO(contents))
    if not df.empty:
        df['Open Time'] = pd.to_datetime(df['Open Time'], format='%Y-%m-%d %H:%M:%S')
        df['Close Time'] = pd.to_datetime(df['Close Time'], format='%Y-%m-%d %H:%M:%S')
    return df

# DataFrame을 CSV 파일으로 저장 (덮어쓰기)
async def save_dataframe_to_csv(df: pd.DataFrame, file_path: Path):
    async with aiofiles.open(file_path, mode='w') as f:
        await f.write(df.to_csv(index=False))
    print(f"CSV 파일이 업데이트되었습니다. 총 데이터 행 수: {len(df)}")

# 새로운 Kline 데이터를 가져와 DataFrame으로 반환하는 함수
async def fetch_new_klines(client, symbol: str, interval: str, start_time: int = None) -> pd.DataFrame:
    """새로운 Kline 데이터를 가져와 DataFrame으로 반환하는 함수"""
    limit = 1000  # 한 번에 가져올 수 있는 최대 데이터 수
    all_fetched = 0
    new_data = pd.DataFrame()

    while True:
        try:
            klines = await client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                startTime=start_time
            )

            if not klines:
                print("더 이상 새로운 데이터가 없습니다.")
                break

            # DataFrame으로 변환
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            # 'Open Time'을 datetime으로 변환 후 ISO 형식 문자열로 변경
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
            # 'Close Time'을 ISO 형식 문자열로 변환
            df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')

            # 숫자 열만 float으로 변환
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume',
                               'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
            df[numeric_columns] = df[numeric_columns].astype(float)

            # 데이터 정렬: 오름차순 (과거 -> 현재)
            df = df.sort_values(by='Open Time', ascending=True)

            # 새로 가져온 데이터를 누적
            new_data = pd.concat([new_data, df])

            all_fetched += len(df)
            print(f"Fetched {len(df)} klines. Total fetched: {all_fetched}")

            # 다음 요청을 위한 시작 시간 설정 (마지막 데이터의 Open Time + 1 ms)
            last_open_time = pd.to_datetime(df['Open Time']).max()
            start_time = int(last_open_time.timestamp() * 1000) + 1
            print(f"Next start_time set to: {datetime.fromtimestamp(start_time / 1000)}")

            # API 호출 제한을 피하기 위해 잠시 대기
            await asyncio.sleep(0.1)

        except BinanceAPIException as e:
            print(f"Binance API 예외 발생: {e}")
            await asyncio.sleep(1)  # 잠시 대기 후 재시도
        except Exception as e:
            print(f"예외 발생: {e}")
            break

    return new_data

async def main():
    try:
        # Binance AsyncClient를 async with로 관리하여 자동으로 종료되도록 함
        client = await AsyncClient.create(API_KEY, API_SECRET)
        try:
            symbol = "BTCUSDT"  # USD-M 선물 심볼
            interval = "4h"     # 4시간 간격

            # CSV 파일에서 마지막 타임스탬프 가져오기
            last_timestamp = await get_last_timestamp(csv_file_path)

            if last_timestamp:
                print(f"가장 최근 타임스탬프: {datetime.fromtimestamp(last_timestamp / 1000)}")
            else:
                # 시작 날짜 설정 (예: 2017-08-17부터 시작)
                start_date = "2017-08-17"
                last_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                print(f"CSV 파일이 없거나 비어있어 시작 날짜: {datetime.fromtimestamp(last_timestamp / 1000)}")

            # 새로운 데이터를 가져와서 DataFrame으로 반환
            new_data = await fetch_new_klines(client, symbol, interval, start_time=last_timestamp)

            if new_data is not None and not new_data.empty:
                print(f"새로 가져온 데이터 행 수: {len(new_data)}")

                # 기존 CSV 데이터 읽기
                existing_data = await read_existing_csv(csv_file_path)

                if not existing_data.empty:
                    # 기존 데이터와 새 데이터를 병합
                    combined_data = pd.concat([existing_data, new_data])

                    # 'Close Time'을 다시 문자열로 유지
                    combined_data['Close Time'] = combined_data['Close Time'].astype(str)

                    # 'Open Time'을 datetime으로 변환
                    combined_data['Open Time'] = pd.to_datetime(combined_data['Open Time'], format='%Y-%m-%d %H:%M:%S')

                    # 'Open Time'을 기준으로 중복 제거 (keep last)
                    combined_data.drop_duplicates(subset=['Open Time'], keep='last', inplace=True)

                    # 'Open Time'을 다시 ISO 형식 문자열로 변환
                    combined_data['Open Time'] = combined_data['Open Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    combined_data = combined_data.sort_values(by='Open Time', ascending=True)

                    # CSV 파일에 저장
                    await save_dataframe_to_csv(combined_data, csv_file_path)
                else:
                    # 기존 데이터가 없으면 새 데이터로 CSV 파일 생성
                    combined_data = new_data.copy()
                    combined_data['Open Time'] = pd.to_datetime(combined_data['Open Time'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')
                    combined_data['Close Time'] = combined_data['Close Time'].astype(str)
                    await save_dataframe_to_csv(combined_data, csv_file_path)
            else:
                print("새로운 데이터가 없습니다.")
        finally:
            # 클라이언트 종료
            await client.close_connection()

    except Exception as e:
        print(f"메인 함수에서 예외 발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())