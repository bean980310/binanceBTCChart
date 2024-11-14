import asyncio
from io import StringIO
from pathlib import Path
import shutil
import pandas as pd
import aiofiles
from datetime import datetime
from binance import AsyncClient

from src.fetch_data import read_existing_csv, fetch_new_klines, calculate_indicators

csv_file_path = Path().parent / 'data' / 'btc_futures_data.csv'

data_cache = {}
cache_lock = asyncio.Lock()

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

def cleanup_backups(backup_dir: Path, max_backups: int = 5):
    backups = sorted(backup_dir.glob(f"{csv_file_path.stem}_backup_*{csv_file_path.suffix}"), reverse=True)
    if len(backups) > max_backups:
        for backup_file in backups[max_backups:]:
            backup_file.unlink()
            print(f"오래된 백업이 삭제되었습니다: {backup_file}")

async def save_dataframe_to_csv(df: pd.DataFrame, file_path: Path):
    if not df.empty:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일이 존재하면 백업 생성
        if file_path.exists():
            backup_dir = file_path.parent / 'backup'
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            backup_file = backup_dir / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            shutil.copy(file_path, backup_file)
            print(f"백업이 생성되었습니다: {backup_file}")
            cleanup_backups(backup_dir, max_backups=5)
        
        # 데이터를 임시 파일에 먼저 저장
        temp_file = file_path.parent / f"{file_path.stem}_temp{file_path.suffix}"
        async with aiofiles.open(temp_file, mode='w') as f:
            await f.write(df.to_csv(index=False))
        
        # 임시 파일을 원본 파일로 교체
        temp_file.replace(file_path)
        print(f"CSV 파일이 업데이트되었습니다. 총 데이터 행 수: {len(df)}")
    else:
        print("저장할 데이터가 없습니다.")

async def fetch_and_update_data(client: AsyncClient, symbol, interval, last_timestamp):
    while True:
        try:
            new_data = await fetch_new_klines(client, symbol, interval, start_time=last_timestamp)

            if new_data is not None and not new_data.empty:
                print(f"새로 가져온 데이터 행 수: {len(new_data)}")

                # 기존 CSV 데이터 읽기
                existing_data = await read_existing_csv(csv_file_path)

                # 숫자 열만 float으로 변환
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                new_data[numeric_columns] = new_data[numeric_columns].astype(float)

                # 기존 데이터와 새로운 데이터 결합
                combined_data = pd.concat([existing_data, new_data])

                # 'Open Time'을 기준으로 중복 제거 (새로운 데이터 우선)
                combined_data.drop_duplicates(subset=['Open Time'], keep='last', inplace=True)

                # 데이터 정렬
                combined_data = combined_data.sort_values(by='Open Time', ascending=True)

                # 지표 계산
                combined_data = calculate_indicators(combined_data)

                # 불필요한 컬럼 제거
                combined_data = combined_data.drop(['Quote Asset Volume', 'Number of Trades',
                                                    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'],
                                                   axis=1, errors='ignore')

                # 'Open Time'과 'Close Time'을 문자열로 변환
                combined_data['Open Time'] = combined_data['Open Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                combined_data['Close Time'] = combined_data['Close Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

                # 컬럼 순서 지정
                column_order = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                                'EMA9', 'EMA60', 'EMA200', 'RSI', 'RSI_SMA',
                                'StochRSI_%K', 'StochRSI_%D', 'MACD', 'MACD_Signal', 'MACD_Hist']
                combined_data = combined_data[column_order]

                # CSV 파일에 저장 (전체 덮어쓰기)
                await save_dataframe_to_csv(combined_data, csv_file_path)
            else:
                print("새로운 데이터가 없습니다.")
        except Exception as e:
            print(f"데이터 페칭 중 예외 발생: {e}")

def get_chart_data():
    data = pd.read_csv(csv_file_path, parse_dates=['Open Time'])
    return data