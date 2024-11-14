from getpass import getpass
from pathlib import Path
from binance import AsyncClient

api_key_file = Path.home() / '.binance' / 'api_key.txt'
api_secret_file = Path.home() / '.binance' / 'api_secret.txt'

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