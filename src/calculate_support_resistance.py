from src.srchannels import SupportResistanceAnalyzer, ChannelAnalyzer
import pandas as pd

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

    return data

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