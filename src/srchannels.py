import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict
from typing import Dict

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
                start_time = data['Open Time'].iloc[int(times[0])].timestamp()
                end_time = data['Open Time'].iloc[int(times[-1])].timestamp()
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
            
            timestamp = data['Open Time'].iloc[t].timestamp()
            
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