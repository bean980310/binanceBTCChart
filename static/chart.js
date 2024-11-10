// 백엔드에서 데이터 가져오기
fetch('/data')
.then(response => response.json())
.then(data => {
    // 데이터 처리 및 차트 렌더링
    const priceChart = LightweightCharts.createChart(document.getElementById('price-chart'), {
        width: 800,
        height: 600,
        layout: {
        backgroundColor: '#ffffff',
        textColor: '#000',
        },
        rightPriceScale: {
            scaleMargins: {
                top: 0.2,
                bottom: 0.2,
            },
        },
        timeScale: {
            borderColor: '#D1D4DC',
        },
    });

    // 볼륨 차트 생성
    const volumeChart = LightweightCharts.createChart(document.getElementById('volume-chart'), {
        width: 800,
        height: 150,
        layout: {
            backgroundColor: '#ffffff',
            textColor: '#000',
        },
        rightPriceScale: {
            scaleMargins: {
                top: 0.05,
                bottom: 0.05,
            },
        },
        timeScale: {
            visible: false, // 볼륨 차트의 시간축을 숨김
        },
    });

    // RSI 차트 생성
    const rsiChart = LightweightCharts.createChart(document.getElementById('rsi-chart'), {
        width: 800,
        height: 150,
        layout: {
            backgroundColor: '#ffffff',
            textColor: '#000',
        },
        rightPriceScale: {
            scaleMargins: {
                top: 0.2,
                bottom: 0.2,
            },
        },
        timeScale: {
            borderColor: '#D1D4DC',
        },
    });

    // MACD 차트 생성
    const macdChart = LightweightCharts.createChart(document.getElementById('macd-chart'), {
        width: 800,
        height: 150,
        layout: {
            backgroundColor: '#ffffff',
            textColor: '#000',
        },
        rightPriceScale: {
            scaleMargins: {
                top: 0.2,
                bottom: 0.2,
            },
        },
        timeScale: {
            borderColor: '#D1D4DC',
        },
    });


    const candlestickSeries = priceChart.addCandlestickSeries();

    const volumeSeries = volumeChart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
            top: 0,
            bottom: 0,
        },
    });

    // 시간과 가격 데이터를 lightweight-charts 형식에 맞게 변환
    const chartData = data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        open: item['Open'],
        high: item['High'],
        low: item['Low'],
        close: item['Close'],
    }));

    candlestickSeries.setData(chartData);

    // 기술 지표 추가 (예: EMA, RSI 등)
    
    // 볼륨 데이터 설정
    const volumeData = data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['Volume'],
        color: item['Close'] > item['Open'] ? '#26a69a' : '#ef5350', // 양봉: 초록색, 음봉: 빨간색
    }));

    volumeSeries.setData(volumeData);

    // 예: EMA 라인 추가
    const emaLineSeries = priceChart.addLineSeries({
        color: 'blue',
        lineWidth: 2,
    });

    const emaData = data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['EMA9'], // EMA9 열 사용
    })).filter(item => item.value !== null); // 결측치 제거

    emaLineSeries.setData(emaData);

    // 추가적인 지표도 동일한 방식으로 추가 가능합니다.

    const emaLineSeries60 = priceChart.addLineSeries({
        color: 'purple',
        lineWidth: 2,
    });

    const emaData60 = data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['EMA60'], // EMA9 열 사용
    })).filter(item => item.value !== null); // 결측치 제거

    emaLineSeries60.setData(emaData60);

    const emaLineSeries200 = priceChart.addLineSeries({
        color: 'cyan',
        lineWidth: 2,
    });

    const emaData200 = data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['EMA200'], // EMA9 열 사용
    })).filter(item => item.value !== null); // 결측치 제거

    emaLineSeries200.setData(emaData200);

    // RSI 시리즈 추가
    const rsiSeries = rsiChart.addLineSeries({
        color: 'purple',
        lineWidth: 1,
    });

    const rsiData = data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['RSI'],
    })).filter(item => item.value !== null);

    rsiSeries.setData(rsiData);

    // RSI 이동평균선(SMA) 시리즈 추가
    const rsiSmaSeries = rsiChart.addLineSeries({
        color: 'orange',
        lineWidth: 1,
    });

    // RSI 이동평균선 데이터 설정
    const rsiSmaData = data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['RSI_SMA'],
    })).filter(item => item.value !== null);

    rsiSmaSeries.setData(rsiSmaData);

    const macdLineSeries = macdChart.addLineSeries({
        color: 'blue',
        lineWidth: 1,
    });

    const macdLineData=data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['MACD'],
    })).filter(item => item.value !== null);

    macdLineSeries.setData(macdLineData)

    const macdSignalSeries = macdChart.addLineSeries({
        color: 'red',
        lineWidth: 1,
    });

    const macdSignalData=data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['MACD_Signal'],
    })).filter(item => item.value !== null);
    
    macdSignalSeries.setData(macdSignalData)

    const macdHistSeries = macdChart.addHistogramSeries({
        priceFormat: {
            type: 'price',
            precision: 5,
            minMove: 0.00001,
        },
        priceScaleId: '',
        scaleMargins: {
            top: 0.2,
            bottom: 0,
        },
    });

    const macdHistData = data.map(item => {
        const time = new Date(item['Open Time']).getTime() / 1000;
        const value = item['MACD_Hist'];
    
        let color = value >= 0 ? '#26a69a' : '#ef5350';
    
        return {
            time: time,
            value: value,
            color: color,
        };
    }).filter(item => item.value !== null);
    

    macdHistSeries.setData(macdHistData)

    function updateData() {
        fetch('/data')
            .then(response => response.json())
            .then(newData => {
                // 새로운 데이터로 시리즈 업데이트
                const updatedChartData = newData.map(item => ({
                    time: new Date(item['Open Time']).getTime() / 1000,
                    open: item['Open'],
                    high: item['High'],
                    low: item['Low'],
                    close: item['Close'],
                }));
                candlestickSeries.setData(updatedChartData);

                // 볼륨 데이터 업데이트
                const updatedVolumeData = data.map(item => ({
                    time: new Date(item['Open Time']).getTime() / 1000,
                    value: item['Volume'],
                    color: item['Close'] > item['Open'] ? '#26a69a' : '#ef5350', // 양봉: 초록색, 음봉: 빨간색
                }));

                volumeSeries.setData(updatedVolumeData);

                // 지표도 동일하게 업데이트
                const updatedEmaData = data.map(item => ({
                    time: new Date(item['Open Time']).getTime() / 1000,
                    value: item['EMA9'], // EMA9 열 사용
                })).filter(item => item.value !== null); // 결측치 제거
                emaLineSeries.setData(updatedEmaData)

                const updatedeEmaData60 = data.map(item => ({
                time: new Date(item['Open Time']).getTime() / 1000,
                value: item['EMA60'], // EMA9 열 사용
                })).filter(item => item.value !== null); // 결측치 제거

                emaLineSeries60.setData(updatedeEmaData60);

                const updatedeEmaData200 = data.map(item => ({
                time: new Date(item['Open Time']).getTime() / 1000,
                value: item['EMA200'], // EMA9 열 사용
                })).filter(item => item.value !== null); // 결측치 제거

                emaLineSeries200.setData(updatedeEmaData200);

                const updatedRsiData = data.map(item => ({
                    time: new Date(item['Open Time']).getTime() / 1000,
                    value: item['RSI'],
                })).filter(item => item.value !== null);

                rsiSeries.setData(updatedRsiData);

                // RSI 이동평균선 데이터 설정
                const updatedRsiSmaData = data.map(item => ({
                    time: new Date(item['Open Time']).getTime() / 1000,
                    value: item['RSI_SMA'],
                })).filter(item => item.value !== null && !isNaN(item.value));

                rsiSmaSeries.setData(updatedRsiSmaData);

                const updatedMacdLineData=data.map(item => ({
                    time: new Date(item['Open Time']).getTime() / 1000,
                    value: item['MACD'],
                })).filter(item => item.value !== null);
            
                macdLineSeries.setData(updatedMacdLineData)

                const updatedMacdSignalData=data.map(item => ({
                    time: new Date(item['Open Time']).getTime() / 1000,
                    value: item['MACD_Signal'],
                })).filter(item => item.value !== null);
                
                macdSignalSeries.setData(updatedMacdSignalData)

                const updatedMacdHistData=data.map(item => {
                    const time = new Date(item['Open Time']).getTime() / 1000;
                    const value = item['MACD_Hist'];
                
                    let color = value >= 0 ? '#26a69a' : '#ef5350';
                
                    return {
                        time: time,
                        value: value,
                        color: color,
                    };
                }).filter(item => item.value !== null);
            
                macdHistSeries.setData(updatedMacdHistData)
            
            });
        }

    // 1분마다 데이터 업데이트
    setInterval(updateData, 100);

    // 시간축 동기화
    const bindScroll = function (chart) {
        return function (e) {
            if (preventScrollEvent) {
                preventScrollEvent = false;
                return;
            }
            preventScrollEvent = true;
            otherChart.timeScale().scrollToPosition(chart.timeScale().scrollPosition(), false);
        };
    };

    const bindVisibleLogicalRangeChange = function (chart) {
        return function (e) {
            otherChart.timeScale().setVisibleLogicalRange(chart.timeScale().getVisibleLogicalRange());
        };
    };

    priceChart.timeScale().subscribeVisibleTimeRangeChange((newVisibleTimeRange) => {
        volumeChart.timeScale().setVisibleRange(newVisibleTimeRange);
    });
});