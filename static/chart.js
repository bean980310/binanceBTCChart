// 백엔드에서 데이터 가져오기
fetch('/data')
.then(response => response.json())
.then(data => {
    const chartOptions = {
        width: 800,
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
    };
    const priceChart = LightweightCharts.createChart(document.getElementById('price-chart'), { ...chartOptions, height: 600 });
    const volumeChart = LightweightCharts.createChart(document.getElementById('volume-chart'), { ...chartOptions, height: 150 });
    const rsiChart = LightweightCharts.createChart(document.getElementById('rsi-chart'), { ...chartOptions, height: 150 });
    const macdChart = LightweightCharts.createChart(document.getElementById('macd-chart'), { ...chartOptions, height: 150 });

    // 시간축 숨김 설정
    volumeChart.timeScale().options().visible = false;
    rsiChart.timeScale().options().visible = false;
    macdChart.timeScale().options().visible = false;

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

     // OHLC 정보를 priceLine으로 추가하는 함수
     function updateOhlcInfo(param) {
        if (param && param.seriesData) {
            const price = param.seriesData.get(candlestickSeries);
            if (price) {
                priceChart.removePriceLine(priceLineOpen);
                priceChart.removePriceLine(priceLineHigh);
                priceChart.removePriceLine(priceLineLow);
                priceChart.removePriceLine(priceLineClose);

                // 시가, 고가, 저가, 종가 표시를 위한 priceLine 설정
                priceLineOpen = priceChart.createPriceLine({
                    price: price.open,
                    color: 'blue',
                    lineWidth: 1,
                    title: `O: ${price.open.toFixed(2)}`,
                });
                priceLineHigh = priceChart.createPriceLine({
                    price: price.high,
                    color: 'green',
                    lineWidth: 1,
                    title: `H: ${price.high.toFixed(2)}`,
                });
                priceLineLow = priceChart.createPriceLine({
                    price: price.low,
                    color: 'red',
                    lineWidth: 1,
                    title: `L: ${price.low.toFixed(2)}`,
                });
                priceLineClose = priceChart.createPriceLine({
                    price: price.close,
                    color: 'purple',
                    lineWidth: 1,
                    title: `C: ${price.close.toFixed(2)}`,
                });

                candlestickSeries.applyOptions({
                    priceLineVisible: true,
                });
            }
        }
    }

    // 초기 priceLine 변수 정의
    let priceLineOpen, priceLineHigh, priceLineLow, priceLineClose;

    // 차트에서 마우스 이동 시 OHLC 정보를 업데이트
    priceChart.subscribeCrosshairMove(updateOhlcInfo);

    // 시간축 동기화
    const synchronizeCharts = (mainChart, linkedCharts) => {
        mainChart.timeScale().subscribeVisibleTimeRangeChange((newVisibleTimeRange) => {
            linkedCharts.forEach(linkedChart => {
                linkedChart.timeScale().setVisibleRange(newVisibleTimeRange);
            });
        });
        // mainChart.priceScale().subscribeVisiblePriceRangeChange((newVisiblePriceRange) => {
        //     linkedCharts.forEach(linkedChart => {
        //         linkedChart.priceScale().setVisiblePriceRange(newVisiblePriceRange);
        //     });
        // });
    };

    synchronizeCharts(priceChart, [volumeChart, rsiChart, macdChart]);

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
});