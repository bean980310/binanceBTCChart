// 백엔드에서 데이터 가져오기
fetch('/data')
.then(response => response.json())
.then(data => {

    const symbol = "BTCUSDT.P";
    const infoBox = initializeInfoBox(symbol, data);

    const { priceChart, volumeChart, rsiChart, macdChart, stochRsiChart } = initializeCharts();

    const candlestickSeries = createCandlestickSeries(priceChart);
    const volumeSeries = createVolumeSeries(volumeChart);

    const ema9Series = createLineSeries(priceChart, 'blue', 2);
    const ema60Series = createLineSeries(priceChart, 'red', 2);
    const ema200Series = createLineSeries(priceChart, 'green', 2);

    const rsiSeries = createLineSeries(rsiChart, 'purple', 1);
    const rsiSmaSeries = createLineSeries(rsiChart, 'orange', 1);

    const macdLineSeries = createLineSeries(macdChart, 'blue', 1)
    const macdSignalSeries = createLineSeries(macdChart, 'orange', 1);
    const macdHistSeries = createHistogramSeries(macdChart);

    const stochRsiKSeries = createLineSeries(stochRsiChart, 'blue', 1)
    const stochRsiDSeries = createLineSeries(stochRsiChart, 'orange', 1);

    const chartData = transformChartData(data);
    const volumeData = transformVolumeData(data);

    const ema9Data = transformIndicatorData(data, 'EMA9');
    const ema60Data = transformIndicatorData(data, 'EMA60');
    const ema200Data = transformIndicatorData(data, 'EMA200');

    const rsiData = transformIndicatorData(data, 'RSI');
    const rsiSmaData = transformIndicatorData(data, 'RSI_SMA');

    const macdLineData = transformIndicatorData(data, 'MACD');
    const macdSignalData = transformIndicatorData(data, 'MACD_Signal');
    const macdHistData = transformMacdHistData(data);

    const stochRsiKData = transformIndicatorData(data, 'StochRSI_%K');
    const stochRsiDData = transformIndicatorData(data, 'StochRSI_%D');

    setSeriesData(candlestickSeries, chartData);
    setSeriesData(volumeSeries, volumeData);

    setSeriesData(ema9Series, ema9Data);
    setSeriesData(ema60Series, ema60Data);
    setSeriesData(ema200Series, ema200Data);

    setSeriesData(rsiSeries, rsiData);
    setSeriesData(rsiSmaSeries, rsiSmaData);

    setSeriesData(macdLineSeries, macdLineData);
    setSeriesData(macdSignalSeries, macdSignalData);
    setSeriesData(macdHistSeries, macdHistData);

    setSeriesData(stochRsiKSeries, stochRsiKData);
    setSeriesData(stochRsiDSeries, stochRsiDData);

    let supportSeriesList = [];
    let resistanceSeriesList = [];

    priceChart.subscribeCrosshairMove(param => updateOhlcInfo(param, infoBox, candlestickSeries, chartData, symbol));

    synchronizeCharts(priceChart, [volumeChart, rsiChart, macdChart, stochRsiChart]);

    // 1분마다 데이터 업데이트
    setInterval(() => updateData(candlestickSeries, volumeSeries, ema9Series, ema60Series, ema200Series, rsiSeries, rsiSmaSeries, macdLineSeries, macdSignalSeries, macdHistSeries, stochRsiKSeries, stochRsiDSeries, infoBox, symbol), 100);

    setInterval(updateSupportResistanceLines(data, supportSeriesList, resistanceSeriesList, priceChart), 60000);
})
.catch(error => {
    console.error('Fetch error:', error);
});

function initializeInfoBox(symbol, data){
    const infoBox = document.createElement('div');
    infoBox.style.position = 'absolute';
    infoBox.style.top = '10px';
    infoBox.style.left = '10px';
    infoBox.style.color = '#000';
    infoBox.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
    infoBox.style.padding = '5px 10px';
    infoBox.style.borderRadius = '5px';
    infoBox.style.fontFamily = 'Arial, sans-serif';
    infoBox.style.fontSize = '14px';
    document.body.appendChild(infoBox);

    let lastData = data.length > 0 ? data[data.length - 1] : null;

    if (lastData) {
        infoBox.innerHTML = `${symbol} - O: ${lastData.Open.toFixed(2)} H: ${lastData.High.toFixed(2)} L: ${lastData.Low.toFixed(2)} C: ${lastData.Close.toFixed(2)}`;
    } else {
        infoBox.innerHTML = `${symbol} - O: N/A H: N/A L: N/A C: N/A`;
    }
    return infoBox;
}

function initializeCharts(){
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
    const stochRsiChart = LightweightCharts.createChart(document.getElementById('stochrsi-chart'), { ...chartOptions, height: 150 });

    // 시간축 숨김 설정
    volumeChart.timeScale().options().visible = false;
    rsiChart.timeScale().options().visible = false;
    macdChart.timeScale().options().visible = false;
    stochRsiChart.timeScale().options().visible = false;

    return { priceChart, volumeChart, rsiChart, macdChart, stochRsiChart };
}

function createCandlestickSeries(chart){
    return chart.addCandlestickSeries();
}

function createVolumeSeries(chart){
    return chart.addHistogramSeries({
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
}

function createLineSeries(chart, color, lineWidth){
    return chart.addLineSeries({
        color: color,
        lineWidth: lineWidth,
    });
}

function createHistogramSeries(chart){
    return chart.addHistogramSeries({
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
}

function transformChartData(data){
    return data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        open: item['Open'],
        high: item['High'],
        low: item['Low'],
        close: item['Close'],
    }));
}

function transformVolumeData(data){
    return data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['Volume'],
        color: item['Close'] > item['Open'] ? '#26a69a' : '#ef5350',
    }));
}

function transformIndicatorData(data, key){
    return data.map(item => ({
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item[key],
    })).filter(item => item.value !== null);
}
function transformMacdHistData(data){
    return data.map(item => ({    
        time: new Date(item['Open Time']).getTime() / 1000,
        value: item['MACD_Hist'],
        color: item['MACD_Hist'] >= 0 ? '#26a69a' : '#ef5350',
    })).filter(item => item.value !== null);
}

function setSeriesData(series, data){
    series.setData(data);
}

function updateOhlcInfo(param, infoBox, candlestickSeries, chartData, symbol) {
    let price = param?.seriesData?.get(candlestickSeries) || chartData[chartData.length - 1];

    if (price) {
        infoBox.innerHTML = `${symbol} - O: ${price.open.toFixed(2)} H: ${price.high.toFixed(2)} L: ${price.low.toFixed(2)} C: ${price.close.toFixed(2)}`;
    } else {
        infoBox.innerHTML = `${symbol} - O: N/A H: N/A L: N/A C: N/A`;
    }
}

function synchronizeCharts(mainChart, linkedCharts) {
    mainChart.timeScale().subscribeVisibleTimeRangeChange((newVisibleTimeRange) => {
        linkedCharts.forEach(linkedChart => {
            linkedChart.timeScale().setVisibleRange(newVisibleTimeRange);
        });
    });
}

function updateSupportResistanceLines(data, supportSeriesList, resistanceSeriesList, priceChart) {
    supportSeriesList.forEach(series => priceChart.removeSeries(series));
    resistanceSeriesList.forEach(series => priceChart.removeSeries(series));
    supportSeriesList = [];
    resistanceSeriesList = [];

    // 1차 및 2차 지지선과 저항선 레벨 가져오기
    const levels = ["Level1", "Level2", "Level3"];
    const supportColors = ['green', 'lightgreen'];
    const resistanceColors = ['red', 'pink'];

    levels.forEach((level, levelIndex) => {
        const support1st = data[0][`Support_1st_${level}`];
        const support2nd = data[0][`Support_2nd_${level}`];
        const resistance1st = data[0][`Resistance_1st_${level}`];
        const resistance2nd = data[0][`Resistance_2nd_${level}`];

        if (support1st !== undefined) {
            const support1stSeries = priceChart.addLineSeries({
                color: supportColors[0],
                lineWidth: 1,
                title: `Support 1차 ${level}`,
            });
            support1stSeries.setData(data.map(item => ({
                time: new Date(item['Open Time']).getTime() / 1000,
                value: support1st,
            })));
            supportSeriesList.push(support1stSeries);
        }

        if (support2nd !== undefined) {
            const support2ndSeries = priceChart.addLineSeries({
                color: supportColors[1],
                lineWidth: 1,
                title: `Support 2차 ${level}`,
            });
            support2ndSeries.setData(data.map(item => ({
                time: new Date(item['Open Time']).getTime() / 1000,
                value: support2nd,
            })));
            supportSeriesList.push(support2ndSeries);
        }

        if (resistance1st !== undefined) {
            const resistance1stSeries = priceChart.addLineSeries({
                color: resistanceColors[0],
                lineWidth: 1,
                title: `Resistance 1차 ${level}`,
            });
            resistance1stSeries.setData(data.map(item => ({
                time: new Date(item['Open Time']).getTime() / 1000,
                value: resistance1st,
            })));
            resistanceSeriesList.push(resistance1stSeries);
        }

        if (resistance2nd !== undefined) {
            const resistance2ndSeries = priceChart.addLineSeries({
                color: resistanceColors[1],
                lineWidth: 1,
                title: `Resistance 2차 ${level}`,
            });
            resistance2ndSeries.setData(data.map(item => ({
                time: new Date(item['Open Time']).getTime() / 1000,
                value: resistance2nd,
            })));
            resistanceSeriesList.push(resistance2ndSeries);
        }
    });
}


function updateData(candlestickSeries, volumeSeries, ema9Series, ema60Series, ema200Series, rsiSeries, rsiSmaSeries, macdLineSeries, macdSignalSeries, macdHistSeries, stochRsiKSeries, stochRsiDSeries, infoBox, symbol) {
    fetch('/data')
    .then(response => response.json())
    .then(newData => {
        const updatedChartData = transformChartData(newData);
        const updatedVolumeData = transformVolumeData(newData);

        const updatedEma9Data = transformIndicatorData(newData, 'EMA9');
        const updatedeEma60Data = transformIndicatorData(newData, 'EMA60');
        const updatedeEma200Data = transformIndicatorData(newData, 'EMA200');

        const updatedRsiData = transformIndicatorData(newData, 'RSI');
        const updatedRsiSmaData = transformIndicatorData(newData, 'RSI_SMA');

        const updatedMacdLineData = transformIndicatorData(newData, 'MACD');
        const updatedMacdSignalData = transformIndicatorData(newData, 'MACD_Signal');
        const updatedMacdHistData = transformMacdHistData(newData)

        const updatedStochRsiKData = transformIndicatorData(newData, 'StochRSI_%K');
        const updatedStochRsiDData = transformIndicatorData(newData, 'StochRSI_%D');

        setSeriesData(candlestickSeries, updatedChartData);
        setSeriesData(volumeSeries, updatedVolumeData);

        setSeriesData(ema9Series, updatedEma9Data);
        setSeriesData(ema60Series, updatedeEma60Data);
        setSeriesData(ema200Series, updatedeEma200Data);

        setSeriesData(rsiSeries, updatedRsiData);
        setSeriesData(rsiSmaSeries, updatedRsiSmaData);

        setSeriesData(macdLineSeries, updatedMacdLineData);
        setSeriesData(macdSignalSeries, updatedMacdSignalData);
        setSeriesData(macdHistSeries, updatedMacdHistData);

        setSeriesData(stochRsiKSeries, updatedStochRsiKData);
        setSeriesData(stochRsiDSeries, updatedStochRsiDData);
        
        const latestData = updatedChartData[updatedChartData.length - 1];
        if (latestData) {
            infoBox.innerHTML = `${symbol} - O: ${latestData.open.toFixed(2)} H: ${latestData.high.toFixed(2)} L: ${latestData.low.toFixed(2)} C: ${latestData.close.toFixed(2)}`;
        } else {
                infoBox.innerHTML = `${symbol} - O: N/A H: N/A L: N/A C: N/A`;
        }
    });
}