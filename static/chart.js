// 메인 함수를 async로 변환
async function initializeChartSystem() {
    try {
        const response = await fetch('/data');
        const { priceData, indicatorData, supportresistance, trendlines, supertrend } = await response.json();

        const symbol = "BTCUSDT";
        const infoBox = initializeInfoBox(symbol, priceData);

        const { priceChart, volumeChart, rsiChart, macdChart, stochRsiChart } = initializeCharts();

        const series = initializeAllSeries(priceChart, volumeChart, rsiChart, macdChart, stochRsiChart);
        
        const chartData = await transformAllData(priceData, indicatorData);
        
        await updateAllSeriesData(series, chartData);

        // const supertrendSeries = initializeSupertrendSeries(priceChart, supertrend);

        let supportSeriesList = [];
        let resistanceSeriesList = [];

        let trendlineSeries = [];
        await updateTrendline(trendlines, trendlineSeries, priceChart)

        priceChart.subscribeCrosshairMove(param => 
            updateOhlcInfo(param, infoBox, series.candlestickSeries, chartData.ohlcData, symbol)
        );

        synchronizeCharts(priceChart, [volumeChart, rsiChart, macdChart, stochRsiChart]);

        // 주기적 업데이트를 async 함수로 변환
        setInterval(async () => {
            await updateAllData(series, infoBox, symbol);
        }, 100);

        // setInterval(async() => {
        //     await updateSupertrendData(supertrendSeries);
        // }, 60000);

        await updateSupportResistanceLines(supportresistance, supportSeriesList, resistanceSeriesList, priceChart);
        
        setInterval(async () => {
            await updateSupportResistanceLines(supportresistance, supportSeriesList, resistanceSeriesList, priceChart);
        }, 60000);

        setInterval(async () => {
            await updateTrendline(trendlines, trendlineSeries, priceChart);
        }, 60000);

    } catch (error) {
        console.error('Initialization error:', error);
    }
}

// 모든 시리즈 초기화를 하나의 함수로 통합
function initializeAllSeries(priceChart, volumeChart, rsiChart, macdChart, stochRsiChart) {
    return {
        candlestickSeries: createCandlestickSeries(priceChart),
        volumeSeries: createVolumeSeries(volumeChart),
        ema9Series: createLineSeries(priceChart, 'blue', 2),
        ema60Series: createLineSeries(priceChart, 'red', 2),
        ema200Series: createLineSeries(priceChart, 'green', 2),
        rsiSeries: createLineSeries(rsiChart, 'purple', 1),
        rsiSmaSeries: createLineSeries(rsiChart, 'orange', 1),
        macdLineSeries: createLineSeries(macdChart, 'blue', 1),
        macdSignalSeries: createLineSeries(macdChart, 'orange', 1),
        macdHistSeries: createHistogramSeries(macdChart),
        stochRsiKSeries: createLineSeries(stochRsiChart, 'blue', 1),
        stochRsiDSeries: createLineSeries(stochRsiChart, 'orange', 1)
    };
}

// 모든 데이터 변환을 하나의 함수로 통합
async function transformAllData(priceData, indicatorData) {
    return {
        ohlcData: transformChartData(priceData),
        volumeData: transformVolumeData(indicatorData),
        ema9Data: transformIndicatorData(indicatorData, 'EMA9'),
        ema60Data: transformIndicatorData(indicatorData, 'EMA60'),
        ema200Data: transformIndicatorData(indicatorData, 'EMA200'),
        rsiData: transformIndicatorData(indicatorData, 'RSI'),
        rsiSmaData: transformIndicatorData(indicatorData, 'RSI_SMA'),
        macdLineData: transformIndicatorData(indicatorData, 'MACD'),
        macdSignalData: transformIndicatorData(indicatorData, 'MACD_Signal'),
        macdHistData: transformMacdHistData(indicatorData),
        stochRsiKData: transformIndicatorData(indicatorData, 'StochRSI_%K'),
        stochRsiDData: transformIndicatorData(indicatorData, 'StochRSI_%D')
    };
}

// 모든 시리즈 데이터 업데이트를 하나의 함수로 통합
async function updateAllSeriesData(series, chartData) {
    setSeriesData(series.candlestickSeries, chartData.ohlcData);
    setSeriesData(series.volumeSeries, chartData.volumeData);
    setSeriesData(series.ema9Series, chartData.ema9Data);
    setSeriesData(series.ema60Series, chartData.ema60Data);
    setSeriesData(series.ema200Series, chartData.ema200Data);
    setSeriesData(series.rsiSeries, chartData.rsiData);
    setSeriesData(series.rsiSmaSeries, chartData.rsiSmaData);
    setSeriesData(series.macdLineSeries, chartData.macdLineData);
    setSeriesData(series.macdSignalSeries, chartData.macdSignalData);
    setSeriesData(series.macdHistSeries, chartData.macdHistData);
    setSeriesData(series.stochRsiKSeries, chartData.stochRsiKData);
    setSeriesData(series.stochRsiDSeries, chartData.stochRsiDData);
}

// 데이터 업데이트 함수를 async로 변환
async function updateAllData(series, infoBox, symbol) {
    try {
        const response = await fetch('/data');
        const { priceData, indicatorData } = await response.json();
        
        const chartData = await transformAllData(priceData, indicatorData);
        await updateAllSeriesData(series, chartData);
        
        const latestData = chartData.ohlcData[chartData.ohlcData.length - 1];
        updateInfoBox(infoBox, symbol, latestData);
    } catch (error) {
        console.error('Update error:', error);
    }
}

async function updateSupportResistanceLines(data, supportSeriesList, resistanceSeriesList, priceChart) {
    try {
        // 기존 라인 제거를 비동기 처리하고 모든 라인 제거가 완료될 때까지 대기
        await Promise.all(supportSeriesList.map(series => priceChart.removeSeries(series)));
        await Promise.all(resistanceSeriesList.map(series => priceChart.removeSeries(series)));

        supportSeriesList.length = 0;
        resistanceSeriesList.length = 0;

        const levels = ["Level1", "Level2", "Level3"];
        const supportColors = ['green', 'lightgreen'];
        const resistanceColors = ['red', 'pink'];

        // 각 레벨에 대한 지지/저항선을 비동기적으로 추가
        await Promise.all(levels.map(level => 
            addSupportResistanceLevel(
                data, level, priceChart, 
                supportSeriesList, resistanceSeriesList,
                supportColors, resistanceColors
            )
        ));
    } catch (error) {
        console.error('Support/Resistance update error:', error);
    }
}

async function addSupportResistanceLevel(data, level, priceChart, supportSeriesList, resistanceSeriesList, supportColors, resistanceColors) {
    const timeData = data.map(item => {
        const timestamp = new Date(item['Open Time']).getTime() / 1000;
        return isNaN(timestamp) ? null : { time: timestamp };
    }).filter(item => item !== null); // 유효하지 않은 타임스탬프는 제외

    const support1st = data[0]?.[`Support_1st_${level}`] ?? null;
    const support2nd = data[0]?.[`Support_2nd_${level}`] ?? null;
    const resistance1st = data[0]?.[`Resistance_1st_${level}`] ?? null;
    const resistance2nd = data[0]?.[`Resistance_2nd_${level}`] ?? null;

    const promises = [];
    if (support1st != null) {
        promises.push(addLevelLine(support1st, `Support 1차 ${level}`, supportColors[0], timeData, priceChart, supportSeriesList));
    }
    if (support2nd != null) {
        promises.push(addLevelLine(support2nd, `Support 2차 ${level}`, supportColors[1], timeData, priceChart, supportSeriesList));
    }
    if (resistance1st != null) {
        promises.push(addLevelLine(resistance1st, `Resistance 1차 ${level}`, resistanceColors[0], timeData, priceChart, resistanceSeriesList));
    }
    if (resistance2nd != null) {
        promises.push(addLevelLine(resistance2nd, `Resistance 2차 ${level}`, resistanceColors[1], timeData, priceChart, resistanceSeriesList));
    }

    await Promise.all(promises);
}

async function addLevelLine(value, label, color, timeData, priceChart, seriesList) {
    const series = priceChart.addLineSeries({
        color: color,
        lineWidth: 1,
        title: label
    });

    const lineData = timeData.map(time => ({
        time: time.time,
        value: value
    }));
    
    await series.setData(lineData);
    seriesList.push(series);
}

async function updateTrendline(data, trendlineSeries, priceChart){ 
    trendlineSeries.forEach(series => priceChart.removeSeries(series));
    trendlineSeries = [];

    // Add resistance lines
    data.resistance.forEach(line => {
        const series = addTrendline(
            priceChart, 
            line.start,
            line.end,
            line.color,
            line.lineWidth
        );
        trendlineSeries.push(series);
    });
    
    // Add support lines
    data.support.forEach(line => {
        const series = addTrendline(
            priceChart,
            line.start,
            line.end,
            line.color,
            line.lineWidth
        );
        trendlineSeries.push(series);
    });

    // Add channel lines
    if (data.channels) {
        data.channels.forEach(channel => {
            const series = priceChart.addLineSeries({
                color: channel.color,
                lineWidth: channel.lineWidth,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            
            series.setData(channel.points);
            trendlineSeries.push(series);
        });
    }
}

// // 슈퍼트렌드 데이터 업데이트
// async function updateSupertrendData(supertrendSeries) {
//     try {
//         const response = await fetch('/data');
//         const { supertrend } = await response.json();

//         const upperbandData = supertrend.map(item => ({
//             time: new Date(item.date).getTime() / 1000,
//             value: item.Upperband,
//         }));

//         const lowerbandData = supertrend.map(item => ({
//             time: new Date(item.date).getTime() / 1000,
//             value: item.Lowerband,
//         }));

//         const buySignals = supertrend
//             .filter(item => item.Signals === 1)
//             .map(item => ({
//                 time: new Date(item.date).getTime() / 1000,
//                 value: item.Close,
//             }));

//         const sellSignals = supertrend
//             .filter(item => item.signals === -1)
//             .map(item => ({
//                 time: new Date(item.date).getTime() / 1000,
//                 value: item.Close,
//             }));

//         supertrendSeries.upperbandSeries.setData(upperbandData);
//         supertrendSeries.lowerbandSeries.setData(lowerbandData);
//         supertrendSeries.buySignalSeries.setData(buySignals);
//         supertrendSeries.sellSignalSeries.setData(sellSignals);
//     } catch (error) {
//         console.error('Supertrend update error:', error);
//     }
// }

function addTrendline(chart, startPoint, endPoint, color, lineWidth) {
    const lineSeries = chart.addLineSeries({
        color: color,
        lineWidth: lineWidth,
        lastValueVisible: false,
        priceLineVisible: false,
    });

    lineSeries.setData([
        { time: startPoint.time, value: startPoint.value },
        { time: endPoint.time, value: endPoint.value }
    ]);

    return lineSeries;
}

function updateInfoBox(infoBox, symbol, latestData) {
    if (latestData) {
        infoBox.innerHTML = `${symbol} - O: ${latestData.open.toFixed(2)} H: ${latestData.high.toFixed(2)} L: ${latestData.low.toFixed(2)} C: ${latestData.close.toFixed(2)}`;
    } else {
        infoBox.innerHTML = `${symbol} - O: N/A H: N/A L: N/A C: N/A`;
    }
}

function initializeInfoBox(symbol, priceData){
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

    let lastData = priceData.length > 0 ? priceData[priceData.length - 1] : null;

    if (lastData) {
        infoBox.innerHTML = `${symbol} - O: ${lastData.Open.toFixed(2)} H: ${lastData.High.toFixed(2)} L: ${lastData.Low.toFixed(2)} C: ${lastData.Close.toFixed(2)}`;
    } else {
        infoBox.innerHTML = `${symbol} - O: N/A H: N/A L: N/A C: N/A`;
    }
    return infoBox;
}

// // 슈퍼트렌드 시리즈 초기화
// function initializeSupertrendSeries(priceChart, supertrendData) {
//     const upperbandSeries = priceChart.addLineSeries({
//         color: 'red',
//         lineWidth: 1,
//         title: 'Supertrend Upper Band',
//     });

//     const lowerbandSeries = priceChart.addLineSeries({
//         color: 'green',
//         lineWidth: 1,
//         title: 'Supertrend Lower Band',
//     });

//     const buySignalSeries = priceChart.addHistogramSeries({
//         color: '#2cf651',
//         priceFormat: { type: 'price' },
//         scaleMargins: { top: 0.7, bottom: 0 },
//     });

//     const sellSignalSeries = priceChart.addHistogramSeries({
//         color: '#f50100',
//         priceFormat: { type: 'price' },
//         scaleMargins: { top: 0.7, bottom: 0 },
//     });

//     const upperbandData = supertrendData.map(item => ({
//         time: new Date(item.date).getTime() / 1000,
//         value: item.upperband,
//     }));

//     const lowerbandData = supertrendData.map(item => ({
//         time: new Date(item.date).getTime() / 1000,
//         value: item.lowerband,
//     }));

//     const buySignals = supertrendData
//         .filter(item => item.Signals === 1)
//         .map(item => ({
//             time: new Date(item.date).getTime() / 1000,
//             value: item.Close,
//         }));

//     const sellSignals = supertrendData
//         .filter(item => item.Signals === -1)
//         .map(item => ({
//             time: new Date(item.date).getTime() / 1000,
//             value: item.Close,
//         }));

//     upperbandSeries.setData(upperbandData);
//     lowerbandSeries.setData(lowerbandData);
//     buySignalSeries.setData(buySignals);
//     sellSignalSeries.setData(sellSignals);

//     return { upperbandSeries, lowerbandSeries, buySignalSeries, sellSignalSeries };
// }

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


initializeChartSystem().catch(error => {
    console.error('Chart system initialization failed:', error);
});