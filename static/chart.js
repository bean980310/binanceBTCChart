// 메인 차트 생성
const chart = LightweightCharts.createChart(document.getElementById('chart'), {
    width: 800,
    height: 600,
    layout: {
        backgroundColor: '#FFFFFF',
        textColor: '#000000',
    },
    grid: {
        vertLines: {
            color: '#e0e0e0',
        },
        horzLines: {
            color: '#e0e0e0',
        },
    },
    priceScale: {
        borderColor: '#cccccc',
    },
    timeScale: {
        borderColor: '#cccccc',
    },
});

const candleSeries = chart.addCandlestickSeries();

// 거래량 차트 생성
const volumeChart = LightweightCharts.createChart(document.getElementById('volume-chart'), {
    width: 800,
    height: 150,
    layout: {
        backgroundColor: '#FFFFFF',
        textColor: '#000000',
    },
    timeScale: {
        visible: false,
    },
});

const volumeSeries = volumeChart.addHistogramSeries({
    color: '#26a69a',
    priceFormat: {
        type: 'volume',
    },
    priceScaleId: '',
});

// RSI 차트 생성
const rsiChart = LightweightCharts.createChart(document.getElementById('rsi-chart'), {
    width: 800,
    height: 200,
    layout: {
        backgroundColor: '#FFFFFF',
        textColor: '#000000',
    },
    timeScale: {
        visible: false,
    },
});

const rsiSeries = rsiChart.addLineSeries({
    color: 'red',
    lineWidth: 2,
});

const rsiSmaSeries = rsiChart.addLineSeries({
    color: 'orange',
    lineWidth: 2,
});

// MACD 차트 생성
const macdChart = LightweightCharts.createChart(document.getElementById('macd-chart'), {
    width: 800,
    height: 200,
    layout: {
        backgroundColor: '#FFFFFF',
        textColor: '#000000',
    },
    timeScale: {
        visible: false,
    },
});

const macdSeries = macdChart.addLineSeries({
    color: 'blue',
    lineWidth: 2,
});

const signalSeries = macdChart.addLineSeries({
    color: 'orange',
    lineWidth: 2,
});

const histSeries = macdChart.addHistogramSeries({
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

// 데이터 가져오기 및 차트 업데이트 함수 정의
async function fetchDataAndUpdateCharts() {
    try {
        // 가격 데이터 가져오기
        const response = await fetch('/data');
        const data = await response.json();
        const chartData = data.map(item => ({
            time: item.time,
            open: item.Open,
            high: item.High,
            low: item.Low,
            close: item.Close,
        }));
        candleSeries.setData(chartData);

        // 거래량 데이터 가져오기
        const volumeResponse = await fetch('/volume_data');
        const volumeDataRaw = await volumeResponse.json();
        const volumeData = volumeDataRaw.map(item => ({
            time: item.time,
            value: item.Volume,
            color: item.Open < item.Close ? 'rgba(0, 150, 136, 0.8)' : 'rgba(255, 82, 82, 0.8)',
        }));
        volumeSeries.setData(volumeData);

        // RSI 데이터 가져오기
        const rsiResponse = await fetch('/rsi_data');
        const rsiDataRaw = await rsiResponse.json();
        const rsiData = rsiDataRaw.map(item => ({
            time: item.time,
            value: item.RSI,
        }));
        rsiSeries.setData(rsiData);

        const rsiSmaData = rsiDataRaw.map(item => ({
            time: item.time,
            value: item.RSI_SMA,
        }));
        rsiSmaSeries.setData(rsiSmaData);

        // EMA 데이터 가져오기 및 차트에 추가
        const emaResponse = await fetch('/ema_data');
        const emaDataRaw = await emaResponse.json();
        const ema9Data = emaDataRaw.map(item => ({
            time: item.time,
            value: item.EMA9,
        }));
        const ema60Data = emaDataRaw.map(item => ({
            time: item.time,
            value: item.EMA60,
        }));
        const ema200Data = emaDataRaw.map(item => ({
            time: item.time,
            value: item.EMA200,
        }));

        const ema9Series = chart.addLineSeries({
            color: 'blue',
            lineWidth: 1,
        });
        ema9Series.setData(ema9Data);

        const ema60Series = chart.addLineSeries({
            color: 'orange',
            lineWidth: 1,
        });
        ema60Series.setData(ema60Data);

        const ema200Series = chart.addLineSeries({
            color: 'purple',
            lineWidth: 1,
        });
        ema200Series.setData(ema200Data);

        // MACD 데이터 가져오기
        const macdResponse = await fetch('/macd_data');
        const macdDataRaw = await macdResponse.json();
        const macdData = macdDataRaw.map(item => ({
            time: item.time,
            value: item.MACD,
        }));
        macdSeries.setData(macdData);

        const signalData = macdDataRaw.map(item => ({
            time: item.time,
            value: item.MACD_Signal,
        }));
        signalSeries.setData(signalData);

        const histData = macdDataRaw.map(item => ({
            time: item.time,
            value: item.MACD_Hist,
            color: item.MACD_Hist >= 0 ? 'rgba(0, 150, 136, 0.8)' : 'rgba(255, 82, 82, 0.8)',
        }));
        histSeries.setData(histData);

    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// 함수 호출
fetchDataAndUpdateCharts();

// 주기적으로 데이터 업데이트 (선택 사항)
setInterval(fetchDataAndUpdateCharts, 1); // 1분마다 업데이트

// 차트 크기 조절
window.addEventListener('resize', () => {
    const width = document.getElementById('chart').clientWidth;
    chart.applyOptions({ width });
    volumeChart.applyOptions({ width });
    rsiChart.applyOptions({ width });
    macdChart.applyOptions({ width });
});