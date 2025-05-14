import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD

def collect_data(symbol, start_date, end_date):
    """Fetch OHLCV history for a single ticker."""
    data = yf.Ticker(symbol).history(start=start_date, end=end_date)
    return None if data.empty else data

def screen_stock(data, volume_threshold=500000,
                 rsi_threshold_lower=30, rsi_threshold_upper=70):
    """Return pass/fail for volume, SMA trend, RSI range, and MACD crossover."""
    avg_vol = data['Volume'].mean()
    vol = 'Pass' if avg_vol >= volume_threshold else 'Fail'

    sma50 = SMAIndicator(data['Close'], 50).sma_indicator().iloc[-1]
    sma200 = SMAIndicator(data['Close'], 200).sma_indicator().iloc[-1]
    sma = 'Pass' if sma50 >= sma200 else 'Fail'

    rsi_val = RSIIndicator(data['Close'], 14).rsi().iloc[-1]
    rsi = 'Pass' if rsi_threshold_lower <= rsi_val <= rsi_threshold_upper else 'Fail'

    macd_ind = MACD(data['Close'])
    macd_val = macd_ind.macd().iloc[-1]
    sig_val = macd_ind.macd_signal().iloc[-1]
    macd = 'Pass' if macd_val > sig_val else 'Fail'

    return {
        'Ticker': data.name,
        'Volume Check': vol,
        'SMA50 vs SMA200': sma,
        'RSI Check': rsi,
        'MACD Check': macd
    }

def load_tickers_from_csv(path):
    """Read a single line of comma-separated tickers."""
    with open(path) as f:
        return [t.strip() for t in f.readline().split(',') if t.strip()]

def analyze_multiple_stocks(tickers, start_date, end_date, workers=5):
    results = []

    def worker(symbol):
        data = collect_data(symbol, start_date, end_date)
        if data is not None:
            data.name = symbol
            results.append(screen_stock(data))

    with ThreadPoolExecutor(max_workers=workers) as exec:
        exec.map(worker, tickers)

    df = pd.DataFrame(results)
    df.to_csv('swing_trading_results_detailed.csv', index=False)
    return df

if __name__ == '__main__':
    tickers = load_tickers_from_csv('tickers.csv')
    df = analyze_multiple_stocks(tickers, '2020-01-01', '2023-01-01')
    print(df)
