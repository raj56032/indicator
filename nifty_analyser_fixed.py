import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt  # NEW: For visualization

class NiftyAnalyzer:
    def __init__(self):
        self.tickers = {
            'Nifty50': '^NSEI',
            'BankNifty': '^NSEBANK'
        }
        self.intervals = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1d': '1d'
        }
    
    def _get_column(self, data, col_name):
        """Helper to get columns from MultiIndex DataFrame"""
        if isinstance(data.columns, pd.MultiIndex):
            return data[(col_name, data.columns.levels[1][0])]
        return data[col_name]
    
    def fetch_data(self, ticker_name, interval, period='1mo'):
        """Fetch data and simplify column names"""
        ticker = self.tickers.get(ticker_name)
        data = yf.download(
            tickers=ticker,
            interval=self.intervals[interval],
            period=period,
            progress=False
        )
        
        # Simplify column names if MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    
    def calculate_rsi(self, data, window=14):
        """Calculate RSI with proper column access"""
        close = self._get_column(data, 'Close')
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_ema(self, data, windows=[20, 50]):
        """Calculate EMA with proper column access"""
        close = self._get_column(data, 'Close')
        for window in windows:
            data[f'EMA_{window}'] = close.ewm(span=window, adjust=False).mean()
        return data
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD with proper column access"""
        close = self._get_column(data, 'Close')
        data['EMA_Fast'] = close.ewm(span=fast, adjust=False).mean()
        data['EMA_Slow'] = close.ewm(span=slow, adjust=False).mean()
        data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
        data['Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        return data
    
    def calculate_supertrend(self, data, period=7, multiplier=3):
        """Calculate SuperTrend with proper column access"""
        high = self._get_column(data, 'High')
        low = self._get_column(data, 'Low')
        close = self._get_column(data, 'Close')
        
        # Calculate ATR
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        
        # Initialize
        st = [upper.iloc[0]]
        direction = [1]
        
        for i in range(1, len(data)):
            if close.iloc[i] > upper.iloc[i-1]:
                direction.append(1)
            elif close.iloc[i] < lower.iloc[i-1]:
                direction.append(-1)
            else:
                direction.append(direction[-1])
            
            if direction[-1] == 1:
                st.append(min(lower.iloc[i], st[-1]) if lower.iloc[i] < st[-1] else lower.iloc[i])
            else:
                st.append(max(upper.iloc[i], st[-1]) if upper.iloc[i] > st[-1] else upper.iloc[i])
        
        data['SuperTrend'] = st
        data['SuperTrend_Direction'] = ['up' if x == 1 else 'down' for x in direction]
        return data
    
    def score_signal(self, data_row):
        score = 0
        if data_row['EMA_20'] > data_row['EMA_50']: score += 1
        if data_row['RSI'] < 30 or data_row['RSI'] > 70: score += 1
        if data_row['MACD'] > data_row['Signal']: score += 1
        if data_row['SuperTrend_Direction'] == 'up': score += 1
        return round(score / 4, 2)

    def generate_signals(self, data):
        """Generate signals with proper column access"""
        signals = []
        close = data['Close']
    
        # More practical signal conditions (less strict)
        # 1. EMA Crossover + Volume filter (50% of average volume allowed)
        if 'EMA_20' in data.columns and 'EMA_50' in data.columns:
            volume_ok = data['Volume'] > (0.5 * data['Volume'].mean())  # 50% of avg volume
            ema_cross_up = (data['EMA_20'] > data['EMA_50']) & (data['EMA_20'].shift() <= data['EMA_50'].shift()) & volume_ok
            ema_cross_down = (data['EMA_20'] < data['EMA_50']) & (data['EMA_20'].shift() >= data['EMA_50'].shift()) & volume_ok
            
            for idx in data[ema_cross_up].index:
                signals.append({
                    'timestamp': idx.isoformat(),
                    'price': float(close.loc[idx]),
                    'signal': 'BUY',
                    'indicator': 'EMA Crossover',
                    'score': self.score_signal(data.loc[idx])
                    })
            
            for idx in data[ema_cross_down].index:
                signals.append({
                    'timestamp': idx.isoformat(),
                    'price': float(close.loc[idx]),
                    'signal': 'SELL',
                    'indicator': 'EMA Crossover',
                    'score': self.score_signal(data.loc[idx])
                })
        
        # 2. RSI: Wider thresholds (25-75 instead of 30-70)
        if 'RSI' in data.columns:
            rsi_overbought = (data['RSI'] > 75) & (data['RSI'].shift() <= 75)
            rsi_oversold = (data['RSI'] < 25) & (data['RSI'].shift() >= 25)
            
            for idx in data[rsi_overbought].index:
                signals.append({
                    'timestamp': idx.isoformat(),
                    'price': float(close.loc[idx]),
                    'signal': 'SELL',
                    'indicator': 'RSI',
                    'score': self.score_signal(data.loc[idx])
                })
            
            for idx in data[rsi_oversold].index:
                signals.append({
                    'timestamp': idx.isoformat(),
                    'price': float(close.loc[idx]),
                    'signal': 'BUY',
                    'indicator': 'RSI',
                    'score': self.score_signal(data.loc[idx])
                })
        
        # 3. MACD: Allow minor crossovers
        if 'MACD' in data.columns and 'Signal' in data.columns:
            macd_cross_up = (data['MACD'] > data['Signal']) 
            macd_cross_down = (data['MACD'] < data['Signal'])
            
            for idx in data[macd_cross_up].index:
                signals.append({
                    'timestamp': idx.isoformat(),
                    'price': float(close.loc[idx]),
                    'signal': 'BUY',
                    'indicator': 'MACD',
                    'score': self.score_signal(data.loc[idx])
                })
            
            for idx in data[macd_cross_down].index:
                signals.append({
                    'timestamp': idx.isoformat(),
                    'price': float(close.loc[idx]),
                    'signal': 'SELL',
                    'indicator': 'MACD',
                    'score': self.score_signal(data.loc[idx])
                })
        
        return signals
    
    
    def analyze(self, ticker_name, interval, period='1mo', plot=False):
        """Complete analysis pipeline"""
        
        data = self.fetch_data(ticker_name, interval, period)
        if data.empty:
            return {"error": "No data"}
        
        # Calculate indicators
        data['RSI'] = self.calculate_rsi(data)
        data = self.calculate_ema(data)
        data = self.calculate_macd(data)
        data = self.calculate_supertrend(data)
        data = self.calculate_bollinger_bands(data)

        signals = self.generate_signals(data)
        
        # NEW: Summarize instead of listing all signals
        summary = {
            'ticker': ticker_name,
            'current_price': round(float(data['Close'].iloc[-1]), 2),
            'trend': 'Bullish' if data['SuperTrend_Direction'].iloc[-1] == 'up' else 'Bearish',
            'indicators': {
                'RSI': round(data['RSI'].iloc[-1], 2),
                'EMA_20_vs_50': 'Above' if data['EMA_20'].iloc[-1] > data['EMA_50'].iloc[-1] else 'Below',
                'MACD': 'Bullish' if data['MACD'].iloc[-1] > data['Signal'].iloc[-1] else 'Bearish',
            },
            'last_signal': signals[-1] if signals else None  # Most recent signal only
        }

        # NEW: Optional plot
        if plot:
            self._plot_data(data, ticker_name)
        
        return summary

    def _plot_data(self, data, ticker_name):  # NEW: Visualization
        plt.figure(figsize=(12, 6))
        # Plot price and indicators
        plt.plot(data['Close'], label='Price', color='blue', alpha=0.8)
        
        if 'EMA_20' in data.columns:
            plt.plot(data['EMA_20'], label='EMA 20', color='orange', linestyle='--', linewidth=1)
        if 'EMA_50' in data.columns:
            plt.plot(data['EMA_50'], label='EMA 50', color='red', linestyle='--', linewidth=1)
        if 'SuperTrend' in data.columns:
            plt.plot(data['SuperTrend'], label='SuperTrend', color='purple', linewidth=1.5)
        
        # Highlight signals if they exist
        signals = self.generate_signals(data)
        if signals:
            buy_signals = [s for s in signals if s['signal'] == 'BUY']
            sell_signals = [s for s in signals if s['signal'] == 'SELL']
            
            if buy_signals:
                buy_times = [pd.to_datetime(s['timestamp']) for s in buy_signals]
                buy_prices = [s['price'] for s in buy_signals]
                plt.scatter(buy_times, buy_prices, color='green', marker='^', label='BUY', s=100)
            
            if sell_signals:
                sell_times = [pd.to_datetime(s['timestamp']) for s in sell_signals]
                sell_prices = [s['price'] for s in sell_signals]
                plt.scatter(sell_times, sell_prices, color='red', marker='v', label='SELL', s=100)
    
        plt.title(f"{ticker_name} Analysis\n(Price + Indicators + Signals)")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        close = self._get_column(data, 'Close')
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        data['Bollinger_Upper'] = rolling_mean + (rolling_std * num_std)
        data['Bollinger_Lower'] = rolling_mean - (rolling_std * num_std)
        return data



if __name__ == "__main__":
    analyzer = NiftyAnalyzer()
    
    nifty = analyzer.analyze('Nifty50', '15m', period='3d', plot=True)  # Try plot=True!
    banknifty = analyzer.analyze('BankNifty', '1d', plot=True)
    
    print("\n=== Results ===")
    print(f"Nifty50 Trend: {nifty['trend']}, RSI: {nifty['indicators']['RSI']}")
    print(f"BankNifty Trend: {banknifty['trend']}, RSI: {banknifty['indicators']['RSI']}")