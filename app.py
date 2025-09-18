#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import joblib
import queue
import random
import logging
import threading
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import requests

from flask import Flask, request, send_from_directory, Response
from flask_cors import CORS

# =====================================================================================
# Logger / Flask
# =====================================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import adfuller
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    logger.warning("statsmodels não disponível. Teste de estacionariedade desativado.")

# ===== Optional scikit-learn (ML) =====
SKLEARN_AVAILABLE = True
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from scipy.stats import randint, uniform
except Exception:
    SKLEARN_AVAILABLE = False

# =====================================================================================
# Config Markets
# =====================================================================================
MARKETS = {
    'BTC': {'ticker': 'BTCUSDT', 'name': 'Bitcoin | BTC/USDT'},
    'ETH': {'ticker': 'ETHUSDT', 'name': 'Ethereum | ETH/USDT'},
    'BNB': {'ticker': 'BNBUSDT', 'name': 'Binance Coin | BNB/USDT'},
    'SOL': {'ticker': 'SOLUSDT', 'name': 'Solana | SOL/USDT'},
    'ADA': {'ticker': 'ADAUSDT', 'name': 'Cardano | ADA/USDT'},
    'XRP': {'ticker': 'XRPUSDT', 'name': 'Ripple | XRP/USDT'},
    'AVAX': {'ticker': 'AVAXUSDT', 'name': 'Avalanche | AVAX/USDT'},
    'DOGE': {'ticker': 'DOGEUSDT', 'name': 'Dogecoin | DOGE/USDT'},
    'MATIC': {'ticker': 'MATICUSDT', 'name': 'Polygon | MATIC/USDT'},
    'LINK': {'ticker': 'LINKUSDT', 'name': 'Chainlink | LINK/USDT'},
}

def normalize_market_key(market: str):
    if not market:
        return None
    m = str(market).strip().upper()
    return m if m in MARKETS else None

# =====================================================================================
# Logger / Flask
# =====================================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# =====================================================================================
# Defaults
# =====================================================================================
DEFAULT_PARAMS = {
    'ml_enabled': True,
    'trailing_enabled': True,
    'trailing_lock_profit_atr': 1.0,   # trava lucro após mover 1x ATR
    'trailing_step_atr': 0.5,          # a cada 0.5x ATR de lucro, aperta stop
    'timeframes': [5, 15],
    'weights': {'ml': 0.5, 'rule': 0.3, 'mtf': 0.2},
    'thresholds': {'buy': 60, 'sell': 60, 'strong': 75, 'rule_min_conditions': 3},
    'vol_filters': {
        'atr_window': 14,
        'atr_percentile_min': 20,
        'atr_percentile_max': 90,
        'volume_ratio_min': 0.9,
        'volume_ratio_max': 3.0,
        'realized_vol_window': 30,
        'realized_vol_min': 0.0004,
        'realized_vol_max': 0.02
    },
    'risk': {
        'atr_stop_mult': 2.5,
        'atr_target_mult': 3.0,
        'min_minutes_between_trades': 15,   # menor refratariedade para aumentar cadência
        'max_trades_per_day': 12,           # mais operações permitidas por dia
        'daily_target_pct_of_initial': 0.5
    },
    'ml': {
        'horizon': 3,
        'n_estimators': 150,
        'max_depth': 6,
        'min_samples_leaf': 5,
        'random_state': 42
    },
    'optimization': {'trials': 20, 'objective': 'sharpe'},
    'data': {'base_interval': '1m', 'max_hist_days': 14, 'allow_simulation': False}
}

# 5 perfis de estratégia
PARAM_PROFILES = {
    'PREDICTIVE': {
        'weights': {'ml': 0.60, 'rule': 0.20, 'mtf': 0.20},
        'thresholds': {'buy': 58, 'sell': 58}
    },
    'BALANCED': {
        'weights': {'ml': 0.40, 'rule': 0.30, 'mtf': 0.30},
        'thresholds': {'buy': 60, 'sell': 60}
    },
    'RULE_HEAVY': {
        'weights': {'ml': 0.20, 'rule': 0.45, 'mtf': 0.35},
        'thresholds': {'buy': 62, 'sell': 62}
    },
    'MTF_HEAVY': {
        'weights': {'ml': 0.30, 'rule': 0.20, 'mtf': 0.50},
        'thresholds': {'buy': 60, 'sell': 60}
    },
    'MEAN_REVERT': {
        'weights': {'ml': 0.25, 'rule': 0.50, 'mtf': 0.25},
        'thresholds': {'buy': 55, 'sell': 58}
    },
}

# =====================================================================================
# Utils
# =====================================================================================
def parse_iso(ts):
    try:
        return datetime.fromisoformat(str(ts).replace('Z', ''))
    except Exception:
        return datetime.now()

def as_iso(dt):
    try:
        return dt.isoformat()
    except Exception:
        return datetime.now().isoformat()

def parse_period_days(period_str, default_days=7):
    try:
        if isinstance(period_str, (int, float)):
            return int(period_str)
        s = str(period_str).strip().lower()
        if s.endswith('d'): return int(s[:-1])
        if s.endswith('w'): return int(s[:-1]) * 7
        if s.endswith('m'): return int(s[:-1]) * 30
        if s.endswith('y'): return int(s[:-1]) * 365
        return int(s)
    except Exception:
        return default_days

def _to_native_types(d):
    if isinstance(d, dict):
        return {k: _to_native_types(v) for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        return [_to_native_types(x) for x in d]
    elif isinstance(d, (np.bool_,)):
        return bool(d)
    elif isinstance(d, (np.integer,)):
        return int(d)
    elif isinstance(d, (np.floating,)):
        return float(d)
    else:
        return d

def json_response(obj, status=200):
    try:
        payload = _to_native_types(obj)
        return Response(json.dumps(payload, ensure_ascii=False), status=status, mimetype='application/json')
    except Exception as e:
        logger.exception("JSON serialization error")
        return Response(json.dumps({'success': False, 'error': 'serialization_error', 'detail': str(e)}),
                        status=500, mimetype='application/json')

@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled exception on %s: %s", request.path, e)
    if request.path.startswith('/api/'):
        return json_response({'success': False, 'error': 'internal_server_error', 'detail': str(e)}, 500)
    return ("Internal Server Error", 500)



# ================ FrozenEstimator (para CalibratedClassifierCV) ================
class FrozenEstimator:
    """
    Wrapper para congelar um estimador já treinado.
    Necessário para uso com CalibratedClassifierCV(cv='prefit').
    Copia atributos essenciais para compatibilidade com scikit-learn.
    """
    def __init__(self, estimator):
        self.estimator = estimator
        # Copia atributos importantes para validação sklearn
        for attr in dir(estimator):
            if not attr.startswith('_') or attr in ['_estimator_type', '_validate_data']:
                try:
                    value = getattr(estimator, attr)
                    if not callable(value) or attr in ['predict_proba', 'decision_function', 'predict']:
                        setattr(self, attr, value)
                except Exception:
                    pass  # ignora se falhar

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)

    def fit(self, X, y=None, **kwargs):
        # Não faz nada — já foi treinado
        return self 

# =====================================================================================
# Binance Provider
# =====================================================================================
class BinanceProvider:
    def __init__(self, symbol='BTCUSDT', base_url='https://api.binance.com'):
        self.symbol = symbol
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (TradingBot)"})

    def _fetch_klines(self, interval='1m', start_ms=None, end_ms=None, limit=1000):
        url = f"{self.base_url}/api/v3/klines"
        params = {'symbol': self.symbol, 'interval': interval, 'limit': limit}
        if start_ms is not None:
            params['startTime'] = int(start_ms)
        if end_ms is not None:
            params['endTime'] = int(end_ms)
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_historical_df(self, days=7, interval='1m'):
        try:
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)
            klines, cur = [], start_ms
            step_ms = {'1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000}.get(interval, 60_000)
            safety = 0
            while cur < end_ms and safety < 6000:
                batch = self._fetch_klines(interval=interval, start_ms=cur, end_ms=end_ms, limit=1000)
                if not batch:
                    break
                klines.extend(batch)
                next_open = batch[-1][0] + step_ms
                if next_open <= cur:
                    next_open = cur + step_ms
                cur = int(next_open)
                safety += 1
                time.sleep(0.02)
            if not klines:
                return pd.DataFrame()
            ot = [int(k[0]) for k in klines]
            o = [float(k[1]) for k in klines]
            h = [float(k[2]) for k in klines]
            l = [float(k[3]) for k in klines]
            c = [float(k[4]) for k in klines]
            v = [float(k[7]) for k in klines]  # quote volume
            idx = pd.to_datetime(ot, unit='ms', utc=True).tz_convert(None)
            df = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}, index=idx)
            df = df[~df.index.duplicated(keep='last')].sort_index()
            return df
        except Exception as e:
            logger.error(f"[{self.symbol}] Erro histórico Binance: {e}")
            return pd.DataFrame()

    def get_latest_df(self, interval='1m'):
        try:
            data = self._fetch_klines(interval=interval, limit=1)
            if not data:
                return pd.DataFrame()
            k = data[-1]
            idx = pd.to_datetime([int(k[0])], unit='ms', utc=True).tz_convert(None)
            return pd.DataFrame({
                'open': [float(k[1])], 'high': [float(k[2])], 'low': [float(k[3])],
                'close': [float(k[4])], 'volume': [float(k[7])]
            }, index=idx)
        except Exception as e:
            logger.warning(f"[{self.symbol}] Erro último kline Binance: {e}")
            return pd.DataFrame()

# =====================================================================================
# Data Collector
# =====================================================================================
class OptimizedDataCollector:
    def __init__(self, provider=None, base_interval='1m', max_hist_days=7, market_name="Market"):
        self.provider = provider or BinanceProvider(symbol='BTCUSDT')
        self.base_interval = base_interval
        self.max_hist_days = max_hist_days
        self.data_source = "Binance"
        self.market_name = market_name

        self.current_data = {}
        self.historical_data = deque(maxlen=12000)
        self.chart_data = deque(maxlen=800)
        self.is_running = False
        self.thread = None
        self.last_update = None
        self.last_fetch_ok = False
        self._last_bar_time = None
        self._initialize_historical_data()

    def _initialize_historical_data(self):
        logger.info(f"[{self.provider.symbol}] Inicializando dados ({self.data_source}) {self.max_hist_days}d {self.base_interval}...")
        df = self.provider.get_historical_df(days=self.max_hist_days, interval=self.base_interval)
        if df is None or df.empty:
            logger.error(f"[{self.provider.symbol}] Histórico inicial vazio.")
            return
        for i, (idx, row) in enumerate(df.iterrows()):
            dp = {
                'timestamp': idx.isoformat(),
                'price': float(row['close']),
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row['volume'])
            }
            self.historical_data.append(dp)
            if i >= len(df) - self.chart_data.maxlen:
                self.chart_data.append(dp)
            self._last_bar_time = idx
        if self.historical_data:
            self.current_data = self.historical_data[-1]
            self.last_fetch_ok = True
        logger.info(f"[{self.provider.symbol}] Histórico pronto: {len(self.historical_data)}")

    def start_collection(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._collect_data_loop, name=f"DataCollector-{self.provider.symbol}", daemon=True)
            self.thread.start()
            logger.info(f"[{self.provider.symbol}] Coleta ({self.base_interval}) iniciada")

    def stop_collection(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"[{self.provider.symbol}] Coleta parada")

    def _collect_data_loop(self):
        while self.is_running:
            try:
                df_latest = self.provider.get_latest_df(interval=self.base_interval)
                if df_latest is not None and not df_latest.empty:
                    idx, row = df_latest.index[-1], df_latest.iloc[-1]
                    if self._last_bar_time is None or idx > self._last_bar_time:
                        data = {
                            'timestamp': idx.isoformat(),
                            'price': float(row['close']),
                            'open': float(row['open']), 'high': float(row['high']),
                            'low': float(row['low']), 'close': float(row['close']),
                            'volume': float(row['volume'])
                        }
                        self.current_data = data
                        self.historical_data.append(data)
                        self.chart_data.append(data)
                        self.last_update = datetime.now()
                        self.last_fetch_ok = True
                        self._last_bar_time = idx
                        logger.info(f"[{self.provider.symbol}] {data['price']:.2f}")
                time.sleep(10)
            except Exception as e:
                self.last_fetch_ok = False
                logger.error(f"[{self.provider.symbol}] Loop coleta: {e}")
                time.sleep(30)

    def get_current_data(self): return self.current_data
    def get_historical_data(self): return list(self.historical_data)
    def get_chart_data(self): return list(self.chart_data)

    def get_historical_dataframe(self):
        if not self.historical_data:
            return pd.DataFrame()
        df = pd.DataFrame(self.historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    def get_extended_historical_dataframe(self, period="7d", interval="1m"):
        days = parse_period_days(period, DEFAULT_PARAMS['data']['max_hist_days'])
        return self.provider.get_historical_df(days=days, interval=interval)

# =====================================================================================
# Indicators (fixes: MACD clássico, consistência)
# =====================================================================================
class TechnicalIndicators:
    @staticmethod
    def calculate_indicators(data_list):
        if len(data_list) < 20:
            return {}
        try:
            prices = np.array([d['close'] for d in data_list], dtype=float)
            highs = np.array([d['high'] for d in data_list], dtype=float)
            lows = np.array([d['low'] for d in data_list], dtype=float)
            volumes = np.array([d['volume'] for d in data_list], dtype=float)

            # 👇 NOVO: Padrões Candlestick Simples
            if len(data_list) > 0:
                last_candle = data_list[-1]
                open_price = float(last_candle['open'])
                close_price = prices[-1]
                high_price = highs[-1]
                low_price = lows[-1]

                body = abs(close_price - open_price)
                upper_wick = high_price - max(close_price, open_price)
                lower_wick = min(close_price, open_price) - low_price

                candle_pattern = 0
                # Martelo Bullish: mecha inferior grande + corpo pequeno + fecha no topo
                if body > 0 and lower_wick > body * 2 and close_price > open_price:
                    candle_pattern = -1  # sinal de COMPRA (reversão de fundo)
                # Estrela Cadente Bearish: mecha superior grande + corpo pequeno + fecha no fundo
                elif body > 0 and upper_wick > body * 2 and close_price < open_price:
                    candle_pattern = 1   # sinal de VENDA (reversão de topo)
            else:
                candle_pattern = 0

            # --- Indicadores Originais ---
            rsi = TechnicalIndicators._rsi(prices, 7)

            bb_period, bb_std_dev = 15, 1.8
            if len(prices) >= bb_period:
                win = prices[-bb_period:]
                mid = float(np.mean(win))
                std = float(np.std(win))
            else:
                mid = float(np.mean(prices))
                std = float(np.std(prices))
            upper, lower = mid + std * bb_std_dev, mid - std * bb_std_dev
            bb_pos = (prices[-1] - lower) / (upper - lower) if upper != lower else 0.5
            bb_pos = float(np.clip(bb_pos, 0, 1))

            ema = TechnicalIndicators._ema(prices, 8)
            p_vs_ema = (prices[-1] - ema) / ema if ema != 0 else 0.0
            momentum = (prices[-1] - prices[-4]) / prices[-4] if len(prices) > 4 and prices[-4] != 0 else 0.0

            macd_line, macd_signal, macd_hist = TechnicalIndicators._macd(prices)

            stoch_k, stoch_d = TechnicalIndicators._stoch(highs, lows, prices, 14)

            vol_sma = np.mean(volumes[-10:]) if len(volumes) >= 10 else (volumes[-1] if len(volumes) else 0.0)
            vol_ratio = volumes[-1] / vol_sma if vol_sma != 0 else 1.0

            atr = TechnicalIndicators._atr(highs, lows, prices, 14)
            ema_slope = TechnicalIndicators._ema_slope(prices, 8)

            # --- NOVAS FEATURES: DIVERGÊNCIA + VOLUME DRYING ---

            # 1. Divergência RSI/Preço (janela de 8 velas)
            rsi_series = []
            for i in range(len(prices)):
                window = prices[max(0, i-6):i+1]
                if len(window) < 2:
                    rsi_series.append(50.0)
                else:
                    deltas = np.diff(window)
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    avg_gain = np.mean(gains[-7:]) if len(gains) >= 7 else (np.mean(gains) if len(gains) > 0 else 0.0)
                    avg_loss = np.mean(losses[-7:]) if len(losses) >= 7 else (np.mean(losses) if len(losses) > 0 else 0.0)
                    if avg_loss == 0:
                        rsiv = 100.0
                    else:
                        rs = avg_gain / avg_loss
                        rsiv = 100.0 - (100.0 / (1.0 + rs))
                    rsi_series.append(float(rsiv))

            divergence = 0
            if len(prices) >= 8:
                recent_prices = prices[-8:]
                recent_rsi = rsi_series[-8:]

                price_high_idx = np.argmax(recent_prices)
                rsi_high_idx = np.argmax(recent_rsi)

                price_low_idx = np.argmin(recent_prices)
                rsi_low_idx = np.argmin(recent_rsi)

                # Divergência de baixa: novo high de preço, mas RSI mais baixo
                if price_high_idx == 7 and rsi_high_idx != 7:
                    if recent_rsi[7] < recent_rsi[rsi_high_idx] * 0.95:
                        divergence = 1  # Sinal de venda

                # Divergência de alta: novo low de preço, mas RSI mais alto
                if price_low_idx == 7 and rsi_low_idx != 7:
                    if recent_rsi[7] > recent_rsi[rsi_low_idx] * 1.05:
                        divergence = -1  # Sinal de compra

            # 2. Volume Drying: volume atual < 70% da média dos últimos 5 candles
            volume_drying = 0
            if len(volumes) >= 5:
                vol_avg_5 = np.mean(volumes[-5:-1])  # exclui o atual
                if volumes[-1] < vol_avg_5 * 0.7 and vol_avg_5 > 0:
                    volume_drying = 1

            # --- Retorno Final ---
            return {
                'rsi': float(rsi),
                'bb_upper': float(upper), 'bb_middle': float(mid), 'bb_lower': float(lower),
                'bb_position': float(bb_pos),
                'ema': float(ema), 'price_vs_ema': float(p_vs_ema),
                'momentum': float(momentum),
                'macd': float(macd_line), 'macd_signal': float(macd_signal), 'macd_histogram': float(macd_hist),
                'stoch_k': float(stoch_k), 'stoch_d': float(stoch_d),
                'volume_ratio': float(vol_ratio), 'atr': float(atr),
                'ema_slope': float(ema_slope),

                # 👇 NOVAS FEATURES
                'divergence': int(divergence),      # -1=compra, 0=neutro, 1=venda
                'volume_drying': int(volume_drying), # 1=volume secando, 0=normal
                'candle_pattern': int(candle_pattern)  # -1=buy reversal, 0=neutro, 1=sell reversal
            }

        except Exception as e:
            logger.error(f"Erro no cálculo de indicadores: {e}")
            return {}

    @staticmethod
    def _rsi(prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if np.any(gains) else 0.0
        avg_loss = np.mean(losses[-period:]) if np.any(losses) else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _ema(prices, period=8):
        if len(prices) < 1:
            return 0.0
        if period <= 1 or len(prices) < period:
            return float(np.mean(prices))
        mult = 2.0 / (period + 1.0)
        ema = prices[0]
        for p in prices[1:]:
            ema = (p - ema) * mult + ema
        return float(ema)

    @staticmethod
    def _macd(prices, fast=12, slow=26, signal=9):
        # MACD clássico: EMA12 - EMA26, sinal = EMA9 da linha MACD
        if len(prices) < slow + signal:
            # fallback simples
            ef = TechnicalIndicators._ema(prices, fast)
            es = TechnicalIndicators._ema(prices, slow)
            line = ef - es
            sig = line
            hist = line - sig
            return float(line), float(sig), float(hist)

        def ema_series(values, period):
            alpha = 2.0 / (period + 1.0)
            out = np.zeros_like(values, dtype=float)
            out[0] = values[0]
            for i in range(1, len(values)):
                out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
            return out

        ef_series = ema_series(np.array(prices, dtype=float), fast)
        es_series = ema_series(np.array(prices, dtype=float), slow)
        macd_line_series = ef_series - es_series
        signal_series = ema_series(macd_line_series, signal)
        hist_series = macd_line_series - signal_series
        return float(macd_line_series[-1]), float(signal_series[-1]), float(hist_series[-1])

    @staticmethod
    def _stoch(highs, lows, closes, period=14):
        if len(closes) < period:
            return 50.0, 50.0
        hh = np.max(highs[-period:])
        ll = np.min(lows[-period:])
        c = closes[-1]
        if hh == ll:
            return 50.0, 50.0
        k = 100 * (c - ll) / (hh - ll)
        d = 0.8 * k + 0.2 * 50.0
        return float(np.clip(k, 0, 100)), float(np.clip(d, 0, 100))

    @staticmethod
    def _atr(highs, lows, closes, period=14):
        if len(closes) < 2:
            return float(closes[-1] * 0.01) if len(closes) else 0.0
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])))
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) else float(np.std(closes) * 0.01)
        return float(np.mean(tr[-period:]))

    @staticmethod
    def _ema_slope(prices, period=8):
        if len(prices) < period + 2:
            return 0.0
        mult = 2.0 / (period + 1.0)
        ema = prices[0]
        vals = []
        for p in prices[1:]:
            ema = (p - ema) * mult + ema
            vals.append(ema)
        if len(vals) < 2:
            return 0.0
        denom = max(1e-9, abs(vals[-2]))
        return float((vals[-1] - vals[-2]) / denom)
    
    @staticmethod
    def _detect_divergence(prices, rsi_values, window=10):
        """
        Detecta divergência de baixa (preço faz novo high, RSI não) → sinal de reversão.
        Retorna: 1 = divergência de baixa (venda), -1 = divergência de alta (compra), 0 = neutro.
        """
        if len(prices) < window or len(rsi_values) < window:
            return 0

        recent_prices = prices[-window:]
        recent_rsi = rsi_values[-window:]

        price_high_idx = np.argmax(recent_prices)
        rsi_high_idx = np.argmax(recent_rsi)

        price_low_idx = np.argmin(recent_prices)
        rsi_low_idx = np.argmin(recent_rsi)

        # Divergência de baixa: preço > anterior, RSI < anterior
        if price_high_idx == len(recent_prices) - 1 and rsi_high_idx != len(recent_rsi) - 1:
            if recent_rsi[-1] < recent_rsi[rsi_high_idx] * 0.95:  # RSI caiu 5%+
                return 1  # Sinal de venda (divergência de baixa)

        # Divergência de alta: preço < anterior, RSI > anterior
        if price_low_idx == len(recent_prices) - 1 and rsi_low_idx != len(recent_rsi) - 1:
            if recent_rsi[-1] > recent_rsi[rsi_low_idx] * 1.05:  # RSI subiu 5%+
                return -1  # Sinal de compra (divergência de alta)

        return 0

# =====================================================================================
# MultiTimeframe Feature Builder (com microestrutura + robustez)
# =====================================================================================
class MultiTimeframeFeatureBuilder:
    def resample_ohlc(self, df, minutes):
        if df is None or df.empty:
            return pd.DataFrame()
        rule = f'{minutes}T'
        ohlc = df[['open', 'high', 'low', 'close']].resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        })
        vol = df['volume'].resample(rule).sum()
        out = pd.concat([ohlc, vol.to_frame('volume')], axis=1).dropna(how='all')
        return out

    def indicators_df(self, df,
                    rsi_period=7, bb_period=15, ema_period=8, stoch_period=14,
                    atr_period=14, vol_sma_period=10):
        if df is None or df.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=df.index.copy())
        c = df['close'].astype(float).values
        h = df['high'].astype(float).values
        l = df['low'].astype(float).values
        v = df['volume'].astype(float).values

        s = pd.Series(c, index=df.index)
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(rsi_period, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(rsi_period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        out['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0).clip(0, 100)

        mid = s.rolling(bb_period, min_periods=2).mean()
        std = s.rolling(bb_period, min_periods=2).std()
        upper = mid + std * 1.8
        lower = mid - std * 1.8
        out['bb_pos'] = ((s - lower) / (upper - lower)).clip(0, 1).fillna(0.5)

        ema = s.ewm(span=ema_period, adjust=False, min_periods=1).mean()
        out['ema'] = ema.values
        out['price_vs_ema'] = ((s - ema) / ema.replace(0, np.nan)).fillna(0.0)

        ema_fast = s.ewm(span=12, adjust=False, min_periods=1).mean()
        ema_slow = s.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        out['macd_hist'] = (macd - sig).values

        hs = pd.Series(h, index=df.index)
        ls = pd.Series(l, index=df.index)
        hh = hs.rolling(stoch_period, min_periods=2).max()
        ll = ls.rolling(stoch_period, min_periods=2).min()
        out['stoch_k'] = (100 * (s - ll) / (hh - ll)).replace([np.inf, -np.inf], np.nan).clip(0, 100).fillna(50.0)

        vol_sma = pd.Series(v, index=df.index).rolling(vol_sma_period, min_periods=1).mean()
        out['volume_ratio'] = (pd.Series(v, index=df.index) / vol_sma.replace(0, np.nan)).fillna(1.0)

        hl = hs - ls
        hc = (hs - s.shift(1)).abs()
        lc = (ls - s.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        out['atr'] = tr.rolling(atr_period, min_periods=2).mean().bfill()

        out['realized_vol'] = s.pct_change().rolling(30, min_periods=5).std()
        out['ema_slope'] = ema.pct_change().fillna(0.0)

        # microestrutura
        o = df['open'].astype(float)
        body = (s - o).fillna(0.0)
        rng = (hs - ls).replace(0, np.nan)
        upper_wick = (hs - s).clip(lower=0).fillna(0.0)
        lower_wick = (s - ls).clip(lower=0).fillna(0.0)
        out['body'] = body
        out['body_pct'] = (body / s.replace(0, np.nan)).fillna(0.0)
        out['range'] = (hs - ls).fillna(0.0)
        out['clv'] = ((s - ls) - (hs - s)) / (hs - ls).replace(0, np.nan)
        out['upper_wick_ratio'] = (upper_wick / rng).fillna(0.0)
        out['lower_wick_ratio'] = (lower_wick / rng).fillna(0.0)
        out['ret_1'] = s.pct_change().fillna(0.0)
        out['ret_2'] = s.pct_change(2).fillna(0.0)
        out['ret_3'] = s.pct_change(3).fillna(0.0)
        out['ret_z20'] = (out['ret_1'] - out['ret_1'].rolling(20).mean()) / (out['ret_1'].rolling(20).std().replace(0, np.nan))
        out['ret_z20'] = out['ret_z20'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out['vol_z20'] = (pd.Series(v, index=df.index) - pd.Series(v, index=df.index).rolling(20).mean()) / \
                        (pd.Series(v, index=df.index).rolling(20).std().replace(0, np.nan))
        out['vol_z20'] = out['vol_z20'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out['atr_n'] = out['atr'] / s.replace(0, np.nan)
        out['atr_n'] = out['atr_n'].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Volume drying at highs
        vol_rolling = pd.Series(v, index=df.index).rolling(5, min_periods=1).mean()
        vol_ratio_5 = pd.Series(v, index=df.index) / vol_rolling.replace(0, np.nan)
        out['volume_drying'] = (vol_ratio_5 < 0.7).astype(int)  # True se volume caiu 30%+ nas últimas 5 velas

        # ================================
        # 👇 NOVAS FEATURES: ORDER FLOW 👇
        # ================================

        # 1. Volume Imbalance: mede se o volume está vindo com força compradora ou vendedora
        price_change = s.diff()
        volume_direction = np.sign(price_change) * v  # positivo = volume com alta, negativo = com baixa
        out['volume_imbalance'] = (
            pd.Series(volume_direction, index=df.index)
            .rolling(20, min_periods=5)
            .sum()
            .fillna(0.0)
        )

        # 2. VPIN Proxy (Volume-Synchronized Probability of Informed Trading) — simplificado
        # Mede desequilíbrio entre "compra agressiva" e "venda agressiva"
        # Aqui usamos: sinal do retorno * volume como proxy de fluxo informado
        signed_volume = price_change * v  # positivo = compra dominante, negativo = venda dominante
        total_volume = pd.Series(v, index=df.index).rolling(20, min_periods=5).sum()
        out['vpin_proxy'] = (
            (pd.Series(signed_volume, index=df.index).rolling(20, min_periods=5).sum())
            / (total_volume.replace(0, np.nan))
        ).fillna(0.0).clip(-1, 1)

        # 3. Spread Proxy: quanto maior o range relativo, menos líquido o candle (maior slippage potencial)
        out['spread_proxy'] = ((hs - ls) / s.replace(0, np.nan)).fillna(0.0)


                # ================================
        # 👇 NOVAS FEATURES AVANÇADAS DE MICROESTRUTURA 👇
        # ================================

        # 1. Order Book Imbalance (Proxy)
        close = s
        open_price = df['open'].astype(float)
        high = hs
        low = ls
        buy_pressure = (close - low) / (high - low + 1e-9)  # 0 a 1: quanto mais perto do topo, mais compra
        sell_pressure = (high - close) / (high - low + 1e-9) # 0 a 1: quanto mais perto do fundo, mais venda
        out['orderbook_imbalance'] = buy_pressure - sell_pressure  # -1 a +1

        # 2. Order Flow (Aggregated Buy/Sell Volume)
        volume_direction = np.where(close > open_price, v, -v)  # positivo = volume comprador, negativo = vendedor
        agg_buy_vol = pd.Series(np.where(volume_direction > 0, v, 0), index=df.index)
        agg_sell_vol = pd.Series(np.where(volume_direction < 0, v, 0), index=df.index)
        out['agg_buy_vol'] = agg_buy_vol.rolling(20, min_periods=5).sum().fillna(0.0)
        out['agg_sell_vol'] = agg_sell_vol.rolling(20, min_periods=5).sum().fillna(0.0)
        total_vol_20 = pd.Series(v, index=df.index).rolling(20, min_periods=5).sum().replace(0, np.nan)
        out['agg_buy_vol_ratio'] = (out['agg_buy_vol'] / total_vol_20).fillna(0.5).clip(0, 1)

        # 3. VPIN / Flow Toxicity (Proxy) — já existe, mas vamos manter consistente
        signed_volume = price_change * v
        rolling_signed_vol = pd.Series(signed_volume, index=df.index).rolling(20, min_periods=5).sum()
        rolling_total_vol = pd.Series(v, index=df.index).rolling(20, min_periods=5).sum()
        out['vpin_proxy'] = (rolling_signed_vol.abs() / (rolling_total_vol.replace(0, np.nan))).fillna(0.0).clip(0, 1)

        # 4. Features de Volatilidade Local
        out['local_vol_5'] = s.pct_change().rolling(5, min_periods=3).std().fillna(0.0)
        out['local_vol_10'] = s.pct_change().rolling(10, min_periods=5).std().fillna(0.0)
        out['local_vol_20'] = s.pct_change().rolling(20, min_periods=10).std().fillna(0.0)
        vol_ma_20 = out['local_vol_20'].rolling(20, min_periods=10).mean()
        vol_std_20 = out['local_vol_20'].rolling(20, min_periods=10).std()
        out['local_vol_zscore'] = ((out['local_vol_20'] - vol_ma_20) / (vol_std_20.replace(0, np.nan))).fillna(0.0)
        out['vol_ratio_5_20'] = (out['local_vol_5'] / (out['local_vol_20'].replace(0, np.nan))).fillna(1.0).clip(0.1, 5.0)


        # ================================
        # 👆 FIM DAS NOVAS FEATURES 👆
        # ================================

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        for col in out.columns:
            if out[col].isna().any():
                median = out[col].median()
                out[col] = out[col].fillna(median if pd.notna(median) else 0.0)
        return out

    def build_mtf_features(self, df, timeframes=[5, 15]):
        if df is None or df.empty:
            return pd.DataFrame()
        f1 = self.indicators_df(df).add_prefix('f1m_')
        if f1.empty:
            return pd.DataFrame()
        merged = f1.reset_index().rename(columns={'index': 'timestamp'})
        for tf in timeframes:
            if tf == 1:
                continue
            dfr = self.resample_ohlc(df, tf)
            if dfr is None or dfr.empty:
                continue
            fi = self.indicators_df(dfr).add_prefix(f'f{tf}m_')
            if fi is None or fi.empty:
                continue
            merged = pd.merge_asof(
                merged.sort_values('timestamp'),
                fi.reset_index().rename(columns={'index': 'timestamp'}).sort_values('timestamp'),
                on='timestamp', direction='backward'
            )
        features_df = merged.set_index('timestamp').sort_index()
        features_df = features_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        for col in features_df.columns:
            if features_df[col].isna().any():
                median = features_df[col].median()
                features_df[col] = features_df[col].fillna(median if pd.notna(median) else 0.0)
        return features_df.loc[:, ~features_df.columns.duplicated()]

    def build_ml_dataset(self, df, horizon=3, timeframes=[5, 15], use_triple_barrier=True,
                        atr_mult_stop=1.5, atr_mult_target=2.5):
        feats = self.build_mtf_features(df, timeframes=timeframes)
        if feats is None or feats.empty:
            feats = self.indicators_df(df).add_prefix('f1m_').replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if feats is None or feats.empty:
            return np.empty((0, 0)), np.array([], dtype=int), [], pd.DataFrame(), pd.DataFrame(), None

        base = df.loc[feats.index].copy()
        aligned = pd.concat([base, feats], axis=1)
        aligned = aligned.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
        if aligned.empty:
            return np.empty((0, 0)), np.array([], dtype=int), [], pd.DataFrame(), pd.DataFrame(), None

        # 👇 NOVO: Teste de Estacionariedade (ADF) em cada feature
        stationary_features = []
        non_stationary_features = []

        if STATS_AVAILABLE:
            for col in feats.columns:
                try:
                    series = aligned[col].dropna()
                    if len(series) < 20:
                        continue
                    result = adfuller(series, autolag='AIC')
                    p_value = result[1]
                    if p_value < 0.05:  # estacionária ao nível de 5%
                        stationary_features.append(col)
                    else:
                        non_stationary_features.append(col)
                except Exception as e:
                    logger.debug(f"Erro no teste ADF para {col}: {e}")
                    stationary_features.append(col)  # fallback: mantém se der erro

            if non_stationary_features:
                logger.info(f"[ADF Filter] Features não estacionárias removidas ({len(non_stationary_features)}): {non_stationary_features[:5]}{'...' if len(non_stationary_features) > 5 else ''}")
                # 👇 ADICIONE AO METRICS
                if hasattr(self, '_last_metrics'):
                    self._last_metrics['non_stationary_removed'] = len(non_stationary_features)
            else:
                if hasattr(self, '_last_metrics'):
                    self._last_metrics['non_stationary_removed'] = 0
            if stationary_features:
                feats = feats[stationary_features]
            else:
                logger.warning("Nenhuma feature passou no teste de estacionariedade. Mantendo todas.")
        else:
            logger.debug("Teste de estacionariedade ignorado (statsmodels indisponível).")

        # Continua normalmente
        if use_triple_barrier:
            y, sample_weights = self._triple_barrier_labels(aligned, horizon=horizon,
                                                            atr_col='f1m_atr', atr_mult_stop=atr_mult_stop,
                                                            atr_mult_target=atr_mult_target)
        else:
            future = aligned['close'].shift(-horizon)
            y = (future > aligned['close']).astype(int).values
            sample_weights = None

        y = y[:len(aligned)]
        mask = ~pd.isna(y)
        y = y[mask]
        X = aligned.loc[mask, feats.columns].values
        sample_weights = sample_weights[mask] if sample_weights is not None else None

        metrics = {
        'non_stationary_removed': len(non_stationary_features) if non_stationary_features else 0
        }

        return X, y.astype(int), list(feats.columns), feats, aligned.loc[mask], sample_weights, metrics

    def _triple_barrier_labels(self, aligned_df, horizon=5, atr_col='f1m_atr',
                            atr_mult_stop=2.0, atr_mult_target=2.0):
        """
        Retorna rótulos binários (0/1) E vetor de confiança (0..1) baseado em:
        - Velocidade de atingir barreira (quanto mais rápido, mais confiável)
        - Magnitude do movimento relativo ao preço de entrada
        """
        close = aligned_df['close'].values
        atr = aligned_df[atr_col].values if atr_col in aligned_df.columns else np.full_like(close, np.std(close) * 0.01)
        n = len(close)
        labels = np.zeros(n, dtype=float) * np.nan
        confidences = np.zeros(n, dtype=float) * np.nan  # ← NOVO: vetor de confiança

        min_volatility = 0.001  # mínimo relativo ao preço para evitar barreiras muito apertadas

        for i in range(n):
            entry = close[i]
            # Garante volatilidade mínima para definir barreiras
            volatility_buffer = max(atr[i], min_volatility * entry)
            up = entry + atr_mult_target * volatility_buffer
            dn = entry - atr_mult_stop * volatility_buffer
            end = min(n - 1, i + horizon)

            outcome = None
            barrier_hit_time = horizon  # por padrão, assume que levou todo o horizonte

            # Verifica barreiras em sequência
            for j in range(i + 1, end + 1):
                if close[j] >= up:
                    outcome = 1
                    barrier_hit_time = j - i
                    break
                if close[j] <= dn:
                    outcome = 0
                    barrier_hit_time = j - i
                    break

            # Se nenhuma barreira foi atingida, usa fechamento no final do horizonte
            if outcome is None:
                outcome = 1 if close[end] > entry else 0
                barrier_hit_time = horizon

            labels[i] = outcome

            # Calcula confiança:
            # 1. Baseado na velocidade: quanto mais rápido, mais confiável
            speed_conf = 1.0 - (barrier_hit_time / max(1, horizon))  # ex: hit em 1 de 5 → 0.8

            # 2. Baseado na magnitude do movimento (normalizado pelo preço)
            magnitude = abs(close[i + barrier_hit_time if i + barrier_hit_time < n else end] - entry) / entry
            mag_conf = min(magnitude / 0.02, 1.0)  # normaliza até 2% de movimento (ajustável)

            # Combina os dois fatores
            confidence_score = (speed_conf * 0.6) + (mag_conf * 0.4)
            confidences[i] = np.clip(confidence_score, 0.0, 1.0)

        return labels, confidences  # ← AGORA RETORNA DOIS VETORES
    
    def latest_feature_row(self, df, timeframes=[5,15]):
            feats = self.build_mtf_features(df, timeframes=timeframes)
            if feats is None or feats.empty:
                return None, []
            row = feats.iloc[-1].to_dict()
            return row, feats.columns.tolist()

# =====================================================================================
# Filters
# =====================================================================================

class MarketRegimeFilter:
    """
    Avalia o regime de mercado global usando BTC como referência.
    Retorna um score de 0.0 (bear forte) a 1.0 (bull forte).
    Usado para ajustar a confiança dos sinais em altcoins.
    """
    def __init__(self, bot_manager):
        self.bot_manager = bot_manager
        self.last_check = None
        self.cache_duration_sec = 60  # atualiza a cada 1 minuto
        self.cached_regime_score = 0.5

    def get_market_regime_score(self):
        now = datetime.now()
        if self.last_check and (now - self.last_check).total_seconds() < self.cache_duration_sec:
            return self.cached_regime_score

        try:
            btc_status = self.bot_manager.status('BTC')
            if not btc_status.get('success') or 'signal' not in btc_status:
                self.cached_regime_score = 0.5
                self.last_check = now
                return 0.5

            sig = btc_status['signal']
            strength = sig.get('strength', 0)
            signal_type = sig.get('signal', 'HOLD')

            if signal_type == 'BUY' and strength >= 70:
                score = 1.0  # bull forte
            elif signal_type == 'SELL' and strength >= 70:
                score = 0.0  # bear forte
            elif signal_type == 'BUY' and strength >= 55:
                score = 0.7  # bull moderado
            elif signal_type == 'SELL' and strength >= 55:
                score = 0.3  # bear moderado
            else:
                score = 0.5  # neutro

            self.cached_regime_score = score
            self.last_check = now
            return score

        except Exception as e:
            logger.warning(f"Erro ao obter regime de mercado: {e}")
            return 0.5



class VolatilityVolumeFilter:
    @staticmethod
    def evaluate(df_1m, params):
        if df_1m is None or df_1m.empty:
            return True, 1.0, ["Filtro: sem dados"]

        p = params.get('vol_filters', DEFAULT_PARAMS['vol_filters'])
        penalty_weights = p.get('penalty_weights', {
            'volatility': 0.6,
            'volume': 0.7,
            'realized_vol': 0.8
        })

        # 👇 NOVO: Configuração de volume mínimo por ticker (opcional)
        min_volume_config = p.get('min_volume_usd_per_ticker', {
            'BTCUSDT': 800000,   # Bitcoin: altíssima liquidez exigida
            'ETHUSDT': 600000,   # Ethereum: também exige segurança máxima
            'BNBUSDT': 300000,   # BNB: bom volume, reduzimos um pouco
            'SOLUSDT': 250000,   # Solana: ativo sólido, volume robusto
            'AVAXUSDT': 150000,  # Avalanche: crescente, volume médio
            'LINKUSDT': 120000,  # Chainlink: bom volume institucional
            'ADAUSDT': 100000,   # Cardano: comunidade grande, volume variável
            'XRPUSDT': 90000,    # Ripple: judicial, mas volume ainda alto
            'MATICUSDT': 80000,  # Polygon: ativo de layer2, volume bom
            'DOGEUSDT': 70000,   # Dogecoin: memecoin, mas ainda com volume considerável
            'default': 100000    # Fallback seguro para qualquer outro
        },
        )

        # 👇 Tenta identificar o ticker do DataFrame (se disponível via atributo)
        ticker = getattr(df_1m, 'ticker', 'default')
        min_volume_threshold = min_volume_config.get(ticker, min_volume_config.get('default', 100000))

        win_atr = p.get('atr_window', 14)
        min_len = max(50, win_atr + 5, p.get('realized_vol_window', 30) + 5, 24)
        if len(df_1m) < min_len:
            return True, 1.0, ["Filtro: dados insuficientes"]

        h, l, c = df_1m['high'].values, df_1m['low'].values, df_1m['close'].values
        s = pd.Series(c, index=df_1m.index)
        hs, ls = pd.Series(h, index=df_1m.index), pd.Series(l, index=df_1m.index)
        hl = hs - ls
        hc = (hs - s.shift(1)).abs()
        lc = (ls - s.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_ser = tr.rolling(win_atr, min_periods=2).mean().bfill()

        if atr_ser.empty:
            return True, 1.0, ["Filtro: ATR indisponível"]

        atr_cur = float(atr_ser.iloc[-1])

        # 👇 MELHORIA 1: Janela adaptativa para percentis (últimos 12h = 720 candles em 1m)
        recent_atr = atr_ser.tail(720)  # últimos 12h
        clean_recent = recent_atr.dropna()
        if len(clean_recent) < 50:      # se poucos dados recentes, usa histórico completo
            clean_recent = atr_ser.dropna()
        if len(clean_recent) < 5:
            return True, 1.0, ["Filtro: ATR insuficiente para cálculo de percentis"]

        atr_min = float(np.percentile(clean_recent, p['atr_percentile_min']))
        atr_max = float(np.percentile(clean_recent, p['atr_percentile_max']))

        # 👇 MELHORIA 2: Volume ratio baseado em mediana dos últimos 60m (não média de 10m)
        vol_median = df_1m['volume'].rolling(60, min_periods=10).median()  # resistente a spikes
        vol_ref = float(vol_median.iloc[-1]) if not math.isnan(vol_median.iloc[-1]) else 1.0
        vol_ratio = float(df_1m['volume'].iloc[-1] / max(1e-9, vol_ref))

        # 👇 MELHORIA 3: Realized Vol com múltiplas janelas + mediana para robustez
        rv1 = s.pct_change().rolling(10, min_periods=5).std().iloc[-1]   # curto prazo
        rv2 = s.pct_change().rolling(30, min_periods=5).std().iloc[-1]   # médio prazo
        rv3 = s.pct_change().rolling(120, min_periods=20).std().iloc[-1] # longo prazo
        rv_list = [x for x in [rv1, rv2, rv3] if not np.isnan(x)]
        rv = float(np.median(rv_list)) if rv_list else 0.0

        # 👇 NOVO: Filtro de Liquidez Mínima em USD
        current_volume_usd = float(df_1m['volume'].iloc[-1]) * float(df_1m['close'].iloc[-1])
        pass_liquidity = current_volume_usd >= min_volume_threshold

        # Limites configuráveis
        pass_vol = bool(atr_min <= atr_cur <= atr_max)
        pass_volume = bool(p.get('volume_ratio_min', 0.9) <= vol_ratio <= p.get('volume_ratio_max', 3.0))
        pass_rv = bool(p.get('realized_vol_min', 0.0004) <= rv <= p.get('realized_vol_max', 0.02))

        reasons = []
        mult = 1.0

        # 👇 MELHORIA 4: Penalidades configuráveis + motivos detalhados
        if not pass_vol:
            reasons.append(f"ATR {atr_cur:.6f} ∉ [{atr_min:.6f}, {atr_max:.6f}]")
            mult *= penalty_weights['volatility']
        if not pass_volume:
            reasons.append(f"VolRatio {vol_ratio:.2f} ∉ [{p.get('volume_ratio_min')}, {p.get('volume_ratio_max')}]")
            mult *= penalty_weights['volume']
        if not pass_rv:
            reasons.append(f"RV {rv:.6f} ∉ [{p.get('realized_vol_min')}, {p.get('realized_vol_max')}]")
            mult *= penalty_weights['realized_vol']

        # 👇 NOVO: Penalidade forte se liquidez insuficiente
        if not pass_liquidity:
            reasons.append(f"Liquidez: ${current_volume_usd:,.0f} < ${min_volume_threshold:,.0f}")
            mult *= 0.1  # penaliza fortemente, mas NÃO BLOQUEIA totalmente (permite trades em memecoins com warning)
            # Não setamos pass_liquidity=False no "pass_all", pois queremos permitir trades em DOGE/SHIB com penalidade

        if all([pass_vol, pass_volume, pass_rv]):
            reasons.append("✅ Filtros OK")

        pass_all = pass_vol and pass_volume and pass_rv
        final_mult = float(np.clip(mult, 0.2, 1.0))
        return pass_all, final_mult, reasons

# =====================================================================================
# Regime Advisor (heurístico) - fix RSI votes e MACD já corrigido no indicador
# =====================================================================================
class RegimeAdvisor:
    def __init__(self, feature_builder: MultiTimeframeFeatureBuilder):
        self.fb = feature_builder

    def _tf_votes(self, fi_last: pd.Series):
        long_votes = 0.0
        short_votes = 0.0
        pve = float(fi_last.get('price_vs_ema', 0.0))
        mh = float(fi_last.get('macd_hist', 0.0))
        rsi = float(fi_last.get('rsi', 50.0))
        # price vs ema
        if pve > 0:
            long_votes += 1
        elif pve < 0:
            short_votes += 1
        # macd hist
        if mh > 0:
            long_votes += 1
        elif mh < 0:
            short_votes += 1
        # rsi (votos simétricos e suaves)
        if rsi >= 55:
            long_votes += 0.5
        elif rsi <= 45:
            short_votes += 0.5
        return long_votes, short_votes

    def evaluate(self, df_1m: pd.DataFrame):
        if df_1m is None or df_1m.empty or len(df_1m) < 60:
            return {
                'profile': 'BALANCED',
                'diagnostics': ['Dados insuficientes'],
                'trend_score': 0.5, 'rv_pct': 50.0, 'bb_width_pct': 50.0,
                'alignment': {'votes_long': 0, 'votes_short': 0, 'used_tfs': []}
            }
        diagnostics = []
        tfs = [1, 5, 15]
        fi_map, used = {}, []
        fi_1 = self.fb.indicators_df(df_1m)
        if fi_1 is not None and not fi_1.empty:
            fi_map[1] = fi_1.iloc[-1]
            used.append(1)
        d5 = self.fb.resample_ohlc(df_1m, 5)
        fi_5 = self.fb.indicators_df(d5) if d5 is not None and not d5.empty else None
        if fi_5 is not None and not fi_5.empty:
            fi_map[5] = fi_5.iloc[-1]
            used.append(5)
        d15 = self.fb.resample_ohlc(df_1m, 15)
        fi_15 = self.fb.indicators_df(d15) if d15 is not None and not d15.empty else None
        if fi_15 is not None and not fi_15.empty:
            fi_map[15] = fi_15.iloc[-1]
            used.append(15)

        long_votes = 0.0
        short_votes = 0.0
        for tf in used:
            lv, sv = self._tf_votes(fi_map[tf])
            long_votes += lv
            short_votes += sv
        total_votes = long_votes + short_votes if (long_votes + short_votes) > 0 else 1.0
        trend_score = np.clip(long_votes / total_votes, 0, 1)

        rv_series = df_1m['close'].pct_change().rolling(30, min_periods=5).std().dropna()
        rv_tail = rv_series.tail(400)
        if rv_tail.empty:
            rv_pct = 50.0
        else:
            rv_cur = rv_tail.iloc[-1]
            rv_pct = float((rv_tail <= rv_cur).sum() / len(rv_tail) * 100.0)

        s = df_1m['close']
        mid = s.rolling(20, min_periods=5).mean()
        std = s.rolling(20, min_periods=5).std()
        upper = mid + std * 2
        lower = mid - std * 2
        bb_width = (upper - lower).dropna()
        bb_tail = bb_width.tail(400)
        bb_width_cur = bb_tail.iloc[-1] if not bb_tail.empty else np.nan
        if bb_tail.empty or pd.isna(bb_width_cur):
            bb_width_pct = 50.0
        else:
            bb_width_pct = float((bb_tail <= bb_width_cur).sum() / len(bb_tail) * 100.0)

        try:
            rsi_last = float(fi_1['rsi'].iloc[-1]) if not fi_1.empty else 50.0
            bbp_last = float(fi_1['bb_pos'].iloc[-1]) if not fi_1.empty else 0.5
        except Exception:
            rsi_last, bbp_last = 50.0, 0.5

        is_range = (0.40 <= bbp_last <= 0.60) and (40 <= rsi_last <= 60) and (0.40 <= trend_score <= 0.60) and (bb_width_pct <= 55)
        is_breakout = (bb_width_pct >= 70) and (trend_score <= 0.35 or trend_score >= 0.65)
        is_choppy = (abs(trend_score - 0.5) <= 0.10) and (rv_pct >= 40) and (45 <= bb_width_pct <= 75)

        profile = 'BALANCED'
        if len(used) >= 2:
            if trend_score >= 0.70 or trend_score <= 0.30:
                profile = 'MTF_HEAVY'
                diagnostics.append("Alinhamento MTF forte")
            if is_breakout:
                profile = 'PREDICTIVE'
                diagnostics.append("Expansão BB/breakout")
            if is_range:
                profile = 'MEAN_REVERT'
                diagnostics.append("Range detectado (RSI~50, BB no meio)")
            if is_choppy:
                profile = 'RULE_HEAVY'
                diagnostics.append("Choppy/contraditório; prioriza regras+consenso")
        else:
            diagnostics.append("MTF insuficiente; Balanceado")

        return {
            'profile': profile,
            'trend_score': float(trend_score),
            'rv_pct': float(rv_pct),
            'bb_width_pct': float(bb_width_pct),
            'alignment': {'votes_long': float(long_votes), 'votes_short': float(short_votes), 'used_tfs': used},
            'diagnostics': diagnostics
        }

# =====================================================================================
# ML Model (com triple-barrier, features micro, RandomizedSearchCV, calibração, gating)
# =====================================================================================
class MLPatternModel:
    def __init__(self, params=None, model_id="DEFAULT"):
        self.params = params or DEFAULT_PARAMS['ml']
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.selected_features = None
        self.trained = False
        self.lock = threading.Lock()
        self.metrics = {}
        self.quality = 0.7
        self.quality_smooth = 0.7
        self.last_trained = None
        self.calibration = 'sigmoid'
        self.model_id = model_id
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.model_path = self.models_dir / f"{self.model_id}_rf.joblib"
        self._max_jobs = max(1, min(4, (os.cpu_count() or 2)))  # Mac M1 friendly
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn não disponível. ML desativado.")
        else:
            self._load_if_exists()

    def _load_if_exists(self):
        try:
            if self.model_path.exists():
                obj = joblib.load(self.model_path)
                self.model = obj.get('model')
                self.scaler = obj.get('scaler')
                self.selector = obj.get('selector')
                self.feature_names = obj.get('feature_names')
                self.selected_features = obj.get('selected_features')
                self.trained = True
                self.metrics = obj.get('metrics', {})
                self.quality = obj.get('quality', 0.7)
                self.quality_smooth = obj.get('quality_smooth', self.quality)
                self.last_trained = obj.get('last_trained')
                self.calibration = obj.get('calibration', 'sigmoid')
                logger.info(f"[ML {self.model_id}] Modelo carregado de {self.model_path}")
        except Exception as e:
            logger.warning(f"[ML {self.model_id}] Falha ao carregar modelo: {e}")

    def _save(self):
        try:
            payload = {
                'model': self.model,
                'scaler': self.scaler,
                'selector': self.selector,
                'feature_names': self.feature_names,
                'selected_features': self.selected_features,
                'metrics': self.metrics,
                'quality': self.quality,
                'quality_smooth': self.quality_smooth,
                'last_trained': self.last_trained,
                'calibration': self.calibration,
            }
            joblib.dump(payload, self.model_path)
        except Exception as e:
            logger.warning(f"[ML {self.model_id}] Falha ao salvar modelo: {e}")

    def _compute_quality(self, cv_auc, hold_auc, oob_auc):
        # mapeia AUC->qualidade [0..1], com pesos
        parts = []
        if hold_auc is not None:
            parts.append(hold_auc)
        if cv_auc is not None:
            parts.append(cv_auc)
        if oob_auc is not None:
            parts.append(oob_auc)
        if not parts:
            return 0.5
        avg_auc = float(np.nanmean(parts))
        # qualidade ~ (AUC - 0.5) escalada e clipada
        q = float(np.clip((avg_auc - 0.5) * 2.0, 0.0, 1.0))
        # suavizar com pequeno bônus se todas acima de 0.55
        if all([(p or 0) >= 0.55 for p in parts]):
            q = float(np.clip(q + 0.05, 0.0, 1.0))
        return q

    def _calibration_method(self, n_samples):
        # isotonic é mais pesado; usar sigmoid por padrão, isotonic se dataset pequeno
        return 'isotonic' if n_samples <= 5000 else 'sigmoid'

    def train(self, X, y, feature_names, sample_weights=None, metrics=None):
        if not SKLEARN_AVAILABLE:
            self.trained = False
            return {'sklearn_available': False}

        with self.lock:
            try:
                if X is None or len(X) == 0 or y is None or len(y) == 0:
                    self.trained = False
                    return {'trained': False, 'error': 'Dataset vazio'}
                if len(np.unique(y)) < 2:
                    self.trained = False
                    return {'trained': False, 'error': 'Labels insuficientes'}

                n = len(X)
                # Bound dataset size to keep CPU/RAM light on M1
                max_n = 20000
                if n > max_n:
                    X = X[-max_n:]
                    y = y[-max_n:]
                    if sample_weights is not None:
                        sample_weights = sample_weights[-max_n:]
                    n = max_n

                self.feature_names = feature_names

                # Split holdout para calibração: últimos 20%
                holdout_ratio = 0.2 if n >= 500 else 0.15
                split_idx = max(10, int(n * (1 - holdout_ratio)))
                X_train, y_train = X[:split_idx], y[:split_idx]
                X_hold, y_hold = X[split_idx:], y[split_idx:]

                sw_train = sample_weights[:split_idx] if sample_weights is not None else None
                sw_hold = sample_weights[split_idx:] if sample_weights is not None else None

                # Escalonador (mesmo que árvores não precisem, mantém pipeline consistente)
                self.scaler = StandardScaler()
                Xs_train = self.scaler.fit_transform(X_train)
                Xs_hold = self.scaler.transform(X_hold)

                # Seleção de features por MI (leve)
                k_select = min(40, max(10, int(Xs_train.shape[1] * 0.5)))
                try:
                    self.selector = SelectKBest(mutual_info_classif, k=k_select)
                    Xsel_train = self.selector.fit_transform(Xs_train, y_train)
                    Xsel_hold = self.selector.transform(Xs_hold)
                    self.selected_features = [f for f, m in zip(self.feature_names, self.selector.get_support()) if m]
                except Exception:
                    self.selector = None
                    Xsel_train = Xs_train
                    Xsel_hold = Xs_hold
                    self.selected_features = self.feature_names[:]

                # TSS para CV
                n_splits = 5 if n >= 2000 else (4 if n >= 1200 else (3 if n >= 600 else 2))
                tss = TimeSeriesSplit(n_splits=n_splits)

                # Hiperparam busca leve + regularização embutida
                rf_base = RandomForestClassifier(
                    n_estimators=self.params.get('n_estimators', 150),
                    max_depth=self.params.get('max_depth', 6),
                    min_samples_leaf=self.params.get('min_samples_leaf', 5),
                    min_samples_split=5,           # ← EVITA SPLIT EM GRUPOS MINÚSCULOS
                    max_features='sqrt',           # ← REDUZ VARIÂNCIA (REGULARIZAÇÃO)
                    ccp_alpha=0.001,               # ← PODA ÁRVORES COMPLEXAS (COST-COMPLEXITY PRUNING)
                    random_state=self.params.get('random_state', 42),
                    class_weight='balanced_subsample',
                    bootstrap=True,
                    n_jobs=self._max_jobs,
                    oob_score=True
                )

                param_distributions = {
                    'n_estimators': randint(80, 180),
                    'max_depth': randint(4, 10),
                    'min_samples_leaf': randint(2, 8),
                    'min_samples_split': randint(3, 10),   # ← AJUSTÁVEL NA BUSCA
                    'max_features': uniform(0.5, 0.5),     # 0.5..1.0
                    'ccp_alpha': uniform(0.0001, 0.005)    # ← AJUSTE FINO DE PRUNING
                }

                search = RandomizedSearchCV(
                    rf_base,
                    param_distributions=param_distributions,
                    n_iter=12,
                    cv=tss,
                    scoring='roc_auc',
                    random_state=self.params.get('random_state', 42),
                    n_jobs=1,  # para não estourar CPU no M1
                    verbose=0
                )

                # Treina com pesos se disponíveis
                fit_params = {}
                if sw_train is not None and len(sw_train) == len(y_train):
                    fit_params['sample_weight'] = sw_train

                search.fit(Xsel_train, y_train, **fit_params)
                best_rf = search.best_estimator_

                # Métricas de CV
                cv_auc = search.best_score_
                # treino final no conjunto de treino para checar overfit (AUC treino via predição OOB ou holdin)
                try:
                    if hasattr(best_rf, 'oob_decision_function_') and best_rf.oob_decision_function_ is not None:
                        oob_proba = best_rf.oob_decision_function_[:, 1]
                        oob_auc = float(roc_auc_score(y_train, oob_proba, sample_weight=sw_train)) if sw_train is not None else float(roc_auc_score(y_train, oob_proba))
                    else:
                        oob_auc = None
                except Exception:
                    oob_auc = None

                # Calibração de probabilidade usando holdout
                self.calibration = self._calibration_method(len(X))

                if len(y_hold) == 0 or Xsel_hold.shape[0] == 0:
                    logger.error("Holdout set vazio, não é possível calibrar.")
                    return {'trained': False, 'error': 'Holdout set vazio.'}
                if len(np.unique(y_hold)) < 2:
                    logger.error("Holdout set does not have at least 2 classes, skipping calibration.")
                    return {'trained': False, 'error': 'Holdout set não tem pelo menos 2 classes para calibrar.'}
                if not hasattr(best_rf, "classes_"):
                    logger.error("RandomForest não treinado corretamente.")
                    return {'trained': False, 'error': 'RandomForest não treinado.'}

                # 👇 Nova forma recomendada
                from sklearn.calibration import CalibratedClassifierCV

                calibrated = CalibratedClassifierCV(
                    estimator=FrozenEstimator(best_rf),
                    method=self.calibration,
                    cv='prefit'
                )

                # Calibra com pesos se disponíveis
                if sw_hold is not None and len(sw_hold) == len(y_hold):
                    calibrated.fit(Xsel_hold, y_hold, sample_weight=sw_hold)
                else:
                    calibrated.fit(Xsel_hold, y_hold)

                if not hasattr(calibrated, "classes_"):
                    logger.error("CalibratedClassifierCV não foi treinado corretamente.")
                    return {'trained': False, 'error': 'CalibratedClassifierCV não foi treinado corretamente.'}

                self.model = calibrated
                self.trained = True
                # Holdout AUC
                hold_proba = calibrated.predict_proba(Xsel_hold)[:, 1]
                hold_auc = float(roc_auc_score(y_hold, hold_proba, sample_weight=sw_hold)) if sw_hold is not None else float(roc_auc_score(y_hold, hold_proba)) if len(np.unique(y_hold)) > 1 else None

                # Ajuste anti-over/underfitting
                gap = None
                if cv_auc is not None and hold_auc is not None:
                    gap = float(cv_auc - hold_auc)
                if gap is not None and gap > 0.08:
                    # Overfit → apertar
                    tuned_rf = RandomForestClassifier(
                        n_estimators=max(80, int(best_rf.n_estimators * 0.8)),
                        max_depth=max(3, int((best_rf.max_depth or 8) * 0.8)),
                        min_samples_leaf=min(10, max(3, int(best_rf.min_samples_leaf * 1.5))),
                        min_samples_split=best_rf.min_samples_split + 2,  # ← MAIS CONSERVADOR
                        max_features='sqrt',
                        ccp_alpha=best_rf.ccp_alpha * 2,                   # ← MAIS PODA
                        random_state=self.params.get('random_state', 42),
                        class_weight='balanced_subsample',
                        bootstrap=True,
                        n_jobs=self._max_jobs,
                        oob_score=True
                    )
                    fit_params_tune = {}
                    if sw_train is not None:
                        fit_params_tune['sample_weight'] = sw_train
                    tuned_rf.fit(Xsel_train, y_train, **fit_params_tune)
                    calibrated = CalibratedClassifierCV(base_estimator=tuned_rf, method=self.calibration, cv='prefit')
                    if sw_hold is not None:
                        calibrated.fit(Xsel_hold, y_hold, sample_weight=sw_hold)
                    else:
                        calibrated.fit(Xsel_hold, y_hold)
                    try:
                        if hasattr(tuned_rf, 'oob_decision_function_') and tuned_rf.oob_decision_function_ is not None:
                            oob_proba = tuned_rf.oob_decision_function_[:, 1]
                            oob_auc = float(roc_auc_score(y_train, oob_proba, sample_weight=sw_train)) if sw_train is not None else float(roc_auc_score(y_train, oob_proba))
                    except Exception:
                        pass
                    best_rf = tuned_rf

                # Underfit → soltar um pouco (se hold_auc muito baixo)
                if (hold_auc or 0) < 0.53 and (cv_auc or 0) < 0.55:
                    tuned_rf2 = RandomForestClassifier(
                        n_estimators=min(220, int(best_rf.n_estimators * 1.3)),
                        max_depth=(best_rf.max_depth or 8) + 1,
                        min_samples_leaf=max(2, int(best_rf.min_samples_leaf * 0.8)),
                        min_samples_split=max(2, best_rf.min_samples_split - 1),
                        max_features='auto',
                        ccp_alpha=best_rf.ccp_alpha * 0.5,                 # ← MENOS PODA
                        random_state=self.params.get('random_state', 42),
                        class_weight='balanced_subsample',
                        bootstrap=True,
                        n_jobs=self._max_jobs,
                        oob_score=True
                    )
                    fit_params_tune2 = {}
                    if sw_train is not None:
                        fit_params_tune2['sample_weight'] = sw_train
                    tuned_rf2.fit(Xsel_train, y_train, **fit_params_tune2)
                    calibrated = CalibratedClassifierCV(base_estimator=tuned_rf2, method=self.calibration, cv='prefit')
                    if sw_hold is not None:
                        calibrated.fit(Xsel_hold, y_hold, sample_weight=sw_hold)
                    else:
                        calibrated.fit(Xsel_hold, y_hold)
                    try:
                        if hasattr(tuned_rf2, 'oob_decision_function_') and tuned_rf2.oob_decision_function_ is not None:
                            oob_proba = tuned_rf2.oob_decision_function_[:, 1]
                            oob_auc = float(roc_auc_score(y_train, oob_proba, sample_weight=sw_train)) if sw_train is not None else float(roc_auc_score(y_train, oob_proba))
                    except Exception:
                        pass
                    best_rf = tuned_rf2

                # empacotar
                self.model = calibrated
                self.trained = True
                self.last_trained = as_iso(datetime.now())

                self.metrics = {
                    'cv_auc': float(cv_auc) if cv_auc is not None else None,
                    'holdout_auc': float(hold_auc) if hold_auc is not None else None,
                    'oob_auc': float(oob_auc) if oob_auc is not None else None,
                    'cv_splits': int(n_splits),
                    'n_samples': int(n),
                    'features_used': int(len(self.selected_features or [])),
                    'gap_cv_holdout': float(gap) if gap is not None else None,
                }

                if metrics:
                    self.metrics.update(metrics)  # 👈 INJETA AS MÉTRICAS EXTERNAS AQUI

                q = self._compute_quality(cv_auc, hold_auc, oob_auc)
                self.quality = q
                self.quality_smooth = float(0.7 * self.quality_smooth + 0.3 * q)

                self._save()

                return {
                    'trained': True,
                    'cv_auc': self.metrics['cv_auc'],
                    'holdout_auc': self.metrics['holdout_auc'],
                    'oob_auc': self.metrics['oob_auc'],
                    'cv_splits': self.metrics['cv_splits'],
                    'n_samples': self.metrics['n_samples'],
                    'features_used': self.metrics['features_used'],
                    'gap_cv_holdout': self.metrics['gap_cv_holdout'],
                    'quality': self.quality,
                    'quality_smooth': self.quality_smooth,
                    'calibration': self.calibration
                }
            except Exception as e:
                logger.error(f"Treino ML: {e}")
                self.trained = False
                return {'trained': False, 'error': str(e)}

    def predict_proba(self, feature_row_dict):
        if not SKLEARN_AVAILABLE or not self.trained or self.model is None or self.scaler is None:
            return 0.5
        with self.lock:
            try:
                # SEMPRE use self.feature_names para manter ordem consistente
                x = np.array([feature_row_dict.get(f, 0.0) for f in self.feature_names], dtype=float).reshape(1, -1)
                xs = self.scaler.transform(x)
                if self.selector is not None:
                    xs = self.selector.transform(xs)  # ← APLICA SELEÇÃO DE FEATURES!
                proba = float(self.model.predict_proba(xs)[0, 1])
                return proba
            except Exception as e:
                logger.warning(f"Predição ML: {e}")
                return 0.5

# =====================================================================================
# Signal Generator (boost de confluência, gatilhos robustos, gating de qualidade)
# =====================================================================================
class AdvancedSignalGenerator:
    def __init__(self, params=None, feature_builder=None, ml_model=None, bot_manager=None, ticker='BTCUSDT'):
        """
        Gerador de sinais avançado com suporte a:
        - Confluência MTF
        - ML calibrado
        - Filtro de regime global (BTC-driven)
        - Gatilhos de entrada
        - Pesos adaptativos
        """
        self.params = deepcopy(DEFAULT_PARAMS) if params is None else deepcopy(params)
        self.feature_builder = feature_builder or MultiTimeframeFeatureBuilder()
        self.ml_model = ml_model or MLPatternModel(self.params.get('ml', {}), model_id="DEFAULT")
        self.last_signal = 'HOLD'
        self.signal_strength = 0
        self.lock = threading.Lock()
        self._regime_advisor = RegimeAdvisor(self.feature_builder)
        
        # 👇 Filtro de regime global baseado em BTC (só ativo se bot_manager for passado e ticker != BTC)
        self._market_regime_filter = MarketRegimeFilter(bot_manager) if bot_manager else None
        self.ticker = ticker  # ← usado para evitar aplicar filtro de regime em BTC

        # ⚠️ NUNCA instancie outro AdvancedSignalGenerator aqui — causa recursão infinita!

    def set_params(self, new_params):
        with self.lock:
            self.params = deepcopy(new_params)

    def get_params(self):
        with self.lock:
            return deepcopy(self.params)

    def _rule_scores(self, ind):
        buy, sell, br, sr = 0.0, 0.0, [], []
        try:
            if ind.get('rsi', 50) < 35:
                buy += 1; br.append(f"RSI {ind['rsi']:.1f}")
            if ind.get('rsi', 50) > 65:
                sell += 1; sr.append(f"RSI {ind['rsi']:.1f}")

            if ind.get('bb_position', 0.5) < 0.25:
                buy += 1; br.append(f"BB low {ind['bb_position']:.2f}")
            if ind.get('bb_position', 0.5) > 0.75:
                sell += 1; sr.append(f"BB high {ind['bb_position']:.2f}")

            if ind.get('price_vs_ema', 0.0) < -0.005:
                buy += 1; br.append(f"<EMA {ind['price_vs_ema']:.3f}")
            if ind.get('price_vs_ema', 0.0) > 0.005:
                sell += 1; sr.append(f">EMA {ind['price_vs_ema']:.3f}")

            if ind.get('macd', 0.0) > ind.get('macd_signal', 0.0):
                buy += 1; br.append("MACD+")
            if ind.get('macd', 0.0) < ind.get('macd_signal', 0.0):
                sell += 1; sr.append("MACD-")

            if ind.get('stoch_k', 50) < 25:
                buy += 1; br.append(f"Stoch {ind['stoch_k']:.1f}")
            if ind.get('stoch_k', 50) > 75:
                sell += 1; sr.append(f"Stoch {ind['stoch_k']:.1f}")

            if ind.get('momentum', 0.0) > 0:
                buy += 0.5
            else:
                sell += 0.5

            maxc = 6.5
            return min(1.0, buy / maxc), min(1.0, sell / maxc), br[:3], sr[:3]
        except Exception:
            return 0.0, 0.0, [], []

    def _mtf_conf(self, df, tfs):
        if df is None or df.empty:
            return 0.0, 0.0, ["MTF: sem dados"]
        cl, cs, total, used = 0.0, 0.0, 0.0, []
        for tf in tfs:
            dfr = df if tf == 1 else self.feature_builder.resample_ohlc(df, tf)
            if dfr is None or dfr.empty or len(dfr) < 20:
                continue
            feats = self.feature_builder.indicators_df(dfr)
            if feats is None or feats.empty:
                continue
            last = feats.iloc[-1]
            if last.get('price_vs_ema', 0) > 0:
                cl += 1
            elif last.get('price_vs_ema', 0) < 0:
                cs += 1
            if last.get('macd_hist', 0) > 0:
                cl += 1
            elif last.get('macd_hist', 0) < 0:
                cs += 1
            r = float(last.get('rsi', 50))
            if r >= 55:
                cl += 0.5
            elif r <= 45:
                cs += 0.5
            total += 2.5
            used.append(tf)
        if total == 0 or not used:
            return 0.0, 0.0, ["MTF indisponível"]
        return float(np.clip(cl / total, 0, 1)), float(np.clip(cs / total, 0, 1)), [
            f"MTF usando {', '.join(f'{x}m' for x in used)}",
            f"MTF long={cl/total:.2f} short={cs/total:.2f}"
        ]

    #def _entry_triggers(self, df, direction='long'):
        # Gatilhos leves: breakout/pullback
        try:
            if df is None or df.empty or len(df) < 25:
                return True, "Sem trigger (dados insuficientes)"
            s = df['close']
            h = df['high']
            l = df['low']
            ema = s.ewm(span=8, adjust=False, min_periods=1).mean()
            atr = (h.combine(l, max) - l.combine(s.shift(1), lambda a, b: abs(a - (b or a))).fillna(0)).rolling(14).mean()
            atr = atr.fillna(atr.rolling(10).mean())

            # Breakout: candle fecha acima da máxima dos últimos N
            N = 12
            recent_high = h.rolling(N).max().shift(1)
            recent_low = l.rolling(N).min().shift(1)

            if direction == 'long':
                cond_breakout = s.iloc[-1] > (recent_high.iloc[-1] or s.iloc[-2])
                cond_pullback = (s.iloc[-1] > ema.iloc[-1]) and \
                                (l.iloc[-5:-1].min() < (ema.iloc[-5:-1].min() + (atr.iloc[-1] or 0) * 0.2))
                if cond_breakout or cond_pullback:
                    return True, ("breakout" if cond_breakout else "pullback")
            else:
                cond_breakout = s.iloc[-1] < (recent_low.iloc[-1] or s.iloc[-2])
                cond_pullback = (s.iloc[-1] < ema.iloc[-1]) and \
                                (h.iloc[-5:-1].max() > (ema.iloc[-5:-1].max() - (atr.iloc[-1] or 0) * 0.2))
                if cond_breakout or cond_pullback:
                    return True, ("breakdown" if cond_breakout else "pullback")
            return False, "sem gatilho"
        except Exception:
            return True, "trigger fallback"

    def generate_signal(self, current_data, indicators, historical_data, df_1m=None):
        try:
            if not indicators or len(historical_data) < 30:
                return {'signal': 'HOLD', 'strength': 0, 'reason': 'Dados insuficientes',
                        'entry_price': None, 'stop_loss': None, 'take_profit': None}

            # 👇 INICIALIZA reasons CEDO para evitar "referenced before assignment"
            reasons = []

            p = self.get_params()
            w = p['weights']
            th = p['thresholds']

            if df_1m is None:
                df_1m = pd.DataFrame(historical_data)
                df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
                df_1m = df_1m.set_index('timestamp').sort_index()[['open', 'high', 'low', 'close', 'volume']].astype(float)

            rb_l, rb_s, rb_lr, rb_sr = self._rule_scores(indicators)
            tf_l, tf_s, mtf_r = self._mtf_conf(df_1m, p['timeframes'])

            if p['ml_enabled'] and SKLEARN_AVAILABLE and self.ml_model.trained:
                feat_row, _ = self.feature_builder.latest_feature_row(df_1m, timeframes=p['timeframes'])
                p_up = self.ml_model.predict_proba(feat_row) if feat_row is not None else 0.5
                quality = getattr(self.ml_model, 'quality_smooth', 0.7)
            else:
                p_up = 0.5
                quality = 0.5

            # Filtro duro: só abre posição se passar
            pass_f, mult_f, filt_r = VolatilityVolumeFilter.evaluate(df_1m.tail(600), p)

            # Gating: peso efetivo da ML multiplicado pela qualidade (suavizada)
            w_ml_eff = w['ml'] * float(np.clip(quality, 0.0, 1.0))
            w_rule = w['rule']
            w_mtf = w['mtf']

            # boost de confluência (sem baixar thresholds)
            conf_long = (rb_l >= 0.66) and (tf_l >= 0.66) and (p_up >= 0.60)
            conf_short = (rb_s >= 0.66) and (tf_s >= 0.66) and ((1 - p_up) >= 0.60)
            boost_long = 1.15 if conf_long else 1.0
            boost_short = 1.15 if conf_short else 1.0

            long_score = 100.0 * (w_rule * rb_l + w_mtf * tf_l + w_ml_eff * p_up) * mult_f * boost_long
            short_score = 100.0 * (w_rule * rb_s + w_mtf * tf_s + w_ml_eff * (1.0 - p_up)) * mult_f * boost_short

            price = float(current_data['price'])
            atr = float(indicators.get('atr', max(0.0001, price * 0.01)))

            # Inicializa reasons com motivos base (MTF + filtros)
            reasons.extend(mtf_r)
            reasons.extend(filt_r)

            # 👇 BOOSTS/PENALIDADES BASEADOS NAS NOVAS FEATURES DE MICROESTRUTURA
            # Aplicados diretamente em long_score / short_score — sem mexer no sniper

            ob_imb = feat_row.get('f1m_orderbook_imbalance', 0.0)
            buy_vol_ratio = feat_row.get('f1m_agg_buy_vol_ratio', 0.5)
            local_vol_z = feat_row.get('f1m_local_vol_zscore', 0.0)
            vol_ratio_5_20 = feat_row.get('f1m_vol_ratio_5_20', 1.0)

            # 1. Order Book Imbalance: > 0.5 = força compradora, < -0.5 = força vendedora
            if ob_imb > 0.5:
                if long_score > short_score:
                    long_score *= 1.15
                    reasons.append(f"📈 OrderBook Bullish ({ob_imb:.2f})")
                else:
                    short_score *= 0.9  # leve penalidade se estiver contra a pressão
            elif ob_imb < -0.5:
                if short_score > long_score:
                    short_score *= 1.15
                    reasons.append(f"📉 OrderBook Bearish ({ob_imb:.2f})")
                else:
                    long_score *= 0.9

            # 2. Aggregated Buy Volume Ratio: > 0.65 = volume comprador dominante
            if buy_vol_ratio > 0.65:
                if long_score > short_score:
                    long_score *= 1.12
                    reasons.append(f"💰 Volume Comprador ({buy_vol_ratio:.2f})")
                else:
                    short_score *= 0.92
            elif buy_vol_ratio < 0.35:
                if short_score > long_score:
                    short_score *= 1.12
                    reasons.append(f"💸 Volume Vendedor ({buy_vol_ratio:.2f})")
                else:
                    long_score *= 0.92

            # 3. Local Vol Z-Score: > 1.5 = volatilidade extrema → favorece trend following
            if local_vol_z > 1.5:
                if tf_l > tf_s and long_score > short_score:
                    long_score *= 1.18
                    reasons.append(f"⚡ Alta Volatilidade → Boost LONG")
                elif tf_s > tf_l and short_score > long_score:
                    short_score *= 1.18
                    reasons.append(f"⚡ Alta Volatilidade → Boost SHORT")

            # 4. Vol Ratio 5/20: > 1.8 = expansão de vol → momentum | < 0.6 = compressão → reversão
            if vol_ratio_5_20 > 1.8:
                if long_score > short_score:
                    long_score *= 1.10
                    reasons.append(f"🚀 Expansão de Vol → Momentum LONG")
                elif short_score > long_score:
                    short_score *= 1.10
                    reasons.append(f"🚀 Expansão de Vol → Momentum SHORT")
            elif vol_ratio_5_20 < 0.6:
                if indicators.get('rsi', 50) < 30 and long_score > short_score:
                    long_score *= 1.15
                    reasons.append("🔄 Compressão + RSI Baixo → Reversão LONG")
                elif indicators.get('rsi', 50) > 70 and short_score > long_score:
                    short_score *= 1.15
                    reasons.append("🔄 Compressão + RSI Alto → Reversão SHORT")

            # 👇 Penalizar LONG se estiver em topo com divergência ou RSI extremo
            if indicators.get('rsi', 50) > 75 or indicators.get('divergence', 0) == 1:
                if long_score > short_score:
                    long_score *= 0.5
                    reasons.append("⛔ Topo Detectado: Penalizando LONG")
                if short_score > 0:
                    short_score *= 1.3
                    reasons.append("✅ Reversão Esperada: Bônus para SHORT")

            # 👇 Penalizar SHORT se estiver em fundo com divergência ou RSI extremo
            if indicators.get('rsi', 50) < 25 or indicators.get('divergence', 0) == -1:
                if short_score > long_score:
                    short_score *= 0.5
                    reasons.append("⛔ Fundo Detectado: Penalizando SHORT")
                if long_score > 0:
                    long_score *= 1.3
                    reasons.append("✅ Reversão Esperada: Bônus para LONG")

            # 👇 BÔNUS para entradas em reversão confirmada
            if indicators.get('divergence', 0) == -1 and long_score > short_score:
                long_score *= 1.3
                reasons.append("✅ BÔNUS: Divergência de Alta")

            if indicators.get('divergence', 0) == 1 and short_score > long_score:
                short_score *= 1.3
                reasons.append("✅ BÔNUS: Divergência de Baixa")

            if long_score > short_score and feat_row.get('f1m_volume_drying', 0) == 1:
                long_score *= 0.5
                reasons.append("💧 Volume secando no topo")

            # Bônus por padrão candlestick de reversão
            if indicators.get('candle_pattern', 0) == -1 and long_score > short_score:
                long_score *= 1.2
                reasons.append("🕯️ Martelo Bullish")
            elif indicators.get('candle_pattern', 0) == 1 and short_score > long_score:
                short_score *= 1.2
                reasons.append("🕯️ Estrela Cadente")

            

            # Bônus por VPIN (fluxo informado)
            vpin = feat_row.get('f1m_vpin_proxy', 0.0)
            if vpin < -0.3 and long_score > short_score:
                long_score *= 1.2
                reasons.append(f"📈 VPIN Comprador ({vpin:.2f})")
            elif vpin > 0.3 and short_score > long_score:
                short_score *= 1.2
                reasons.append(f"📉 VPIN Vendedor ({vpin:.2f})")

            # ==============================
            # 👇 SNIPER MODE: Força entrada se 3+ condições de reversão se alinharem
            # ==============================

            long_sniper_conditions = 0
            short_sniper_conditions = 0
            sniper_reasons = []

            # Condições para LONG (reversão de fundo)
            if vpin < -0.3:
                long_sniper_conditions += 1
                sniper_reasons.append("VPIN Comprador")
            if indicators.get('divergence', 0) == -1:
                long_sniper_conditions += 1
                sniper_reasons.append("Divergência de Alta")
            if indicators.get('rsi', 50) < 25:
                long_sniper_conditions += 1
                sniper_reasons.append("RSI < 25")
            if indicators.get('candle_pattern', 0) == -1:
                long_sniper_conditions += 1
                sniper_reasons.append("Martelo Bullish")
            if indicators.get('volume_drying', 0) == 1 and indicators.get('rsi', 50) < 40:
                long_sniper_conditions += 1
                sniper_reasons.append("Volume Secando no Fundo")

            # Condições para SHORT (reversão de topo)
            if vpin > 0.3:
                short_sniper_conditions += 1
                sniper_reasons.append("VPIN Vendedor")
            if indicators.get('divergence', 0) == 1:
                short_sniper_conditions += 1
                sniper_reasons.append("Divergência de Baixa")
            if indicators.get('rsi', 50) > 75:
                short_sniper_conditions += 1
                sniper_reasons.append("RSI > 75")
            if indicators.get('candle_pattern', 0) == 1:
                short_sniper_conditions += 1
                sniper_reasons.append("Estrela Cadente")
            if indicators.get('volume_drying', 0) == 1 and indicators.get('rsi', 50) > 60:
                short_sniper_conditions += 1
                sniper_reasons.append("Volume Secando no Topo")

            sniper_triggered = False
            sniper_signal = 'HOLD'
            sniper_strength = 0

            if long_sniper_conditions >= 3:
                sniper_triggered = True
                sniper_signal = 'BUY'
                sniper_strength = 95.0
                reasons = [f"🎯 SNIPER LONG ({long_sniper_conditions} condições):"] + sniper_reasons[:5]
                logger.info(f"[SNIPER] Entrada forçada em LONG: {reasons}")

            elif short_sniper_conditions >= 3:
                sniper_triggered = True
                sniper_signal = 'SELL'
                sniper_strength = 95.0
                reasons = [f"🎯 SNIPER SHORT ({short_sniper_conditions} condições):"] + sniper_reasons[:5]
                logger.info(f"[SNIPER] Entrada forçada em SHORT: {reasons}")

            # 👇 SE SNIPER FOI ATIVADO, RETORNA IMEDIATAMENTE — NÃO CONTINUA!
            if sniper_triggered:
                # 👇 REGIME ADVISOR — MOVIDO PARA CIMA, ANTES DO PRIMEIRO USO DE details_common
                regime = self._regime_advisor.evaluate(df_1m.tail(600))
                regime_profile = regime.get('profile', 'BALANCED')
                trend_score = regime.get('trend_score', 0.5)
                rv_pct = regime.get('rv_pct', 50.0)

                # Ajuste de pesos por regime
                if trend_score >= 0.7:
                    w_ml_eff *= 1.2
                    w_rule *= 0.8
                elif trend_score <= 0.3:
                    w_ml_eff *= 1.2
                    w_rule *= 0.8
                else:
                    w_ml_eff *= 0.7
                    w_rule *= 1.3
                regime_good = (regime_profile in ['PREDICTIVE', 'MTF_HEAVY']) or (trend_score <= 0.35 or trend_score >= 0.65)

                # Aplica filtro de regime global (baseado em BTC)
                regime_global_score = 1.0
                if self._market_regime_filter and self.ticker != 'BTCUSDT':
                    regime_score = self._market_regime_filter.get_market_regime_score()
                    regime_global_score = float(regime_score)
                    if long_score > short_score and regime_score <= 0.3:
                        long_score *= 0.5
                        reasons.append(f"⚠️ Bear Market Global (BTC={regime_score:.1f})")
                    elif short_score > long_score and regime_score >= 0.7:
                        short_score *= 0.5
                        reasons.append(f"⚠️ Bull Market Global (BTC={regime_score:.1f})")

                # 👇 AGORA details_common É DEFINIDO ANTES DE QUALQUER USO!
                details_common = {
                    'rb_long': float(rb_l), 'rb_short': float(rb_s),
                    'tf_long': float(tf_l), 'tf_short': float(tf_s),
                    'ml_p_up': float(p_up),
                    'ml_quality': float(quality),
                    'ml_weight_effective': float(w_ml_eff),
                    'filter_ok': bool(pass_f),
                    'filter_mult': float(mult_f),
                    'conf_boost_long': float(boost_long),
                    'conf_boost_short': float(boost_short),
                    'regime_profile': regime_profile,
                    'trend_score': float(trend_score),
                    'volume_imbalance': float(feat_row.get('f1m_volume_imbalance', 0.0)),
                    'vpin_proxy': float(feat_row.get('f1m_vpin_proxy', 0.0)),
                    'spread_proxy': float(feat_row.get('f1m_spread_proxy', 0.0)),
                    'regime_global_score': regime_global_score,
                    'local_vol_zscore': float(feat_row.get('f1m_local_vol_zscore', 0.0)),
                    'vol_ratio_5_20': float(feat_row.get('f1m_vol_ratio_5_20', 1.0)),
                    'orderbook_imbalance': float(feat_row.get('f1m_orderbook_imbalance', 0.0)),
                    'agg_buy_vol_ratio': float(feat_row.get('f1m_agg_buy_vol_ratio', 0.5)),
                }

                # 👇 AGORA PODEMOS USAR details_common SEM PROBLEMAS
                if not pass_f:
                    return {
                        'signal': 'HOLD', 'strength': 0,
                        'reason': 'Filtros não aprovados; ' + '; '.join(reasons),
                        'entry_price': None, 'stop_loss': None, 'take_profit': None,
                        'details': details_common
                    }

                # Define stop e target corretos conforme o lado
                if sniper_signal == 'BUY':
                    stop_loss = price - atr * DEFAULT_PARAMS['risk']['atr_stop_mult']
                    take_profit = price + atr * DEFAULT_PARAMS['risk']['atr_target_mult']
                else:  # sniper_signal == 'SELL'
                    stop_loss = price + atr * DEFAULT_PARAMS['risk']['atr_stop_mult']
                    take_profit = price - atr * DEFAULT_PARAMS['risk']['atr_target_mult']

                return {
                    'signal': sniper_signal,
                    'strength': float(sniper_strength),
                    'reason': '; '.join(reasons[:6]),
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'details': details_common
                }

            # 👇 SÓ CHEGA AQUI SE NÃO HOUVE SNIPER

            # Regime Advisor (se não foi chamado no sniper)
            regime = self._regime_advisor.evaluate(df_1m.tail(600))
            regime_profile = regime.get('profile', 'BALANCED')
            trend_score = regime.get('trend_score', 0.5)
            rv_pct = regime.get('rv_pct', 50.0)

            if trend_score >= 0.7:
                w_ml_eff *= 1.2
                w_rule *= 0.8
            elif trend_score <= 0.3:
                w_ml_eff *= 1.2
                w_rule *= 0.8
            else:
                w_ml_eff *= 0.7
                w_rule *= 1.3
            regime_good = (regime_profile in ['PREDICTIVE', 'MTF_HEAVY']) or (trend_score <= 0.35 or trend_score >= 0.65)

            # Aplica filtro de regime global (baseado em BTC)
            regime_global_score = 1.0
            if self._market_regime_filter and self.ticker != 'BTCUSDT':
                regime_score = self._market_regime_filter.get_market_regime_score()
                regime_global_score = float(regime_score)
                if long_score > short_score and regime_score <= 0.3:
                    long_score *= 0.5
                    reasons.append(f"⚠️ Bear Market Global (BTC={regime_score:.1f})")
                elif short_score > long_score and regime_score >= 0.7:
                    short_score *= 0.5
                    reasons.append(f"⚠️ Bull Market Global (BTC={regime_score:.1f})")

            details_common = {
                'rb_long': float(rb_l), 'rb_short': float(rb_s),
                'tf_long': float(tf_l), 'tf_short': float(tf_s),
                'ml_p_up': float(p_up),
                'ml_quality': float(quality),
                'ml_weight_effective': float(w_ml_eff),
                'filter_ok': bool(pass_f),
                'filter_mult': float(mult_f),
                'conf_boost_long': float(boost_long),
                'conf_boost_short': float(boost_short),
                'regime_profile': regime_profile,
                'trend_score': float(trend_score),
                'volume_imbalance': float(feat_row.get('f1m_volume_imbalance', 0.0)),
                'vpin_proxy': float(feat_row.get('f1m_vpin_proxy', 0.0)),
                'spread_proxy': float(feat_row.get('f1m_spread_proxy', 0.0)),
                'regime_global_score': regime_global_score,
            }

            # ============ ENTRADA PARA LONG ============
            if (long_score >= th['buy']) and (long_score > short_score):
                if not pass_f:
                    return {
                        'signal': 'HOLD', 'strength': 0,
                        'reason': 'Filtros não aprovados; ' + '; '.join(reasons),
                        'entry_price': None, 'stop_loss': None, 'take_profit': None,
                        'details': details_common
                    }

                trigger_ok = True
                trig_reason = "livre"
                if regime_good:
                    trigger_ok, trig_reason = self._entry_triggers(df_1m, 'long')

                if trigger_ok:
                    long_score *= 1.10
                    trigger_note = f"✅ Trigger: {trig_reason}"
                else:
                    trigger_note = f"⚠️ Sem gatilho [{trig_reason}]"

                return {
                    'signal': 'BUY',
                    'strength': float(np.clip(long_score, 0, 100)),
                    'reason': '; '.join((rb_lr + [trigger_note] + reasons)[:6]),
                    'entry_price': price,
                    'stop_loss': price - atr * DEFAULT_PARAMS['risk']['atr_stop_mult'],
                    'take_profit': price + atr * DEFAULT_PARAMS['risk']['atr_target_mult'],
                    'details': details_common
                }

            # ============ ENTRADA PARA SHORT ============
            elif (short_score >= th['sell']) and (short_score > long_score):
                if not pass_f:
                    return {
                        'signal': 'HOLD', 'strength': 0,
                        'reason': 'Filtros não aprovados; ' + '; '.join(reasons),
                        'entry_price': None, 'stop_loss': None, 'take_profit': None,
                        'details': details_common
                    }

                return {
                    'signal': 'SELL',
                    'strength': float(np.clip(short_score, 0, 100)),
                    'reason': '; '.join((rb_sr + reasons)[:6]),
                    'entry_price': price,
                    'stop_loss': price + atr * DEFAULT_PARAMS['risk']['atr_stop_mult'],
                    'take_profit': price - atr * DEFAULT_PARAMS['risk']['atr_target_mult'],
                    'details': details_common
                }

            # ============ SEM SINAL VÁLIDO ============
            return {
                'signal': 'HOLD',
                'strength': 0,
                'reason': 'Aguardando condições favoráveis; ' + '; '.join(reasons),
                'entry_price': None, 'stop_loss': None, 'take_profit': None,
                'details': details_common
            }

        except Exception as e:
            logger.error(f"Sinal: {e}")
            return {'signal': 'HOLD', 'strength': 0, 'reason': f'Erro: {str(e)}',
                    'entry_price': None, 'stop_loss': None, 'take_profit': None}
        
        
# =====================================================================================
# Trading Engine (maior cadência, filtro duro, exits prudentes)
# =====================================================================================
class TradingEngine:
    def __init__(self, initial_capital=100, symbol='MARKET'):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.trades_history = []
        self.current_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_today = 0
        self.trade_points = []
        self.position_size = 0.0
        self.last_trade_time = None
        self.daily_target_reached = False
        self.symbol = symbol

    def execute_trade(self, signal, current_data, indicators, params=None):
        try:
            params = params or DEFAULT_PARAMS
            price = float(current_data['price'])
            ts = str(current_data['timestamp'])
            atr = float(indicators.get('atr', max(0.0001, price * 0.01)))

            # 👇 NOVO: Atualiza trailing stop se estiver em posição
            if self.position != 0:
                self._update_trailing_stop(price, atr, signal.get('details', {}))

            # meta diária
            if self.total_pnl >= self.initial_capital * (params['risk']['daily_target_pct_of_initial'] / 100.0):
                if not self.daily_target_reached:
                    logger.info(f"🎯 Meta diária atingida [{self.symbol}]! P&L: ${self.total_pnl:.2f}")
                    self.daily_target_reached = True
                return
            
            # Se for sinal sniper, aumenta position size em 20%
            is_sniper = "🎯 SNIPER" in signal.get('reason', '')
            size_factor = 1.2 if is_sniper else 1.0

            self.position_size = max(0.0, (self.current_capital * size_factor) / max(1e-9, price))

            # gerenciar posição atual
            if self.position != 0:
                if self._check_exit_conditions(price, atr, signal):
                    self._close_position(price, ts, "Condições de saída")

            # abrir se flat
            if self.position == 0:
                if self.last_trade_time:
                    delta = parse_iso(ts) - parse_iso(self.last_trade_time)
                    if delta.total_seconds() < params['risk']['min_minutes_between_trades'] * 60:
                        return
                if self.trades_today >= params['risk']['max_trades_per_day']:
                    return

                if signal['signal'] == 'BUY' and signal['strength'] >= 50:
                    self._open_long(price, ts, signal, atr)   # <-- invertido
                elif signal['signal'] == 'SELL' and signal['strength'] >= 50:    
                    self._open_short(price, ts, signal, atr)

            if self.position != 0:
                self._update_current_pnl(price)
        except Exception as e:
            logger.error(f"[{self.symbol}] Exec trade: {e}")

    def _open_long(self, price, ts, signal, atr):
        self.position = 1
        self.entry_price = price
        self.entry_time = ts
        self.last_trade_time = ts
        self.position_size = max(0.0, self.current_capital / max(1e-9, price))
        sl = price - atr * DEFAULT_PARAMS['risk']['atr_stop_mult']
        tp = price + atr * DEFAULT_PARAMS['risk']['atr_target_mult']
        self.trades_history.append({
            'type': 'BUY', 'entry_price': price, 'entry_time': ts,
            'stop_loss': sl, 'take_profit': tp,
            'signal_strength': signal['strength'], 'reason': signal['reason'],
            'position_size': self.position_size, 'status': 'open'
        })
        self.trades_today += 1
        self.trade_points.append({'timestamp': ts, 'price': price, 'type': 'BUY', 'strength': signal['strength']})
        logger.info(f"🟢 BUY {self.symbol}: {price:.2f} | qty: {self.position_size:.6f}")

    def _open_short(self, price, ts, signal, atr):
        self.position = -1
        self.entry_price = price
        self.entry_time = ts
        self.last_trade_time = ts
        self.position_size = max(0.0, self.current_capital / max(1e-9, price))
        sl = price + atr * DEFAULT_PARAMS['risk']['atr_stop_mult']
        tp = price - atr * DEFAULT_PARAMS['risk']['atr_target_mult']
        self.trades_history.append({
            'type': 'SELL', 'entry_price': price, 'entry_time': ts,
            'stop_loss': sl, 'take_profit': tp,
            'signal_strength': signal['strength'], 'reason': signal['reason'],
            'position_size': self.position_size, 'status': 'open'
        })
        self.trades_today += 1
        self.trade_points.append({'timestamp': ts, 'price': price, 'type': 'SELL', 'strength': signal['strength']})
        logger.info(f"🔴 SELL {self.symbol}: {price:.2f} | qty: {self.position_size:.6f}")

    def _close_position(self, price, ts, reason):
        if self.position == 0:
            return
        if self.position == 1:
            pnl_abs = (price - self.entry_price) * self.position_size
        else:
            pnl_abs = (self.entry_price - price) * self.position_size
        capital_used = self.entry_price * self.position_size if self.entry_price > 0 else 1.0
        pnl_pct = (pnl_abs / capital_used) * 100 if capital_used != 0 else 0.0
        self.current_capital += pnl_abs
        self.total_pnl += pnl_abs
        if self.trades_history:
            self.trades_history[-1].update({
                'exit_price': price, 'exit_time': ts,
                'pnl_absolute': pnl_abs, 'pnl_percent': pnl_pct,
                'capital_after': self.current_capital, 'exit_reason': reason, 'status': 'closed'
            })
        exit_type = 'SELL_EXIT' if self.position == 1 else 'BUY_EXIT'
        self.trade_points.append({'timestamp': ts, 'price': price, 'type': exit_type, 'pnl': pnl_abs, 'pnl_percent': pnl_pct})
        logger.info(f"✅ EXIT {self.symbol} @ {price:.2f} | P&L: ${pnl_abs:.2f} ({pnl_pct:.2f}%) | Capital: ${self.current_capital:.2f} | {reason}")
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_pnl = 0.0
        self.position_size = 0.0

    def _check_exit_conditions(self, price, atr, signal):
        if self.position == 0 or not self.trades_history:
            return False
        last = self.trades_history[-1]
        if self.position == 1:
            if price <= last['stop_loss'] or price >= last['take_profit']:
                return True
        else:
            if price >= last['stop_loss'] or price <= last['take_profit']:
                return True
        # reversão forte
        if signal['strength'] >= 70:
            if self.position == 1 and signal['signal'] == 'SELL':
                return True
            if self.position == -1 and signal['signal'] == 'BUY':
                return True
        # tempo máximo na posição (3h)
        if self.entry_time:
            try:
                if (datetime.now() - parse_iso(self.entry_time)).total_seconds() > 3 * 3600:
                    return True
            except Exception:
                pass
        return False

    def _update_trailing_stop(self, price, atr, signal_details):
        """
        Atualiza dinamicamente o stop loss com base em:
        - Movimento do preço (trailing)
        - Mudança de regime ou fluxo de ordem
        - Realização parcial de lucro
        """
        if self.position == 0 or not self.trades_history:
            return

        last_trade = self.trades_history[-1]
        entry_price = last_trade['entry_price']
        current_sl = last_trade['stop_loss']
        current_tp = last_trade['take_profit']

        # 1. Trailing Stop Clássico (protege lucro)
        if self.position == 1:  # LONG
            new_sl_trailing = max(current_sl, price - atr * DEFAULT_PARAMS['risk']['atr_stop_mult'])
            # Se preço subiu, aperta o stop
            if price > entry_price + atr * 1.0:  # já está com 1x ATR de lucro
                new_sl_trailing = max(new_sl_trailing, entry_price + atr * 0.5)  # lock 0.5x ATR de lucro
            if price > entry_price + atr * 2.0:  # 2x ATR de lucro → aperta mais
                new_sl_trailing = max(new_sl_trailing, entry_price + atr * 1.5)

        else:  # SHORT
            new_sl_trailing = min(current_sl, price + atr * DEFAULT_PARAMS['risk']['atr_stop_mult'])
            if price < entry_price - atr * 1.0:
                new_sl_trailing = min(new_sl_trailing, entry_price - atr * 0.5)
            if price < entry_price - atr * 2.0:
                new_sl_trailing = min(new_sl_trailing, entry_price - atr * 1.5)

        # 2. Aperto por mudança de regime/fluxo (se disponível)
        if signal_details:
            # Se regime virou contra você
            if self.position == 1 and signal_details.get('trend_score', 0.5) < 0.4:
                new_sl_trailing = max(new_sl_trailing, price - atr * 1.0)  # aperta stop
            elif self.position == -1 and signal_details.get('trend_score', 0.5) > 0.6:
                new_sl_trailing = min(new_sl_trailing, price + atr * 1.0)

            # Se order flow virou contra você
            ob_imb = signal_details.get('orderbook_imbalance', 0.0)
            if self.position == 1 and ob_imb < -0.3:  # pressão vendedora aumentando
                new_sl_trailing = max(new_sl_trailing, price - atr * 1.2)
            elif self.position == -1 and ob_imb > 0.3:  # pressão compradora aumentando
                new_sl_trailing = min(new_sl_trailing, price + atr * 1.2)

            # Se VPIN mostra toxicidade contra sua posição
            vpin = signal_details.get('vpin_proxy', 0.0)
            if self.position == 1 and vpin > 0.4:  # venda agressiva dominando
                new_sl_trailing = max(new_sl_trailing, price - atr * 1.0)
            elif self.position == -1 and vpin < -0.4:  # compra agressiva dominando
                new_sl_trailing = min(new_sl_trailing, price + atr * 1.0)

        # 3. Atualiza apenas se for mais protetivo
        if self.position == 1 and new_sl_trailing > current_sl:
            last_trade['stop_loss'] = new_sl_trailing
            logger.debug(f"[{self.symbol}] Trailing Stop atualizado para {new_sl_trailing:.2f} (LONG)")
        elif self.position == -1 and new_sl_trailing < current_sl:
            last_trade['stop_loss'] = new_sl_trailing
            logger.debug(f"[{self.symbol}] Trailing Stop atualizado para {new_sl_trailing:.2f} (SHORT)")

    def _update_current_pnl(self, price):
        if self.position == 1:
            self.current_pnl = (price - self.entry_price) * self.position_size
        else:
            self.current_pnl = (self.entry_price - price) * self.position_size

    def get_trade_points(self): return self.trade_points[-80:]

    def get_performance(self):
        closed = [t for t in self.trades_history if t.get('status') == 'closed']
        if not closed:
            return {'total_trades': len(self.trades_history), 'closed_trades': 0, 'win_rate': 0, 'avg_pnl': 0,
                    'best_trade': 0, 'worst_trade': 0}
        wins = [t for t in closed if t.get('pnl_absolute', 0) > 0]
        pnls = [t.get('pnl_absolute', 0) for t in closed]
        return {
            'total_trades': len(self.trades_history), 'closed_trades': len(closed),
            'win_rate': len(wins) / len(closed) * 100, 'avg_pnl': float(np.mean(pnls)),
            'best_trade': float(np.max(pnls)), 'worst_trade': float(np.min(pnls))
        }

# =====================================================================================
# Backtest & Optimize
# =====================================================================================
class Backtester:
    def __init__(self, feature_builder, signal_generator_class, params, market_symbol='MARKET'):
        self.feature_builder = feature_builder
        self.signal_generator_class = signal_generator_class
        self.params = deepcopy(params)
        self.market_symbol = market_symbol

    def run(self, df, walkforward=False, retrain_every=500, initial_train_ratio=0.6, initial_capital=500):
        if df is None or df.empty or len(df) < 400:
            return {'success': False, 'error': 'Dados insuficientes (>=400 barras)'}
        eng = TradingEngine(initial_capital=initial_capital, symbol=self.market_symbol)
        sig = self.signal_generator_class(params=self.params,
                                          feature_builder=self.feature_builder,
                                          ml_model=MLPatternModel(self.params.get('ml', {}), model_id=f"{self.market_symbol}_BT"))
        tfs = self.params.get('timeframes', [5, 15])
        horizon = self.params.get('ml', {}).get('horizon', 3)
        X, y, fcols, feats, aligned, sample_weights = self.feature_builder.build_ml_dataset(
            df, horizon=horizon, timeframes=tfs, use_triple_barrier=True,
            atr_mult_stop=DEFAULT_PARAMS['risk']['atr_stop_mult'],
            atr_mult_target=DEFAULT_PARAMS['risk']['atr_target_mult']
        )
        rep = {'trained': False}
        if self.params.get('ml_enabled', True) and SKLEARN_AVAILABLE and len(y) > 0 and len(np.unique(y)) >= 2:
            cutoff = int(len(X) * initial_train_ratio)
            sig.ml_model.train(X[:cutoff], y[:cutoff], fcols, sample_weights=sample_weights[:cutoff] if sample_weights is not None else None)
            rep = {'trained': True}
        start = int(len(aligned) * initial_train_ratio)
        times = aligned.index
        last_train = start
        for i in range(start, len(aligned) - 1):
            if walkforward and self.params.get('ml_enabled', True) and SKLEARN_AVAILABLE:
                if (i - last_train) >= retrain_every:
                    Xw, yw, _, _, _ = self.feature_builder.build_ml_dataset(
                        df.loc[:times[i]],
                        horizon=horizon, timeframes=tfs, use_triple_barrier=True,
                        atr_mult_stop=DEFAULT_PARAMS['risk']['atr_stop_mult'],
                        atr_mult_target=DEFAULT_PARAMS['risk']['atr_target_mult']
                    )
                    if len(yw) > 0 and len(np.unique(yw)) >= 2:
                        sig.ml_model.train(Xw, yw, fcols)
                    last_train = i

            hist_slice = df.loc[:times[i]].copy()
            if len(hist_slice) < 40:
                continue
            row = aligned.iloc[i]
            cur = {
                'timestamp': as_iso(times[i].to_pydatetime()), 'price': float(row['close']),
                'open': float(row['open']), 'high': float(row['high']), 'low': float(row['low']),
                'close': float(row['close']), 'volume': float(row['volume'])
            }
            hlist = self._df_to_list(hist_slice)
            ind = TechnicalIndicators.calculate_indicators(hlist)
            if not ind:
                continue
            signal = sig.generate_signal(cur, ind, hlist, df_1m=hist_slice)
            eng.execute_trade(signal, cur, ind, params=self.params)
            if i == len(aligned) - 2 and eng.position != 0:
                eng._close_position(cur['price'], cur['timestamp'], "Fechamento Backtest")

        perf = eng.get_performance()
        eq = [t.get('capital_after', eng.initial_capital) for t in eng.trades_history if t.get('status') == 'closed']
        eq = [eng.initial_capital] + eq
        ret = pd.Series(eq).pct_change().dropna()
        sharpe = float(np.sqrt(252 * 24 * 60) * (ret.mean() / (ret.std() + 1e-9))) if not ret.empty else 0.0
        mdd = self._max_drawdown(eq)
        return {'success': True, 'train_report': rep, 'performance': perf,
                'final_capital': float(eng.current_capital),
                'total_pnl': float(eng.total_pnl), 'sharpe': sharpe,
                'max_drawdown': float(mdd), 'trades': eng.trades_history}

    def _df_to_list(self, df):
        out = []
        for idx, row in df.iterrows():
            out.append({
                'timestamp': as_iso(idx.to_pydatetime()), 'price': float(row['close']),
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row['volume'])
            })
        return out

    def _max_drawdown(self, equity):
        arr = np.array(equity, dtype=float)
        if len(arr) < 2:
            return 0.0
        roll = np.maximum.accumulate(arr)
        dd = (arr - roll) / roll
        return float(np.min(dd))

class Optimizer:
    def __init__(self, backtester): self.backtester = backtester
    def random_search(self, df, trials=20, objective='sharpe'):
        best = None; best_score = -1e9; cands = []
        for _ in range(trials):
            p = deepcopy(self.backtester.params)
            w_ml = random.choice([0.3, 0.4, 0.5, 0.6])
            w_mtf = random.choice([0.1, 0.2, 0.3])
            w_rule = float(np.clip(1.0 - w_ml - w_mtf, 0.1, 0.8))
            p['weights'] = {'ml': w_ml, 'mtf': w_mtf, 'rule': w_rule}
            p['thresholds']['buy'] = random.choice([55, 60, 65, 70])
            p['thresholds']['sell'] = random.choice([55, 60, 65, 70])
            p['vol_filters']['volume_ratio_min'] = random.choice([0.8, 0.9, 1.0, 1.1])
            p['vol_filters']['volume_ratio_max'] = random.choice([2.5, 3.0, 3.5])
            p['ml']['n_estimators'] = random.choice([100, 150, 180])
            p['ml']['max_depth'] = random.choice([4, 6, 8])
            p['ml']['min_samples_leaf'] = random.choice([3, 5, 8])

            bt = Backtester(self.backtester.feature_builder, self.backtester.signal_generator_class, p,
                            market_symbol=self.backtester.market_symbol)
            res = bt.run(df, walkforward=False, initial_train_ratio=0.6)
            if not res.get('success'):
                continue
            score = res.get('sharpe', 0.0) if objective == 'sharpe' else res.get('total_pnl', 0.0)
            cands.append({'params': deepcopy(p), 'result': res, 'score': score})
            if score > best_score:
                best_score = score
                best = cands[-1]
                logger.info(f"[{self.backtester.market_symbol}] Novo melhor {objective}: {best_score:.4f}")
        return {'best': best, 'candidates': cands}

# =====================================================================================
# Auto Param Scheduler (re-treino periódico e por mudança de regime)
# =====================================================================================
class AutoParamScheduler:
    """
    Roda em thread por mercado, avalia o regime e aplica automaticamente um dos 5 perfis.
    E dispara retreino de ML (se disponível) periodicamente e quando o regime muda.
    """
    def __init__(self, market_bot, interval_sec=120, min_hold_sec=600, retrain_minutes=90):
        self.market_bot = market_bot
        self.feature_builder = market_bot.feature_builder
        self.advisor = RegimeAdvisor(self.feature_builder)
        self.interval_sec = interval_sec
        self.min_hold_sec = min_hold_sec
        self.enabled = True
        self.current_profile = 'BALANCED'
        self.last_switch = None
        self.diagnostics = []
        self.thread = None
        self._stop = False
        self.lock = threading.Lock()
        self.retrain_every = max(999999, int(retrain_minutes))  # minutos
        self._last_retrain = None

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self._stop = False
            self.thread = threading.Thread(target=self._loop, name=f"AutoParams-{self.market_bot.ticker}", daemon=True)
            self.thread.start()
            logger.info(f"[{self.market_bot.ticker}] AutoParamScheduler iniciado")

    def stop(self):
        with self.lock:
            self._stop = True
            if self.thread:
                self.thread.join(timeout=2)
            logger.info(f"[{self.market_bot.ticker}] AutoParamScheduler parado")

    def set_config(self, enabled=None, interval_sec=None, min_hold_sec=None):
        with self.lock:
            if enabled is not None: self.enabled = bool(enabled)
            if interval_sec is not None: self.interval_sec = max(30, int(interval_sec))
            if min_hold_sec is not None: self.min_hold_sec = max(60, int(min_hold_sec))
            return self.get_state()

    def get_state(self):
        return {
            'enabled': self.enabled,
            'interval_sec': self.interval_sec,
            'min_hold_sec': self.min_hold_sec,
            'current_profile': self.current_profile,
            'last_switch': as_iso(self.last_switch) if self.last_switch else None,
            'diagnostics': list(self.diagnostics)
        }

    def _apply_profile(self, profile_name):
        base = self.market_bot.signal_generator.get_params()
        prof = PARAM_PROFILES.get(profile_name, PARAM_PROFILES['BALANCED'])
        base['weights'] = deepcopy(prof['weights'])
        base['thresholds']['buy'] = prof['thresholds']['buy']
        base['thresholds']['sell'] = prof['thresholds']['sell']
        self.market_bot.signal_generator.set_params(base)

    def _eligible_to_switch(self):
        if self.last_switch is None:
            return True
        return (datetime.now() - (self.last_switch or datetime.now())).total_seconds() >= self.min_hold_sec

    def _maybe_retrain(self, reason="periodic"):
        if not SKLEARN_AVAILABLE or not self.market_bot.signal_generator.get_params().get('ml_enabled', True):
            return
        # controle de frequência
        now = datetime.now()
        if self._last_retrain and (now - self._last_retrain).total_seconds() < self.retrain_every * 60:
            return
        self._last_retrain = now
        threading.Thread(target=self.market_bot.train_ml,
                         kwargs={'period': '14d', 'interval': '1m'},
                         name=f"ML-Retrain-{self.market_bot.ticker}",
                         daemon=True).start()
        logger.info(f"[{self.market_bot.ticker}] Retreino de ML disparado ({reason})")

    def _loop(self):
        while not self._stop:
            try:
                if not self.enabled:
                    time.sleep(self.interval_sec); continue
                df1 = self.market_bot.data_collector.get_historical_dataframe()
                if df1 is None or df1.empty or len(df1) < 120:
                    time.sleep(self.interval_sec); continue
                res = self.advisor.evaluate(df1.tail(600))
                proposed = res.get('profile', 'BALANCED')
                self.diagnostics = [
                    f"trend_score={res['trend_score']:.2f}",
                    f"rv_pct={res['rv_pct']:.1f}",
                    f"bb_width_pct={res['bb_width_pct']:.1f}"
                ] + res.get('diagnostics', [])

                # re-treino periódico
                self._maybe_retrain(reason="periodic")

                if proposed != self.current_profile and self._eligible_to_switch():
                    self._apply_profile(proposed)
                    self.current_profile = proposed
                    self.last_switch = datetime.now()
                    logger.info(f"[{self.market_bot.ticker}] AutoParams -> {proposed} | {self.diagnostics}")
                    # re-treino oportunístico por mudança de regime
                    self._maybe_retrain(reason="regime-change")
                time.sleep(self.interval_sec)
            except Exception as e:
                logger.error(f"[{self.market_bot.ticker}] AutoParams loop: {e}")
                time.sleep(self.interval_sec)

# =====================================================================================
# Bot (por Mercado)
# =====================================================================================
class MarketTradingBot:
    def __init__(self, ticker='BTCUSDT', market_name='BTC/USDT (Binance) | Crypto', bot_manager=None):
        self.ticker = ticker
        self.market_name = market_name
        self.data_collector = OptimizedDataCollector(BinanceProvider(self.ticker),
                                                     DEFAULT_PARAMS['data']['base_interval'],
                                                     DEFAULT_PARAMS['data']['max_hist_days'],
                                                     market_name=self.market_name)
        self.indicators_calculator = TechnicalIndicators()
        self.feature_builder = MultiTimeframeFeatureBuilder()
        self.ml_model = MLPatternModel(DEFAULT_PARAMS['ml'], model_id=self.ticker)
        self.signal_generator = AdvancedSignalGenerator(params=DEFAULT_PARAMS,
                                                        feature_builder=self.feature_builder,
                                                        ml_model=self.ml_model,
                                                        bot_manager=bot_manager if 'bot_manager' in globals() else None)
        self.trading_engine = TradingEngine(symbol=self.ticker)
        self.is_running = False
        self.trading_thread = None

        # Auto-params com retreino
        self.auto_scheduler = AutoParamScheduler(self, interval_sec=120, min_hold_sec=600, retrain_minutes=90)

    def start_bot(self):
        if not self.is_running:
            self.is_running = True
            self.data_collector.start_collection()
            self.trading_thread = threading.Thread(target=self._trading_loop, name=f"TradingLoop-{self.ticker}", daemon=True)
            self.trading_thread.start()
            self.auto_scheduler.start()
            logger.info(f"[{self.ticker}] Bot iniciado")
            return True
        return False

    def stop_bot(self):
        if self.is_running:
            self.is_running = False
            self.data_collector.stop_collection()
            if self.trading_thread:
                self.trading_thread.join(timeout=2)
            self.auto_scheduler.stop()
            logger.info(f"[{self.ticker}] Bot parado")
            return True
        return False

    def _trading_loop(self):
        while self.is_running:
            try:
                cur = self.data_collector.get_current_data()
                hist = self.data_collector.get_historical_data()
                df1 = self.data_collector.get_historical_dataframe()
                if cur and len(hist) >= 40 and df1 is not None and not df1.empty:
                    ind = self.indicators_calculator.calculate_indicators(hist)
                    if ind:
                        sig = self.signal_generator.generate_signal(cur, ind, hist, df_1m=df1)
                        self.trading_engine.execute_trade(sig, cur, ind, params=self.signal_generator.get_params())
                time.sleep(5)
            except Exception as e:
                logger.error(f"[{self.ticker}] Loop trading: {e}")
                time.sleep(15)

    def get_status(self):
        cur = self.data_collector.get_current_data()
        cur_native = {}
        if cur:
            for k in ['price', 'open', 'high', 'low', 'close', 'volume']:
                if k in cur and cur[k] is not None:
                    cur_native[k] = float(cur[k])
            if 'timestamp' in cur:
                cur_native['timestamp'] = str(cur['timestamp'])
        hist = self.data_collector.get_historical_data()
        df1 = self.data_collector.get_historical_dataframe()
        ind = {}
        if hist and len(hist) >= 20:
            ind = self.indicators_calculator.calculate_indicators(hist)
        sig = {'signal': 'HOLD', 'strength': 0, 'reason': 'Sem dados'}
        if cur and ind and df1 is not None and not df1.empty:
            sig = self.signal_generator.generate_signal(cur, ind, hist, df_1m=df1)
        perf = self.trading_engine.get_performance()
        ml_enabled_flag = bool(self.signal_generator.get_params().get('ml_enabled', False) and SKLEARN_AVAILABLE)
        ml_info = {
            'enabled': ml_enabled_flag,
            'trained': bool(self.ml_model.trained),
            'sklearn_available': bool(SKLEARN_AVAILABLE),
            'quality': float(getattr(self.ml_model, 'quality', 0.0)),
            'quality_smooth': float(getattr(self.ml_model, 'quality_smooth', 0.0)),
            'metrics': self.ml_model.metrics,
            'last_trained': self.ml_model.last_trained,
            'features_used': int(self.ml_model.metrics.get('features_used', 0) if self.ml_model.metrics else 0),
            'model_path': str(self.ml_model.model_path)
        }
        auto_state = self.auto_scheduler.get_state()
        # Define last_trade before using it
        last_trade = self.trading_engine.trades_history[-1] if self.trading_engine.trades_history else None
        return {
            'is_running': self.is_running, 'market': self.market_name, 'ticker': self.ticker,
            'data_source': self.data_collector.data_source, 'last_fetch_ok': bool(self.data_collector.last_fetch_ok),
            'current_data': cur_native, 'indicators': ind, 'signal': sig, 'position': int(self.trading_engine.position),
            'entry_price': float(self.trading_engine.entry_price),
            'current_pnl': float(self.trading_engine.current_pnl),
            'total_pnl': float(self.trading_engine.total_pnl),
            'current_capital': float(self.trading_engine.current_capital),
            'trades_today': int(self.trading_engine.trades_today),
            'data_points': int(len(hist) if hist else 0),
            'performance': perf,
            'ml': ml_info,
            'params': self.signal_generator.get_params(),
            'current_stop_loss': float(last_trade['stop_loss']) if last_trade else None,
            'auto_params': auto_state

        }

    def get_chart_data(self):
        candles = self.data_collector.get_chart_data()
        tps = self.trading_engine.get_trade_points()
        out = []
        for d in candles:
            out.append({
                'timestamp': d['timestamp'], 'open': float(d['open']), 'high': float(d['high']),
                'low': float(d['low']), 'close': float(d['close']), 'volume': float(d['volume'])
            })
        return {'ticker': self.ticker, 'candlesticks': out, 'trade_points': tps}

    # Fallback dataset builder citado no pseudo original – reimplementado mais simples
    def _build_dataset_fallback(self, df, base_horizon, base_tfs):
        tf_candidates = [base_tfs, [1, 5], [1, 15], [5, 15], [1], [5], [15]]
        horizons = [base_horizon, 1, 5]
        for tfs in tf_candidates:
            for hz in horizons:
                X, y, fn, feats, aligned, sample_weights, metrics = self.feature_builder.build_ml_dataset(
                    df, horizon=hz, timeframes=tfs, use_triple_barrier=True,
                    atr_mult_stop=DEFAULT_PARAMS['risk']['atr_stop_mult'],
                    atr_mult_target=DEFAULT_PARAMS['risk']['atr_target_mult']
                )
                if X.size > 0 and len(y) >= 120 and len(np.unique(y)) >= 2:
                    return X, y, fn, aligned.index, hz, tfs, 'triple_barrier', metrics
        return np.empty((0, 0)), np.array([]), [], [], base_horizon, base_tfs, 'failed'

    def train_ml(self, period="14d", interval="1m"):
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'scikit-learn indisponível'}
        attempts = [(period, interval)]
        last_err = "Desconhecido"
        for per, inter in attempts:
            df = self.data_collector.get_extended_historical_dataframe(period=per, interval=inter)
            if df is None or df.empty or len(df) < 500:
                last_err = f"Histórico insuficiente ({per}/{inter})"
                continue
            tfs = self.signal_generator.get_params().get('timeframes', [5, 15])
            base_hz = self.signal_generator.get_params().get('ml', {}).get('horizon', 3)
            X, y, fn, idx, hz_used, tfs_used, method, metrics = self._build_dataset_fallback(df, base_hz, tfs)
            if X.size == 0 or len(y) < 120 or len(np.unique(y)) < 2:
                last_err = f"Dataset vazio ({per}, horizon={hz_used})"
                continue
            # sample_weights is not returned by _build_dataset_fallback, so set to None
            sample_weights = None
            rep = self.ml_model.train(X, y, fn, sample_weights=sample_weights, metrics=metrics)
            if rep.get('trained'):
                return {'success': True, 'report': {**rep, 'period': per, 'interval': inter,
                                                    'horizon': hz_used, 'timeframes_used': tfs_used,
                                                    'method': method}}
            last_err = rep.get('error', 'Falha treino ML')
        return {'success': False, 'error': last_err}

    def backtest(self, period="7d", interval="1m", walkforward=False):
        df = self.data_collector.get_extended_historical_dataframe(period=period, interval=interval)
        if df is None or df.empty:
            return {'success': False, 'error': 'Histórico indisponível'}
        bt = Backtester(self.feature_builder, AdvancedSignalGenerator, self.signal_generator.get_params(), market_symbol=self.ticker)
        return bt.run(df, walkforward=walkforward, initial_train_ratio=0.6)

    def optimize(self, period="7d", interval="1m", trials=20, objective='sharpe'):
        df = self.data_collector.get_extended_historical_dataframe(period=period, interval=interval)
        if df is None or df.empty:
            return {'success': False, 'error': 'Histórico indisponível'}
        bt = Backtester(self.feature_builder, AdvancedSignalGenerator, self.signal_generator.get_params(), market_symbol=self.ticker)
        opt = Optimizer(bt); res = opt.random_search(df, trials=trials, objective=objective)
        if res.get('best'):
            self.signal_generator.set_params(res['best']['params'])
            return {'success': True, 'best': res['best'], 'count': len(res['candidates'])}
        return {'success': False, 'error': 'Otimização sem solução válida'}

# =====================================================================================
# Manager (Multi-Mercado)
# =====================================================================================
class BotManager:
    def __init__(self):
        """
        Gerenciador central de todos os bots de trading.
        Cria uma instância de MarketTradingBot para cada mercado em MARKETS.
        Não possui seus próprios data_collector, signal_generator, etc. — isso é responsabilidade dos bots individuais.
        """
        self.bots = {}
        for mk, cfg in MARKETS.items():
            try:
                self.bots[mk] = MarketTradingBot(
                    ticker=cfg['ticker'],
                    market_name=cfg['name'],
                    bot_manager=self  # ← passa a si mesmo para permitir regime filter global
                )
                logger.info(f"[BotManager] Bot {mk} ({cfg['ticker']}) inicializado.")
            except Exception as e:
                logger.error(f"[BotManager] Falha ao inicializar bot {mk}: {e}")
                self.bots[mk] = None

    def _get_bot(self, market):
        mk = normalize_market_key(market)
        if mk is None:
            return None, {'success': False, 'error': 'Mercado inválido. Use BTC, ETH ou BNB.'}
        return self.bots[mk], None

    def list_markets(self):
        out = []
        for mk, cfg in MARKETS.items():
            try:
                bot = self.bots[mk]
                st = bot.get_status()
                out.append({'market': mk, 'ticker': cfg['ticker'], 'name': cfg['name'],
                            'is_running': bool(st.get('is_running', False)),
                            'ml_trained': bool(st.get('ml', {}).get('trained', False)),
                            'last_fetch_ok': bool(st.get('last_fetch_ok', False))})
            except Exception as e:
                out.append({'market': mk, 'ticker': cfg['ticker'], 'name': cfg['name'],
                            'is_running': False, 'ml_trained': False, 'last_fetch_ok': False,
                            'error': str(e)})
        return out

    def start(self, market):
        if str(market).strip().upper() == 'ALL':
            res = {}
            for mk in MARKETS.keys():
                try:
                    res[mk] = {'success': self.bots[mk].start_bot()}
                except Exception as e:
                    logger.exception("start(%s) error", mk)
                    res[mk] = {'success': False, 'error': str(e)}
            return res
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return {'success': bot.start_bot()}
        except Exception as e:
            logger.exception("start(%s) error", market)
            return {'success': False, 'error': str(e)}

    def stop(self, market):
        if str(market).strip().upper() == 'ALL':
            res = {}
            for mk in MARKETS.keys():
                try:
                    res[mk] = {'success': self.bots[mk].stop_bot()}
                except Exception as e:
                    logger.exception("stop(%s) error", mk)
                    res[mk] = {'success': False, 'error': str(e)}
            return res
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return {'success': bot.stop_bot()}
        except Exception as e:
            logger.exception("stop(%s) error", market)
            return {'success': False, 'error': str(e)}

    def reset(self, market):
        if str(market).strip().upper() == 'ALL':
            for mk, cfg in MARKETS.items():
                try:
                    self.bots[mk].stop_bot()
                    time.sleep(0.3)
                    self.bots[mk] = MarketTradingBot(ticker=cfg['ticker'], market_name=cfg['name'])
                except Exception as e:
                    logger.exception("reset(%s) error", mk)
            return {'success': True}
        bot, err = self._get_bot(market)
        if err: return err
        mk = normalize_market_key(market)
        cfg = MARKETS[mk]
        try:
            self.bots[mk].stop_bot()
            time.sleep(0.3)
            self.bots[mk] = MarketTradingBot(ticker=cfg['ticker'], market_name=cfg['name'])
            return {'success': True}
        except Exception as e:
            logger.exception("reset(%s) error", market)
            return {'success': False, 'error': str(e)}

    def status(self, market):
        if str(market).strip().upper() == 'ALL':
            res = {}
            for mk in MARKETS.keys():
                try:
                    st = self.bots[mk].get_status()
                    res[mk] = st
                except Exception as e:
                    logger.exception("status(%s) error", mk)
                    res[mk] = {'success': False, 'error': str(e)}
            return res
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return bot.get_status()
        except Exception as e:
            logger.exception("status(%s) error", market)
            return {'success': False, 'error': str(e)}

    def chart(self, market):
        if str(market).strip().upper() == 'ALL':
            res = {}
            for mk in MARKETS.keys():
                try:
                    res[mk] = self.bots[mk].get_chart_data()
                except Exception as e:
                    logger.exception("chart(%s) error", mk)
                    res[mk] = {'success': False, 'error': str(e)}
            return res
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return bot.get_chart_data()
        except Exception as e:
            logger.exception("chart(%s) error", market)
            return {'success': False, 'error': str(e)}

    def train_ml(self, market, period='14d', interval='1m'):
        if str(market).strip().upper() == 'ALL':
            results = {}
            for mk in MARKETS.keys():
                try:
                    results[mk] = self.bots[mk].train_ml(period=period, interval=interval)
                except Exception as e:
                    logger.exception("train_ml(%s) error", mk)
                    results[mk] = {'success': False, 'error': str(e)}
            return {'success': True, 'results': results}
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return bot.train_ml(period=period, interval=interval)
        except Exception as e:
            logger.exception("train_ml(%s) error", market)
            return {'success': False, 'error': str(e)}

    def backtest(self, market, period='7d', interval='1m', walkforward=False):
        if str(market).strip().upper() == 'ALL':
            results = {}
            for mk in MARKETS.keys():
                try:
                    results[mk] = self.bots[mk].backtest(period=period, interval=interval, walkforward=walkforward)
                except Exception as e:
                    logger.exception("backtest(%s) error", mk)
                    results[mk] = {'success': False, 'error': str(e)}
            return {'success': True, 'results': results}
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return bot.backtest(period=period, interval=interval, walkforward=walkforward)
        except Exception as e:
            logger.exception("backtest(%s) error", market)
            return {'success': False, 'error': str(e)}

    def optimize(self, market, period='14d', interval='1m', trials=20, objective='sharpe'):
        if str(market).strip().upper() == 'ALL':
            results = {}
            for mk in MARKETS.keys():
                try:
                    results[mk] = self.bots[mk].optimize(period=period, interval=interval, trials=trials, objective=objective)
                except Exception as e:
                    logger.exception("optimize(%s) error", mk)
                    results[mk] = {'success': False, 'error': str(e)}
            return {'success': True, 'results': results}
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return bot.optimize(period=period, interval=interval, trials=trials, objective=objective)
        except Exception as e:
            logger.exception("optimize(%s) error", market)
            return {'success': False, 'error': str(e)}

    def get_params(self, market):
        if str(market).strip().upper() == 'ALL':
            res = {}
            for mk in MARKETS.keys():
                try:
                    res[mk] = self.bots[mk].signal_generator.get_params()
                except Exception as e:
                    logger.exception("get_params(%s) error", mk)
                    res[mk] = {'success': False, 'error': str(e)}
            return res
        bot, err = self._get_bot(market)
        if err: return err
        try:
            return bot.signal_generator.get_params()
        except Exception as e:
            logger.exception("get_params(%s) error", market)
            return {'success': False, 'error': str(e)}

    def set_params(self, market, payload):
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    deep_update(d[k], v)
                else:
                    d[k] = v
        if str(market).strip().upper() == 'ALL':
            out = {}
            for mk in MARKETS.keys():
                try:
                    current = self.bots[mk].signal_generator.get_params()
                    deep_update(current, payload)
                    self.bots[mk].signal_generator.set_params(current)
                    out[mk] = current
                except Exception as e:
                    logger.exception("set_params(%s) error", mk)
                    out[mk] = {'success': False, 'error': str(e)}
            return {'success': True, 'params': out}
        bot, err = self._get_bot(market)
        if err: return err
        try:
            current = bot.signal_generator.get_params()
            deep_update(current, payload)
            bot.signal_generator.set_params(current)
            return {'success': True, 'params': bot.signal_generator.get_params()}
        except Exception as e:
            logger.exception("set_params(%s) error", market)
            return {'success': False, 'error': str(e)}

    # Auto-params endpoints helpers
    def auto_get(self, market):
        bot, err = self._get_bot(market)
        if err: return err
        return {'success': True, 'auto_params': bot.auto_scheduler.get_state()}

    def auto_set(self, market, payload):
        bot, err = self._get_bot(market)
        if err: return err
        enabled = payload.get('enabled', None)
        interval_sec = payload.get('interval_sec', None)
        min_hold_sec = payload.get('min_hold_sec', None)
        st = bot.auto_scheduler.set_config(enabled=enabled, interval_sec=interval_sec, min_hold_sec=min_hold_sec)
        return {'success': True, 'auto_params': st}

# Manager global
bot_manager = BotManager()

# =====================================================================================
# Routes
# =====================================================================================
@app.route('/')
def index():
    return send_from_directory('.', 'interface_advanced.html')

@app.route('/api/markets')
def api_markets():
    return json_response({'markets': bot_manager.list_markets()})

# Start/Stop/Status/Chart/Reset por mercado ou all
@app.route('/api/bot/<market>/start', methods=['POST'])
def api_start_market(market):
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.start('ALL'))
    return json_response(bot_manager.start(m))

@app.route('/api/bot/<market>/stop', methods=['POST'])
def api_stop_market(market):
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.stop('ALL'))
    return json_response(bot_manager.stop(m))

@app.route('/api/bot/<market>/status')
def api_status_market(market):
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.status('ALL'))
    return json_response(bot_manager.status(m))

@app.route('/api/bot/<market>/chart')
def api_chart_market(market):
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.chart('ALL'))
    return json_response(bot_manager.chart(m))

@app.route('/api/bot/<market>/reset', methods=['POST'])
def api_reset_market(market):
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.reset('ALL'))
    return json_response(bot_manager.reset(m))

# Train ML / Backtest / Optimize
@app.route('/api/bot/<market>/train_ml', methods=['POST'])
def api_train_ml_market(market):
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.train_ml('ALL', period=period, interval=interval))
    return json_response(bot_manager.train_ml(m, period=period, interval=interval))

@app.route('/api/bot/<market>/backtest', methods=['POST'])
def api_backtest_market(market):
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    walkforward = bool(payload.get('walkforward', False))
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.backtest('ALL', period=period, interval=interval, walkforward=walkforward))
    return json_response(bot_manager.backtest(m, period=period, interval=interval, walkforward=walkforward))

@app.route('/api/bot/<market>/optimize', methods=['POST'])
def api_optimize_market(market):
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    trials = int(payload.get('trials', DEFAULT_PARAMS['optimization']['trials']))
    objective = payload.get('objective', DEFAULT_PARAMS['optimization']['objective'])
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.optimize('ALL', period=period, interval=interval, trials=trials, objective=objective))
    return json_response(bot_manager.optimize(m, period=period, interval=interval, trials=trials, objective=objective))

# Params
@app.route('/api/bot/<market>/params', methods=['GET'])
def api_params_market(market):
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.get_params('ALL'))
    return json_response(bot_manager.get_params(m))

@app.route('/api/bot/<market>/set_params', methods=['POST'])
def api_set_params_market(market):
    payload = request.get_json(force=True, silent=True) or {}
    m = str(market).strip().upper()
    if m == 'ALL':
        return json_response(bot_manager.set_params('ALL', payload))
    return json_response(bot_manager.set_params(m, payload))

# Auto-params (novo)
@app.route('/api/bot/<market>/auto_params', methods=['GET', 'POST'])
def api_auto_params(market):
    m = str(market).strip().upper()
    if request.method == 'GET':
        if m == 'ALL':
            return json_response({'success': False, 'error': 'Use mercado específico (BTC/ETH/BNB) para auto_params'})
        return json_response(bot_manager.auto_get(m))
    else:
        payload = request.get_json(force=True, silent=True) or {}
        if m == 'ALL':
            return json_response({'success': False, 'error': 'Use mercado específico (BTC/ETH/BNB) para auto_params'})
        return json_response(bot_manager.auto_set(m, payload))

# Rotas legadas (mantidas para compatibilidade; usam BTC como padrão)
@app.route('/api/bot/start', methods=['POST'])
def legacy_start_bot(): return json_response(bot_manager.start('BTC'))

@app.route('/api/bot/stop', methods=['POST'])
def legacy_stop_bot(): return json_response(bot_manager.stop('BTC'))

@app.route('/api/bot/status')
def legacy_get_bot_status(): return json_response(bot_manager.status('BTC'))

@app.route('/api/bot/chart')
def legacy_get_chart_data(): return json_response(bot_manager.chart('BTC'))

@app.route('/api/bot/reset', methods=['POST'])
def legacy_reset_bot(): return json_response(bot_manager.reset('BTC'))

@app.route('/api/bot/train_ml', methods=['POST'])
def legacy_api_train_ml():
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    return json_response(bot_manager.train_ml('BTC', period=period, interval=interval))

@app.route('/api/bot/backtest', methods=['POST'])
def legacy_api_backtest():
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    walkforward = bool(payload.get('walkforward', False))
    return json_response(bot_manager.backtest('BTC', period=period, interval=interval, walkforward=walkforward))

@app.route('/api/bot/optimize', methods=['POST'])
def legacy_api_optimize():
    payload = request.get_json(force=True, silent=True) or {}
    period = payload.get('period', '7d'); interval = payload.get('interval', '1m')
    trials = int(payload.get('trials', DEFAULT_PARAMS['optimization']['trials']))
    objective = payload.get('objective', DEFAULT_PARAMS['optimization']['objective'])
    return json_response(bot_manager.optimize('BTC', period=period, interval=interval, trials=trials, objective=objective))

@app.route('/api/bot/params', methods=['GET'])
def legacy_api_params(): return json_response(bot_manager.get_params('BTC'))

@app.route('/api/bot/set_params', methods=['POST'])
def legacy_api_set_params():
    payload = request.get_json(force=True, silent=True) or {}
    return json_response(bot_manager.set_params('BTC', payload))

if __name__ == '__main__':
    print("🚀 Trading Bot Avançado - Multi-Mercados (BTC, ETH, BNB)")
    print("✅ Dados reais (Binance) 1m | MTF + ML robusto por mercado")
    print(f"🧠 scikit-learn disponível: {SKLEARN_AVAILABLE}")
    print("🌐 API: http://localhost:5000")
    app.run(host='0.0.0.0', port=3030, debug=False)