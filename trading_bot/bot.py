import ccxt
import pandas as pd
import time
import logging
import random
import sys
import numpy as np
import json
import os
from datetime import datetime

# 設定日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class BinanceTrendGrid:
    def __init__(self, config):
        self.config = config
        api_key = config.get('api_key', 'NONE')
        api_secret = config.get('api_secret', 'NONE')
        symbol = config.get('symbol', 'BTC/USDT')
        settings = config.get('settings', {})
        dry_run = config.get('dry_run', True)

        self.dry_run = dry_run
        
        # 1. 交易所初始化
        exchange_config = {
            'apiKey': api_key if api_key != "NONE" else '',
            'secret': api_secret if api_secret != "NONE" else '',
            'enableRateLimit': True,
            'timeout': 10000,
            'options': {'defaultType': 'spot'}
        }
        self.exchange = ccxt.binance(exchange_config)

        self.backup_exchanges = [
            ccxt.kucoin({'enableRateLimit': True, 'timeout': 10000}),
            ccxt.bitget({'enableRateLimit': True, 'timeout': 10000}),
            ccxt.kraken({'enableRateLimit': True, 'timeout': 10000})
        ]

        # 2. 策略參數 (V24.0 動能突破與手續費防護版)
        self.symbol = symbol
        self.grid_count = settings.get('grid_count', 5)
        self.base_tp_percent = settings.get('tp_percent', 0.015)
        self.ma_fast_period = settings.get('ma_period', 50)        # V24 建議: 50
        self.ma_slow_period = settings.get('ma_slow_period', 200)   # V24 建議: 200
        self.atr_multiplier = settings.get('atr_multiplier', 1.0) # V24: 寬間距防禦

        # 3. 槓桿與風險控管
        self.base_leverage = settings.get('leverage', 8)
        self.account_sl_percent = settings.get('account_sl_percent', 0.20)

        self.order_size_ratio = settings.get('order_size_ratio', 0.02) # 降低初始比例
        self.fee_rate = 0.001

        # 4. 狀態管理
        self.current_trend = None
        self.data_source = "None"
        self.pending_orders = []
        self.is_halted = False
        self.peak_unrealized = 0
        self.total_fees_paid = 0.0 # 新增：追蹤手續費損耗

        # 5. 資產統計
        self.initial_balance = 10000.0
        self.balance_usdt = self.initial_balance
        self.position_amount = 0.0
        self.avg_entry_price = 0.0

        # 6. 統計數據
        self.trade_history = []
        self.grid_pnl = 0.0
        self.trend_pnl = 0.0
        self.max_equity = self.initial_balance
        self.current_mdd = 0.0

        print(f"Bot initialized for {self.symbol} (Dry Run: {self.dry_run})")

    def get_total_equity(self, current_price):
        unrealized = 0
        if abs(self.position_amount) > 1e-9:
            if self.position_amount > 0:
                unrealized = (current_price - self.avg_entry_price) * self.position_amount
            else:
                unrealized = (self.avg_entry_price - current_price) * abs(self.position_amount)

        pos_val = abs(self.position_amount) * (self.avg_entry_price if self.avg_entry_price > 0 else current_price)
        used_margin = pos_val / self.base_leverage
        return self.balance_usdt + used_margin + unrealized

    def _calculate_tema(self, series, period):
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    def calculate_indicators(self, df):
        """指標進化：增加動能強度(ADX)與波動過濾"""
        df = df.copy()
        df['tema_fast'] = self._calculate_tema(df['close'], self.ma_fast_period)
        df['tema_long'] = self._calculate_tema(df['close'], self.ma_slow_period)

        # MACD 與 動量變化
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_hist'] = exp1 - exp2 - (exp1 - exp2).ewm(span=9, adjust=False).mean()
        df['macd_slope'] = df['macd_hist'].diff()

        # ATR 波動過濾
        high_low = df['high'] - df['low']
        high_cp = (df['high'] - df['close'].shift(1)).abs()
        low_cp = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=20).mean()
        df['atr_ma'] = df['atr'].rolling(window=200).mean()

        # ADX (趨勢強度)
        df['plus_dm'] = np.where((df['high'].diff() > df['low'].diff()) & (df['high'].diff() > 0), df['high'].diff(), 0)
        df['minus_dm'] = np.where((df['low'].diff() > df['high'].diff()) & (df['low'].diff() > 0), df['low'].diff(), 0)
        # Using a simple rolling mean approach for TR in ADX calc as per original code logic
        # Ideally wildcard import or explicit might be better but sticking to original logic structure
        tr_sm = df['tr'].rolling(14).mean() + 1e-10
        # This ADX calculation in original code used rolling(14) on the components directly
        # Preserving original logic
        df['adx'] = 100 * (abs(df['plus_dm'].rolling(14).mean() - df['minus_dm'].rolling(14).mean()) / (df['plus_dm'].rolling(14).mean() + df['minus_dm'].rolling(14).mean() + 1e-10)).rolling(14).mean()

        return df

    def fetch_recent_data(self, timeframe='1m', limit=300):
        """Fetch recent candles for live execution"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def record_trade(self, time, side, price, amount, label="", pnl=0.0):
        fee = (price * amount) * self.fee_rate
        self.total_fees_paid += fee
        
        trade_info = {
            'time': str(time), 'side': side, 'price': price,
            'amount': amount, 'label': label, 'pnl': pnl, 'fee': fee
        }
        self.trade_history.append(trade_info)
        
        log_msg = f"[TRADE] {side.upper()} {amount} @ {price} | Label: {label} | PnL: {pnl:.2f}"
        if any(kw in label for kw in ["TP", "Breakout", "Shield"]):
            self.grid_pnl += pnl
        else:
            self.trend_pnl += pnl
        
        logging.info(log_msg)

    def deploy_grid(self, current_price, trend, indicators, timestamp=None):
        if self.is_halted: return
        atr = indicators['atr']
        adx = indicators['adx']

        # 趨勢反轉平倉 (增加 ADX 緩衝)
        if self.current_trend is not None and trend != self.current_trend:
            if abs(self.position_amount) > 1e-8:
                side = 'sell' if self.position_amount > 0 else 'buy'
                margin_released = (abs(self.position_amount) * self.avg_entry_price) / self.base_leverage
                pnl = ((current_price - self.avg_entry_price) * self.position_amount if self.position_amount > 0 else
                       (self.avg_entry_price - current_price) * abs(self.position_amount))
                pnl -= (abs(self.position_amount) * current_price * self.fee_rate)

                self.balance_usdt += margin_released + pnl
                self.record_trade(timestamp, side, current_price, abs(self.position_amount), label="Trend-Reverse-Exit", pnl=pnl)
                self.position_amount, self.avg_entry_price, self.peak_unrealized = 0, 0, 0
                logging.info(f"Trend Reversal: Closed position. PnL: {pnl:.2f}")

        if self.current_trend != trend:
            logging.info(f"Trend Changed to {trend}")
            
        self.current_trend = trend

        # V24 重點：不再提前佈買單，僅根據突破動能決定是否加倉
        self.pending_orders = []
        if adx > 30: # 僅在強勢區間掛出突破加碼單
            interval = atr * self.atr_multiplier
            for i in range(1, self.grid_count + 1):
                # 做多網格：在當前價格「上方」掛買單 (追漲)
                # 做空網格：在當前價格「下方」掛賣單 (追跌)
                price = current_price + (i * interval) if trend == 'Long' else current_price - (i * interval)
                side = 'buy' if trend == 'Long' else 'sell'
                self.pending_orders.append({'price': price, 'side': side, 'label': f'V24-Breakout-{i}'})
            if len(self.pending_orders) > 0:
                logging.info(f"Deployed {len(self.pending_orders)} breakout orders (Trend: {trend}, ATR: {atr:.2f})")

    def handle_execution(self, current_price, high=None, low=None, timestamp=None, indicators=None):
        if self.is_halted: return False
        new_pending = []
        executed = False
        ch, cl = (high, low) if high is not None else (current_price, current_price)

        macd_slope = indicators['macd_slope']
        # adx = indicators['adx'] # Unused variable
        atr = indicators['atr']
        atr_ma = indicators['atr_ma']

        # --- V24.0 核心：手續費防護與動能收割 ---
        if abs(self.position_amount) > 1e-8:
            unrealized_pct = (current_price / self.avg_entry_price - 1) if self.position_amount > 0 else (self.avg_entry_price / current_price - 1)
            self.peak_unrealized = max(self.peak_unrealized, unrealized_pct)

            # 1. 動態利潤鎖定：浮盈達 0.6% 且動能轉弱，立即全平。
            # 目標：獲利必須 > 2 倍手續費
            if unrealized_pct > 0.006 and abs(macd_slope) < 0:
                self.execute_partial_close(timestamp, current_price, 1.0, "Momentum-Fee-Shield-Exit")
                return True

            # 2. 嚴格止損：單層浮虧破 1.5% 直接切斷
            if unrealized_pct < -0.015:
                self.execute_partial_close(timestamp, current_price, 1.0, "Hard-Risk-SL")
                return True

        # 突破撮合邏輯
        for order in self.pending_orders:
            is_filled = False

            # V24: 判斷是否發生突破 (追漲/追跌)
            if (order['side'] == 'buy' and ch >= order['price']) or (order['side'] == 'sell' and cl <= order['price']):

                # 波動過濾：若 ATR 是平均的 2.5 倍，視為異常插針，拒絕進場
                if atr > (atr_ma * 2.5):
                    new_pending.append(order)
                    continue

                is_filled = True
                equity = self.get_total_equity(current_price)

                # 複投下單
                dynamic_order_size = equity * self.order_size_ratio
                amount = dynamic_order_size / order['price']
                margin = dynamic_order_size / self.base_leverage
                fee = dynamic_order_size * self.fee_rate

                if order['side'] == 'buy':
                    if self.position_amount >= 0:
                        self.avg_entry_price = ((self.avg_entry_price * self.position_amount) + (order['price'] * amount)) / (self.position_amount + amount)
                        self.position_amount += amount
                        self.balance_usdt -= (margin + fee)
                        pnl = 0
                    else: # 平空
                        pnl = (self.avg_entry_price - order['price']) * amount - fee
                        self.balance_usdt += ((amount * self.avg_entry_price) / self.base_leverage + pnl)
                        self.position_amount += amount

                    if self.position_amount >= 0:
                        # 提高止盈空間至 1.2%
                        tp_target = order['price'] * (1 + max(self.base_tp_percent, 0.012))
                        new_pending.append({'price': tp_target, 'side': 'sell', 'label': f"TP-Breakout", 'entry_price': order['price']})

                else: # Sell Side
                    if self.position_amount > 1e-8:
                        entry_p = order.get('entry_price', self.avg_entry_price)
                        pnl = (order['price'] - entry_p) * amount - (fee + (amount * entry_p * self.fee_rate))
                        self.balance_usdt += ((amount * entry_p) / self.base_leverage + pnl)
                        self.position_amount -= amount
                    else:
                        self.avg_entry_price = ((abs(self.avg_entry_price * self.position_amount)) + (order['price'] * amount)) / abs(self.position_amount - amount)
                        self.position_amount -= amount
                        self.balance_usdt -= (margin + fee)
                        pnl = 0

                    if self.position_amount <= 0:
                        tp_target = order['price'] * (1 - max(self.base_tp_percent, 0.012))
                        new_pending.append({'price': tp_target, 'side': 'buy', 'label': f"RE-Breakout", 'entry_price': order['price']})

                self.record_trade(timestamp, order['side'], order['price'], amount, order['label'], pnl=pnl)
                executed = True

            if not is_filled: new_pending.append(order)

        self.pending_orders = new_pending
        return executed

    def execute_partial_close(self, timestamp, price, pct, label):
        if abs(self.position_amount) < 1e-9: return
        close_amt = abs(self.position_amount) * pct
        side = 'sell' if self.position_amount > 0 else 'buy'
        margin_released = (close_amt * self.avg_entry_price) / self.base_leverage
        pnl = ((price - self.avg_entry_price) * close_amt if self.position_amount > 0 else (self.avg_entry_price - price) * close_amt)
        pnl -= (close_amt * price * self.fee_rate)
        self.balance_usdt += margin_released + pnl
        self.record_trade(timestamp, side, price, close_amt, label=label, pnl=pnl)
        self.position_amount = self.position_amount * (1 - pct)
        if abs(self.position_amount) < 1e-8:
            self.position_amount, self.avg_entry_price, self.peak_unrealized = 0, 0, 0
        logging.info(f"Partial Close ({pct*100}%): {label} | PnL: {pnl:.2f}")

    def run_forever(self):
        print(f"Starting continuous loop for {self.symbol}...")
        while True:
            try:
                # 1. Fetch data
                df = self.fetch_recent_data(limit=300)
                if len(df) < self.ma_slow_period + 10:
                    logging.warning("Not enough data yet, waiting...")
                    time.sleep(60)
                    continue

                # 2. Calculate Indicators
                df = self.calculate_indicators(df)
                latest_row = df.iloc[-1]
                
                # Check if it's a new candle or just update
                # For safety, we just run logic on latest completed info or close price
                # Here using latest close as "current price"
                
                current_price = latest_row['close']
                current_time = latest_row['timestamp']
                
                # V24 趨勢共振判斷
                is_long = (latest_row['tema_fast'] > latest_row['tema_long']) and (latest_row['close'] > latest_row['tema_fast'])
                is_short = (latest_row['tema_fast'] < latest_row['tema_long']) and (latest_row['close'] < latest_row['tema_fast'])
                trend = 'Long' if is_long else 'Short' if is_short else self.current_trend

                inds = {
                    'macd_slope': latest_row['macd_slope'], 'atr': latest_row['atr'],
                    'adx': latest_row['adx'], 'atr_ma': latest_row['atr_ma']
                }

                # 3. Strategy Execution Logic
                if trend != self.current_trend:
                    self.deploy_grid(current_price, trend, inds, timestamp=current_time)

                self.handle_execution(current_price, high=latest_row['high'], low=latest_row['low'], timestamp=current_time, indicators=inds)

                # 4. Status Update Log
                equity = self.get_total_equity(current_price)
                logging.info(f"Ping | Price: {current_price} | Equity: {equity:.2f} | Trend: {trend} | Pos: {self.position_amount}")
                
                # Wait for next minute
                # Sleep 60 seconds (simple loop)
                time.sleep(60)

            except Exception as e:
                logging.error(f"Loop Error: {e}")
                time.sleep(60)


def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("config.json not found, using defaults")
        return {}
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

if __name__ == "__main__":
    config = load_config()
    
    # Check if we should run backtest or live mode
    # For docker deployment, we simply start the live loop
    
    bot = BinanceTrendGrid(config)
    
    # In a real scenario, you might want a flag to switch modes
    # But for "continuous execution in docker", we default to run_forever
    bot.run_forever()
