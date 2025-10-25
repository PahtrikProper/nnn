import sys
import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import random
import time
import traceback
import threading
import warnings
from datetime import datetime, timezone
from binance.spot import Spot as Client
import concurrent.futures
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QHBoxLayout, QLineEdit, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCursor

warnings.filterwarnings("ignore", message="resource_tracker")

# ===== CONFIG =====
SYMBOL = "ASTERUSDT"
QUOTE_ASSET = "USDT"
BASE_ASSET = "ASTER"
MIN_TRADE_AMOUNT = 5.0
OPTIMIZATION_DAYS = 3
N_RANDOM_TRIALS = 3000
MAX_CPU = max(1, int(os.cpu_count() * 0.8))

CLIENT = None
API_KEY = ""
API_SECRET = ""

def create_client(api_key, api_secret):
    return Client(api_key=api_key, api_secret=api_secret)

# ===== Paper Trading State =====
class PaperWallet:
    def __init__(self, usdt=1000.0, aster=0.0):
        self.reset(usdt, aster)
    def reset(self, usdt=1000.0, aster=0.0):
        self.usdt = usdt
        self.aster = aster
        self.last_entry = None
        self.active_params = None
        self.open_pos = False

PAPER = PaperWallet()

# ===== Core Trading Logic (Unchanged) =====
async def fetch_binance_historical_async(symbol, interval="1m", days=3):
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current = start_time
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while current < end_time:
            params = {"symbol": symbol, "interval": interval, "startTime": current, "endTime": end_time, "limit": 1000}
            async with session.get(url, params=params) as r:
                data = await r.json()
                if not data:
                    break
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "n", "tbbav", "tbqav", "ignore"
                ])
                df = df[["open_time", "open", "high", "low", "close", "volume"]]
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                all_data.append(df)
                current = data[-1][0] + 60_000
            await asyncio.sleep(0.03)
    df = pd.concat(all_data).drop_duplicates(subset="open_time")
    df = df.set_index("open_time")
    return df

def aggregate_to_14m(df):
    return df.resample("14min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

def ema(series, length): return series.ewm(span=length, adjust=False).mean()
def sma(series, length): return series.rolling(length).mean()

def triple_ema_backtest(args):
    df, tp_pct, sl_pct, fast_len, mid_len, slow_len, trend_len = args
    df = df.copy()
    df["fast"] = ema(df["close"], fast_len)
    df["mid"] = ema(df["close"], mid_len)
    df["slow"] = ema(df["close"], slow_len)
    df["trend"] = sma(df["close"], trend_len)
    df["crossUp"] = (df["fast"].shift(1) <= df["mid"].shift(1)) & (df["fast"] > df["mid"])
    df["trendOK"] = (df["mid"] > df["slow"]) & (df["close"] > df["trend"])
    df["enter"] = df["crossUp"] & df["trendOK"]
    equity = 1_000
    entry_price = None
    tp = sl = None
    pos = 0
    trades = []
    for _, r in df.iterrows():
        h, l, c = r.high, r.low, r.close
        if pos > 0:
            if h >= tp:
                pnl = (tp - entry_price) / entry_price
                equity *= (1 + pnl - 0.002)
                trades.append(pnl)
                pos = 0
            elif l <= sl:
                pnl = (sl - entry_price) / entry_price
                equity *= (1 + pnl - 0.002)
                trades.append(pnl)
                pos = 0
        if pos == 0 and r.enter:
            entry_price = c
            tp = entry_price * (1 + tp_pct / 100)
            sl = entry_price * (1 - sl_pct / 100)
            pos = 1
    total = len(trades)
    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / total * 100 if total > 0 else 0
    pnl = equity - 1_000
    return pnl, win_rate, (tp_pct, sl_pct, fast_len, mid_len, slow_len, trend_len)

def optimize_params_parallel(df, trials=N_RANDOM_TRIALS):
    print(f"🔄 Parallel backtest: {trials} trials, {MAX_CPU} CPU cores...")
    args_list = []
    for i in range(trials):
        tp = round(random.uniform(1.0, 6.0), 1)
        sl = round(random.uniform(5.0, 20.0), 1)
        fast = random.randint(5, 20)
        mid = random.randint(15, 40)
        slow = random.randint(60, 200)
        trend = random.randint(80, 250)
        args_list.append((df, tp, sl, fast, mid, slow, trend))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CPU) as executor:
        for idx, out in enumerate(executor.map(triple_ema_backtest, args_list), 1):
            results.append(out)
            if idx % 100 == 0 or idx == trials:
                print(f"   ...{idx}/{trials} trials complete")
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return results[:3]

def smart_quantity(price, balance):
    qty = (balance / price) * 0.99
    return float(f"{qty:.3f}")

def get_balance(asset=QUOTE_ASSET, paper=False):
    global CLIENT
    if paper:
        if asset == "USDT":
            return PAPER.usdt
        elif asset == "ASTER":
            return PAPER.aster
        else:
            return 0
    else:
        acc = CLIENT.account()
        for b in acc["balances"]:
            if b["asset"] == asset:
                return float(b["free"])
        return 0

def get_base_balance(paper=False):
    return get_balance(asset=BASE_ASSET, paper=paper)

def place_order(side, qty, price=None, paper=False):
    global CLIENT
    if not paper:
        try:
            order = CLIENT.new_order(symbol=SYMBOL, side=side, type="MARKET", quantity=qty)
            print(f"✅ LIVE {side} {qty} {SYMBOL}")
            return order
        except Exception as e:
            print("❌ Order failed:", e)
            traceback.print_exc()
            return None
    else:
        # Paper trading: simulate balance change
        if side == "BUY":
            cost = qty * price * 1.002  # with fees
            if PAPER.usdt >= cost:
                PAPER.usdt -= cost
                PAPER.aster += qty
                PAPER.last_entry = price
                PAPER.open_pos = True
                print(f"✅ PAPER BUY {qty:.3f} ASTER at {price:.5f} (cost: {cost:.2f}), USDT left: {PAPER.usdt:.2f}")
                return True
            else:
                print("❌ PAPER BUY failed: not enough USDT")
                return None
        elif side == "SELL":
            proceeds = qty * price * 0.998  # with fees
            if PAPER.aster >= qty:
                PAPER.aster -= qty
                PAPER.usdt += proceeds
                PAPER.open_pos = False
                print(f"✅ PAPER SELL {qty:.3f} ASTER at {price:.5f} (proceeds: {proceeds:.2f}), USDT now: {PAPER.usdt:.2f}")
                return True
            else:
                print("❌ PAPER SELL failed: not enough ASTER")
                return None

def add_indicators(df, params):
    tp, sl, fast, mid, slow, trend = params
    df["fast"] = ema(df["close"], fast)
    df["mid"] = ema(df["close"], mid)
    df["slow"] = ema(df["close"], slow)
    df["trend"] = sma(df["close"], trend)
    df["crossUp"] = (df["fast"].shift(1) <= df["mid"].shift(1)) & (df["fast"] > df["mid"])
    df["trendOK"] = (df["mid"] > df["slow"]) & (df["close"] > df["trend"])
    df["enter"] = df["crossUp"] & df["trendOK"]
    return df

# ===== MAIN LOOP (patched for proper paper/live) =====
async def main_loop(paper=False):
    global CLIENT
    if not paper and (not API_KEY or not API_SECRET or CLIENT is None):
        print("❗ ERROR: Please enter and apply your Binance API keys before starting LIVE trading.\n")
        return

    if paper:
        PAPER.reset(1000.0, 0.0)
        print("💸 PAPER TRADER: Starting with 1000.00 USDT, 0.00 ASTER")

    print(f"📥 Backtesting for {OPTIMIZATION_DAYS} days (3 days) to get top 3 parameter sets...")
    df_hist = await fetch_binance_historical_async(SYMBOL, days=OPTIMIZATION_DAYS)
    df_agg = aggregate_to_14m(df_hist)
    top3 = optimize_params_parallel(df_agg, N_RANDOM_TRIALS)
    param_sets = [x[2] for x in top3]

    print("\n🏆 Top 3 optimized parameter sets:")
    for rank, (pnl, wr, params) in enumerate(top3, 1):
        print(f"{rank}. PNL=${pnl:.2f} | WR={wr:.2f}% | Params={params}")

    open_pos = False
    entry_price = None
    active_params = None
    last_1m_candle_time = None
    last_14m_candle_time = None

    while True:
        try:
            if not paper:
                klines = CLIENT.klines(SYMBOL, "1m", limit=1000)
            else:
                url = "https://api.binance.com/api/v3/klines"
                params = {"symbol": SYMBOL, "interval": "1m", "limit": 1000}
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                    async with session.get(url, params=params) as r:
                        klines = await r.json()
            df_1m = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "n", "tbbav", "tbqav", "ignore"
            ])
            df_1m["open_time"] = pd.to_datetime(df_1m["open_time"], unit="ms")
            df_1m[["open", "high", "low", "close", "volume"]] = df_1m[["open", "high", "low", "close", "volume"]].astype(float)
            df_1m = df_1m.set_index("open_time")

            latest_1m_time = df_1m.index[-1]
            latest_1m_close = df_1m["close"].iloc[-1]
            if last_1m_candle_time != latest_1m_time:
                print(f"🕛 1m candle closed at {latest_1m_time} | Close: {latest_1m_close:.5f}")
                last_1m_candle_time = latest_1m_time

            df_agg = aggregate_to_14m(df_1m)
            df_agg = add_indicators(df_agg, param_sets[0])

            latest_14m_time = df_agg.index[-1]
            if last_14m_candle_time != latest_14m_time:
                row = df_agg.iloc[-1]
                print(f"\n⏰ 14m candle closed at {latest_14m_time}")
                print(f"  ↳ Fast EMA: {row.fast:.5f}, Mid EMA: {row.mid:.5f}, Slow EMA: {row.slow:.5f}, Trend SMA: {row.trend:.5f}")
                print(f"  ↳ Trend OK: {row.trendOK} | CrossUp: {row.crossUp} | Enter: {row.enter}")
                usdt_bal = get_balance("USDT", paper=paper)
                ASTER_bal = get_balance("ASTER", paper=paper)
                print(f"  ↳ USDT balance: {usdt_bal:.2f}, ASTER balance: {ASTER_bal:.5f}\n")
                last_14m_candle_time = latest_14m_time

            price = df_agg["close"].iloc[-1]
            bal = get_balance(paper=paper)

            if not open_pos:
                if bal >= MIN_TRADE_AMOUNT:
                    for p in param_sets:
                        df_live_ind = add_indicators(df_agg.copy(), p)
                        if df_live_ind["enter"].iloc[-1]:
                            qty = smart_quantity(price, bal)
                            order = place_order("BUY", qty, price=price, paper=paper)
                            if order:
                                entry_price = price
                                active_params = p
                                open_pos = True
                                if paper:
                                    PAPER.open_pos = True
                                    PAPER.last_entry = price
                                    PAPER.active_params = p
                            break
            else:
                tp, sl, *_ = active_params
                tp_level = entry_price * (1 + tp/100)
                sl_level = entry_price * (1 - sl/100)
                trigger = (price >= tp_level) or (price <= sl_level)
                if trigger:
                    base_qty = get_base_balance(paper=paper)
                    if base_qty > 0:
                        place_order("SELL", float(f"{base_qty:.3f}"), price=price, paper=paper)
                    open_pos = False
                    entry_price = None
                    active_params = None
                    if paper:
                        PAPER.open_pos = False
                        PAPER.active_params = None
                        PAPER.last_entry = None

                    print("\n🔄 Re-running optimizer after sell signal...")
                    df_hist = await fetch_binance_historical_async(SYMBOL, days=OPTIMIZATION_DAYS)
                    df_agg = aggregate_to_14m(df_hist)
                    top3 = optimize_params_parallel(df_agg, N_RANDOM_TRIALS)
                    param_sets = [x[2] for x in top3]
                    print("\n🏆 New optimized parameter sets:")
                    for rank, (pnl, wr, params) in enumerate(top3, 1):
                        print(f"{rank}. PNL=${pnl:.2f} | WR={wr:.2f}% | Params={params}")

        except Exception as e:
            print("⚠️ Error:", e)
            traceback.print_exc()

        await asyncio.sleep(20)

# ===== GUI WRAPPER =====
class TradingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.running = False
        self.loop = None
        self.task = None
        self.mode = None # "live" or "paper"
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("📈 ASTERUSDT Trading Bot")
        self.resize(950, 750)
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        api_box = QGroupBox("Binance API Keys (for LIVE only)")
        api_form = QFormLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_secret_edit = QLineEdit()
        self.api_secret_edit.setEchoMode(QLineEdit.EchoMode.Password)
        api_form.addRow("API Key:", self.api_key_edit)
        api_form.addRow("Secret:", self.api_secret_edit)
        api_btn_row = QHBoxLayout()
        self.api_apply_btn = QPushButton("Apply Keys")
        self.api_clear_btn = QPushButton("Clear")
        for btn in (self.api_apply_btn, self.api_clear_btn):
            btn.setFixedHeight(30)
            btn.setFont(QFont("Arial", 11))
            btn.setStyleSheet("""
                QPushButton {
                    border-radius: 8px;
                    background-color: #444;
                    color: white;
                }
                QPushButton:hover { background-color: #666; }
                QPushButton:pressed { background-color: #222; }
            """)
        api_form.addRow(api_btn_row)
        api_box.setLayout(api_form)
        layout.addWidget(api_box)

        # Status + Log
        self.status_label = QLabel("Status: 💤 Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14))
        layout.addWidget(self.status_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("""
            background-color: #111;
            color: #EAEAEA;
            font-family: Consolas;
            border-radius: 12px;
            padding: 10px;
        """)
        layout.addWidget(self.log_box)

        btn_row = QHBoxLayout()
        self.start_paper_btn = QPushButton("💸 Start Paper Trader")
        self.start_live_btn = QPushButton("▶️ Start LIVE Trader")
        self.stop_btn = QPushButton("⏹ Stop")
        self.exit_btn = QPushButton("❌ Exit")

        for btn in (self.start_paper_btn, self.start_live_btn, self.stop_btn, self.exit_btn):
            btn.setFixedHeight(45)
            btn.setFont(QFont("Arial", 12))
            btn.setStyleSheet("""
                QPushButton {
                    border-radius: 10px;
                    background-color: #0078D4;
                    color: white;
                }
                QPushButton:hover { background-color: #0098F0; }
                QPushButton:pressed { background-color: #005999; }
            """)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)
        self.setLayout(layout)
        self.setStyleSheet("background-color: #1E1E1E; color: white;")

        # API key logic
        self.api_apply_btn.clicked.connect(self.apply_api_keys)
        self.api_clear_btn.clicked.connect(self.clear_api_keys)

        # Main control logic
        self.start_paper_btn.clicked.connect(lambda: self.start_trading(mode="paper"))
        self.start_live_btn.clicked.connect(lambda: self.start_trading(mode="live"))
        self.stop_btn.clicked.connect(self.stop_trading)
        self.exit_btn.clicked.connect(self.close)

        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        self.log_box.insertPlainText(text)
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)

    def flush(self): pass

    def apply_api_keys(self):
        global API_KEY, API_SECRET, CLIENT
        key = self.api_key_edit.text().strip()
        sec = self.api_secret_edit.text().strip()
        if key and sec:
            API_KEY = key
            API_SECRET = sec
            CLIENT = create_client(API_KEY, API_SECRET)
            self.status_label.setText("API Keys applied.")
        else:
            self.status_label.setText("API Keys cannot be blank.")

    def clear_api_keys(self):
        global API_KEY, API_SECRET, CLIENT
        self.api_key_edit.clear()
        self.api_secret_edit.clear()
        API_KEY = ""
        API_SECRET = ""
        CLIENT = None
        self.status_label.setText("API Keys cleared.")

    def start_trading(self, mode):
        if self.running:
            print("⚠️ Bot already running.\n")
            return
        self.running = True
        self.mode = mode
        if mode == "paper":
            self.status_label.setText("Status: 💸 PAPER Trader Running")
            print("💸 Starting PAPER TRADER...\n")
        else:
            self.status_label.setText("Status: ▶️ LIVE Trader Running")
            print("▶️ Starting LIVE TRADER...\n")
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.run_asyncio, daemon=True).start()

    def run_asyncio(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.task = self.loop.create_task(main_loop(paper=(self.mode == "paper")))
            self.loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"⚠️ Error in loop: {e}")
        finally:
            self.loop.stop()
            self.running = False
            self.status_label.setText("Status: 💤 Idle")

    def stop_trading(self):
        if not self.running:
            print("🛑 Bot not running.\n")
            return
        self.running = False
        self.status_label.setText("Status: 💤 Stopped")
        print("🛑 Stopping bot...\n")
        if self.task:
            self.task.cancel()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TradingGUI()
    gui.show()
    sys.exit(app.exec())
