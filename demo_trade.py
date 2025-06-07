"""仮想通貨デモトレーディングボット
リアルマネーを使わずに取引戦略をテストするためのシミュレーション環境
"""

import queue
import random
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========================= データクラス =========================


@dataclass
class DemoTrade:
    """デモ取引を表すクラス"""

    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    fee: float
    pnl: float = 0.0
    balance_after: float = 0.0


@dataclass
class DemoPosition:
    """デモポジションを表すクラス"""

    symbol: str
    entry_price: float
    amount: float
    side: str
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class DemoAccount:
    """デモアカウントを表すクラス"""

    initial_balance: float
    current_balance: float
    available_balance: float
    positions: Dict[str, DemoPosition] = field(default_factory=dict)
    trade_history: List[DemoTrade] = field(default_factory=list)
    equity_history: List[Tuple[datetime, float]] = field(default_factory=list)


# ========================= 価格データプロバイダー =========================


class PriceDataProvider:
    """価格データを提供するクラス（実際の価格またはシミュレーション）"""

    def __init__(self, mode="simulation", symbol="BTC-USD"):
        self.mode = mode
        self.symbol = symbol
        self.current_price = 45000.0  # 初期価格
        self.price_history = []
        self.volatility = 0.02  # 2%のボラティリティ

        if mode == "historical":
            self._load_historical_data()

    def _load_historical_data(self):
        """過去データを読み込む"""
        try:
            # Yahoo Financeから過去30日のデータを取得
            ticker = yf.Ticker(self.symbol)
            self.historical_data = ticker.history(period="30d", interval="5m")
            if not self.historical_data.empty:
                self.current_price = self.historical_data["Close"].iloc[-1]
        except Exception as e:
            print(f"過去データの取得エラー: {e}")
            self.mode = "simulation"

    def get_current_price(self) -> float:
        """現在の価格を取得"""
        if self.mode == "simulation":
            # ランダムウォークでシミュレート
            change = np.random.normal(0, self.volatility)
            self.current_price *= 1 + change

            # 時々大きな動きを追加（リアリティのため）
            if random.random() < 0.05:  # 5%の確率
                spike = np.random.uniform(-0.03, 0.03)
                self.current_price *= 1 + spike

        elif self.mode == "historical" and hasattr(self, "historical_data"):
            # 過去データを順番に返す
            if len(self.price_history) < len(self.historical_data):
                self.current_price = self.historical_data["Close"].iloc[
                    len(self.price_history)
                ]

        self.price_history.append((datetime.now(), self.current_price))
        return self.current_price

    def get_ohlcv_data(self, periods: int = 100) -> pd.DataFrame:
        """OHLCVデータを生成または取得"""
        if self.mode == "historical" and hasattr(self, "historical_data"):
            return self.historical_data.tail(periods)

        # シミュレーションモードではランダムに生成
        data = []
        base_price = self.current_price

        for i in range(periods):
            open_price = base_price
            high_price = open_price * (1 + np.random.uniform(0, 0.01))
            low_price = open_price * (1 - np.random.uniform(0, 0.01))
            close_price = np.random.uniform(low_price, high_price)
            volume = np.random.uniform(1000, 10000)

            data.append(
                {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                }
            )

            base_price = close_price

        df = pd.DataFrame(data)
        df.index = pd.date_range(end=datetime.now(), periods=periods, freq="5T")
        return df


# ========================= デモ取引エンジン =========================


class DemoTradingEngine:
    """デモ取引を実行するエンジン"""

    def __init__(self, initial_balance: float = 10000.0, fee_rate: float = 0.001):
        self.account = DemoAccount(
            initial_balance=initial_balance,
            current_balance=initial_balance,
            available_balance=initial_balance,
        )
        self.fee_rate = fee_rate
        self.price_provider = PriceDataProvider(mode="simulation")

    def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        limit_price: float = None,
    ) -> Dict:
        """注文を実行"""
        current_price = self.price_provider.get_current_price()

        # 市場注文の場合はスリッページを追加
        if order_type == "market":
            slippage = np.random.uniform(0.0001, 0.0005)  # 0.01% - 0.05%
            if side == "buy":
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)
        else:
            execution_price = limit_price or current_price

        # 注文金額を計算
        order_value = amount * execution_price
        fee = order_value * self.fee_rate
        total_cost = order_value + fee if side == "buy" else fee

        # 残高チェック
        if side == "buy" and total_cost > self.account.available_balance:
            return {"status": "rejected", "reason": "残高不足"}

        if side == "sell" and symbol not in self.account.positions:
            return {"status": "rejected", "reason": "ポジションなし"}

        # 取引を実行
        if side == "buy":
            # 買い注文
            self.account.current_balance -= total_cost
            self.account.available_balance -= total_cost

            # ポジションを作成または追加
            if symbol in self.account.positions:
                position = self.account.positions[symbol]
                # 平均価格を計算
                total_amount = position.amount + amount
                position.entry_price = (
                    position.entry_price * position.amount + execution_price * amount
                ) / total_amount
                position.amount = total_amount
            else:
                self.account.positions[symbol] = DemoPosition(
                    symbol=symbol,
                    entry_price=execution_price,
                    amount=amount,
                    side="long",
                    entry_time=datetime.now(),
                    current_price=execution_price,
                )
        else:
            # 売り注文
            position = self.account.positions[symbol]

            # 損益を計算
            pnl = (execution_price - position.entry_price) * amount - fee

            # 残高を更新
            self.account.current_balance += amount * execution_price - fee
            self.account.available_balance += amount * execution_price - fee

            # ポジションを更新または削除
            if position.amount > amount:
                position.amount -= amount
            else:
                del self.account.positions[symbol]

        # 取引履歴に追加
        trade = DemoTrade(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            price=execution_price,
            amount=amount,
            fee=fee,
            pnl=pnl if side == "sell" else 0.0,
            balance_after=self.account.current_balance,
        )
        self.account.trade_history.append(trade)

        # 資産履歴を更新
        total_equity = self.calculate_total_equity()
        self.account.equity_history.append((datetime.now(), total_equity))

        return {
            "status": "filled",
            "trade": trade,
            "execution_price": execution_price,
            "fee": fee,
        }

    def update_positions(self):
        """ポジションの未実現損益を更新"""
        current_price = self.price_provider.get_current_price()

        for symbol, position in self.account.positions.items():
            position.current_price = current_price
            position.unrealized_pnl = (
                current_price - position.entry_price
            ) * position.amount

    def calculate_total_equity(self) -> float:
        """総資産（残高＋未実現損益）を計算"""
        self.update_positions()
        total_equity = self.account.current_balance

        for position in self.account.positions.values():
            total_equity += position.amount * position.current_price

        return total_equity

    def get_performance_metrics(self) -> Dict:
        """パフォーマンス指標を計算"""
        if not self.account.trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_pnl_percent": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
            }

        # 勝率を計算
        winning_trades = [t for t in self.account.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.account.trade_history if t.pnl < 0]
        total_trades = len([t for t in self.account.trade_history if t.side == "sell"])

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # 総損益を計算
        total_pnl = sum(t.pnl for t in self.account.trade_history)
        total_pnl_percent = (total_pnl / self.account.initial_balance) * 100

        # 最大ドローダウンを計算
        if self.account.equity_history:
            equity_values = [e[1] for e in self.account.equity_history]
            peak = equity_values[0]
            max_drawdown = 0

            for value in equity_values:
                peak = max(peak, value)
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0

        # シャープレシオを計算（簡易版）
        if len(self.account.equity_history) > 1:
            returns = []
            for i in range(1, len(self.account.equity_history)):
                prev_equity = self.account.equity_history[i - 1][1]
                curr_equity = self.account.equity_history[i][1]
                returns.append((curr_equity - prev_equity) / prev_equity)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (
                    (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
                )
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_pnl_percent": total_pnl_percent,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
        }


# ========================= シンプルな取引戦略 =========================


class SimpleTradingStrategy:
    """デモ用のシンプルな取引戦略"""

    def __init__(self, engine: DemoTradingEngine):
        self.engine = engine
        self.position_size = 0.1  # 10%のポジションサイズ

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """テクニカル指標を計算"""
        # RSI
        rsi = talib.RSI(df["Close"], timeperiod=14)

        # MACD
        macd, signal, hist = talib.MACD(df["Close"])

        # ボリンジャーバンド
        upper, middle, lower = talib.BBANDS(df["Close"])

        # 移動平均
        sma_20 = talib.SMA(df["Close"], timeperiod=20)
        sma_50 = talib.SMA(df["Close"], timeperiod=50)

        return {
            "rsi": rsi.iloc[-1],
            "macd": macd.iloc[-1],
            "macd_signal": signal.iloc[-1],
            "bb_upper": upper.iloc[-1],
            "bb_lower": lower.iloc[-1],
            "sma_20": sma_20.iloc[-1],
            "sma_50": sma_50.iloc[-1],
            "close": df["Close"].iloc[-1],
        }

    def generate_signal(self, indicators: Dict) -> str:
        """取引シグナルを生成"""
        # RSI戦略
        if indicators["rsi"] < 30:
            return "buy"
        if indicators["rsi"] > 70:
            return "sell"

        # MACD戦略
        if indicators["macd"] > indicators["macd_signal"]:
            return "buy"
        if indicators["macd"] < indicators["macd_signal"]:
            return "sell"

        return "hold"

    def execute_trade(self):
        """取引を実行"""
        # 価格データを取得
        df = self.engine.price_provider.get_ohlcv_data(100)

        # 指標を計算
        indicators = self.calculate_indicators(df)

        # シグナルを生成
        signal = self.generate_signal(indicators)

        # 現在のポジションを確認
        symbol = "BTC-USD"
        has_position = symbol in self.engine.account.positions

        # 取引実行
        if signal == "buy" and not has_position:
            # 買い注文
            available_capital = self.engine.account.available_balance
            position_value = available_capital * self.position_size
            current_price = self.engine.price_provider.get_current_price()
            amount = position_value / current_price

            result = self.engine.place_order(symbol, "buy", amount)
            return result, indicators

        if signal == "sell" and has_position:
            # 売り注文
            position = self.engine.account.positions[symbol]
            result = self.engine.place_order(symbol, "sell", position.amount)
            return result, indicators

        return None, indicators


# ========================= GUI アプリケーション =========================


class DemoTradingGUI:
    """デモトレーディングのGUIアプリケーション"""

    def __init__(self, root):
        self.root = root
        self.root.title("仮想通貨デモトレーディングボット")
        self.root.geometry("1200x800")

        # デモエンジンと戦略を初期化
        self.engine = DemoTradingEngine(initial_balance=10000.0)
        self.strategy = SimpleTradingStrategy(self.engine)

        # 取引スレッドの制御
        self.trading_active = False
        self.trading_thread = None
        self.update_queue = queue.Queue()

        # GUI要素を作成
        self.create_widgets()

        # 定期更新を開始
        self.update_display()

    def create_widgets(self):
        """GUI要素を作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # アカウント情報フレーム
        account_frame = ttk.LabelFrame(main_frame, text="アカウント情報", padding="10")
        account_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.balance_label = ttk.Label(account_frame, text="残高: $10,000.00")
        self.balance_label.grid(row=0, column=0, padx=10)

        self.equity_label = ttk.Label(account_frame, text="総資産: $10,000.00")
        self.equity_label.grid(row=0, column=1, padx=10)

        self.pnl_label = ttk.Label(account_frame, text="損益: $0.00 (0.00%)")
        self.pnl_label.grid(row=0, column=2, padx=10)

        # コントロールフレーム
        control_frame = ttk.LabelFrame(main_frame, text="コントロール", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.start_button = ttk.Button(
            control_frame, text="取引開始", command=self.start_trading
        )
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(
            control_frame, text="取引停止", command=self.stop_trading, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        self.reset_button = ttk.Button(
            control_frame, text="リセット", command=self.reset_account
        )
        self.reset_button.grid(row=0, column=2, padx=5)

        # 取引間隔設定
        ttk.Label(control_frame, text="取引間隔（秒）:").grid(
            row=1, column=0, padx=5, pady=5
        )
        self.interval_var = tk.StringVar(value="5")
        interval_spinbox = ttk.Spinbox(
            control_frame, from_=1, to=60, textvariable=self.interval_var, width=10
        )
        interval_spinbox.grid(row=1, column=1, padx=5, pady=5)

        # パフォーマンス指標フレーム
        metrics_frame = ttk.LabelFrame(
            main_frame, text="パフォーマンス指標", padding="10"
        )
        metrics_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.trades_label = ttk.Label(metrics_frame, text="総取引数: 0")
        self.trades_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)

        self.winrate_label = ttk.Label(metrics_frame, text="勝率: 0.00%")
        self.winrate_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)

        self.drawdown_label = ttk.Label(metrics_frame, text="最大DD: 0.00%")
        self.drawdown_label.grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)

        self.sharpe_label = ttk.Label(metrics_frame, text="シャープレシオ: 0.00")
        self.sharpe_label.grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)

        # 取引履歴フレーム
        history_frame = ttk.LabelFrame(main_frame, text="取引履歴", padding="10")
        history_frame.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5
        )

        # スクロールバー付きのテキストウィジェット
        self.history_text = tk.Text(history_frame, height=10, width=80)
        scrollbar = ttk.Scrollbar(
            history_frame, orient="vertical", command=self.history_text.yview
        )
        self.history_text.configure(yscrollcommand=scrollbar.set)

        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # チャートフレーム
        chart_frame = ttk.LabelFrame(main_frame, text="資産推移チャート", padding="10")
        chart_frame.grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5
        )

        # Matplotlibフィギュアを作成
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # グリッドの重みを設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

    def start_trading(self):
        """取引を開始"""
        self.trading_active = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.reset_button.config(state="disabled")

        # 取引スレッドを開始
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

        self.add_log("取引を開始しました")

    def stop_trading(self):
        """取引を停止"""
        self.trading_active = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.reset_button.config(state="normal")

        self.add_log("取引を停止しました")

    def reset_account(self):
        """アカウントをリセット"""
        self.engine = DemoTradingEngine(initial_balance=10000.0)
        self.strategy = SimpleTradingStrategy(self.engine)
        self.history_text.delete(1.0, tk.END)
        self.update_chart()
        self.add_log("アカウントをリセットしました")

    def trading_loop(self):
        """取引ループ（別スレッドで実行）"""
        while self.trading_active:
            try:
                # 取引を実行
                result, indicators = self.strategy.execute_trade()

                if result and result["status"] == "filled":
                    trade = result["trade"]
                    message = f"{trade.side.upper()} {trade.amount:.4f} @ ${trade.price:.2f} (手数料: ${trade.fee:.2f})"
                    if trade.side == "sell" and trade.pnl != 0:
                        message += f" - 損益: ${trade.pnl:.2f}"
                    self.update_queue.put(("log", message))

                # 指定された間隔で待機
                interval = int(self.interval_var.get())
                time.sleep(interval)

            except Exception as e:
                self.update_queue.put(("log", f"エラー: {e!s}"))
                time.sleep(5)

    def add_log(self, message):
        """ログメッセージを追加"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.history_text.see(tk.END)

    def update_display(self):
        """表示を更新"""
        # キューからメッセージを処理
        try:
            while True:
                msg_type, message = self.update_queue.get_nowait()
                if msg_type == "log":
                    self.add_log(message)
        except queue.Empty:
            pass

        # アカウント情報を更新
        total_equity = self.engine.calculate_total_equity()
        pnl = total_equity - self.engine.account.initial_balance
        pnl_percent = (pnl / self.engine.account.initial_balance) * 100

        self.balance_label.config(
            text=f"残高: ${self.engine.account.current_balance:,.2f}"
        )
        self.equity_label.config(text=f"総資産: ${total_equity:,.2f}")

        # 損益の色を設定
        pnl_color = "green" if pnl >= 0 else "red"
        self.pnl_label.config(
            text=f"損益: ${pnl:,.2f} ({pnl_percent:.2f}%)", foreground=pnl_color
        )

        # パフォーマンス指標を更新
        metrics = self.engine.get_performance_metrics()
        self.trades_label.config(text=f"総取引数: {metrics['total_trades']}")
        self.winrate_label.config(text=f"勝率: {metrics['win_rate'] * 100:.2f}%")
        self.drawdown_label.config(text=f"最大DD: {metrics['max_drawdown'] * 100:.2f}%")
        self.sharpe_label.config(text=f"シャープレシオ: {metrics['sharpe_ratio']:.2f}")

        # チャートを更新（10秒ごと）
        if hasattr(self, "last_chart_update"):
            if datetime.now() - self.last_chart_update > timedelta(seconds=10):
                self.update_chart()
                self.last_chart_update = datetime.now()
        else:
            self.update_chart()
            self.last_chart_update = datetime.now()

        # 100ms後に再度実行
        self.root.after(100, self.update_display)

    def update_chart(self):
        """チャートを更新"""
        if not self.engine.account.equity_history:
            return

        # データを準備
        times = [e[0] for e in self.engine.account.equity_history]
        values = [e[1] for e in self.engine.account.equity_history]

        # チャートをクリア
        self.ax.clear()

        # 資産推移をプロット
        self.ax.plot(times, values, "b-", linewidth=2)
        self.ax.axhline(
            y=self.engine.account.initial_balance,
            color="gray",
            linestyle="--",
            alpha=0.5,
        )

        # フォーマット
        self.ax.set_xlabel("時間")
        self.ax.set_ylabel("総資産 ($)")
        self.ax.set_title("資産推移")
        self.ax.grid(True, alpha=0.3)

        # 背景色を設定
        if values:
            if values[-1] >= self.engine.account.initial_balance:
                self.ax.axhspan(
                    self.engine.account.initial_balance,
                    max(values),
                    alpha=0.1,
                    color="green",
                )
            else:
                self.ax.axhspan(
                    min(values),
                    self.engine.account.initial_balance,
                    alpha=0.1,
                    color="red",
                )

        # 日付フォーマット
        self.fig.autofmt_xdate()

        # キャンバスを更新
        self.canvas.draw()


# ========================= メイン関数 =========================


def main():
    """メインエントリーポイント"""
    # GUIアプリケーションを起動
    root = tk.Tk()
    app = DemoTradingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
