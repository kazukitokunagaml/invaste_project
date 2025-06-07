"""Cryptocurrency Trading Bot Framework
A modular, extensible trading bot with multiple strategies based on comprehensive research
"""

import asyncio
import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import talib

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ========================= Data Classes =========================


@dataclass
class Trade:
    """Represents a single trade"""

    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    amount: float
    fee: float
    order_id: str
    strategy: str


@dataclass
class Position:
    """Represents an open position"""

    symbol: str
    entry_price: float
    amount: float
    side: str
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None


@dataclass
class MarketData:
    """Represents market data for a symbol"""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"


# ========================= Risk Management =========================


class RiskManager:
    """Handles all risk management operations"""

    def __init__(self, config: Dict):
        self.max_position_size = config.get("max_position_size", 0.02)  # 2% per trade
        self.max_positions = config.get("max_positions", 3)
        self.max_drawdown = config.get("max_drawdown", 0.10)  # 10%
        self.stop_loss_percent = config.get("stop_loss_percent", 0.03)  # 3%
        self.take_profit_percent = config.get("take_profit_percent", 0.08)  # 8%
        self.use_trailing_stop = config.get("use_trailing_stop", False)
        self.trailing_stop_percent = config.get("trailing_stop_percent", 0.02)
        self.cooldown_period = config.get("cooldown_period", 300)  # 5 minutes
        self.consecutive_losses = 0
        self.max_consecutive_losses = config.get("max_consecutive_losses", 3)
        self.initial_capital = 0
        self.current_capital = 0
        self.last_trade_time = None

    def calculate_position_size(
        self, capital: float, price: float, volatility: float = 1.0
    ) -> float:
        """Calculate position size based on Kelly Criterion or fixed percentage"""
        base_size = capital * self.max_position_size

        # Adjust for volatility (lower size in high volatility)
        volatility_adjusted_size = base_size / volatility

        return min(volatility_adjusted_size, base_size)

    def check_drawdown(self) -> bool:
        """Check if maximum drawdown has been exceeded"""
        if self.initial_capital == 0:
            return False

        current_drawdown = (
            self.initial_capital - self.current_capital
        ) / self.initial_capital
        return current_drawdown >= self.max_drawdown

    def update_stop_loss(
        self, position: Position, current_price: float
    ) -> Optional[float]:
        """Update trailing stop loss if applicable"""
        if not self.use_trailing_stop:
            return position.stop_loss

        if position.side == "long":
            if position.highest_price is None or current_price > position.highest_price:
                position.highest_price = current_price
                new_stop = current_price * (1 - self.trailing_stop_percent)
                if position.stop_loss is None or new_stop > position.stop_loss:
                    return new_stop
        elif position.lowest_price is None or current_price < position.lowest_price:
            position.lowest_price = current_price
            new_stop = current_price * (1 + self.trailing_stop_percent)
            if position.stop_loss is None or new_stop < position.stop_loss:
                return new_stop

        return position.stop_loss

    def should_trade(self) -> bool:
        """Check if trading should be allowed based on risk rules"""
        # Check drawdown
        if self.check_drawdown():
            logger.warning("Maximum drawdown reached. Trading suspended.")
            return False

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(
                f"Consecutive losses limit reached ({self.consecutive_losses}). Cooling down."
            )
            if (
                self.last_trade_time
                and datetime.now() - self.last_trade_time
                < timedelta(seconds=self.cooldown_period)
            ):
                return False

        return True


# ========================= Exchange Interface =========================


class ExchangeInterface:
    """Unified interface for cryptocurrency exchanges using ccxt"""

    def __init__(self, exchange_name: str, config: Dict):
        self.exchange_name = exchange_name
        self.config = config

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_name)
        self.exchange = exchange_class(
            {
                "apiKey": config.get("api_key"),
                "secret": config.get("api_secret"),
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",  # or 'future' for futures trading
                },
            }
        )

        # Load markets
        self.markets = self.exchange.load_markets()

    async def fetch_balance(self) -> Dict:
        """Fetch account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return pd.DataFrame()

    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: OrderType = OrderType.MARKET,
        price: float = None,
    ) -> Dict:
        """Place an order"""
        try:
            if order_type == OrderType.MARKET:
                order = self.exchange.create_market_order(symbol, side, amount)
            elif order_type == OrderType.LIMIT:
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            logger.info(f"Order placed: {order}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {}

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def calculate_fee(self, amount: float, price: float, side: str) -> float:
        """Calculate trading fee"""
        fee_rate = self.config.get("taker_fee", 0.001)  # 0.1% default
        return amount * price * fee_rate


# ========================= Technical Indicators =========================


class TechnicalIndicators:
    """Calculate technical indicators using TA-Lib"""

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        return talib.RSI(df["close"], timeperiod=period)

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        macd, macdsignal, macdhist = talib.MACD(
            df["close"], fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return macd, macdsignal, macdhist

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(
            df["close"], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
        )
        return upper, middle, lower

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        return talib.ATR(df["high"], df["low"], df["close"], timeperiod=period)

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        return talib.ADX(df["high"], df["low"], df["close"], timeperiod=period)

    @staticmethod
    def calculate_dmi(
        df: pd.DataFrame, period: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate DMI (+DI and -DI)"""
        plus_di = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
        minus_di = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=period)
        return plus_di, minus_di

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate EMA"""
        return talib.EMA(df["close"], timeperiod=period)


# ========================= Trading Strategies =========================


class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""

    def __init__(self, config: Dict):
        self.config = config
        self.indicators = TechnicalIndicators()

    @abstractmethod
    async def analyze(
        self, df: pd.DataFrame, position: Optional[Position] = None
    ) -> Dict[str, Any]:
        """Analyze market data and return trading signal"""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name"""


class RSIDMIStrategy(TradingStrategy):
    """RSI + DMI Strategy (Based on the 'Chimpanzee Level' Bot)"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.adx_period = config.get("adx_period", 14)
        self.adx_threshold = config.get("adx_threshold", 25)
        self.use_ema_filter = config.get("use_ema_filter", True)
        self.ema_period = config.get("ema_period", 50)

    async def analyze(
        self, df: pd.DataFrame, position: Optional[Position] = None
    ) -> Dict[str, Any]:
        """Analyze using RSI and DMI indicators"""
        # Calculate indicators
        rsi = self.indicators.calculate_rsi(df, self.rsi_period)
        adx = self.indicators.calculate_adx(df, self.adx_period)
        plus_di, minus_di = self.indicators.calculate_dmi(df, self.adx_period)

        # Optional EMA filter
        ema = None
        if self.use_ema_filter:
            ema = self.indicators.calculate_ema(df, self.ema_period)

        # Get latest values
        latest_rsi = rsi.iloc[-1]
        latest_adx = adx.iloc[-1]
        latest_plus_di = plus_di.iloc[-1]
        latest_minus_di = minus_di.iloc[-1]
        latest_close = df["close"].iloc[-1]
        latest_ema = ema.iloc[-1] if ema is not None else None

        signal = {
            "action": "hold",
            "confidence": 0.0,
            "indicators": {
                "rsi": latest_rsi,
                "adx": latest_adx,
                "plus_di": latest_plus_di,
                "minus_di": latest_minus_di,
                "close": latest_close,
                "ema": latest_ema,
            },
        }

        # Check for existing position
        if position:
            # Exit conditions
            if position.side == "long" and latest_rsi > self.rsi_overbought:
                signal["action"] = "sell"
                signal["confidence"] = 0.8
                signal["reason"] = "RSI overbought - taking profit"
            elif position.side == "short" and latest_rsi < self.rsi_oversold:
                signal["action"] = "buy"
                signal["confidence"] = 0.8
                signal["reason"] = "RSI oversold - taking profit"
        # Entry conditions
        # Long signal
        elif (
            latest_rsi < self.rsi_oversold
            and latest_adx > self.adx_threshold
            and latest_plus_di > latest_minus_di
        ):
            if not self.use_ema_filter or latest_close > latest_ema:
                signal["action"] = "buy"
                signal["confidence"] = 0.7
                signal["reason"] = "RSI oversold with strong uptrend"

        # Short signal (if enabled)
        elif (
            latest_rsi > self.rsi_overbought
            and latest_adx > self.adx_threshold
            and latest_minus_di > latest_plus_di
        ):
            if not self.use_ema_filter or latest_close < latest_ema:
                signal["action"] = "sell"
                signal["confidence"] = 0.7
                signal["reason"] = "RSI overbought with strong downtrend"

        return signal

    def get_strategy_name(self) -> str:
        return "RSI_DMI_Strategy"


class GridTradingStrategy(TradingStrategy):
    """Grid Trading Strategy for ranging markets"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.upper_price = config.get("upper_price")
        self.lower_price = config.get("lower_price")
        self.num_grids = config.get("num_grids", 10)
        self.grid_spacing_type = config.get(
            "grid_spacing_type", "arithmetic"
        )  # or 'geometric'
        self.grid_levels = self._calculate_grid_levels()
        self.active_orders = {}

    def _calculate_grid_levels(self) -> List[float]:
        """Calculate grid price levels"""
        if self.grid_spacing_type == "arithmetic":
            return np.linspace(self.lower_price, self.upper_price, self.num_grids + 1)
        # geometric
        return np.geomspace(self.lower_price, self.upper_price, self.num_grids + 1)

    async def analyze(
        self, df: pd.DataFrame, position: Optional[Position] = None
    ) -> Dict[str, Any]:
        """Grid trading doesn't use traditional signals, it places orders at grid levels"""
        latest_price = df["close"].iloc[-1]

        signal = {
            "action": "grid_update",
            "current_price": latest_price,
            "grid_levels": self.grid_levels.tolist(),
            "orders_to_place": [],
        }

        # Check if price is within grid range
        if latest_price < self.lower_price or latest_price > self.upper_price:
            signal["action"] = "hold"
            signal["reason"] = "Price outside grid range"
            return signal

        # Determine which orders need to be placed
        for i, level in enumerate(self.grid_levels):
            level_key = f"grid_{i}"

            # If we don't have an order at this level
            if level_key not in self.active_orders:
                # Buy orders below current price
                if level < latest_price * 0.995:  # Small buffer
                    signal["orders_to_place"].append(
                        {
                            "side": "buy",
                            "price": level,
                            "grid_level": i,
                        }
                    )
                # Sell orders above current price
                elif level > latest_price * 1.005:  # Small buffer
                    signal["orders_to_place"].append(
                        {
                            "side": "sell",
                            "price": level,
                            "grid_level": i,
                        }
                    )

        return signal

    def get_strategy_name(self) -> str:
        return "Grid_Trading_Strategy"


class DCAStrategy(TradingStrategy):
    """Dollar Cost Averaging Strategy with dynamic adjustments"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_amount = config.get("base_amount", 100)  # USD per interval
        self.interval_hours = config.get("interval_hours", 24)
        self.use_dynamic_dca = config.get("use_dynamic_dca", True)
        self.price_drop_threshold = config.get("price_drop_threshold", 0.05)  # 5%
        self.amount_multiplier = config.get("amount_multiplier", 1.5)
        self.last_buy_time = None
        self.last_buy_price = None

    async def analyze(
        self, df: pd.DataFrame, position: Optional[Position] = None
    ) -> Dict[str, Any]:
        """DCA analysis - time and price based"""
        latest_price = df["close"].iloc[-1]
        current_time = datetime.now()

        signal = {
            "action": "hold",
            "amount": self.base_amount,
            "reason": "",
        }

        # Check if it's time for regular DCA
        if self.last_buy_time is None or (
            current_time - self.last_buy_time
        ) >= timedelta(hours=self.interval_hours):
            signal["action"] = "buy"
            signal["reason"] = "Regular DCA interval"

            # Dynamic DCA - buy more on dips
            if self.use_dynamic_dca and self.last_buy_price:
                price_change = (
                    latest_price - self.last_buy_price
                ) / self.last_buy_price

                if price_change <= -self.price_drop_threshold:
                    signal["amount"] = self.base_amount * self.amount_multiplier
                    signal["reason"] = f"Dynamic DCA - price dropped {price_change:.1%}"

        return signal

    def get_strategy_name(self) -> str:
        return "DCA_Strategy"


class MACDStrategy(TradingStrategy):
    """MACD Crossover Strategy"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.fast_period = config.get("macd_fast", 12)
        self.slow_period = config.get("macd_slow", 26)
        self.signal_period = config.get("macd_signal", 9)
        self.use_histogram = config.get("use_histogram", True)

    async def analyze(
        self, df: pd.DataFrame, position: Optional[Position] = None
    ) -> Dict[str, Any]:
        """Analyze using MACD indicator"""
        macd, signal_line, histogram = self.indicators.calculate_macd(
            df,
            self.fast_period,
            self.slow_period,
            self.signal_period,
        )

        # Get latest values
        latest_macd = macd.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        latest_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2]

        signal = {
            "action": "hold",
            "confidence": 0.0,
            "indicators": {
                "macd": latest_macd,
                "signal_line": latest_signal,
                "histogram": latest_histogram,
            },
        }

        # MACD crossover signals
        if position is None:
            # Bullish crossover
            if latest_macd > latest_signal and macd.iloc[-2] <= signal_line.iloc[-2]:
                signal["action"] = "buy"
                signal["confidence"] = 0.7
                signal["reason"] = "MACD bullish crossover"

                # Stronger signal if histogram is increasing
                if self.use_histogram and latest_histogram > prev_histogram:
                    signal["confidence"] = 0.8
        # Bearish crossover for exit
        elif latest_macd < latest_signal and macd.iloc[-2] >= signal_line.iloc[-2]:
            signal["action"] = "sell"
            signal["confidence"] = 0.7
            signal["reason"] = "MACD bearish crossover"

        return signal

    def get_strategy_name(self) -> str:
        return "MACD_Strategy"


# ========================= Backtesting Engine =========================


class BacktestEngine:
    """Backtesting engine for strategy evaluation"""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.trades = []
        self.positions = []
        self.equity_curve = []

    async def backtest(
        self,
        strategy: TradingStrategy,
        df: pd.DataFrame,
        risk_manager: RiskManager,
        fee_rate: float = 0.001,
    ) -> Dict:
        """Run backtest on historical data"""
        capital = self.initial_capital
        position = None

        # Ensure we have enough data
        if len(df) < 100:
            logger.error("Insufficient data for backtesting")
            return {}

        # Iterate through historical data
        for i in range(100, len(df)):
            current_data = df.iloc[: i + 1]
            current_price = current_data["close"].iloc[-1]
            current_time = current_data.index[-1]

            # Get strategy signal
            signal = await strategy.analyze(current_data, position)

            # Execute trades based on signal
            if signal["action"] == "buy" and position is None:
                # Calculate position size
                amount = (
                    risk_manager.calculate_position_size(capital, current_price)
                    / current_price
                )
                fee = amount * current_price * fee_rate

                # Open position
                position = Position(
                    symbol="BTC/USDT",  # Placeholder
                    entry_price=current_price,
                    amount=amount,
                    side="long",
                    entry_time=current_time,
                    stop_loss=current_price * (1 - risk_manager.stop_loss_percent),
                    take_profit=current_price * (1 + risk_manager.take_profit_percent),
                )

                capital -= amount * current_price + fee

                self.trades.append(
                    Trade(
                        timestamp=current_time,
                        symbol="BTC/USDT",
                        side="buy",
                        price=current_price,
                        amount=amount,
                        fee=fee,
                        order_id=f"backtest_{i}",
                        strategy=strategy.get_strategy_name(),
                    )
                )

            elif signal["action"] == "sell" and position:
                # Close position
                pnl = (current_price - position.entry_price) * position.amount
                fee = position.amount * current_price * fee_rate
                capital += position.amount * current_price - fee

                self.trades.append(
                    Trade(
                        timestamp=current_time,
                        symbol="BTC/USDT",
                        side="sell",
                        price=current_price,
                        amount=position.amount,
                        fee=fee,
                        order_id=f"backtest_{i}",
                        strategy=strategy.get_strategy_name(),
                    )
                )

                position = None

            # Check stop loss and take profit
            if position:
                if current_price <= position.stop_loss:
                    # Stop loss hit
                    capital += position.amount * current_price * (1 - fee_rate)
                    position = None
                elif current_price >= position.take_profit:
                    # Take profit hit
                    capital += position.amount * current_price * (1 - fee_rate)
                    position = None

            # Record equity
            total_equity = capital
            if position:
                total_equity += position.amount * current_price
            self.equity_curve.append(
                {
                    "timestamp": current_time,
                    "equity": total_equity,
                }
            )

        # Calculate performance metrics
        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.trades or not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index("timestamp", inplace=True)

        # Calculate returns
        returns = equity_df["equity"].pct_change().dropna()

        # Basic metrics
        total_return = (
            equity_df["equity"].iloc[-1] - self.initial_capital
        ) / self.initial_capital
        num_trades = len(self.trades)

        # Win rate
        winning_trades = 0
        total_pnl = 0

        for i in range(0, len(self.trades), 2):  # Pairs of buy/sell
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                pnl = (
                    (sell_trade.price - buy_trade.price) * buy_trade.amount
                    - buy_trade.fee
                    - sell_trade.fee
                )
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1

        win_rate = winning_trades / (num_trades // 2) if num_trades > 0 else 0

        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(equity_df["equity"])

        return {
            "total_return": total_return,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_equity": equity_df["equity"].iloc[-1],
        }

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return (
            np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            if excess_returns.std() > 0
            else 0
        )

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown.min()


# ========================= Main Trading Bot =========================


class CryptoTradingBot:
    """Main trading bot orchestrator"""

    def __init__(self, config_file: str):
        # Load configuration
        with open(config_file) as f:
            self.config = json.load(f)

        # Initialize components
        self.exchange = ExchangeInterface(
            self.config["exchange"]["name"],
            self.config["exchange"],
        )

        self.risk_manager = RiskManager(self.config["risk_management"])

        # Initialize strategies
        self.strategies = self._initialize_strategies()
        self.active_strategy = self.config.get("active_strategy", "RSI_DMI")

        # State management
        self.positions = {}
        self.is_running = False

        # Performance tracking
        self.start_time = datetime.now()
        self.trade_history = []

    def _initialize_strategies(self) -> Dict[str, TradingStrategy]:
        """Initialize all available strategies"""
        strategies = {}

        if "rsi_dmi" in self.config["strategies"]:
            strategies["RSI_DMI"] = RSIDMIStrategy(self.config["strategies"]["rsi_dmi"])

        if "grid" in self.config["strategies"]:
            strategies["GRID"] = GridTradingStrategy(self.config["strategies"]["grid"])

        if "dca" in self.config["strategies"]:
            strategies["DCA"] = DCAStrategy(self.config["strategies"]["dca"])

        if "macd" in self.config["strategies"]:
            strategies["MACD"] = MACDStrategy(self.config["strategies"]["macd"])

        return strategies

    async def start(self):
        """Start the trading bot"""
        logger.info(f"Starting {self.active_strategy} trading bot...")
        self.is_running = True

        # Get initial balance
        balance = await self.exchange.fetch_balance()
        if balance:
            total_balance = sum(
                [
                    b["free"] + b["used"]
                    for b in balance["total"].values()
                    if isinstance(b, dict)
                ]
            )
            self.risk_manager.initial_capital = total_balance
            self.risk_manager.current_capital = total_balance
            logger.info(f"Initial capital: ${total_balance:.2f}")

        # Main trading loop
        while self.is_running:
            try:
                await self._trading_cycle()

                # Sleep based on timeframe
                timeframe = self.config.get("timeframe", "5m")
                sleep_seconds = self._get_sleep_seconds(timeframe)
                await asyncio.sleep(sleep_seconds)

            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def _get_sleep_seconds(self, timeframe: str) -> int:
        """Convert timeframe to seconds"""
        timeframe_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        return timeframe_map.get(timeframe, 300)

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        # Check if we should trade
        if not self.risk_manager.should_trade():
            logger.info("Risk manager preventing trades")
            return

        # Get active strategy
        strategy = self.strategies.get(self.active_strategy)
        if not strategy:
            logger.error(f"Strategy {self.active_strategy} not found")
            return

        # Fetch market data
        symbol = self.config.get("symbol", "BTC/USDT")
        timeframe = self.config.get("timeframe", "5m")

        df = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        if df.empty:
            logger.warning("No market data available")
            return

        # Get current position
        position = self.positions.get(symbol)

        # Get strategy signal
        signal = await strategy.analyze(df, position)

        # Log signal
        logger.info(f"Signal: {signal}")

        # Execute based on signal
        await self._execute_signal(signal, symbol, df["close"].iloc[-1])

        # Update position stop losses if needed
        if position and self.risk_manager.use_trailing_stop:
            new_stop = self.risk_manager.update_stop_loss(
                position, df["close"].iloc[-1]
            )
            if new_stop != position.stop_loss:
                position.stop_loss = new_stop
                logger.info(f"Updated trailing stop to {new_stop:.2f}")

    async def _execute_signal(self, signal: Dict, symbol: str, current_price: float):
        """Execute trading signal"""
        if signal["action"] == "buy" and symbol not in self.positions:
            # Check if we have room for more positions
            if len(self.positions) >= self.risk_manager.max_positions:
                logger.warning("Maximum positions reached")
                return

            # Calculate position size
            balance = await self.exchange.fetch_balance()
            if not balance:
                return

            # Get available capital (assuming USDT quote currency)
            available_capital = balance["USDT"]["free"] if "USDT" in balance else 0

            if available_capital < 10:  # Minimum trade size
                logger.warning("Insufficient capital for trade")
                return

            # Calculate volatility for position sizing
            volatility = 1.0  # Placeholder - could use ATR
            position_size = self.risk_manager.calculate_position_size(
                available_capital,
                current_price,
                volatility,
            )

            amount = position_size / current_price

            # Place order
            order = await self.exchange.place_order(symbol, "buy", amount)

            if order:
                # Create position
                position = Position(
                    symbol=symbol,
                    entry_price=current_price,
                    amount=amount,
                    side="long",
                    entry_time=datetime.now(),
                    stop_loss=current_price * (1 - self.risk_manager.stop_loss_percent),
                    take_profit=current_price
                    * (1 + self.risk_manager.take_profit_percent),
                )

                self.positions[symbol] = position

                # Record trade
                self.trade_history.append(
                    Trade(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        side="buy",
                        price=current_price,
                        amount=amount,
                        fee=self.exchange.calculate_fee(amount, current_price, "buy"),
                        order_id=order.get("id", ""),
                        strategy=self.active_strategy,
                    )
                )

                logger.info(f"Opened position: {position}")

        elif signal["action"] == "sell" and symbol in self.positions:
            position = self.positions[symbol]

            # Place sell order
            order = await self.exchange.place_order(symbol, "sell", position.amount)

            if order:
                # Calculate PnL
                pnl = (current_price - position.entry_price) * position.amount
                fee = self.exchange.calculate_fee(
                    position.amount, current_price, "sell"
                )
                net_pnl = pnl - fee

                # Update risk manager
                if net_pnl < 0:
                    self.risk_manager.consecutive_losses += 1
                else:
                    self.risk_manager.consecutive_losses = 0

                self.risk_manager.last_trade_time = datetime.now()

                # Record trade
                self.trade_history.append(
                    Trade(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        side="sell",
                        price=current_price,
                        amount=position.amount,
                        fee=fee,
                        order_id=order.get("id", ""),
                        strategy=self.active_strategy,
                    )
                )

                # Remove position
                del self.positions[symbol]

                logger.info(f"Closed position with PnL: ${net_pnl:.2f}")

        elif signal["action"] == "grid_update":
            # Special handling for grid strategy
            await self._handle_grid_orders(signal, symbol)

    async def _handle_grid_orders(self, signal: Dict, symbol: str):
        """Handle grid trading orders"""
        # This is a simplified implementation
        # In production, you'd track all grid orders and their states

        for order_info in signal.get("orders_to_place", []):
            amount = (
                self.risk_manager.calculate_position_size(
                    self.risk_manager.current_capital,
                    order_info["price"],
                )
                / order_info["price"]
                / len(signal["grid_levels"])
            )

            await self.exchange.place_order(
                symbol,
                order_info["side"],
                amount,
                OrderType.LIMIT,
                order_info["price"],
            )

            logger.info(f"Placed grid order: {order_info}")

    async def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False

        # Close all positions
        for symbol, position in self.positions.items():
            logger.info(f"Closing position: {symbol}")
            await self.exchange.place_order(symbol, "sell", position.amount)

        # Generate performance report
        self._generate_report()

    def _generate_report(self):
        """Generate performance report"""
        if not self.trade_history:
            logger.info("No trades to report")
            return

        # Calculate metrics
        total_trades = len(self.trade_history)
        buy_trades = [t for t in self.trade_history if t.side == "buy"]
        sell_trades = [t for t in self.trade_history if t.side == "sell"]

        # Simple P&L calculation
        total_pnl = 0
        for i, sell_trade in enumerate(sell_trades):
            if i < len(buy_trades):
                pnl = (sell_trade.price - buy_trades[i].price) * buy_trades[i].amount
                pnl -= buy_trades[i].fee + sell_trade.fee
                total_pnl += pnl

        runtime = datetime.now() - self.start_time

        report = f"""
        ===== Trading Bot Performance Report =====
        Strategy: {self.active_strategy}
        Runtime: {runtime}
        Total Trades: {total_trades}
        Total P&L: ${total_pnl:.2f}
        Win Rate: {(total_pnl > 0):.1%}
        =========================================
        """

        logger.info(report)

        # Save detailed trade history
        trades_df = pd.DataFrame([t.__dict__ for t in self.trade_history])
        trades_df.to_csv(
            f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False
        )


# ========================= Configuration Template =========================


def create_config_template():
    """Create a configuration template file"""
    config = {
        "exchange": {
            "name": "binance",
            "api_key": "YOUR_API_KEY",
            "api_secret": "YOUR_API_SECRET",
            "taker_fee": 0.001,
        },
        "symbol": "BTC/USDT",
        "timeframe": "5m",
        "active_strategy": "RSI_DMI",
        "risk_management": {
            "max_position_size": 0.02,
            "max_positions": 2,
            "max_drawdown": 0.10,
            "stop_loss_percent": 0.03,
            "take_profit_percent": 0.08,
            "use_trailing_stop": True,
            "trailing_stop_percent": 0.02,
            "cooldown_period": 300,
            "max_consecutive_losses": 3,
        },
        "strategies": {
            "rsi_dmi": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "adx_period": 14,
                "adx_threshold": 25,
                "use_ema_filter": True,
                "ema_period": 50,
            },
            "grid": {
                "upper_price": 50000,
                "lower_price": 40000,
                "num_grids": 20,
                "grid_spacing_type": "arithmetic",
            },
            "dca": {
                "base_amount": 100,
                "interval_hours": 24,
                "use_dynamic_dca": True,
                "price_drop_threshold": 0.05,
                "amount_multiplier": 1.5,
            },
            "macd": {
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "use_histogram": True,
            },
        },
    }

    with open("config_template.json", "w") as f:
        json.dump(config, f, indent=4)

    logger.info("Configuration template created: config_template.json")


# ========================= Main Entry Point =========================


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Configuration file"
    )
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument(
        "--create-config", action="store_true", help="Create config template"
    )

    args = parser.parse_args()

    if args.create_config:
        create_config_template()
        return

    # Check if config exists
    if not os.path.exists(args.config):
        logger.error(
            f"Configuration file {args.config} not found. Run with --create-config first."
        )
        return

    # Initialize bot
    bot = CryptoTradingBot(args.config)

    if args.backtest:
        # Run backtest
        logger.info("Running backtest...")

        # Load historical data
        symbol = bot.config.get("symbol", "BTC/USDT")
        timeframe = bot.config.get("timeframe", "5m")

        # Fetch more data for backtesting
        df = await bot.exchange.fetch_ohlcv(symbol, timeframe, limit=1000)

        if df.empty:
            logger.error("No historical data available for backtesting")
            return

        # Run backtest for each strategy
        for strategy_name, strategy in bot.strategies.items():
            logger.info(f"Backtesting {strategy_name}...")

            backtest_engine = BacktestEngine(initial_capital=10000)
            results = await backtest_engine.backtest(
                strategy,
                df,
                bot.risk_manager,
                fee_rate=bot.config["exchange"].get("taker_fee", 0.001),
            )

            if results:
                logger.info(f"\n{strategy_name} Backtest Results:")
                logger.info(f"Total Return: {results['total_return']:.2%}")
                logger.info(f"Number of Trades: {results['num_trades']}")
                logger.info(f"Win Rate: {results['win_rate']:.2%}")
                logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
                logger.info(f"Final Equity: ${results['final_equity']:.2f}")
                logger.info("-" * 40)
    else:
        # Run live trading
        try:
            await bot.start()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            await bot.stop()
        except Exception as e:
            logger.error(f"Bot error: {e}")
            await bot.stop()


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())
