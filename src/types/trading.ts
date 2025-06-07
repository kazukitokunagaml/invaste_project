export interface DemoTrade {
  timestamp: Date;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  amount: number;
  fee: number;
  pnl: number;
  balanceAfter: number;
}

export interface DemoPosition {
  symbol: string;
  entryPrice: number;
  amount: number;
  side: 'long' | 'short';
  entryTime: Date;
  currentPrice: number;
  unrealizedPnl: number;
  stopLoss?: number;
  takeProfit?: number;
}

export interface DemoAccount {
  initialBalance: number;
  currentBalance: number;
  availableBalance: number;
  positions: Map<string, DemoPosition>;
  tradeHistory: DemoTrade[];
  equityHistory: Array<{ timestamp: Date; equity: number }>;
}

export interface PerformanceMetrics {
  totalTrades: number;
  winRate: number;
  totalPnl: number;
  totalPnlPercent: number;
  maxDrawdown: number;
  sharpeRatio: number;
  winningTrades: number;
  losingTrades: number;
}

export interface TechnicalIndicators {
  rsi: number;
  macd: number;
  macdSignal: number;
  bbUpper: number;
  bbLower: number;
  sma20: number;
  sma50: number;
  close: number;
}

export interface OHLCV {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OrderResult {
  status: 'filled' | 'rejected';
  trade?: DemoTrade;
  executionPrice?: number;
  fee?: number;
  reason?: string;
}