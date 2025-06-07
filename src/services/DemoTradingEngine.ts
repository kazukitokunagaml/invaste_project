import { 
  DemoAccount, 
  DemoTrade, 
  DemoPosition, 
  OrderResult, 
  PerformanceMetrics,
  TechnicalIndicators,
  OHLCV
} from '../types/trading';
import { PriceDataProvider } from './PriceDataProvider';
import { TechnicalIndicatorCalculator } from '../utils/technicalIndicators';

export class DemoTradingEngine {
  private account: DemoAccount;
  private feeRate: number;
  private priceProvider: PriceDataProvider;
  private subscribers: Array<(account: DemoAccount) => void> = [];

  constructor(initialBalance: number = 10000, feeRate: number = 0.001) {
    this.account = {
      initialBalance,
      currentBalance: initialBalance,
      availableBalance: initialBalance,
      positions: new Map(),
      tradeHistory: [],
      equityHistory: []
    };
    this.feeRate = feeRate;
    this.priceProvider = new PriceDataProvider();
    
    this.priceProvider.subscribe(() => {
      this.updatePositions();
      this.updateEquityHistory();
      this.notifySubscribers();
    });
  }

  placeOrder(
    symbol: string,
    side: 'buy' | 'sell',
    amount: number,
    orderType: 'market' | 'limit' = 'market',
    limitPrice?: number
  ): OrderResult {
    const currentPrice = this.priceProvider.getCurrentPrice();
    
    let executionPrice = currentPrice;
    if (orderType === 'market') {
      const slippage = (Math.random() * 0.0005) + 0.0001;
      executionPrice = side === 'buy' 
        ? currentPrice * (1 + slippage)
        : currentPrice * (1 - slippage);
    } else if (limitPrice) {
      executionPrice = limitPrice;
    }

    const orderValue = amount * executionPrice;
    const fee = orderValue * this.feeRate;
    const totalCost = side === 'buy' ? orderValue + fee : fee;

    if (side === 'buy' && totalCost > this.account.availableBalance) {
      return { status: 'rejected', reason: '残高不足' };
    }

    if (side === 'sell' && !this.account.positions.has(symbol)) {
      return { status: 'rejected', reason: 'ポジションなし' };
    }

    let pnl = 0;

    if (side === 'buy') {
      this.account.currentBalance -= totalCost;
      this.account.availableBalance -= totalCost;

      const existingPosition = this.account.positions.get(symbol);
      if (existingPosition) {
        const totalAmount = existingPosition.amount + amount;
        existingPosition.entryPrice = 
          (existingPosition.entryPrice * existingPosition.amount + executionPrice * amount) / totalAmount;
        existingPosition.amount = totalAmount;
      } else {
        this.account.positions.set(symbol, {
          symbol,
          entryPrice: executionPrice,
          amount,
          side: 'long',
          entryTime: new Date(),
          currentPrice: executionPrice,
          unrealizedPnl: 0
        });
      }
    } else {
      const position = this.account.positions.get(symbol)!;
      const sellAmount = Math.min(amount, position.amount);
      
      pnl = (executionPrice - position.entryPrice) * sellAmount - fee;
      
      this.account.currentBalance += sellAmount * executionPrice - fee;
      this.account.availableBalance += sellAmount * executionPrice - fee;

      if (position.amount > sellAmount) {
        position.amount -= sellAmount;
      } else {
        this.account.positions.delete(symbol);
      }
    }

    const trade: DemoTrade = {
      timestamp: new Date(),
      symbol,
      side,
      price: executionPrice,
      amount: side === 'buy' ? amount : Math.min(amount, this.account.positions.get(symbol)?.amount || amount),
      fee,
      pnl,
      balanceAfter: this.account.currentBalance
    };

    this.account.tradeHistory.push(trade);
    this.updateEquityHistory();
    this.notifySubscribers();

    return {
      status: 'filled',
      trade,
      executionPrice,
      fee
    };
  }

  private updatePositions() {
    const currentPrice = this.priceProvider.getCurrentPrice();
    
    for (const [, position] of this.account.positions) {
      position.currentPrice = currentPrice;
      position.unrealizedPnl = (currentPrice - position.entryPrice) * position.amount;
    }
  }

  private updateEquityHistory() {
    const totalEquity = this.calculateTotalEquity();
    this.account.equityHistory.push({
      timestamp: new Date(),
      equity: totalEquity
    });

    if (this.account.equityHistory.length > 1000) {
      this.account.equityHistory.shift();
    }
  }

  calculateTotalEquity(): number {
    let total = this.account.currentBalance;
    
    for (const [, position] of this.account.positions) {
      total += position.amount * position.currentPrice;
    }
    
    return total;
  }

  getPerformanceMetrics(): PerformanceMetrics {
    const sellTrades = this.account.tradeHistory.filter(t => t.side === 'sell');
    const winningTrades = sellTrades.filter(t => t.pnl > 0);
    const losingTrades = sellTrades.filter(t => t.pnl < 0);
    
    const totalTrades = sellTrades.length;
    const winRate = totalTrades > 0 ? winningTrades.length / totalTrades : 0;
    
    const totalPnl = this.account.tradeHistory.reduce((sum, t) => sum + t.pnl, 0);
    const totalPnlPercent = (totalPnl / this.account.initialBalance) * 100;
    
    let maxDrawdown = 0;
    if (this.account.equityHistory.length > 0) {
      let peak = this.account.equityHistory[0].equity;
      for (const { equity } of this.account.equityHistory) {
        peak = Math.max(peak, equity);
        const drawdown = peak > 0 ? (peak - equity) / peak : 0;
        maxDrawdown = Math.max(maxDrawdown, drawdown);
      }
    }

    let sharpeRatio = 0;
    if (this.account.equityHistory.length > 1) {
      const returns: number[] = [];
      for (let i = 1; i < this.account.equityHistory.length; i++) {
        const prev = this.account.equityHistory[i - 1].equity;
        const curr = this.account.equityHistory[i].equity;
        if (prev > 0) {
          returns.push((curr - prev) / prev);
        }
      }
      
      if (returns.length > 0) {
        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
        const stdReturn = Math.sqrt(variance);
        sharpeRatio = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(252) : 0;
      }
    }

    return {
      totalTrades,
      winRate,
      totalPnl,
      totalPnlPercent,
      maxDrawdown,
      sharpeRatio,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length
    };
  }

  getAccount(): DemoAccount {
    return { ...this.account };
  }

  subscribe(callback: (account: DemoAccount) => void) {
    this.subscribers.push(callback);
    return () => {
      this.subscribers = this.subscribers.filter(sub => sub !== callback);
    };
  }

  private notifySubscribers() {
    this.subscribers.forEach(callback => callback(this.getAccount()));
  }

  getPriceProvider(): PriceDataProvider {
    return this.priceProvider;
  }

  reset() {
    this.account = {
      initialBalance: this.account.initialBalance,
      currentBalance: this.account.initialBalance,
      availableBalance: this.account.initialBalance,
      positions: new Map(),
      tradeHistory: [],
      equityHistory: []
    };
    this.notifySubscribers();
  }

  destroy() {
    this.priceProvider.destroy();
    this.subscribers = [];
  }
}