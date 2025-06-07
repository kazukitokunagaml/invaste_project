import { DemoTradingEngine } from './DemoTradingEngine';
import { TechnicalIndicators, OrderResult } from '../types/trading';
import { TechnicalIndicatorCalculator } from '../utils/technicalIndicators';

export class SimpleTradingStrategy {
  private engine: DemoTradingEngine;
  private positionSize: number = 0.1; // 10% of available balance
  private symbol: string = 'BTC-USD';
  
  constructor(engine: DemoTradingEngine) {
    this.engine = engine;
  }

  calculateIndicators(): TechnicalIndicators | null {
    const ohlcvData = this.engine.getPriceProvider().getOHLCVData(100);
    
    if (ohlcvData.length < 50) {
      return null;
    }

    const closePrices = ohlcvData.map(d => d.close);
    
    try {
      const rsi = TechnicalIndicatorCalculator.RSI(closePrices, 14);
      const { macd, signal } = TechnicalIndicatorCalculator.MACD(closePrices);
      const { upper, middle, lower } = TechnicalIndicatorCalculator.BollingerBands(closePrices);
      const sma20 = TechnicalIndicatorCalculator.SMA(closePrices, 20);
      const sma50 = TechnicalIndicatorCalculator.SMA(closePrices, 50);

      const lastIndex = closePrices.length - 1;
      
      if (
        isNaN(rsi[lastIndex]) ||
        isNaN(macd[lastIndex]) ||
        isNaN(signal[lastIndex]) ||
        isNaN(upper[lastIndex]) ||
        isNaN(lower[lastIndex]) ||
        isNaN(sma20[lastIndex]) ||
        isNaN(sma50[lastIndex])
      ) {
        return null;
      }

      return {
        rsi: rsi[lastIndex],
        macd: macd[lastIndex],
        macdSignal: signal[lastIndex],
        bbUpper: upper[lastIndex],
        bbLower: lower[lastIndex],
        sma20: sma20[lastIndex],
        sma50: sma50[lastIndex],
        close: closePrices[lastIndex]
      };
    } catch (error) {
      console.error('Error calculating indicators:', error);
      return null;
    }
  }

  generateSignal(indicators: TechnicalIndicators): 'buy' | 'sell' | 'hold' {
    // RSI strategy
    if (indicators.rsi < 30) {
      return 'buy';
    }
    if (indicators.rsi > 70) {
      return 'sell';
    }

    // MACD strategy
    if (indicators.macd > indicators.macdSignal) {
      return 'buy';
    }
    if (indicators.macd < indicators.macdSignal) {
      return 'sell';
    }

    // Bollinger Bands strategy
    if (indicators.close < indicators.bbLower) {
      return 'buy';
    }
    if (indicators.close > indicators.bbUpper) {
      return 'sell';
    }

    // Moving averages strategy
    if (indicators.sma20 > indicators.sma50 && indicators.close > indicators.sma20) {
      return 'buy';
    }
    if (indicators.sma20 < indicators.sma50 && indicators.close < indicators.sma20) {
      return 'sell';
    }

    return 'hold';
  }

  executeTrade(): { result: OrderResult | null; indicators: TechnicalIndicators | null } {
    const indicators = this.calculateIndicators();
    
    if (!indicators) {
      return { result: null, indicators: null };
    }

    const signal = this.generateSignal(indicators);
    const account = this.engine.getAccount();
    const hasPosition = account.positions.has(this.symbol);

    if (signal === 'buy' && !hasPosition) {
      const availableCapital = account.availableBalance;
      const positionValue = availableCapital * this.positionSize;
      const currentPrice = this.engine.getPriceProvider().getCurrentPrice();
      
      if (currentPrice > 0) {
        const amount = positionValue / currentPrice;
        const result = this.engine.placeOrder(this.symbol, 'buy', amount);
        return { result, indicators };
      }
    }

    if (signal === 'sell' && hasPosition) {
      const position = account.positions.get(this.symbol);
      if (position) {
        const result = this.engine.placeOrder(this.symbol, 'sell', position.amount);
        return { result, indicators };
      }
    }

    return { result: null, indicators };
  }

  setPositionSize(size: number) {
    this.positionSize = Math.max(0.01, Math.min(1, size));
  }

  getPositionSize(): number {
    return this.positionSize;
  }
}