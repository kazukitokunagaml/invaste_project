export class TechnicalIndicatorCalculator {
  static SMA(data: number[], period: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        result.push(NaN);
      } else {
        const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        result.push(sum / period);
      }
    }
    return result;
  }

  static EMA(data: number[], period: number): number[] {
    const result: number[] = [];
    const multiplier = 2 / (period + 1);
    
    for (let i = 0; i < data.length; i++) {
      if (i === 0) {
        result.push(data[i]);
      } else {
        result.push((data[i] - result[i - 1]) * multiplier + result[i - 1]);
      }
    }
    return result;
  }

  static RSI(data: number[], period: number = 14): number[] {
    const result: number[] = [];
    const deltas: number[] = [];
    
    for (let i = 1; i < data.length; i++) {
      deltas.push(data[i] - data[i - 1]);
    }
    
    for (let i = 0; i < deltas.length; i++) {
      if (i < period - 1) {
        result.push(NaN);
      } else {
        const gains = deltas.slice(i - period + 1, i + 1).filter(d => d > 0);
        const losses = deltas.slice(i - period + 1, i + 1).filter(d => d < 0).map(d => Math.abs(d));
        
        const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
        const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;
        
        if (avgLoss === 0) {
          result.push(100);
        } else {
          const rs = avgGain / avgLoss;
          result.push(100 - (100 / (1 + rs)));
        }
      }
    }
    
    return [NaN, ...result];
  }

  static MACD(data: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) {
    const emaFast = this.EMA(data, fastPeriod);
    const emaSlow = this.EMA(data, slowPeriod);
    
    const macd = emaFast.map((fast, i) => fast - emaSlow[i]);
    const signal = this.EMA(macd.filter(v => !isNaN(v)), signalPeriod);
    
    const paddedSignal = [...Array(macd.length - signal.length).fill(NaN), ...signal];
    const histogram = macd.map((m, i) => m - paddedSignal[i]);
    
    return { macd, signal: paddedSignal, histogram };
  }

  static BollingerBands(data: number[], period: number = 20, stdDev: number = 2) {
    const sma = this.SMA(data, period);
    const result = {
      upper: [] as number[],
      middle: sma,
      lower: [] as number[]
    };
    
    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        result.upper.push(NaN);
        result.lower.push(NaN);
      } else {
        const slice = data.slice(i - period + 1, i + 1);
        const mean = sma[i];
        const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
        const std = Math.sqrt(variance);
        
        result.upper.push(mean + (std * stdDev));
        result.lower.push(mean - (std * stdDev));
      }
    }
    
    return result;
  }
}