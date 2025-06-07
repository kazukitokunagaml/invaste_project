import { OHLCV } from '../types/trading';

export class PriceDataProvider {
  private currentPrice: number = 45000;
  private priceHistory: Array<{ timestamp: Date; price: number }> = [];
  private volatility: number = 0.02;
  private minPrice: number = 1000;
  private intervalId?: number;
  private subscribers: Array<(price: number) => void> = [];

  constructor(
    private mode: 'simulation' | 'historical' = 'simulation',
    private symbol: string = 'BTC-USD'
  ) {
    this.startPriceSimulation();
  }

  private startPriceSimulation() {
    if (this.mode === 'simulation') {
      this.intervalId = window.setInterval(() => {
        this.updatePrice();
      }, 1000);
    }
  }

  private updatePrice() {
    const change = (Math.random() - 0.5) * this.volatility * 2;
    this.currentPrice *= (1 + change);
    
    if (Math.random() < 0.05) {
      const spike = (Math.random() - 0.5) * 0.06;
      this.currentPrice *= (1 + spike);
    }
    
    this.currentPrice = Math.max(this.currentPrice, this.minPrice);
    
    const now = new Date();
    this.priceHistory.push({ timestamp: now, price: this.currentPrice });
    
    if (this.priceHistory.length > 1000) {
      this.priceHistory.shift();
    }
    
    this.notifySubscribers(this.currentPrice);
  }

  getCurrentPrice(): number {
    return this.currentPrice;
  }

  subscribe(callback: (price: number) => void) {
    this.subscribers.push(callback);
    return () => {
      this.subscribers = this.subscribers.filter(sub => sub !== callback);
    };
  }

  private notifySubscribers(price: number) {
    this.subscribers.forEach(callback => callback(price));
  }

  getOHLCVData(periods: number = 100): OHLCV[] {
    const data: OHLCV[] = [];
    const now = new Date();
    
    for (let i = periods - 1; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 5 * 60 * 1000);
      const basePrice = this.currentPrice * (1 + (Math.random() - 0.5) * 0.1);
      
      const open = basePrice;
      const high = open * (1 + Math.random() * 0.02);
      const low = open * (1 - Math.random() * 0.02);
      const close = low + Math.random() * (high - low);
      const volume = Math.random() * 10000 + 1000;
      
      data.push({
        timestamp,
        open,
        high,
        low,
        close,
        volume
      });
    }
    
    return data;
  }

  getPriceHistory(): Array<{ timestamp: Date; price: number }> {
    return [...this.priceHistory];
  }

  destroy() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    this.subscribers = [];
  }
}