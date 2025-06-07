import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface PriceDisplayProps {
  currentPrice: number;
  symbol: string;
}

export const PriceDisplay: React.FC<PriceDisplayProps> = ({ currentPrice, symbol }) => {
  const [prevPrice, setPrevPrice] = useState(currentPrice);
  const [priceChange, setPriceChange] = useState(0);
  const [isFlashing, setIsFlashing] = useState(false);

  useEffect(() => {
    if (currentPrice !== prevPrice) {
      setPriceChange(currentPrice - prevPrice);
      setIsFlashing(true);
      
      const timer = setTimeout(() => {
        setIsFlashing(false);
        setPrevPrice(currentPrice);
      }, 500);

      return () => clearTimeout(timer);
    }
  }, [currentPrice, prevPrice]);

  const changePercent = prevPrice > 0 ? ((currentPrice - prevPrice) / prevPrice) * 100 : 0;
  const isPositive = priceChange >= 0;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">{symbol}</h3>
          <div className="flex items-center gap-2">
            <span className={`text-3xl font-bold transition-colors duration-300 ${
              isFlashing 
                ? isPositive ? 'text-green-500' : 'text-red-500'
                : 'text-gray-900'
            }`}>
              ${currentPrice.toFixed(2)}
            </span>
            {Math.abs(priceChange) > 0 && (
              <div className={`flex items-center gap-1 ${
                isPositive ? 'text-green-600' : 'text-red-600'
              }`}>
                {isPositive ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                <span className="text-sm font-medium">
                  {isPositive ? '+' : ''}${priceChange.toFixed(2)} ({changePercent.toFixed(2)}%)
                </span>
              </div>
            )}
          </div>
        </div>
        
        <div className={`w-3 h-3 rounded-full transition-all duration-300 ${
          isFlashing 
            ? isPositive ? 'bg-green-400 animate-pulse' : 'bg-red-400 animate-pulse'
            : 'bg-gray-300'
        }`} />
      </div>
    </div>
  );
};