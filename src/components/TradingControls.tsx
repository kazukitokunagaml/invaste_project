import React, { useState } from 'react';
import { Play, Square, RotateCcw } from 'lucide-react';

interface TradingControlsProps {
  isTrading: boolean;
  onStartTrading: () => void;
  onStopTrading: () => void;
  onReset: () => void;
  tradingInterval: number;
  onIntervalChange: (interval: number) => void;
}

export const TradingControls: React.FC<TradingControlsProps> = ({
  isTrading,
  onStartTrading,
  onStopTrading,
  onReset,
  tradingInterval,
  onIntervalChange
}) => {
  const [intervalInput, setIntervalInput] = useState(tradingInterval.toString());

  const handleIntervalChange = (value: string) => {
    setIntervalInput(value);
    const numValue = parseInt(value);
    if (!isNaN(numValue) && numValue >= 1 && numValue <= 60) {
      onIntervalChange(numValue);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">コントロール</h2>
      
      <div className="flex flex-wrap gap-4 items-center">
        <button
          onClick={onStartTrading}
          disabled={isTrading}
          className={`flex items-center gap-2 px-4 py-2 rounded font-medium ${
            isTrading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-600 text-white hover:bg-green-700'
          }`}
        >
          <Play size={16} />
          取引開始
        </button>

        <button
          onClick={onStopTrading}
          disabled={!isTrading}
          className={`flex items-center gap-2 px-4 py-2 rounded font-medium ${
            !isTrading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-red-600 text-white hover:bg-red-700'
          }`}
        >
          <Square size={16} />
          取引停止
        </button>

        <button
          onClick={onReset}
          disabled={isTrading}
          className={`flex items-center gap-2 px-4 py-2 rounded font-medium ${
            isTrading
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          <RotateCcw size={16} />
          リセット
        </button>

        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700">
            取引間隔（秒）:
          </label>
          <input
            type="number"
            min="1"
            max="60"
            value={intervalInput}
            onChange={(e) => handleIntervalChange(e.target.value)}
            className="w-20 px-2 py-1 border rounded text-center"
            disabled={isTrading}
          />
        </div>
      </div>
    </div>
  );
};