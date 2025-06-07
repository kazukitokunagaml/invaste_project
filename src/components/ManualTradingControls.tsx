import React, { useState } from 'react';
import { TrendingUp, TrendingDown, DollarSign, Percent } from 'lucide-react';
import { DemoAccount } from '../types/trading';

interface ManualTradingControlsProps {
  account: DemoAccount;
  currentPrice: number;
  onBuy: (amount: number) => void;
  onSell: (amount: number) => void;
  onReset: () => void;
}

export const ManualTradingControls: React.FC<ManualTradingControlsProps> = ({
  account,
  currentPrice,
  onBuy,
  onSell,
  onReset
}) => {
  const [buyAmount, setBuyAmount] = useState('0.01');
  const [sellAmount, setSellAmount] = useState('0.01');
  const [buyType, setBuyType] = useState<'amount' | 'usd'>('amount');
  const [sellType, setSellType] = useState<'amount' | 'percent'>('amount');

  const position = account.positions.get('BTC-USD');
  const maxBuyUSD = account.availableBalance;
  const maxBuyAmount = currentPrice > 0 ? maxBuyUSD / currentPrice : 0;
  const maxSellAmount = position?.amount || 0;

  const calculateBuyAmount = () => {
    if (buyType === 'usd') {
      const usdValue = parseFloat(buyAmount) || 0;
      return currentPrice > 0 ? usdValue / currentPrice : 0;
    }
    return parseFloat(buyAmount) || 0;
  };

  const calculateSellAmount = () => {
    if (sellType === 'percent') {
      const percent = parseFloat(sellAmount) || 0;
      return maxSellAmount * (percent / 100);
    }
    return parseFloat(sellAmount) || 0;
  };

  const handleBuy = () => {
    const amount = calculateBuyAmount();
    if (amount > 0 && amount <= maxBuyAmount) {
      onBuy(amount);
      setBuyAmount('0.01');
    }
  };

  const handleSell = () => {
    const amount = calculateSellAmount();
    if (amount > 0 && amount <= maxSellAmount) {
      onSell(amount);
      setSellAmount('0.01');
    }
  };

  const getBuyButtonText = () => {
    const amount = calculateBuyAmount();
    const cost = amount * currentPrice;
    return `買い - $${cost.toFixed(2)}`;
  };

  const getSellButtonText = () => {
    const amount = calculateSellAmount();
    const value = amount * currentPrice;
    return `売り - $${value.toFixed(2)}`;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold">手動取引</h2>
        <button
          onClick={onReset}
          className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 text-sm"
        >
          リセット
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 買い注文 */}
        <div className="border rounded-lg p-4">
          <h3 className="text-lg font-medium text-green-600 mb-4 flex items-center gap-2">
            <TrendingUp size={20} />
            買い注文
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                注文タイプ
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => setBuyType('amount')}
                  className={`flex-1 px-3 py-2 rounded text-sm ${
                    buyType === 'amount'
                      ? 'bg-green-100 text-green-800 border-green-300'
                      : 'bg-gray-100 text-gray-700 border-gray-300'
                  } border`}
                >
                  数量指定
                </button>
                <button
                  onClick={() => setBuyType('usd')}
                  className={`flex-1 px-3 py-2 rounded text-sm ${
                    buyType === 'usd'
                      ? 'bg-green-100 text-green-800 border-green-300'
                      : 'bg-gray-100 text-gray-700 border-gray-300'
                  } border`}
                >
                  金額指定
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {buyType === 'amount' ? '数量 (BTC)' : '金額 (USD)'}
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={buyAmount}
                  onChange={(e) => setBuyAmount(e.target.value)}
                  step="0.0001"
                  min="0"
                  max={buyType === 'amount' ? maxBuyAmount : maxBuyUSD}
                  className="w-full px-3 py-2 border rounded-md pr-10"
                  placeholder={buyType === 'amount' ? '0.01' : '100'}
                />
                <div className="absolute right-3 top-2 text-gray-500 text-sm">
                  {buyType === 'amount' ? <span>BTC</span> : <DollarSign size={16} />}
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                最大: {buyType === 'amount' ? `${maxBuyAmount.toFixed(4)} BTC` : `$${maxBuyUSD.toFixed(2)}`}
              </p>
            </div>

            <button
              onClick={handleBuy}
              disabled={calculateBuyAmount() <= 0 || calculateBuyAmount() > maxBuyAmount}
              className="w-full px-4 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
            >
              {getBuyButtonText()}
            </button>
          </div>
        </div>

        {/* 売り注文 */}
        <div className="border rounded-lg p-4">
          <h3 className="text-lg font-medium text-red-600 mb-4 flex items-center gap-2">
            <TrendingDown size={20} />
            売り注文
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                注文タイプ
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => setSellType('amount')}
                  className={`flex-1 px-3 py-2 rounded text-sm ${
                    sellType === 'amount'
                      ? 'bg-red-100 text-red-800 border-red-300'
                      : 'bg-gray-100 text-gray-700 border-gray-300'
                  } border`}
                >
                  数量指定
                </button>
                <button
                  onClick={() => setSellType('percent')}
                  className={`flex-1 px-3 py-2 rounded text-sm ${
                    sellType === 'percent'
                      ? 'bg-red-100 text-red-800 border-red-300'
                      : 'bg-gray-100 text-gray-700 border-gray-300'
                  } border`}
                >
                  割合指定
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {sellType === 'amount' ? '数量 (BTC)' : '割合 (%)'}
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={sellAmount}
                  onChange={(e) => setSellAmount(e.target.value)}
                  step={sellType === 'amount' ? '0.0001' : '1'}
                  min="0"
                  max={sellType === 'amount' ? maxSellAmount : 100}
                  className="w-full px-3 py-2 border rounded-md pr-10"
                  placeholder={sellType === 'amount' ? '0.01' : '25'}
                />
                <div className="absolute right-3 top-2 text-gray-500 text-sm">
                  {sellType === 'amount' ? <span>BTC</span> : <Percent size={16} />}
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                最大: {sellType === 'amount' ? `${maxSellAmount.toFixed(4)} BTC` : '100%'}
              </p>
            </div>

            {/* クイック売却ボタン */}
            {sellType === 'percent' && maxSellAmount > 0 && (
              <div className="flex gap-2">
                {[25, 50, 75, 100].map(percent => (
                  <button
                    key={percent}
                    onClick={() => setSellAmount(percent.toString())}
                    className="flex-1 px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs hover:bg-gray-200"
                  >
                    {percent}%
                  </button>
                ))}
              </div>
            )}

            <button
              onClick={handleSell}
              disabled={!position || calculateSellAmount() <= 0 || calculateSellAmount() > maxSellAmount}
              className="w-full px-4 py-3 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
            >
              {position ? getSellButtonText() : 'ポジションなし'}
            </button>
          </div>
        </div>
      </div>

      {/* 現在のポジション情報 */}
      {position && (
        <div className="mt-6 bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-3">現在のポジション</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-600">数量</p>
              <p className="font-medium">{position.amount.toFixed(4)} BTC</p>
            </div>
            <div>
              <p className="text-gray-600">平均取得価格</p>
              <p className="font-medium">${position.entryPrice.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-600">現在価値</p>
              <p className="font-medium">${(position.amount * currentPrice).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-600">未実現損益</p>
              <p className={`font-medium ${
                position.unrealizedPnl >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                ${position.unrealizedPnl.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};