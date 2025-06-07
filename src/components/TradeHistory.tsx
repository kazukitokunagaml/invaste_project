import React from 'react';
import { format } from 'date-fns';
import { DemoTrade } from '../types/trading';

interface TradeHistoryProps {
  trades: DemoTrade[];
}

export const TradeHistory: React.FC<TradeHistoryProps> = ({ trades }) => {
  const recentTrades = trades.slice(-10).reverse();

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">取引履歴</h2>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2">時刻</th>
              <th className="text-left py-2">方向</th>
              <th className="text-right py-2">価格</th>
              <th className="text-right py-2">数量</th>
              <th className="text-right py-2">手数料</th>
              <th className="text-right py-2">損益</th>
            </tr>
          </thead>
          <tbody>
            {recentTrades.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-4 text-gray-500">
                  取引履歴がありません
                </td>
              </tr>
            ) : (
              recentTrades.map((trade, index) => (
                <tr key={index} className="border-b hover:bg-gray-50">
                  <td className="py-2">
                    {format(trade.timestamp, 'HH:mm:ss')}
                  </td>
                  <td className="py-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      trade.side === 'buy' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {trade.side === 'buy' ? '買い' : '売り'}
                    </span>
                  </td>
                  <td className="text-right py-2">
                    ${trade.price.toFixed(2)}
                  </td>
                  <td className="text-right py-2">
                    {trade.amount.toFixed(4)}
                  </td>
                  <td className="text-right py-2">
                    ${trade.fee.toFixed(2)}
                  </td>
                  <td className={`text-right py-2 font-medium ${
                    trade.pnl > 0 ? 'text-green-600' : 
                    trade.pnl < 0 ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {trade.pnl !== 0 ? `$${trade.pnl.toFixed(2)}` : '-'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      {trades.length > 10 && (
        <p className="text-sm text-gray-500 mt-2">
          最新の10件を表示中（全{trades.length}件）
        </p>
      )}
    </div>
  );
};