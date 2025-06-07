import React from 'react';
import { PerformanceMetrics as IPerformanceMetrics } from '../types/trading';

interface PerformanceMetricsProps {
  metrics: IPerformanceMetrics;
}

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ metrics }) => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">パフォーマンス指標</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">総取引数</p>
          <p className="text-lg font-semibold">{metrics.totalTrades}</p>
        </div>
        
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">勝率</p>
          <p className="text-lg font-semibold text-blue-600">
            {(metrics.winRate * 100).toFixed(2)}%
          </p>
        </div>
        
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">最大ドローダウン</p>
          <p className="text-lg font-semibold text-red-600">
            {(metrics.maxDrawdown * 100).toFixed(2)}%
          </p>
        </div>
        
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">シャープレシオ</p>
          <p className="text-lg font-semibold text-purple-600">
            {metrics.sharpeRatio.toFixed(2)}
          </p>
        </div>
        
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">勝ちトレード</p>
          <p className="text-lg font-semibold text-green-600">
            {metrics.winningTrades}
          </p>
        </div>
        
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">負けトレード</p>
          <p className="text-lg font-semibold text-red-600">
            {metrics.losingTrades}
          </p>
        </div>
        
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">総損益</p>
          <p className={`text-lg font-semibold ${
            metrics.totalPnl >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            ${metrics.totalPnl.toFixed(2)}
          </p>
        </div>
        
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-sm text-gray-600">総損益率</p>
          <p className={`text-lg font-semibold ${
            metrics.totalPnlPercent >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {metrics.totalPnlPercent.toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
};