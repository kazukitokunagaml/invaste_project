import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { format } from 'date-fns';
import { DemoAccount } from '../types/trading';

interface EquityChartProps {
  account: DemoAccount;
}

export const EquityChart: React.FC<EquityChartProps> = ({ account }) => {
  const chartData = account.equityHistory.map(point => ({
    timestamp: point.timestamp.getTime(),
    equity: point.equity,
    formattedTime: format(point.timestamp, 'HH:mm:ss')
  }));

  const formatTooltipLabel = (value: number) => {
    return format(new Date(value), 'HH:mm:ss');
  };

  const formatYAxis = (value: number) => {
    return `$${value.toLocaleString()}`;
  };

  const currentEquity = chartData.length > 0 ? chartData[chartData.length - 1].equity : account.initialBalance;
  const isProfit = currentEquity >= account.initialBalance;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">資産推移チャート</h2>
        <div className="text-sm text-gray-600">
          現在価格: 
          <span className={`ml-2 font-semibold ${isProfit ? 'text-green-600' : 'text-red-600'}`}>
            ${currentEquity.toFixed(2)}
          </span>
        </div>
      </div>
      
      <div className="h-80">
        {chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            取引データがありません
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="timestamp"
                type="number"
                scale="time"
                domain={['dataMin', 'dataMax']}
                tickFormatter={formatTooltipLabel}
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                tickFormatter={formatYAxis}
                domain={['dataMin - 100', 'dataMax + 100']}
                tick={{ fontSize: 12 }}
              />
              <Tooltip 
                labelFormatter={formatTooltipLabel}
                formatter={(value: number) => [`$${value.toFixed(2)}`, '総資産']}
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #ccc',
                  borderRadius: '4px'
                }}
              />
              <ReferenceLine 
                y={account.initialBalance} 
                stroke="#94a3b8" 
                strokeDasharray="5 5" 
                label={{ value: "開始資産", position: "insideTopRight" }}
              />
              <Line 
                type="monotone" 
                dataKey="equity" 
                stroke={isProfit ? "#10b981" : "#ef4444"}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, stroke: isProfit ? "#10b981" : "#ef4444", strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};