import React from 'react';
import { DemoAccount } from '../types/trading';

interface AccountInfoProps {
  account: DemoAccount;
  totalEquity: number;
}

export const AccountInfo: React.FC<AccountInfoProps> = ({ account, totalEquity }) => {
  const pnl = totalEquity - account.initialBalance;
  const pnlPercent = (pnl / account.initialBalance) * 100;
  const pnlColor = pnl >= 0 ? 'text-green-600' : 'text-red-600';

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4">アカウント情報</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-50 p-4 rounded">
          <p className="text-sm text-gray-600">残高</p>
          <p className="text-lg font-semibold">${account.currentBalance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
        </div>
        <div className="bg-gray-50 p-4 rounded">
          <p className="text-sm text-gray-600">総資産</p>
          <p className="text-lg font-semibold">${totalEquity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
        </div>
        <div className="bg-gray-50 p-4 rounded">
          <p className="text-sm text-gray-600">損益</p>
          <p className={`text-lg font-semibold ${pnlColor}`}>
            ${pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ({pnlPercent.toFixed(2)}%)
          </p>
        </div>
      </div>
    </div>
  );
};