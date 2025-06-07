import React, { useState, useEffect } from 'react';
import { DemoTradingEngine } from './services/DemoTradingEngine';
import { DemoAccount } from './types/trading';
import { AccountInfo } from './components/AccountInfo';
import { ManualTradingControls } from './components/ManualTradingControls';
import { PerformanceMetrics } from './components/PerformanceMetrics';
import { TradeHistory } from './components/TradeHistory';
import { EquityChart } from './components/EquityChart';
import { PriceDisplay } from './components/PriceDisplay';

const App: React.FC = () => {
  const [engine] = useState(() => new DemoTradingEngine(10000));
  const [account, setAccount] = useState<DemoAccount>(engine.getAccount());
  const [currentPrice, setCurrentPrice] = useState(45000);
  const [logs, setLogs] = useState<string[]>(['手動取引モードが開始されました']);

  useEffect(() => {
    const unsubscribeEngine = engine.subscribe((updatedAccount) => {
      setAccount(updatedAccount);
    });

    const unsubscribePrice = engine.getPriceProvider().subscribe((price) => {
      setCurrentPrice(price);
    });

    return () => {
      unsubscribeEngine();
      unsubscribePrice();
      engine.destroy();
    };
  }, [engine]);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-50), `[${timestamp}] ${message}`]);
  };

  const handleBuy = (amount: number) => {
    try {
      const result = engine.placeOrder('BTC-USD', 'buy', amount);
      if (result.status === 'filled' && result.trade) {
        const trade = result.trade;
        addLog(`買い注文実行: ${trade.amount.toFixed(4)} BTC @ $${trade.price.toFixed(2)} (手数料: $${trade.fee.toFixed(2)})`);
      } else if (result.status === 'rejected') {
        addLog(`買い注文失敗: ${result.reason}`);
      }
    } catch (error) {
      addLog(`エラー: ${error}`);
    }
  };

  const handleSell = (amount: number) => {
    try {
      const result = engine.placeOrder('BTC-USD', 'sell', amount);
      if (result.status === 'filled' && result.trade) {
        const trade = result.trade;
        addLog(`売り注文実行: ${trade.amount.toFixed(4)} BTC @ $${trade.price.toFixed(2)} (損益: $${trade.pnl.toFixed(2)})`);
      } else if (result.status === 'rejected') {
        addLog(`売り注文失敗: ${result.reason}`);
      }
    } catch (error) {
      addLog(`エラー: ${error}`);
    }
  };

  const resetAccount = () => {
    engine.reset();
    setLogs([]);
    addLog('アカウントをリセットしました');
  };

  const totalEquity = engine.calculateTotalEquity();
  const metrics = engine.getPerformanceMetrics();

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl font-bold text-gray-900">
              仮想通貨デモトレーディング
            </h1>
            <div className="text-sm text-gray-500">
              手動取引モード
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-6">
          {/* 価格表示とアカウント情報 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PriceDisplay currentPrice={currentPrice} symbol="BTC-USD" />
            <AccountInfo account={account} totalEquity={totalEquity} />
          </div>

          {/* 手動取引コントロール */}
          <ManualTradingControls
            account={account}
            currentPrice={currentPrice}
            onBuy={handleBuy}
            onSell={handleSell}
            onReset={resetAccount}
          />

          {/* パフォーマンス指標 */}
          <PerformanceMetrics metrics={metrics} />

          {/* チャート */}
          <EquityChart account={account} />

          {/* 取引履歴とログ */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <TradeHistory trades={account.tradeHistory} />
            
            {/* 取引ログ */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">取引ログ</h2>
              <div className="bg-gray-50 rounded p-4 h-64 overflow-y-auto">
                {logs.length === 0 ? (
                  <p className="text-gray-500">ログがありません</p>
                ) : (
                  <div className="space-y-1">
                    {logs.map((log, index) => (
                      <div key={index} className="text-sm font-mono text-gray-700">
                        {log}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;