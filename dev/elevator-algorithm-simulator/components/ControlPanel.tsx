import React from 'react';
import { AlgorithmType, Direction } from '../types';
import { NUM_FLOORS } from '../constants';

interface ControlPanelProps {
  algorithm: AlgorithmType;
  setAlgorithm: (algo: AlgorithmType) => void;
  onReset: () => void;
  onScenario: (type: 'RANDOM' | 'MORNING' | 'DOWN_PEAK' | 'OPPOSITE') => void;
  isPaused: boolean;
  setIsPaused: (p: boolean) => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ 
  algorithm, 
  setAlgorithm, 
  onReset, 
  onScenario,
  isPaused,
  setIsPaused
}) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 flex flex-col gap-6 h-fit w-full">
      
      <div>
        <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">演算法策略選擇</h3>
        <div className="grid grid-cols-1 gap-3">
            <button
                onClick={() => setAlgorithm(AlgorithmType.GREEDY)}
                className={`relative p-4 rounded-xl text-left border-2 transition-all duration-200 ${
                    algorithm === AlgorithmType.GREEDY 
                    ? 'bg-blue-50 border-blue-500 shadow-md' 
                    : 'bg-white border-slate-200 hover:border-blue-300 hover:bg-slate-50'
                }`}
            >
                <div className="flex justify-between items-center mb-1">
                    <span className={`font-bold text-lg ${algorithm === AlgorithmType.GREEDY ? 'text-blue-700' : 'text-slate-700'}`}>
                        貪婪演算法 (Greedy)
                    </span>
                    {algorithm === AlgorithmType.GREEDY && <span className="text-blue-600 text-xl">✓</span>}
                </div>
                <div className="text-xs text-slate-500 leading-relaxed">
                   總是派遣<strong>物理距離最近</strong>的電梯，完全忽略運行方向與當前負載。
                   <br/>
                   <span className="text-red-500 font-medium">缺點：</span> 容易造成反向攔截，增加乘客等待時間。
                </div>
            </button>

            <button
                onClick={() => setAlgorithm(AlgorithmType.SMART_ETA)}
                className={`relative p-4 rounded-xl text-left border-2 transition-all duration-200 ${
                    algorithm === AlgorithmType.SMART_ETA 
                    ? 'bg-purple-50 border-purple-500 shadow-md' 
                    : 'bg-white border-slate-200 hover:border-purple-300 hover:bg-slate-50'
                }`}
            >
                <div className="flex justify-between items-center mb-1">
                     <span className={`font-bold text-lg ${algorithm === AlgorithmType.SMART_ETA ? 'text-purple-700' : 'text-slate-700'}`}>
                        智慧調度 (Smart ETA)
                    </span>
                    {algorithm === AlgorithmType.SMART_ETA && <span className="text-purple-600 text-xl">✓</span>}
                </div>
                <div className="text-xs text-slate-500 leading-relaxed">
                    計算<strong>預估抵達時間 (ETA)</strong>，納入中途停靠與轉向成本。
                    <br/>
                    <span className="text-green-600 font-medium">優點：</span> 優先順路載客，避免無效移動。
                </div>
            </button>
        </div>
      </div>

      <div className="border-t border-slate-100 pt-4">
        <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">模擬控制</h3>
        
        <div className="grid grid-cols-2 gap-3 mb-4">
            <button 
                onClick={() => setIsPaused(!isPaused)}
                className={`px-4 py-3 rounded-lg font-bold text-sm transition-colors flex items-center justify-center gap-2 ${
                    isPaused 
                    ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                    : 'bg-orange-100 text-orange-700 hover:bg-orange-200'
                }`}
            >
                {isPaused ? '▶ 繼續模擬' : '⏸ 暫停模擬'}
            </button>
            <button 
                onClick={onReset}
                className="px-4 py-3 rounded-lg font-bold text-sm bg-slate-100 text-slate-600 hover:bg-slate-200 flex items-center justify-center gap-2"
            >
                ↺ 重置系統
            </button>
        </div>

        <h4 className="text-xs font-semibold text-slate-400 mb-2">快速測試情境</h4>
        <div className="grid grid-cols-1 gap-2">
            <button 
                onClick={() => onScenario('RANDOM')}
                className="px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg hover:bg-slate-50 hover:border-slate-300 text-left transition-colors text-slate-700"
            >
                🎲 <strong>隨機人流</strong> (隨機產生 5 組呼叫)
            </button>
            <button 
                onClick={() => onScenario('MORNING')}
                className="px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg hover:bg-slate-50 hover:border-slate-300 text-left transition-colors text-slate-700"
            >
                🌅 <strong>早晨尖峰</strong> (1樓大量上樓需求)
            </button>
            <button 
                onClick={() => onScenario('DOWN_PEAK')}
                className="px-3 py-2 text-sm bg-white border border-slate-200 rounded-lg hover:bg-slate-50 hover:border-slate-300 text-left transition-colors text-slate-700"
            >
                🌇 <strong>下班尖峰</strong> (高樓層集中下樓)
            </button>
             <button 
                onClick={() => onScenario('OPPOSITE')}
                className="px-3 py-2 text-sm bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 text-left transition-colors text-red-800"
            >
                ⚠️ <strong>反向陷阱</strong> (測試演算法智商)
                <div className="text-[10px] opacity-75 font-normal mt-1">
                    情境: 電梯B在高樓層下行時，呼叫其上方樓層的下樓需求。貪婪法會錯誤指派B。
                </div>
            </button>
        </div>
      </div>

    </div>
  );
};

export default ControlPanel;