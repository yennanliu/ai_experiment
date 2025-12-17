import React, { useEffect, useRef } from 'react';
import { SimulationStats, LogEntry } from '../types';

interface StatsPanelProps {
  stats: SimulationStats;
  logs: LogEntry[];
}

const StatsPanel: React.FC<StatsPanelProps> = ({ stats, logs }) => {
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [logs]);

  // Derived stats
  const avgWaitTime = stats.completedCalls > 0 
    ? (stats.totalWaitTime / stats.completedCalls).toFixed(1) 
    : '0.0';

  const efficiency = stats.totalDistance > 0 && stats.completedCalls > 0
    ? (stats.totalDistance / stats.completedCalls).toFixed(1)
    : '0.0';

  return (
    <div className="flex flex-col gap-4 w-full">
      
      {/* Metrics Cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col items-center text-center">
          <div className="text-xs text-slate-500 uppercase font-bold tracking-wide">已完成載客</div>
          <div className="text-3xl font-mono font-bold text-slate-800">{stats.completedCalls}</div>
          <div className="text-[10px] text-slate-400">人次</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col items-center text-center">
          <div className="text-xs text-slate-500 uppercase font-bold tracking-wide">總移動樓層</div>
          <div className="text-3xl font-mono font-bold text-blue-600">{stats.totalDistance}</div>
          <div className="text-[10px] text-slate-400">層</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col items-center text-center">
           <div className="text-xs text-slate-500 uppercase font-bold tracking-wide">平均等待時間</div>
           <div className={`text-3xl font-mono font-bold ${Number(avgWaitTime) > 100 ? 'text-red-500' : 'text-green-600'}`}>
             {avgWaitTime}
           </div>
           <div className="text-[10px] text-slate-400">時間單位 (Ticks)</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col items-center text-center">
           <div className="text-xs text-slate-500 uppercase font-bold tracking-wide">移動效率</div>
           <div className="text-3xl font-mono font-bold text-purple-600">{efficiency}</div>
           <div className="text-[10px] text-slate-400">樓層 / 每趟</div>
        </div>
      </div>

      {/* Logs */}
      <div className="bg-slate-900 rounded-lg p-4 flex-grow flex flex-col h-80 shadow-inner border border-slate-700">
         <div className="flex justify-between items-center mb-2">
            <h3 className="text-slate-400 text-xs font-bold uppercase">系統運行日誌</h3>
            <span className="text-[10px] text-slate-600">即時更新</span>
         </div>
         <div 
            ref={logContainerRef}
            className="overflow-y-auto flex-grow flex flex-col gap-1 font-mono text-xs pr-2"
        >
            {logs.map((log) => (
                <div key={log.id} className={`break-words border-b border-slate-800 pb-1 last:border-0 ${
                    log.type === 'info' ? 'text-slate-300' : 
                    log.type === 'success' ? 'text-green-400' : 'text-yellow-400'
                }`}>
                    <span className="opacity-40 mr-2 text-[10px] inline-block w-14">[{log.time}]</span>
                    {log.message}
                </div>
            ))}
            {logs.length === 0 && <span className="text-slate-600 italic text-center mt-10">系統已就緒，等待呼叫指令...</span>}
         </div>
      </div>
    </div>
  );
};

export default StatsPanel;