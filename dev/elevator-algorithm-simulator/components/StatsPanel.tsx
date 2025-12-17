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
    <div className="flex flex-col gap-4 w-full max-w-sm">
      
      {/* Metrics Cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <div className="text-xs text-slate-500 uppercase font-bold">Completed Trips</div>
          <div className="text-2xl font-mono font-bold text-slate-800">{stats.completedCalls}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <div className="text-xs text-slate-500 uppercase font-bold">Total Floors Moved</div>
          <div className="text-2xl font-mono font-bold text-blue-600">{stats.totalDistance}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
           <div className="text-xs text-slate-500 uppercase font-bold">Avg Wait Time (ticks)</div>
           <div className={`text-2xl font-mono font-bold ${Number(avgWaitTime) > 100 ? 'text-red-500' : 'text-green-600'}`}>
             {avgWaitTime}
           </div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
           <div className="text-xs text-slate-500 uppercase font-bold">Floors / Trip</div>
           <div className="text-2xl font-mono font-bold text-purple-600">{efficiency}</div>
        </div>
      </div>

      {/* Logs */}
      <div className="bg-slate-900 rounded-lg p-4 flex-grow flex flex-col h-64 shadow-inner">
         <h3 className="text-slate-400 text-xs font-bold uppercase mb-2">System Log</h3>
         <div 
            ref={logContainerRef}
            className="overflow-y-auto flex-grow flex flex-col gap-1 font-mono text-xs pr-2"
        >
            {logs.map((log) => (
                <div key={log.id} className={`break-words ${
                    log.type === 'info' ? 'text-slate-300' : 
                    log.type === 'success' ? 'text-green-400' : 'text-yellow-400'
                }`}>
                    <span className="opacity-50 mr-2">[{log.time}]</span>
                    {log.message}
                </div>
            ))}
            {logs.length === 0 && <span className="text-slate-600 italic">System initialized. Waiting for calls...</span>}
         </div>
      </div>
    </div>
  );
};

export default StatsPanel;