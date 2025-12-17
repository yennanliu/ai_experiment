import React from 'react';
import { AlgorithmType, Direction } from '../types';
import { NUM_FLOORS } from '../constants';

interface ControlPanelProps {
  algorithm: AlgorithmType;
  setAlgorithm: (algo: AlgorithmType) => void;
  onReset: () => void;
  onScenario: (type: 'RANDOM' | 'MORNING' | 'DOWN_PEAK') => void;
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
    <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col gap-6 h-fit">
      
      <div>
        <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">Algorithm Strategy</h3>
        <div className="flex flex-col gap-2">
            <button
                onClick={() => setAlgorithm(AlgorithmType.GREEDY)}
                className={`p-3 rounded-lg text-left text-sm border transition-all ${
                    algorithm === AlgorithmType.GREEDY 
                    ? 'bg-blue-50 border-blue-500 text-blue-700 ring-1 ring-blue-500' 
                    : 'bg-white border-slate-200 hover:border-blue-300'
                }`}
            >
                <div className="font-bold">Greedy (Nearest Car)</div>
                <div className="text-xs text-slate-500 mt-1">
                    Always picks the physically closest elevator. Ignores direction and load. Fast response, poor efficiency under load.
                </div>
            </button>

            <button
                onClick={() => setAlgorithm(AlgorithmType.SMART_ETA)}
                className={`p-3 rounded-lg text-left text-sm border transition-all ${
                    algorithm === AlgorithmType.SMART_ETA 
                    ? 'bg-purple-50 border-purple-500 text-purple-700 ring-1 ring-purple-500' 
                    : 'bg-white border-slate-200 hover:border-purple-300'
                }`}
            >
                <div className="font-bold">Smart ETA (Cost Function)</div>
                <div className="text-xs text-slate-500 mt-1">
                    Calculates estimated arrival time including stops and direction penalties. Groups passengers to reduce total travel time.
                </div>
            </button>
        </div>
      </div>

      <div className="border-t border-slate-100 pt-4">
        <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-3">Simulation Controls</h3>
        
        <div className="grid grid-cols-2 gap-2 mb-4">
            <button 
                onClick={() => setIsPaused(!isPaused)}
                className={`px-4 py-2 rounded font-bold text-sm ${isPaused ? 'bg-green-100 text-green-700' : 'bg-orange-100 text-orange-700'}`}
            >
                {isPaused ? '▶ Resume' : '⏸ Pause'}
            </button>
            <button 
                onClick={onReset}
                className="px-4 py-2 rounded font-bold text-sm bg-slate-100 text-slate-600 hover:bg-slate-200"
            >
                ↺ Reset
            </button>
        </div>

        <h4 className="text-xs font-semibold text-slate-400 mb-2">Scenarios</h4>
        <div className="flex flex-col gap-2">
            <button 
                onClick={() => onScenario('RANDOM')}
                className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded hover:bg-slate-100 text-left"
            >
                🎲 Random Burst (5 people)
            </button>
            <button 
                onClick={() => onScenario('MORNING')}
                className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded hover:bg-slate-100 text-left"
            >
                🌅 Morning Rush (Everyone at 1F going up)
            </button>
            <button 
                onClick={() => onScenario('DOWN_PEAK')}
                className="px-3 py-2 text-xs bg-slate-50 border border-slate-200 rounded hover:bg-slate-100 text-left"
            >
                🌇 Down Peak (High floors going down)
            </button>
        </div>
      </div>

    </div>
  );
};

export default ControlPanel;