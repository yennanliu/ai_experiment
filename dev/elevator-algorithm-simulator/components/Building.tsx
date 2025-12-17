import React from 'react';
import { Elevator, HallCall, Direction, ElevatorState } from '../types';
import { NUM_FLOORS, COLORS, TICKS_PER_FLOOR } from '../constants';

interface BuildingProps {
  elevators: Elevator[];
  hallCalls: HallCall[];
  onCall: (floor: number, direction: Direction) => void;
}

const Building: React.FC<BuildingProps> = ({ elevators, hallCalls, onCall }) => {
  
  const isCallActive = (floor: number, direction: Direction) => {
    return hallCalls.some(c => c.floor === floor && c.direction === direction);
  };

  return (
    <div className="flex flex-row gap-6 items-end p-6 bg-white rounded-xl shadow-xl border border-slate-200">
      {/* Floor Labels & Buttons */}
      <div className="flex flex-col-reverse">
        {Array.from({ length: NUM_FLOORS }, (_, i) => i + 1).map(floor => (
          <div key={floor} className="h-12 flex items-center gap-4 justify-end border-b border-slate-100 last:border-0">
            <span className="font-mono text-slate-400 font-bold w-8 text-right text-sm">{floor}F</span>
            <div className="flex gap-1">
              {floor < NUM_FLOORS && (
                <button
                  onClick={() => onCall(floor, Direction.UP)}
                  className={`w-8 h-8 rounded-lg text-xs font-bold transition-all duration-200 flex items-center justify-center ${
                    isCallActive(floor, Direction.UP) ? COLORS.btnActiveUp : COLORS.btnIdle
                  }`}
                  title="上樓"
                >
                  ▲
                </button>
              )}
              {floor === NUM_FLOORS && <div className="w-8 h-8"></div>} {/* Spacer */}
              
              {floor > 1 && (
                <button
                  onClick={() => onCall(floor, Direction.DOWN)}
                  className={`w-8 h-8 rounded-lg text-xs font-bold transition-all duration-200 flex items-center justify-center ${
                    isCallActive(floor, Direction.DOWN) ? COLORS.btnActiveDown : COLORS.btnIdle
                  }`}
                  title="下樓"
                >
                  ▼
                </button>
              )}
               {floor === 1 && <div className="w-8 h-8"></div>} {/* Spacer */}
            </div>
          </div>
        ))}
      </div>

      {/* Shafts */}
      <div className="flex gap-6 px-4 py-2 bg-slate-50 rounded-lg border-inner shadow-inner">
        {elevators.map((elevator, index) => {
          const FLOOR_HEIGHT = 48; // px
          
          let bottomPx = (elevator.floor - 1) * FLOOR_HEIGHT;
          
          if (elevator.state === ElevatorState.MOVING) {
             const progress = (TICKS_PER_FLOOR - elevator.moveTimer) / TICKS_PER_FLOOR;
             if (elevator.direction === Direction.UP) {
                bottomPx += progress * FLOOR_HEIGHT;
             } else {
                bottomPx -= progress * FLOOR_HEIGHT;
             }
          }

          const bgColor = index === 0 ? COLORS.elevatorA : COLORS.elevatorB;
          const statusChar = elevator.direction === Direction.UP ? '▲' : elevator.direction === Direction.DOWN ? '▼' : '●';
          const doorsOpen = elevator.state === ElevatorState.DOORS_OPEN || elevator.state === ElevatorState.DOORS_OPENING || elevator.state === ElevatorState.DOORS_CLOSING;

          return (
            <div key={elevator.id} className="relative w-20 bg-slate-200 rounded border-x-2 border-slate-300" style={{ height: `${NUM_FLOORS * FLOOR_HEIGHT}px` }}>
              {/* Grid lines */}
              {Array.from({ length: NUM_FLOORS }).map((_, i) => (
                <div key={i} className="absolute w-full border-b border-slate-300/40" style={{ bottom: `${i * FLOOR_HEIGHT}px`, height: '1px' }}></div>
              ))}
              
              {/* Elevator Car */}
               <div
                className={`absolute left-1 right-1 h-10 rounded shadow-md flex items-center justify-center text-white font-bold text-xs flex-col transition-all duration-75 linear ${bgColor} z-10 border border-white/20`}
                style={{ 
                    bottom: `${bottomPx + 2}px`, 
                }}
              >
                <div className="flex items-center gap-1">
                    <span>{elevator.id}</span>
                    <span className="text-[10px]">{statusChar}</span>
                </div>
                {elevator.targets.length > 0 && (
                   <span className="text-[9px] bg-black/20 px-1 rounded-sm mt-0.5">待停: {elevator.targets.length}</span>
                )}
                
                {/* Doors Visual */}
                <div className={`absolute inset-0 bg-slate-800/80 flex transition-all duration-300 rounded overflow-hidden ${doorsOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
                   <div className="w-1/2 h-full bg-slate-300 border-r border-slate-400 transition-all origin-left duration-500 ease-in-out" style={{ transform: doorsOpen && elevator.state !== ElevatorState.DOORS_CLOSING ? 'scaleX(0.1)' : 'scaleX(1)' }}></div>
                   <div className="w-1/2 h-full bg-slate-300 border-l border-slate-400 transition-all origin-right duration-500 ease-in-out" style={{ transform: doorsOpen && elevator.state !== ElevatorState.DOORS_CLOSING ? 'scaleX(0.1)' : 'scaleX(1)' }}></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Building;