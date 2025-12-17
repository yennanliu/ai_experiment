import React from 'react';
import { Elevator, HallCall, Direction, ElevatorState } from '../types';
import { NUM_FLOORS, COLORS, TICKS_PER_FLOOR } from '../constants';

interface BuildingProps {
  elevators: Elevator[];
  hallCalls: HallCall[];
  onCall: (floor: number, direction: Direction) => void;
}

const Building: React.FC<BuildingProps> = ({ elevators, hallCalls, onCall }) => {
  
  // Helper to check if a button is active
  const isCallActive = (floor: number, direction: Direction) => {
    return hallCalls.some(c => c.floor === floor && c.direction === direction);
  };

  return (
    <div className="flex flex-row gap-8 items-end p-4 bg-white rounded-xl shadow-lg border border-slate-200">
      {/* Floor Labels & Buttons */}
      <div className="flex flex-col-reverse">
        {Array.from({ length: NUM_FLOORS }, (_, i) => i + 1).map(floor => (
          <div key={floor} className="h-12 flex items-center gap-4 justify-end border-b border-slate-100 last:border-0">
            <span className="font-mono text-slate-400 font-bold w-6 text-right">{floor}</span>
            <div className="flex gap-1">
              {floor < NUM_FLOORS && (
                <button
                  onClick={() => onCall(floor, Direction.UP)}
                  className={`w-8 h-8 rounded text-xs font-bold transition-all ${
                    isCallActive(floor, Direction.UP) ? COLORS.btnActiveUp : COLORS.btnIdle
                  }`}
                >
                  ▲
                </button>
              )}
              {floor > 1 && (
                <button
                  onClick={() => onCall(floor, Direction.DOWN)}
                  className={`w-8 h-8 rounded text-xs font-bold transition-all ${
                    isCallActive(floor, Direction.DOWN) ? COLORS.btnActiveDown : COLORS.btnIdle
                  }`}
                >
                  ▼
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Shafts */}
      <div className="flex gap-4 p-2 bg-slate-50 rounded-lg border-inner shadow-inner">
        {elevators.map((elevator, index) => {
          // Calculate precise bottom position for smooth animation
          // Standard floor height is 48px (h-12)
          const FLOOR_HEIGHT = 48; // px
          
          let bottomPx = (elevator.floor - 1) * FLOOR_HEIGHT;
          
          // Interpolate movement if MOVING
          if (elevator.state === ElevatorState.MOVING) {
             const progress = (TICKS_PER_FLOOR - elevator.moveTimer) / TICKS_PER_FLOOR;
             // If moving UP, we add progress. If DOWN, we subtract.
             // Wait, logic updates floor only on arrival. So current floor is start.
             // If UP: Current is 1. moving to 2. 
             // Logic: Floor is incremented at END of move timer in my logic.
             // So visually we are moving towards target.
             // Actually, my logic updates floor at END of timer. So elevator.floor is "Previous Floor".
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
            <div key={elevator.id} className="relative w-16 bg-slate-200 rounded border-x-2 border-slate-300" style={{ height: `${NUM_FLOORS * FLOOR_HEIGHT}px` }}>
              {/* Grid lines in shaft */}
              {Array.from({ length: NUM_FLOORS }).map((_, i) => (
                <div key={i} className="absolute w-full border-b border-slate-300/50" style={{ bottom: `${i * FLOOR_HEIGHT}px`, height: '1px' }}></div>
              ))}
              
              {/* The Elevator Car */}
              <div
                className={`absolute left-1 right-1 h-10 rounded shadow-md flex items-center justify-center text-white font-bold text-sm transition-transform duration-75 ${bgColor}`}
                style={{ 
                    bottom: '4px', // slight offset for centering in floor slot
                    transform: `translateY(${-bottomPx}px)` // Using negative translate to move UP because DOM flow is top-down usually, but here we used flex-col-reverse or relative bottom?
                    // Actually standard 'bottom' CSS property is easier here.
                }}
              >
                  {/* Override styles for 'bottom' approach */}
              </div>
              
              {/* Re-implementing using 'bottom' property directly avoids transform confusion with flex-col-reverse contexts sometimes */}
               <div
                className={`absolute left-1 right-1 h-10 rounded shadow-md flex items-center justify-center text-white font-bold text-xs flex-col transition-all duration-75 linear ${bgColor} z-10`}
                style={{ 
                    bottom: `${bottomPx + 2}px`, 
                }}
              >
                <span>{elevator.id} {statusChar}</span>
                <span className="text-[10px] opacity-80">{elevator.targets.length} req</span>
                
                {/* Doors Visual */}
                <div className={`absolute inset-0 bg-black/20 flex transition-all duration-300 ${doorsOpen ? 'w-full' : 'w-0 opacity-0'}`}>
                   <div className="w-1/2 h-full bg-slate-300 border-r border-slate-400 transition-all origin-left" style={{ transform: doorsOpen ? 'scaleX(0.1)' : 'scaleX(1)' }}></div>
                   <div className="w-1/2 h-full bg-slate-300 border-l border-slate-400 transition-all origin-right" style={{ transform: doorsOpen ? 'scaleX(0.1)' : 'scaleX(1)' }}></div>
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