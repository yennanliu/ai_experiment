export const NUM_FLOORS = 15;
export const NUM_ELEVATORS = 2;

// Timing (in Ticks)
export const TICKS_PER_FLOOR = 20; 
export const TICKS_DOOR_OPERATION = 10;
export const TICKS_DOOR_OPEN_WAIT = 30;

// Simulation Speed
export const TICK_RATE_MS = 50; // Update every 50ms

export const COLORS = {
  elevatorA: 'bg-blue-600',
  elevatorB: 'bg-purple-600',
  floorActive: 'bg-yellow-100',
  btnActiveUp: 'bg-green-500 text-white ring-2 ring-green-300',
  btnActiveDown: 'bg-red-500 text-white ring-2 ring-red-300',
  btnIdle: 'bg-slate-200 text-slate-500 hover:bg-slate-300',
};