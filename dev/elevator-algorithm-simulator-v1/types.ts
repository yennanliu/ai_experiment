export enum Direction {
  UP = 'UP',
  DOWN = 'DOWN',
  IDLE = 'IDLE',
}

export enum ElevatorState {
  IDLE = 'IDLE',
  MOVING = 'MOVING',
  DOORS_OPENING = 'DOORS_OPENING',
  DOORS_OPEN = 'DOORS_OPEN',
  DOORS_CLOSING = 'DOORS_CLOSING',
}

export enum AlgorithmType {
  GREEDY = 'GREEDY', // Nearest Car
  SMART_ETA = 'SMART_ETA', // Cost-based/ETA
}

export interface Elevator {
  id: string;
  floor: number;
  direction: Direction;
  state: ElevatorState;
  targets: number[]; // Floors the elevator needs to stop at (Set converted to array for easier handling)
  doorTimer: number; // Ticks remaining for door operations
  moveTimer: number; // Ticks remaining to move to next floor
  totalMovedFloors: number;
}

export interface HallCall {
  floor: number;
  direction: Direction;
  timestamp: number; // To calculate wait time
}

export interface SimulationStats {
  totalWaitTime: number;
  completedCalls: number;
  totalDistance: number;
  startTime: number;
}

export interface LogEntry {
  id: string;
  time: string;
  message: string;
  type: 'info' | 'success' | 'warning';
}