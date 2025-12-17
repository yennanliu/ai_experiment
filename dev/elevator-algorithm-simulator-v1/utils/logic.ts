import { AlgorithmType, Direction, Elevator, ElevatorState, HallCall } from '../types';
import { NUM_FLOORS, TICKS_PER_FLOOR, TICKS_DOOR_OPERATION, TICKS_DOOR_OPEN_WAIT } from '../constants';

// --- Helper: Get next target based on direction ---
export const getNextTarget = (elevator: Elevator): number | null => {
  if (elevator.targets.length === 0) return null;

  const current = elevator.floor;
  const sortedTargets = [...elevator.targets].sort((a, b) => a - b);

  if (elevator.direction === Direction.UP) {
    const nextAbove = sortedTargets.find((t) => t >= current);
    if (nextAbove !== undefined) return nextAbove;
    // If nothing above, wrap around (logic handles direction switch later)
    return sortedTargets[sortedTargets.length - 1]; 
  }

  if (elevator.direction === Direction.DOWN) {
    const nextBelow = [...sortedTargets].reverse().find((t) => t <= current);
    if (nextBelow !== undefined) return nextBelow;
    return sortedTargets[0];
  }

  // If IDLE, go to nearest
  return sortedTargets.reduce((prev, curr) => 
    Math.abs(curr - current) < Math.abs(prev - current) ? curr : prev
  );
};

// --- Algorithm 1: Greedy (Nearest Car) ---
// Simply picks the elevator with the smallest physical distance to the call, ignoring direction.
const calculateGreedyCost = (elevator: Elevator, callFloor: number): number => {
  return Math.abs(elevator.floor - callFloor);
};

// --- Algorithm 2: Smart ETA ---
// Considers direction, current stops, and penalties for switching direction.
const calculateETACost = (elevator: Elevator, callFloor: number, callDir: Direction): number => {
  const currentFloor = elevator.floor;
  const currentDir = elevator.direction;
  const state = elevator.state;
  let cost = 0;

  // Base distance cost
  cost += Math.abs(currentFloor - callFloor) * TICKS_PER_FLOOR;

  // Penalty if the elevator is moving away from the call
  if (state !== ElevatorState.IDLE) {
    if (currentDir === Direction.UP) {
      if (callFloor < currentFloor || (callFloor > currentFloor && callDir === Direction.DOWN)) {
        // Going up, but call is below OR call is above but wants to go down (needs to finish up trip first)
        cost += (NUM_FLOORS * TICKS_PER_FLOOR) / 2; // Heuristic penalty
      }
    } else if (currentDir === Direction.DOWN) {
      if (callFloor > currentFloor || (callFloor < currentFloor && callDir === Direction.UP)) {
        // Going down, but call is above OR call is below but wants to go up
        cost += (NUM_FLOORS * TICKS_PER_FLOOR) / 2;
      }
    }
  }

  // Penalty for each stop currently in the queue between elevator and call
  // This prevents assigning too many people to one elevator
  const stopsBetween = elevator.targets.filter(t => {
      if (currentDir === Direction.UP) return t > currentFloor && t < callFloor;
      if (currentDir === Direction.DOWN) return t < currentFloor && t > callFloor;
      return false;
  }).length;

  cost += stopsBetween * (TICKS_DOOR_OPERATION * 2 + TICKS_DOOR_OPEN_WAIT);

  // Small bias for IDLE elevators to distribute load
  if (state === ElevatorState.IDLE) {
    cost -= 10;
  }

  return cost;
};

// --- Dispatcher ---
export const findBestElevator = (
  elevators: Elevator[],
  floor: number,
  direction: Direction,
  algorithm: AlgorithmType
): string => {
  let bestId = elevators[0].id;
  let minCost = Infinity;

  for (const elevator of elevators) {
    let cost = 0;
    if (algorithm === AlgorithmType.GREEDY) {
      cost = calculateGreedyCost(elevator, floor);
    } else {
      cost = calculateETACost(elevator, floor, direction);
    }

    // Tie-breaker: Randomize slightly or pick lower ID to avoid lockstep
    if (cost < minCost) {
      minCost = cost;
      bestId = elevator.id;
    }
  }

  return bestId;
};

// --- Physics/State Update Tick ---
export const updateElevatorState = (elevator: Elevator, hallCalls: HallCall[]): { updatedElevator: Elevator, servicedCalls: HallCall[] } => {
  const e = { ...elevator };
  let serviced: HallCall[] = [];

  // 1. Check if we have arrived at a target
  const isAtTarget = e.targets.includes(e.floor);
  
  // Logic Flow based on State
  switch (e.state) {
    case ElevatorState.IDLE:
      if (e.targets.length > 0) {
        const nextTarget = getNextTarget(e);
        if (nextTarget !== null) {
          if (nextTarget === e.floor) {
             e.state = ElevatorState.DOORS_OPENING;
             e.doorTimer = TICKS_DOOR_OPERATION;
          } else {
            e.direction = nextTarget > e.floor ? Direction.UP : Direction.DOWN;
            e.state = ElevatorState.MOVING;
            e.moveTimer = TICKS_PER_FLOOR;
          }
        }
      } else {
        e.direction = Direction.IDLE;
      }
      break;

    case ElevatorState.MOVING:
      if (e.moveTimer > 0) {
        e.moveTimer--;
      } else {
        // Arrived at a new floor
        if (e.direction === Direction.UP) e.floor++;
        else if (e.direction === Direction.DOWN) e.floor--;
        
        e.totalMovedFloors++;
        e.moveTimer = TICKS_PER_FLOOR; // Reset for next movement

        // Check if we should stop here
        if (e.targets.includes(e.floor)) {
            // STOP!
            e.state = ElevatorState.DOORS_OPENING;
            e.doorTimer = TICKS_DOOR_OPERATION;
        }
      }
      break;

    case ElevatorState.DOORS_OPENING:
      if (e.doorTimer > 0) {
        e.doorTimer--;
      } else {
        e.state = ElevatorState.DOORS_OPEN;
        e.doorTimer = TICKS_DOOR_OPEN_WAIT;
        
        // Remove current floor from targets
        e.targets = e.targets.filter(t => t !== e.floor);

        // Find calls at this floor that match our direction (or if we are empty/turning)
        // Simplification: Elevators pick up everyone at the floor if doors open
        const callsAtFloor = hallCalls.filter(c => c.floor === e.floor);
        serviced = [...callsAtFloor];
      }
      break;

    case ElevatorState.DOORS_OPEN:
      if (e.doorTimer > 0) {
        e.doorTimer--;
      } else {
        e.state = ElevatorState.DOORS_CLOSING;
        e.doorTimer = TICKS_DOOR_OPERATION;
      }
      break;

    case ElevatorState.DOORS_CLOSING:
      if (e.doorTimer > 0) {
        e.doorTimer--;
      } else {
        // Done closing. Decide next move.
        if (e.targets.length > 0) {
           const nextTarget = getNextTarget(e);
           if (nextTarget !== null && nextTarget !== e.floor) {
             e.direction = nextTarget > e.floor ? Direction.UP : Direction.DOWN;
             e.state = ElevatorState.MOVING;
             e.moveTimer = TICKS_PER_FLOOR;
           } else if (nextTarget === e.floor) {
             // Someone pushed button for current floor while closing
             e.state = ElevatorState.DOORS_OPENING;
             e.doorTimer = TICKS_DOOR_OPERATION;
           }
        } else {
          e.state = ElevatorState.IDLE;
          e.direction = Direction.IDLE;
        }
      }
      break;
  }

  return { updatedElevator: e, servicedCalls: serviced };
};
