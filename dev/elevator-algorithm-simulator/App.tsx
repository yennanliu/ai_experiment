import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Elevator, 
  HallCall, 
  Direction, 
  ElevatorState, 
  AlgorithmType, 
  SimulationStats, 
  LogEntry 
} from './types';
import { NUM_FLOORS, NUM_ELEVATORS, TICK_RATE_MS } from './constants';
import { findBestElevator, updateElevatorState } from './utils/logic';
import Building from './components/Building';
import ControlPanel from './components/ControlPanel';
import StatsPanel from './components/StatsPanel';

const INITIAL_STATS: SimulationStats = {
  totalWaitTime: 0,
  completedCalls: 0,
  totalDistance: 0,
  startTime: Date.now(),
};

const App: React.FC = () => {
  // --- State ---
  const [elevators, setElevators] = useState<Elevator[]>([]);
  const [hallCalls, setHallCalls] = useState<HallCall[]>([]);
  const [algorithm, setAlgorithm] = useState<AlgorithmType>(AlgorithmType.GREEDY);
  const [stats, setStats] = useState<SimulationStats>(INITIAL_STATS);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [tickCount, setTickCount] = useState(0);

  // --- Refs ---
  // Using refs for values needed inside the interval to avoid stale closures 
  // without re-creating the interval constanty
  const elevatorsRef = useRef<Elevator[]>([]);
  const hallCallsRef = useRef<HallCall[]>([]);
  const statsRef = useRef<SimulationStats>(INITIAL_STATS);

  // --- Initialization ---
  const initSimulation = useCallback(() => {
    const newElevators: Elevator[] = Array.from({ length: NUM_ELEVATORS }, (_, i) => ({
      id: String.fromCharCode(65 + i), // 'A', 'B'
      floor: i === 0 ? 1 : Math.floor(NUM_FLOORS / 2), // Spread them out initially
      direction: Direction.IDLE,
      state: ElevatorState.IDLE,
      targets: [],
      doorTimer: 0,
      moveTimer: 0,
      totalMovedFloors: 0
    }));
    
    setElevators(newElevators);
    elevatorsRef.current = newElevators;

    setHallCalls([]);
    hallCallsRef.current = [];

    setStats(INITIAL_STATS);
    statsRef.current = INITIAL_STATS;

    setLogs([{ id: 'init', time: new Date().toLocaleTimeString(), message: 'Simulation initialized.', type: 'info' }]);
    setTickCount(0);
  }, []);

  useEffect(() => {
    initSimulation();
  }, [initSimulation]);


  // --- Logging Helper ---
  const addLog = (message: string, type: 'info' | 'success' | 'warning' = 'info') => {
    const newLog: LogEntry = {
      id: Math.random().toString(36).substr(2, 9),
      time: new Date().toLocaleTimeString(),
      message,
      type
    };
    setLogs(prev => [newLog, ...prev].slice(0, 50));
  };

  // --- Core Actions ---

  const handleCall = (floor: number, direction: Direction) => {
    // Check if call already exists
    if (hallCallsRef.current.some(c => c.floor === floor && c.direction === direction)) {
      return;
    }

    const newCall: HallCall = { floor, direction, timestamp: tickCount };
    hallCallsRef.current = [...hallCallsRef.current, newCall];
    setHallCalls(hallCallsRef.current);
    
    addLog(`Call registered: Floor ${floor} ${direction}`, 'info');

    // Trigger dispatch immediately
    dispatchElevator(newCall);
  };

  const dispatchElevator = (call: HallCall) => {
    const bestElevatorId = findBestElevator(elevatorsRef.current, call.floor, call.direction, algorithm);
    
    const elevatorIndex = elevatorsRef.current.findIndex(e => e.id === bestElevatorId);
    if (elevatorIndex === -1) return;

    const elevator = { ...elevatorsRef.current[elevatorIndex] };
    
    // Add target if not present
    if (!elevator.targets.includes(call.floor)) {
      elevator.targets = [...elevator.targets, call.floor];
      addLog(`Elevator ${elevator.id} dispatched to Floor ${call.floor} (${algorithm})`, 'info');
    }

    // Update ref and state
    const newElevators = [...elevatorsRef.current];
    newElevators[elevatorIndex] = elevator;
    elevatorsRef.current = newElevators;
    setElevators(newElevators);
  };

  const handleScenario = (type: 'RANDOM' | 'MORNING' | 'DOWN_PEAK') => {
    addLog(`Starting Scenario: ${type}`, 'warning');
    
    if (type === 'RANDOM') {
        for(let i=0; i<5; i++) {
            const floor = Math.floor(Math.random() * NUM_FLOORS) + 1;
            const dir = Math.random() > 0.5 ? Direction.UP : Direction.DOWN;
            if (floor === 1 && dir === Direction.DOWN) continue;
            if (floor === NUM_FLOORS && dir === Direction.UP) continue;
            handleCall(floor, dir);
        }
    } else if (type === 'MORNING') {
        // Many people at floor 1 going up
        for(let i=0; i<3; i++) { // Simulate 3 distinct calls (batching logic simplifies this to 1 usually, but let's assume rapid fire)
           handleCall(1, Direction.UP);
        }
        // Add random calls to floors to simulate people needing to go places
        // In this model, call buttons are unique per floor/dir. So 1 UP is 1 UP.
        // To simulate load, we just treat the single call as "active".
        // Let's create destinations. 
        // Note: The simple model here is "Hall Call -> Elevator Arrives -> Doors Open".
        // It doesn't model "Internal Car Calls" explicitly until someone enters.
        // For visual simplicity, when an elevator services a hall call, we assume passengers press buttons inside.
        // See the 'serviced' logic in the loop.
    } else if (type === 'DOWN_PEAK') {
        // High floors going down
        [15, 14, 13, 10, 8].forEach(f => handleCall(f, Direction.DOWN));
    }
  };


  // --- Simulation Loop ---
  useEffect(() => {
    if (isPaused) return;

    const interval = setInterval(() => {
      setTickCount(t => t + 1);

      let updatedElevators = [...elevatorsRef.current];
      let updatedHallCalls = [...hallCallsRef.current];
      let updatedStats = { ...statsRef.current };
      let anyUpdate = false;

      updatedElevators = updatedElevators.map(elev => {
        // Run physics/logic for this elevator
        const { updatedElevator, servicedCalls } = updateElevatorState(elev, updatedHallCalls);
        
        // If elevator moved, update distance stat
        if (updatedElevator.totalMovedFloors > elev.totalMovedFloors) {
            updatedStats.totalDistance += (updatedElevator.totalMovedFloors - elev.totalMovedFloors);
        }

        // If calls were serviced
        if (servicedCalls.length > 0) {
            servicedCalls.forEach(call => {
                // Calculate wait time
                const waitTime = tickCount - call.timestamp;
                updatedStats.totalWaitTime += waitTime;
                updatedStats.completedCalls += 1;
                addLog(`Elevator ${elev.id} serviced Floor ${call.floor} (Wait: ${waitTime} ticks)`, 'success');

                // Remove from hall calls
                updatedHallCalls = updatedHallCalls.filter(c => c !== call);
                
                // SIMULATE CAR CALLS:
                // If we picked someone up, they likely want to go somewhere.
                // If Going UP from 1, they go to random high floor.
                // If Going DOWN, they likely go to 1.
                // If Random, go random.
                let newTarget = -1;
                if (updatedElevator.direction === Direction.UP) {
                    newTarget = Math.floor(Math.random() * (NUM_FLOORS - updatedElevator.floor)) + updatedElevator.floor + 1;
                } else if (updatedElevator.direction === Direction.DOWN) {
                    newTarget = 1; // Most down traffic goes to lobby
                    if (updatedElevator.floor === 1) newTarget = -1; // Already at lobby
                } else {
                    // Was idle, picked someone up.
                    // If at 1, go up. If at top, go down.
                    if (updatedElevator.floor === 1) newTarget = Math.floor(Math.random() * (NUM_FLOORS - 1)) + 2;
                    else newTarget = 1; 
                }

                if (newTarget > 0 && newTarget <= NUM_FLOORS && newTarget !== updatedElevator.floor) {
                     if (!updatedElevator.targets.includes(newTarget)) {
                         updatedElevator.targets.push(newTarget);
                         // addLog(`Passenger in ${elev.id} pressed Floor ${newTarget}`, 'info');
                     }
                }
            });
            anyUpdate = true;
        }

        // Re-dispatch logic check?
        // In a real system, Dispatcher runs constantly.
        // Here, we dispatch on call. But if elevator becomes IDLE, it needs to check if it should help others?
        // Or if Greedy algorithm, calls might stick to a busy elevator.
        // For this demo, dispatch-on-call is usually sufficient, but let's check for "Orphaned" calls occasionally?
        // Actually, calls are assigned to a "Best Elevator" implicitly by the fact that the elevator accepted the target.
        // Wait, 'dispatchElevator' adds targets to specific elevators.
        // If an elevator is busy, and a new call comes, dispatchElevator adds it to the queue.
        
        return updatedElevator;
      });

      // Update refs
      elevatorsRef.current = updatedElevators;
      hallCallsRef.current = updatedHallCalls;
      statsRef.current = updatedStats;

      // Sync State for Render
      setElevators(updatedElevators);
      setHallCalls(updatedHallCalls);
      setStats(updatedStats);

    }, TICK_RATE_MS);

    return () => clearInterval(interval);
  }, [isPaused, tickCount, algorithm]); // Dependencies trigger restart of interval if changed, but we use refs for mutable data so it's safe.


  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col items-center gap-8 font-sans">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-slate-800">Elevator Algorithm Simulator</h1>
        <p className="text-slate-500 mt-1">Comparing Dispatch Strategies</p>
      </div>

      <div className="flex flex-col xl:flex-row gap-8 items-start justify-center w-full max-w-7xl">
        {/* Left: Building Visualization */}
        <div className="flex-grow flex justify-center">
             <Building 
                elevators={elevators} 
                hallCalls={hallCalls} 
                onCall={handleCall} 
             />
        </div>

        {/* Right: Controls & Stats */}
        <div className="flex flex-col gap-6 w-full xl:w-96">
            <ControlPanel 
                algorithm={algorithm}
                setAlgorithm={(algo) => {
                    setAlgorithm(algo);
                    addLog(`Switched algorithm to ${algo}`, 'warning');
                }}
                onReset={initSimulation}
                onScenario={handleScenario}
                isPaused={isPaused}
                setIsPaused={setIsPaused}
            />
            
            <StatsPanel stats={stats} logs={logs} />
        </div>
      </div>
    </div>
  );
};

export default App;