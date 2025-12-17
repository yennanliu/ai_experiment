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
  const elevatorsRef = useRef<Elevator[]>([]);
  const hallCallsRef = useRef<HallCall[]>([]);
  const statsRef = useRef<SimulationStats>(INITIAL_STATS);

  // --- Initialization ---
  const initSimulation = useCallback(() => {
    const newElevators: Elevator[] = Array.from({ length: NUM_ELEVATORS }, (_, i) => ({
      id: String.fromCharCode(65 + i), // 'A', 'B'
      floor: i === 0 ? 1 : Math.floor(NUM_FLOORS / 2),
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

    setLogs([{ id: 'init', time: new Date().toLocaleTimeString(), message: '系統已初始化', type: 'info' }]);
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
    if (hallCallsRef.current.some(c => c.floor === floor && c.direction === direction)) {
      return;
    }

    const newCall: HallCall = { floor, direction, timestamp: tickCount };
    hallCallsRef.current = [...hallCallsRef.current, newCall];
    setHallCalls(hallCallsRef.current);
    
    const dirStr = direction === Direction.UP ? '上樓' : '下樓';
    addLog(`收到呼叫: ${floor}樓 ${dirStr}`, 'info');

    dispatchElevator(newCall);
  };

  const dispatchElevator = (call: HallCall) => {
    const bestElevatorId = findBestElevator(elevatorsRef.current, call.floor, call.direction, algorithm);
    
    const elevatorIndex = elevatorsRef.current.findIndex(e => e.id === bestElevatorId);
    if (elevatorIndex === -1) return;

    const elevator = { ...elevatorsRef.current[elevatorIndex] };
    
    if (!elevator.targets.includes(call.floor)) {
      elevator.targets = [...elevator.targets, call.floor];
      const algoName = algorithm === AlgorithmType.GREEDY ? '貪婪' : '智慧';
      addLog(`指派電梯 ${elevator.id} 至 ${call.floor}樓 (${algoName})`, 'warning');
    }

    const newElevators = [...elevatorsRef.current];
    newElevators[elevatorIndex] = elevator;
    elevatorsRef.current = newElevators;
    setElevators(newElevators);
  };

  const handleScenario = (type: 'RANDOM' | 'MORNING' | 'DOWN_PEAK' | 'OPPOSITE') => {
    addLog(`--- 啟動情境: ${type === 'OPPOSITE' ? '反向陷阱測試' : type} ---`, 'warning');
    
    if (type === 'RANDOM') {
        for(let i=0; i<5; i++) {
            const floor = Math.floor(Math.random() * NUM_FLOORS) + 1;
            const dir = Math.random() > 0.5 ? Direction.UP : Direction.DOWN;
            if (floor === 1 && dir === Direction.DOWN) continue;
            if (floor === NUM_FLOORS && dir === Direction.UP) continue;
            handleCall(floor, dir);
        }
    } else if (type === 'MORNING') {
        // Burst at lobby
        for(let i=0; i<3; i++) handleCall(1, Direction.UP);
    } else if (type === 'DOWN_PEAK') {
        [15, 14, 13, 10, 8].forEach(f => handleCall(f, Direction.DOWN));
    } else if (type === 'OPPOSITE') {
        // Reset elevators for specific setup
        // Elevator A: Floor 1, IDLE
        // Elevator B: Floor 8, Moving DOWN to 1
        
        // We need to manually force state to set up the trap visual
        const setupElevators = [...elevatorsRef.current];
        
        // Setup A
        setupElevators[0].floor = 1;
        setupElevators[0].direction = Direction.IDLE;
        setupElevators[0].state = ElevatorState.IDLE;
        setupElevators[0].targets = [];

        // Setup B
        setupElevators[1].floor = 8;
        setupElevators[1].direction = Direction.DOWN;
        setupElevators[1].state = ElevatorState.MOVING;
        setupElevators[1].targets = [1]; // B is going to lobby
        setupElevators[1].moveTimer = 0; // Ready to move

        elevatorsRef.current = setupElevators;
        setElevators(setupElevators);

        // Trap: Call at 9F DOWN
        // Greedy: |8-9|=1 (B) vs |1-9|=8 (A) -> Picks B. 
        // But B is going down to 1 first! Wait = (8->1) + (1->9) + (9->dest). Huge wait.
        // Smart: A is idle at 1. Cost for B is huge (wrong direction). Picks A.
        
        setTimeout(() => {
             handleCall(9, Direction.DOWN);
        }, 500);
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
        const { updatedElevator, servicedCalls } = updateElevatorState(elev, updatedHallCalls);
        
        if (updatedElevator.totalMovedFloors > elev.totalMovedFloors) {
            updatedStats.totalDistance += (updatedElevator.totalMovedFloors - elev.totalMovedFloors);
        }

        if (servicedCalls.length > 0) {
            servicedCalls.forEach(call => {
                const waitTime = tickCount - call.timestamp;
                updatedStats.totalWaitTime += waitTime;
                updatedStats.completedCalls += 1;
                addLog(`電梯 ${elev.id} 抵達 ${call.floor}樓 (等待: ${waitTime} ticks)`, 'success');

                updatedHallCalls = updatedHallCalls.filter(c => c !== call);
                
                // Simulate Passenger Destination
                let newTarget = -1;
                if (updatedElevator.direction === Direction.UP) {
                    newTarget = Math.floor(Math.random() * (NUM_FLOORS - updatedElevator.floor)) + updatedElevator.floor + 1;
                } else if (updatedElevator.direction === Direction.DOWN) {
                    newTarget = 1; 
                    if (updatedElevator.floor === 1) newTarget = -1;
                } else {
                    if (updatedElevator.floor === 1) newTarget = Math.floor(Math.random() * (NUM_FLOORS - 1)) + 2;
                    else newTarget = 1; 
                }

                if (newTarget > 0 && newTarget <= NUM_FLOORS && newTarget !== updatedElevator.floor) {
                     if (!updatedElevator.targets.includes(newTarget)) {
                         updatedElevator.targets.push(newTarget);
                     }
                }
            });
            anyUpdate = true;
        }
        
        return updatedElevator;
      });

      elevatorsRef.current = updatedElevators;
      hallCallsRef.current = updatedHallCalls;
      statsRef.current = updatedStats;

      setElevators(updatedElevators);
      setHallCalls(updatedHallCalls);
      setStats(updatedStats);

    }, TICK_RATE_MS);

    return () => clearInterval(interval);
  }, [isPaused, tickCount, algorithm]); 


  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col items-center gap-8 font-sans bg-slate-50 text-slate-800">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-slate-900 tracking-tight">雙電梯演算法模擬器</h1>
        <p className="text-slate-500 mt-2 font-medium">比較「貪婪演算法」與「智慧調度 (Smart ETA)」的效率差異</p>
      </div>

      <div className="flex flex-col xl:flex-row gap-8 items-start justify-center w-full max-w-7xl">
        {/* Left: Building Visualization */}
        <div className="flex-grow flex justify-center w-full xl:w-auto overflow-x-auto pb-4">
             <Building 
                elevators={elevators} 
                hallCalls={hallCalls} 
                onCall={handleCall} 
             />
        </div>

        {/* Right: Controls & Stats */}
        <div className="flex flex-col gap-6 w-full xl:w-96 flex-shrink-0">
            <ControlPanel 
                algorithm={algorithm}
                setAlgorithm={(algo) => {
                    setAlgorithm(algo);
                    addLog(`切換策略為: ${algo === AlgorithmType.GREEDY ? '貪婪 (Greedy)' : '智慧 (Smart ETA)'}`, 'warning');
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