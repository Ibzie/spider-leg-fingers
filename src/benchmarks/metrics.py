"""
Benchmark metrics for Spider Leg Finger experiment
Computational (C1-C6) and Manipulation (M1-M6) metrics
"""

import time
import numpy as np
import torch
from typing import Dict, List
import sys
sys.path.insert(0, '/mnt/stuff-first/Projects/SpiderLegFingers/spider_leg_fingers/src')

from models.spider_leg import SpiderLegFinger
from physics.mujoco_env import create_env


def benchmark_ik_solve(model_type: str = 'slfm', n_solves: int = 10000) -> Dict:
    """C1: IK solve time (microseconds per solve)"""
    
    if model_type == 'slfm':
        finger = SpiderLegFinger()
        solve_times = []
        
        for _ in range(n_solves):
            target = np.random.randn(3) * 0.02 + np.array([0.05, 0, 0])
            start = time.perf_counter()
            finger.solve_ik(target)
            elapsed = (time.perf_counter() - start) * 1e6  # microseconds
            solve_times.append(elapsed)
    else:
        # HFM uses numerical IK (slower)
        from scipy.optimize import minimize
        finger = SpiderLegFinger()  # Use as proxy
        solve_times = []
        
        for _ in range(n_solves):
            target = np.random.randn(3) * 0.02 + np.array([0.05, 0, 0])
            start = time.perf_counter()
            # Numerical IK simulation
            result = minimize(
                lambda x: np.linalg.norm(finger.kinematics.forward_kinematics(x) - target),
                x0=np.zeros(3),
                method='L-BFGS-B'
            )
            elapsed = (time.perf_counter() - start) * 1e6
            solve_times.append(elapsed)
    
    return {
        'mean_us': np.mean(solve_times),
        'std_us': np.std(solve_times),
        'min_us': np.min(solve_times),
        'max_us': np.max(solve_times),
    }


def benchmark_simulation_rate(model_type: str = 'slfm', duration: float = 5.0) -> Dict:
    """C2: Maximum stable simulation tick rate (Hz)"""
    
    env = create_env(model_type, 'power_grasp')
    
    obs = env.reset()
    steps = 0
    start = time.time()
    
    while time.time() - start < duration:
        action = np.random.randn(env.action_dim) * 0.1
        obs, _, done, _ = env.step(action)
        steps += 1
        if done:
            obs = env.reset()
    
    elapsed = time.time() - start
    rate = steps / elapsed
    
    return {
        'sim_hz': rate,
        'total_steps': steps,
        'elapsed_time': elapsed,
    }


def benchmark_memory_footprint(model_type: str = 'slfm') -> Dict:
    """C4: Memory footprint (MB)"""
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create environment
    env = create_env(model_type, 'power_grasp')
    after_env = process.memory_info().rss / 1024 / 1024
    
    return {
        'baseline_mb': baseline,
        'env_mb': after_env,
        'delta_mb': after_env - baseline,
    }


def run_all_benchmarks():
    """Run complete benchmark suite"""
    
    results = {}
    
    for model in ['slfm', 'hfm']:
        print(f"\nBenchmarking {model.upper()}...")
        results[model] = {}
        
        # C1: IK solve time
        print("  C1: IK Solve Time...")
        results[model]['ik_solve'] = benchmark_ik_solve(model)
        print(f"    Mean: {results[model]['ik_solve']['mean_us']:.2f} µs")
        
        # C2: Simulation rate
        print("  C2: Simulation Rate...")
        results[model]['sim_rate'] = benchmark_simulation_rate(model, duration=2.0)
        print(f"    Rate: {results[model]['sim_rate']['sim_hz']:.1f} Hz")
        
        # C4: Memory
        print("  C4: Memory Footprint...")
        results[model]['memory'] = benchmark_memory_footprint(model)
        print(f"    Memory: {results[model]['memory']['env_mb']:.1f} MB")
    
    # Compare
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    
    ik_speedup = results['hfm']['ik_solve']['mean_us'] / results['slfm']['ik_solve']['mean_us']
    rate_speedup = results['slfm']['sim_rate']['sim_hz'] / results['hfm']['sim_rate']['sim_hz']
    memory_reduction = results['hfm']['memory']['env_mb'] / results['slfm']['memory']['env_mb']
    
    print(f"IK Solve Speedup: {ik_speedup:.1f}x")
    print(f"Simulation Rate Speedup: {rate_speedup:.1f}x")
    print(f"Memory Efficiency: {memory_reduction:.2f}x smaller")
    
    return results


if __name__ == '__main__':
    results = run_all_benchmarks()
