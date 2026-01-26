#!/usr/bin/env python3
"""
compare_benchmark_results.py

Compare benchmark results from before and after optimization.
Reads from benchmark_results.txt and shows improvement percentage.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent


def parse_benchmark_results(file_path: Path) -> List[Dict]:
    """Parse benchmark results from file."""
    results = []
    current_result = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Start of new benchmark run
            if '=' * 70 in line or 'Benchmark Run:' in line:
                if current_result:
                    results.append(current_result)
                    current_result = {}
                continue
            
            # Parse key-value pairs
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'Number of tickers':
                    current_result['num_tickers'] = int(value)
                elif key == 'Number of runs':
                    current_result['num_runs'] = int(value)
                elif key == 'Average total time':
                    current_result['avg_total_time'] = float(value.split()[0])
                elif key == 'Average time per ticker':
                    current_result['avg_time_per_ticker'] = float(value.split()[0])
                elif key == 'SPY loading time':
                    current_result['spy_loading_time'] = float(value.split()[0])
                elif key == 'Feature set':
                    current_result['feature_set'] = value
    
    # Add last result
    if current_result:
        results.append(current_result)
    
    return results


def find_baseline_and_optimized(results: List[Dict]) -> tuple:
    """Find baseline (oldest) and optimized (newest) results with same config."""
    if len(results) < 2:
        return None, None
    
    # Group by configuration
    configs = {}
    for i, result in enumerate(results):
        config_key = (
            result.get('num_tickers', 0),
            result.get('feature_set', 'unknown')
        )
        if config_key not in configs:
            configs[config_key] = []
        configs[config_key].append((i, result))
    
    # Find config with at least 2 results
    for config_key, config_results in configs.items():
        if len(config_results) >= 2:
            # Sort by index (order in file)
            config_results.sort(key=lambda x: x[0])
            baseline = config_results[0][1]  # Oldest
            optimized = config_results[-1][1]  # Newest
            return baseline, optimized
    
    return None, None


def main():
    results_file = PROJECT_ROOT / "outputs" / "benchmarks" / "benchmark_results.txt"
    
    if not results_file.exists():
        print(f"Error: Benchmark results file not found: {results_file}")
        print("\nRun benchmark_feature_pipeline.py first to generate results.")
        return
    
    results = parse_benchmark_results(results_file)
    
    if len(results) == 0:
        print("No benchmark results found in file.")
        return
    
    print("=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    
    if len(results) == 1:
        print(f"\nOnly one benchmark result found:")
        result = results[0]
        print(f"  Feature set: {result.get('feature_set', 'unknown')}")
        print(f"  Tickers: {result.get('num_tickers', 'unknown')}")
        print(f"  Average total time: {result.get('avg_total_time', 0):.2f} seconds")
        print(f"  Average time per ticker: {result.get('avg_time_per_ticker', 0):.3f} seconds")
        print("\nRun another benchmark to compare!")
        return
    
    baseline, optimized = find_baseline_and_optimized(results)
    
    if not baseline or not optimized:
        print("\nCould not find matching baseline and optimized results.")
        print(f"Found {len(results)} benchmark runs:")
        for i, result in enumerate(results):
            print(f"  Run {i+1}: {result.get('num_tickers', '?')} tickers, "
                  f"{result.get('avg_total_time', 0):.2f}s")
        return
    
    # Calculate improvements
    total_time_improvement = ((baseline['avg_total_time'] - optimized['avg_total_time']) 
                              / baseline['avg_total_time'] * 100)
    per_ticker_improvement = ((baseline['avg_time_per_ticker'] - optimized['avg_time_per_ticker'])
                              / baseline['avg_time_per_ticker'] * 100)
    
    print(f"\nBaseline (Before Optimization):")
    print(f"  Feature set: {baseline.get('feature_set', 'unknown')}")
    print(f"  Tickers: {baseline.get('num_tickers', 'unknown')}")
    print(f"  Average total time: {baseline['avg_total_time']:.2f} seconds")
    print(f"  Average time per ticker: {baseline['avg_time_per_ticker']:.3f} seconds")
    
    print(f"\nOptimized (After Optimization):")
    print(f"  Feature set: {optimized.get('feature_set', 'unknown')}")
    print(f"  Tickers: {optimized.get('num_tickers', 'unknown')}")
    print(f"  Average total time: {optimized['avg_total_time']:.2f} seconds")
    print(f"  Average time per ticker: {optimized['avg_time_per_ticker']:.3f} seconds")
    
    print(f"\nImprovement:")
    print(f"  Total time: {total_time_improvement:+.1f}% "
          f"({baseline['avg_total_time'] - optimized['avg_total_time']:.2f}s faster)")
    print(f"  Per ticker: {per_ticker_improvement:+.1f}% "
          f"({baseline['avg_time_per_ticker'] - optimized['avg_time_per_ticker']:.3f}s faster)")
    
    if 'spy_loading_time' in optimized:
        spy_savings = optimized.get('spy_loading_time', 0) * optimized.get('num_tickers', 1) / 1000
        print(f"\nEstimated SPY loading savings:")
        print(f"  SPY load time: {optimized['spy_loading_time']:.2f} ms")
        print(f"  Estimated total savings: {spy_savings:.2f} seconds "
              f"(if each worker loaded SPY independently)")


if __name__ == "__main__":
    main()
