#!/usr/bin/env python3
"""
benchmark_feature_pipeline.py

Benchmark script to measure feature pipeline performance improvements.
Tests optimization 1.1: Pre-load and Share SPY Data Across Processes.

Usage:
    python benchmark_feature_pipeline.py [--num-tickers N] [--feature-set SET] [--runs N]
    
Example:
    # Test with 20 tickers, 3 runs for averaging
    python benchmark_feature_pipeline.py --num-tickers 20 --runs 3
"""

import argparse
import sys
import time
import statistics
import shutil
from pathlib import Path
from typing import List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from src.feature_pipeline import main as feature_pipeline_main
from features.shared.utils import _load_spy_data


def get_sample_tickers(input_dir: Path, num_tickers: int) -> List[Path]:
    """Get a sample of ticker files for testing."""
    all_files = sorted(input_dir.glob("*.parquet"))
    if len(all_files) == 0:
        raise ValueError(f"No Parquet files found in {input_dir}")
    
    # Take first N tickers (sorted alphabetically for consistency)
    sample_files = all_files[:num_tickers]
    return sample_files


def create_temp_input_dir(input_dir: Path, sample_files: List[Path], run_id: int) -> Path:
    """Create a temporary input directory with only sample ticker files."""
    temp_input = input_dir.parent / f"clean_benchmark_run{run_id}"
    temp_input.mkdir(parents=True, exist_ok=True)
    
    # Copy sample files to temp directory
    for file_path in sample_files:
        shutil.copy2(file_path, temp_input / file_path.name)
    
    return temp_input


def create_temp_output_dir(base_output: Path, run_id: int) -> Path:
    """Create a temporary output directory for this benchmark run."""
    temp_dir = base_output.parent / f"{base_output.name}_benchmark_run{run_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_dir(temp_dir: Path):
    """Remove temporary benchmark directory."""
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def benchmark_spy_loading(num_iterations: int = 10) -> Dict[str, float]:
    """Benchmark SPY data loading time."""
    print("\n" + "="*70)
    print("BENCHMARK: SPY Data Loading")
    print("="*70)
    
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        spy_data = _load_spy_data()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f} ms")
    
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    print(f"\n  Average: {avg_time*1000:.2f} ms")
    print(f"  Min:     {min_time*1000:.2f} ms")
    print(f"  Max:     {max_time*1000:.2f} ms")
    print(f"  Std Dev: {std_time*1000:.2f} ms")
    
    return {
        'average': avg_time,
        'min': min_time,
        'max': max_time,
        'std': std_time,
        'all_times': times
    }


def benchmark_feature_pipeline(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    sample_files: List[Path],
    run_id: int
) -> Dict[str, float]:
    """Run feature pipeline benchmark for a single run."""
    # Create temporary input directory with only sample files
    temp_input = create_temp_input_dir(input_dir, sample_files, run_id)
    temp_output = create_temp_output_dir(output_dir, run_id)
    
    try:
        # Measure total time
        start_time = time.perf_counter()
        
        # Run feature pipeline with full refresh (no caching)
        feature_pipeline_main(
            input_dir=str(temp_input),
            output_dir=str(temp_output),
            config_path=str(config_path),
            full_refresh=True  # Force recomputation to avoid cache effects
        )
        
        elapsed = time.perf_counter() - start_time
        
        # Count how many files were actually processed
        output_files = list(temp_output.glob("*.parquet"))
        num_processed = len(output_files)
        
        return {
            'total_time': elapsed,
            'num_processed': num_processed,
            'time_per_ticker': elapsed / num_processed if num_processed > 0 else 0
        }
    finally:
        # Cleanup temporary directories
        cleanup_temp_dir(temp_input)
        cleanup_temp_dir(temp_output)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark feature pipeline performance improvements"
    )
    parser.add_argument(
        "--num-tickers",
        type=int,
        default=20,
        help="Number of tickers to test with (default: 20)"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="v1",
        help="Feature set to use (default: v1)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs to average (default: 3)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory (default: data/clean)"
    )
    parser.add_argument(
        "--benchmark-spy",
        action="store_true",
        help="Also benchmark SPY loading time separately"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = PROJECT_ROOT / "data" / "clean"
    
    # Determine config path based on feature set
    if args.feature_set == "v1":
        config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    elif args.feature_set == "v2":
        config_path = PROJECT_ROOT / "config" / "features_v2.yaml"
    else:
        # Try to find config file
        config_path = PROJECT_ROOT / "config" / f"features_{args.feature_set}.yaml"
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
    
    # Default output (we'll use temp dirs for actual runs)
    output_dir = PROJECT_ROOT / "data" / f"features_labeled_{args.feature_set}"
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Get sample ticker files
    try:
        sample_files = get_sample_tickers(input_dir, args.num_tickers)
        sample_ticker_names = [f.stem for f in sample_files]
        print(f"\nTesting with {len(sample_files)} tickers:")
        print(f"  Sample: {', '.join(sample_ticker_names[:5])}...")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Benchmark SPY loading if requested
    spy_stats = None
    if args.benchmark_spy:
        spy_stats = benchmark_spy_loading()
    
    # Run feature pipeline benchmarks
    print("\n" + "="*70)
    print(f"BENCHMARK: Feature Pipeline ({args.num_tickers} tickers, {args.runs} runs)")
    print("="*70)
    
    all_times = []
    all_times_per_ticker = []
    
    for run in range(1, args.runs + 1):
        print(f"\nRun {run}/{args.runs}...")
        result = benchmark_feature_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            config_path=config_path,
            sample_files=sample_files,
            run_id=run
        )
        
        all_times.append(result['total_time'])
        all_times_per_ticker.append(result['time_per_ticker'])
        
        print(f"  Total time: {result['total_time']:.2f} seconds")
        print(f"  Time per ticker: {result['time_per_ticker']:.3f} seconds")
        print(f"  Tickers processed: {result['num_processed']}")
    
    # Calculate statistics
    avg_total = statistics.mean(all_times)
    avg_per_ticker = statistics.mean(all_times_per_ticker)
    min_total = min(all_times)
    max_total = max(all_times)
    std_total = statistics.stdev(all_times) if len(all_times) > 1 else 0
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Feature set: {args.feature_set}")
    print(f"  Number of tickers: {args.num_tickers}")
    print(f"  Number of runs: {args.runs}")
    print(f"  Input directory: {input_dir}")
    
    print(f"\nTotal Time (all tickers):")
    print(f"  Average: {avg_total:.2f} seconds ({avg_total/60:.2f} minutes)")
    print(f"  Min:     {min_total:.2f} seconds")
    print(f"  Max:     {max_total:.2f} seconds")
    print(f"  Std Dev: {std_total:.2f} seconds")
    
    print(f"\nTime Per Ticker:")
    print(f"  Average: {avg_per_ticker:.3f} seconds")
    print(f"  Min:     {min(all_times_per_ticker):.3f} seconds")
    print(f"  Max:     {max(all_times_per_ticker):.3f} seconds")
    
    if spy_stats:
        print(f"\nSPY Loading Time:")
        print(f"  Average: {spy_stats['average']*1000:.2f} ms")
        print(f"  Estimated savings per worker: {spy_stats['average']*1000:.2f} ms")
        print(f"  (With {args.num_tickers} workers, total savings: {spy_stats['average']*args.num_tickers*1000:.2f} ms)")
    
    print("\n" + "="*70)
    print("NOTE: Compare these results with baseline (before optimization)")
    print("="*70)
    
    # Save results to file
    results_file = PROJECT_ROOT / "benchmark_results.txt"
    with open(results_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"Benchmark Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Feature set: {args.feature_set}\n")
        f.write(f"Number of tickers: {args.num_tickers}\n")
        f.write(f"Number of runs: {args.runs}\n")
        f.write(f"Average total time: {avg_total:.2f} seconds\n")
        f.write(f"Average time per ticker: {avg_per_ticker:.3f} seconds\n")
        if spy_stats:
            f.write(f"SPY loading time: {spy_stats['average']*1000:.2f} ms\n")
        f.write(f"\n")
    
    print(f"\nResults appended to: {results_file}")


if __name__ == "__main__":
    main()
