#!/usr/bin/env python3
"""
Convergence Study Script for Navier-Stokes-Brinkman Solver

Warning: check that the test_convergence.c has the manufactured solution and parameters that you want to use

This script automates spatial and temporal convergence tests by:
1. Compiling the code with different grid sizes / time steps
2. Running the convergence test
3. Collecting results and computing convergence rates
4. Generating CSV output and summary report

Usage:
    python run_convergence_study.py <name> [--spatial] [--temporal] [--all]
    
Options:
    <name>      Name of the output folder (required). Results will be saved in convergence_test/<name>/
    --spatial   Run spatial convergence study (varying DX with small fixed DT)
    --temporal  Run temporal convergence study (varying DT with fine fixed grid)
    --all       Run both studies (default)
"""

import subprocess
import os
import json
import math
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# ==================== Configuration ====================

# Project paths
PROJECT_DIR = Path(__file__).parent.absolute()
BUILD_DIR = PROJECT_DIR / "build_convergence"
CONVERGENCE_TEST_DIR = PROJECT_DIR / "convergence_test"

# Spatial convergence: vary grid size, keep DT small
SPATIAL_CONFIGS = [
    {"width": 16,  "dt": 0.0005, "total_time": 0.005},
    {"width": 32,  "dt": 0.0005, "total_time": 0.005},
    {"width": 64,  "dt": 0.0005, "total_time": 0.005},
    {"width": 128, "dt": 0.0005, "total_time": 0.005},
    {"width": 256, "dt": 0.0005, "total_time": 0.005},
]

# Temporal convergence: vary DT, keep grid fine
TEMPORAL_CONFIGS = [
    {"width": 128, "dt": 0.1,    "total_time": 0.5},
    {"width": 128, "dt": 0.05,   "total_time": 0.5},
    {"width": 128, "dt": 0.025,  "total_time": 0.5},
    {"width": 128, "dt": 0.0125, "total_time": 0.5},
    {"width": 128, "dt": 0.00625, "total_time": 0.5},
]

# ==================== Helper Functions ====================

def cmake_build(config, build_subdir):
    """Configure and build with given parameters."""
    build_path = BUILD_DIR / build_subdir
    build_path.mkdir(parents=True, exist_ok=True)
    
    cmake_args = [
        "cmake",
        "-S", str(PROJECT_DIR),
        "-B", str(build_path),
        f"-DGRID_WIDTH={config['width']}",
        f"-DGRID_HEIGHT={config['width']}",
        f"-DGRID_DEPTH={config['width']}",
        f"-DTIME_STEP={config['dt']}",
        f"-DSIM_TOTAL_TIME={config['total_time']}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DENABLE_EIGEN_TESTS=OFF"
    ]
    
    print(f"  Configuring: WIDTH={config['width']}, DT={config['dt']}")
    result = subprocess.run(cmake_args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CMake configure failed:\n{result.stderr}")
        return None
    
    print("  Building...")
    result = subprocess.run(["make", "-j4", "test_convergence"], cwd=build_path, 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Build failed:\n{result.stderr}")
        return None
    
    return build_path


def run_test(build_path, output_file):
    """Run the convergence test and capture output."""
    test_exe = build_path / "test_convergence"
    
    print("  Running test...")
    result = subprocess.run(
        [str(test_exe), str(output_file)],
        cwd=PROJECT_DIR / "test" / "C_test",
        capture_output=True,
        text=True
    )
    
    # Print stderr (progress info)
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            print(f"    {line}")
    
    return result.returncode == 0


def compute_convergence_rate(error_coarse, error_fine, h_coarse, h_fine):
    """Compute convergence rate: rate = log(e_c/e_f) / log(h_c/h_f)"""
    if error_fine <= 0 or error_coarse <= 0:
        return float('nan')
    if h_fine >= h_coarse:
        return float('nan')
    return math.log(error_coarse / error_fine) / math.log(h_coarse / h_fine)


def load_results(output_file):
    """Load JSON results from file."""
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def analyze_results(results, study_type):
    """Analyze results and compute convergence rates."""
    print(f"\n{'='*60}")
    print(f"  {study_type.upper()} CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    if len(results) < 2:
        print("  Not enough data points for convergence analysis")
        return results
    
    # Sort by dx (spatial) or dt (temporal)
    if study_type == "spatial":
        results.sort(key=lambda r: r['dx'], reverse=True)  # coarse to fine
        h_key = 'dx'
    else:
        results.sort(key=lambda r: r['dt'], reverse=True)  # coarse to fine
        h_key = 'dt'
    
    # Compute convergence rates
    for i in range(1, len(results)):
        h_coarse = results[i-1][h_key]
        h_fine = results[i][h_key]
        
        # Velocity L2 error (combined)
        vel_L2_coarse = math.sqrt(
            results[i-1]['vel_x_L2']**2 + 
            results[i-1]['vel_y_L2']**2 + 
            results[i-1]['vel_z_L2']**2
        )
        vel_L2_fine = math.sqrt(
            results[i]['vel_x_L2']**2 + 
            results[i]['vel_y_L2']**2 + 
            results[i]['vel_z_L2']**2
        )
        
        results[i]['vel_rate'] = compute_convergence_rate(
            vel_L2_coarse, vel_L2_fine, h_coarse, h_fine
        )
        results[i]['pres_rate'] = compute_convergence_rate(
            results[i-1]['pres_L2'], results[i]['pres_L2'], h_coarse, h_fine
        )
    
    # Print table
    print(f"\n  {'Grid':<10} {h_key.upper():<12} {'Vel L2':<12} {'Vel Rate':<10} {'Pres L2':<12} {'Pres Rate':<10}")
    print(f"  {'-'*66}")
    
    for i, r in enumerate(results):
        grid = f"{r['width']}³"
        h = r[h_key]
        vel_L2 = math.sqrt(r['vel_x_L2']**2 + r['vel_y_L2']**2 + r['vel_z_L2']**2)
        vel_rate = r.get('vel_rate', float('nan'))
        pres_L2 = r['pres_L2']
        pres_rate = r.get('pres_rate', float('nan'))
        
        rate_str_v = f"{vel_rate:.2f}" if not math.isnan(vel_rate) else "-"
        rate_str_p = f"{pres_rate:.2f}" if not math.isnan(pres_rate) else "-"
        
        print(f"  {grid:<10} {h:<12.2e} {vel_L2:<12.4e} {rate_str_v:<10} {pres_L2:<12.4e} {rate_str_p:<10}")
    
    return results


def save_csv(results, filename, study_type):
    """Save results to CSV file."""
    with open(filename, 'w') as f:
        # Header
        h_key = 'dx' if study_type == 'spatial' else 'dt'
        f.write(f"width,height,depth,{h_key},vel_x_L2,vel_y_L2,vel_z_L2,vel_total_L2,vel_rate,pres_L2,pres_rate\n")
        
        for r in results:
            vel_total = math.sqrt(r['vel_x_L2']**2 + r['vel_y_L2']**2 + r['vel_z_L2']**2)
            vel_rate = r.get('vel_rate', '')
            pres_rate = r.get('pres_rate', '')
            
            f.write(f"{r['width']},{r['height']},{r['depth']},{r[h_key]},"
                    f"{r['vel_x_L2']},{r['vel_y_L2']},{r['vel_z_L2']},{vel_total},"
                    f"{vel_rate},{r['pres_L2']},{pres_rate}\n")
    
    print(f"\n  Results saved to: {filename}")


# ==================== Main Study Functions ====================

def run_study(configs, study_name, study_type, output_dir):
    """Run a convergence study with given configurations."""
    print(f"\n{'#'*60}")
    print(f"# {study_name}")
    print(f"{'#'*60}")
    
    output_file = output_dir / f"convergence_{study_type}.jsonl"
    
    # Clear previous results
    if output_file.exists():
        output_file.unlink()
    
    # Run each configuration
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Configuration:")
        build_subdir = f"{study_type}_{config['width']}_{config['dt']}"
        
        build_path = cmake_build(config, build_subdir)
        if build_path is None:
            print("  FAILED - skipping")
            continue
        
        success = run_test(build_path, output_file)
        if not success:
            print("  TEST FAILED")
    
    # Analyze results
    if output_file.exists():
        results = load_results(output_file)
        results = analyze_results(results, study_type)
        
        # Save CSV
        csv_file = output_dir / f"convergence_{study_type}.csv"
        save_csv(results, csv_file, study_type)
        
        return results
    
    return []


def main():
    parser = argparse.ArgumentParser(description="Run convergence study")
    parser.add_argument('name', help="Name of the output folder (results saved in convergence_test/<name>/)")
    parser.add_argument('--spatial', action='store_true', help="Run spatial convergence study")
    parser.add_argument('--temporal', action='store_true', help="Run temporal convergence study")
    parser.add_argument('--all', action='store_true', help="Run all studies (default)")
    parser.add_argument('--clean', action='store_true', help="Clean build directory before running")
    args = parser.parse_args()
    
    # Default to --all if nothing specified
    if not (args.spatial or args.temporal):
        args.all = True
    
    # Create output directory
    output_dir = CONVERGENCE_TEST_DIR / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  CONVERGENCE STUDY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Project: {PROJECT_DIR}")
    print(f"  Output: {output_dir}")
    
    # Clean if requested
    if args.clean and BUILD_DIR.exists():
        print(f"  Cleaning build directory...")
        shutil.rmtree(BUILD_DIR)
    
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run studies
    if args.all or args.spatial:
        run_study(SPATIAL_CONFIGS, "SPATIAL CONVERGENCE STUDY", "spatial", output_dir)
    
    if args.all or args.temporal:
        run_study(TEMPORAL_CONFIGS, "TEMPORAL CONVERGENCE STUDY", "temporal", output_dir)
    
    print(f"\n{'='*60}")
    print("  CONVERGENCE STUDY COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
