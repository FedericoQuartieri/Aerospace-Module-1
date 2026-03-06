#!/usr/bin/env python3
"""
Convergence Plotting Script for Navier-Stokes-Brinkman Solver

Generates convergence plots from test results stored in convergence_test/<folder_name>/

Usage:
    python plot_convergence.py <folder_name>
    
Example:
    python plot_convergence.py paper
"""

import argparse
import sys
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).parent.absolute()
CONVERGENCE_TEST_DIR = PROJECT_DIR / "convergence_test"


def load_csv(filepath):
    """Load CSV file and return list of dictionaries."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if row[key] == '':
                    row[key] = None
                else:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            results.append(row)
    return results


def plot_spatial_convergence(results, output_dir):
    """Plot spatial convergence (error vs dx)."""
    if not results:
        print("No spatial results to plot")
        return
    
    # Sort by dx (coarse to fine)
    results.sort(key=lambda r: r['dx'], reverse=True)
    
    dx = np.array([r['dx'] for r in results])
    vel_L2 = np.array([r['vel_total_L2'] for r in results])
    pres_L2 = np.array([r['pres_L2'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity plot
    ax = axes[0]
    ax.loglog(dx, vel_L2, 'bo-', linewidth=2, markersize=8, label='Velocity L2 error')
    
    # Reference slopes
    dx_ref = np.array([dx[0], dx[-1]])
    # First order reference
    ref_1 = vel_L2[0] * (dx_ref / dx[0])**1
    ax.loglog(dx_ref, ref_1, 'k--', alpha=0.5, label='1st order')
    # Second order reference
    ref_2 = vel_L2[0] * (dx_ref / dx[0])**2
    ax.loglog(dx_ref, ref_2, 'k:', alpha=0.5, label='2nd order')
    
    ax.set_xlabel('dx', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Velocity Spatial Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add convergence rate annotation
    if len(results) > 1:
        rates = [r['vel_rate'] for r in results[1:] if r['vel_rate'] is not None]
        if rates:
            avg_rate = np.mean(rates)
            ax.annotate(f'Avg rate: {avg_rate:.2f}', xy=(0.05, 0.05), 
                       xycoords='axes fraction', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Pressure plot
    ax = axes[1]
    ax.loglog(dx, pres_L2, 'rs-', linewidth=2, markersize=8, label='Pressure L2 error')
    
    # Reference slopes
    ref_1 = pres_L2[0] * (dx_ref / dx[0])**1
    ax.loglog(dx_ref, ref_1, 'k--', alpha=0.5, label='1st order')
    ref_2 = pres_L2[0] * (dx_ref / dx[0])**2
    ax.loglog(dx_ref, ref_2, 'k:', alpha=0.5, label='2nd order')
    
    ax.set_xlabel('dx', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Pressure Spatial Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add convergence rate annotation
    if len(results) > 1:
        rates = [r['pres_rate'] for r in results[1:] if r['pres_rate'] is not None]
        if rates:
            avg_rate = np.mean(rates)
            ax.annotate(f'Avg rate: {avg_rate:.2f}', xy=(0.05, 0.05), 
                       xycoords='axes fraction', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / "spatial_convergence.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def plot_temporal_convergence(results, output_dir):
    """Plot temporal convergence (error vs dt)."""
    if not results:
        print("No temporal results to plot")
        return
    
    # Sort by dt (coarse to fine)
    results.sort(key=lambda r: r['dt'], reverse=True)
    
    dt = np.array([r['dt'] for r in results])
    vel_L2 = np.array([r['vel_total_L2'] for r in results])
    pres_L2 = np.array([r['pres_L2'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity plot
    ax = axes[0]
    ax.loglog(dt, vel_L2, 'bo-', linewidth=2, markersize=8, label='Velocity L2 error')
    
    # Reference slopes
    dt_ref = np.array([dt[0], dt[-1]])
    ref_1 = vel_L2[0] * (dt_ref / dt[0])**1
    ax.loglog(dt_ref, ref_1, 'k--', alpha=0.5, label='1st order')
    ref_2 = vel_L2[0] * (dt_ref / dt[0])**2
    ax.loglog(dt_ref, ref_2, 'k:', alpha=0.5, label='2nd order')
    
    ax.set_xlabel('dt', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Velocity Temporal Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add convergence rate annotation
    if len(results) > 1:
        rates = [r['vel_rate'] for r in results[1:] if r['vel_rate'] is not None]
        if rates:
            avg_rate = np.mean(rates)
            ax.annotate(f'Avg rate: {avg_rate:.2f}', xy=(0.05, 0.05), 
                       xycoords='axes fraction', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Pressure plot
    ax = axes[1]
    ax.loglog(dt, pres_L2, 'rs-', linewidth=2, markersize=8, label='Pressure L2 error')
    
    # Reference slopes
    ref_1 = pres_L2[0] * (dt_ref / dt[0])**1
    ax.loglog(dt_ref, ref_1, 'k--', alpha=0.5, label='1st order')
    ref_2 = pres_L2[0] * (dt_ref / dt[0])**2
    ax.loglog(dt_ref, ref_2, 'k:', alpha=0.5, label='2nd order')
    
    ax.set_xlabel('dt', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Pressure Temporal Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add convergence rate annotation
    if len(results) > 1:
        rates = [r['pres_rate'] for r in results[1:] if r['pres_rate'] is not None]
        if rates:
            avg_rate = np.mean(rates)
            ax.annotate(f'Avg rate: {avg_rate:.2f}', xy=(0.05, 0.05), 
                       xycoords='axes fraction', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = output_dir / "temporal_convergence.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot convergence results")
    parser.add_argument('folder', help="Folder name under convergence_test/")
    args = parser.parse_args()
    
    # Check folder exists
    folder_path = CONVERGENCE_TEST_DIR / args.folder
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)
    
    print(f"Plotting convergence results from: {folder_path}")
    
    # Plot spatial convergence
    spatial_csv = folder_path / "convergence_spatial.csv"
    if spatial_csv.exists():
        results = load_csv(spatial_csv)
        plot_spatial_convergence(results, folder_path)
    else:
        print(f"No spatial results found ({spatial_csv})")
    
    # Plot temporal convergence
    temporal_csv = folder_path / "convergence_temporal.csv"
    if temporal_csv.exists():
        results = load_csv(temporal_csv)
        plot_temporal_convergence(results, folder_path)
    else:
        print(f"No temporal results found ({temporal_csv})")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
