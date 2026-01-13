"""
Metrics and analysis utilities for PCCS.

Provides quantitative measures of emergent phenomena.
"""

from typing import Optional

import mlx.core as mx
import numpy as np

from .state import CellState
from .phase import kuramoto_order_parameter, local_order_parameter
from .bonds import count_bonds, find_bond_clusters, detect_closed_membranes


def total_mass(state: CellState) -> float:
    """
    Compute total mass (sum of all concentrations).
    
    Args:
        state: Cell state
    
    Returns:
        Total mass across grid
    """
    return float(mx.sum(state.A) + mx.sum(state.B) + mx.sum(state.C))


def mass_by_species(state: CellState) -> dict[str, float]:
    """
    Compute mass for each species.
    
    Args:
        state: Cell state
    
    Returns:
        Dictionary with A, B, C masses
    """
    return {
        "A": float(mx.sum(state.A)),
        "B": float(mx.sum(state.B)),
        "C": float(mx.sum(state.C)),
    }


def mean_concentrations(state: CellState) -> dict[str, float]:
    """
    Compute mean concentration for each species.
    
    Args:
        state: Cell state
    
    Returns:
        Dictionary with mean A, B, C concentrations
    """
    return {
        "A": float(mx.mean(state.A)),
        "B": float(mx.mean(state.B)),
        "C": float(mx.mean(state.C)),
    }


def concentration_variance(state: CellState) -> dict[str, float]:
    """
    Compute variance of each species concentration.
    
    Higher variance indicates spatial structure.
    
    Args:
        state: Cell state
    
    Returns:
        Dictionary with A, B, C variances
    """
    return {
        "A": float(mx.var(state.A)),
        "B": float(mx.var(state.B)),
        "C": float(mx.var(state.C)),
    }


def phase_statistics(state: CellState) -> dict[str, float]:
    """
    Compute phase-related statistics.
    
    Args:
        state: Cell state
    
    Returns:
        Dictionary with phase statistics
    """
    R_global = kuramoto_order_parameter(state.phase)
    R_local = local_order_parameter(state.phase)
    
    return {
        "global_sync": R_global,
        "mean_local_sync": float(mx.mean(R_local)),
        "max_local_sync": float(mx.max(R_local)),
        "phase_mean": float(mx.mean(state.phase)),
        "phase_std": float(mx.std(state.phase)),
    }


def bond_statistics(state: CellState) -> dict[str, float]:
    """
    Compute bond-related statistics.
    
    Args:
        state: Cell state
    
    Returns:
        Dictionary with bond statistics
    """
    total_bonds = count_bonds(state)
    H, W = state.shape
    max_bonds = 2 * H * W  # Each cell can have 4 bonds, but counted twice
    
    # Bond density
    bond_density = total_bonds / max_bonds
    
    # Bonds per cell
    bonds_per_cell = mx.sum(state.bonds, axis=-1)
    
    return {
        "total_bonds": total_bonds,
        "bond_density": bond_density,
        "mean_bonds_per_cell": float(mx.mean(bonds_per_cell)),
        "max_bonds_per_cell": float(mx.max(bonds_per_cell)),
        "cells_with_bonds": int(mx.sum(bonds_per_cell > 0)),
    }


def membrane_statistics(state: CellState) -> dict:
    """
    Compute membrane-related statistics.
    
    Args:
        state: Cell state
    
    Returns:
        Dictionary with membrane statistics
    """
    membranes = detect_closed_membranes(state.bonds)
    
    if not membranes:
        return {
            "num_closed_membranes": 0,
            "total_membrane_cells": 0,
            "mean_membrane_size": 0,
            "max_membrane_size": 0,
        }
    
    sizes = [len(m) for m in membranes]
    
    return {
        "num_closed_membranes": len(membranes),
        "total_membrane_cells": sum(sizes),
        "mean_membrane_size": np.mean(sizes),
        "max_membrane_size": max(sizes),
        "membrane_sizes": sizes,
    }


def compartment_analysis(state: CellState) -> dict:
    """
    Analyze concentration differences between compartments.
    
    Compares interior vs exterior of detected membranes.
    
    Args:
        state: Cell state
    
    Returns:
        Dictionary with compartment statistics
    """
    membranes = detect_closed_membranes(state.bonds)
    
    if not membranes:
        return {"num_compartments": 0}
    
    results = []
    
    for i, membrane in enumerate(membranes):
        # For simplicity, compute mean concentrations at membrane cells
        # Full interior detection would require flood fill
        
        membrane_A = []
        membrane_B = []
        membrane_C = []
        
        A_np = np.array(state.A)
        B_np = np.array(state.B)
        C_np = np.array(state.C)
        
        for y, x in membrane:
            membrane_A.append(A_np[y, x])
            membrane_B.append(B_np[y, x])
            membrane_C.append(C_np[y, x])
        
        results.append({
            "id": i,
            "size": len(membrane),
            "mean_A": np.mean(membrane_A),
            "mean_B": np.mean(membrane_B),
            "mean_C": np.mean(membrane_C),
        })
    
    return {
        "num_compartments": len(results),
        "compartments": results,
    }


def compute_all_metrics(state: CellState) -> dict:
    """
    Compute all available metrics.
    
    Args:
        state: Cell state
    
    Returns:
        Comprehensive dictionary of all metrics
    """
    return {
        "mass": {
            "total": total_mass(state),
            "by_species": mass_by_species(state),
        },
        "concentrations": {
            "mean": mean_concentrations(state),
            "variance": concentration_variance(state),
        },
        "phase": phase_statistics(state),
        "bonds": bond_statistics(state),
        "membranes": membrane_statistics(state),
    }


def print_metrics_summary(metrics: dict) -> None:
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Output from compute_all_metrics
    """
    print("\n=== PCCS Metrics Summary ===\n")
    
    print("Mass:")
    print(f"  Total: {metrics['mass']['total']:.4f}")
    for species, mass in metrics['mass']['by_species'].items():
        print(f"  {species}: {mass:.4f}")
    
    print("\nConcentrations:")
    for species, mean in metrics['concentrations']['mean'].items():
        var = metrics['concentrations']['variance'][species]
        print(f"  {species}: mean={mean:.4f}, var={var:.6f}")
    
    print("\nPhase Synchronization:")
    phase = metrics['phase']
    print(f"  Global sync (R): {phase['global_sync']:.4f}")
    print(f"  Mean local sync: {phase['mean_local_sync']:.4f}")
    
    print("\nBonds:")
    bonds = metrics['bonds']
    print(f"  Total bonds: {bonds['total_bonds']}")
    print(f"  Bond density: {bonds['bond_density']:.4f}")
    print(f"  Cells with bonds: {bonds['cells_with_bonds']}")
    
    print("\nMembranes:")
    membranes = metrics['membranes']
    print(f"  Closed membranes: {membranes['num_closed_membranes']}")
    if membranes['num_closed_membranes'] > 0:
        print(f"  Mean size: {membranes['mean_membrane_size']:.1f}")
        print(f"  Max size: {membranes['max_membrane_size']}")
    
    print()
