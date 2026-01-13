#!/usr/bin/env python3
"""
Basic PCCS simulation example.

This script demonstrates:
1. Creating a simulation with custom parameters
2. Running the simulation
3. Measuring emergent phenomena
4. Visualizing the results
"""

import mlx.core as mx

from pccs import Config
from pccs.simulation import Simulation
from pccs.metrics import compute_all_metrics, print_metrics_summary
from pccs.phase import kuramoto_order_parameter
from pccs.bonds import count_bonds


def main():
    print("=" * 60)
    print("PCCS - Phase-Coupled Catalytic Substrate")
    print("Basic Simulation Example")
    print("=" * 60)
    print()
    
    # Check MLX device
    print(f"MLX device: {mx.default_device()}")
    print()
    
    # Create configuration
    config = Config(
        grid_size=64,          # Smaller grid for quick demo
        D_base=0.1,            # Diffusion rate
        alpha=0.9,             # Membrane impermeability
        K_phase=0.5,           # Phase coupling strength
        kappa=2.0,             # Phase gate sharpness
        injection_mode="boundary",  # Resource injection at edges
        injection_rate=0.01,
    )
    
    print("Configuration:")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Diffusion rate: {config.D_base}")
    print(f"  Phase coupling: {config.K_phase}")
    print()
    
    # Create simulation
    sim = Simulation(config, seed=42)
    
    # Initial measurements
    print("Initial state:")
    R_initial = kuramoto_order_parameter(sim.state.phase)
    bonds_initial = count_bonds(sim.state)
    print(f"  Phase synchronization (R): {R_initial:.4f}")
    print(f"  Bond count: {bonds_initial}")
    print()
    
    # Run simulation
    print("Running simulation for 1000 steps...")
    
    def progress_callback(s: Simulation):
        R = kuramoto_order_parameter(s.state.phase)
        bonds = count_bonds(s.state)
        print(f"  Step {s.step_count}: R={R:.4f}, bonds={bonds}")
    
    sim.run(
        steps=1000,
        callback=progress_callback,
        callback_interval=200,
        show_progress=True,
    )
    print()
    
    # Final measurements
    print("Final state:")
    mx.eval(sim.state.A)  # Ensure all computations complete
    
    metrics = compute_all_metrics(sim.state)
    print_metrics_summary(metrics)
    
    # Check for emergent structures
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    R_final = metrics['phase']['global_sync']
    if R_final > R_initial * 1.5:
        print("✓ Phase synchronization increased!")
    else:
        print("✗ Limited phase synchronization")
    
    bonds_final = metrics['bonds']['total_bonds']
    if bonds_final > 0:
        print(f"✓ Bonds formed: {bonds_final}")
    else:
        print("✗ No bonds formed")
    
    membranes = metrics['membranes']['num_closed_membranes']
    if membranes > 0:
        print(f"✓ Closed membranes detected: {membranes}")
    else:
        print("✗ No closed membranes yet (try running longer)")
    
    print()
    print("To visualize, run:")
    print("  python -m pccs.main --visualize --steps 5000")
    print()


if __name__ == "__main__":
    main()
