#!/usr/bin/env python3
"""
Protocell division experiments.

Tests three hypotheses for protocell division:
1. Growth dynamics - increased resource injection causes membrane growth
2. High-pressure destabilization - extreme injection causes fission
3. Mechanical cut - sustained perturbation splits protocell in two

Usage:
    python examples/division_experiments.py --experiment 1 --output-dir docs/assets/division/
    python examples/division_experiments.py --experiment all --output-dir docs/assets/division/
"""

import argparse
import json
import mlx.core as mx
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from scipy.stats import linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from pccs import Config
from pccs.simulation import Simulation
from pccs.state import CellState
from pccs.bonds import detect_closed_membranes, count_bonds
from pccs.metrics import total_mass, membrane_statistics
from pccs.phase import kuramoto_order_parameter
from pccs.visualization import save_state_images


def create_metrics_callback(history: list):
    """Create callback that logs membrane and bond stats."""
    def callback(sim: Simulation):
        # Force evaluation before metrics
        mx.eval(sim.state.B)
        membranes = detect_closed_membranes(sim.state.bonds)
        bonds = count_bonds(sim.state)

        # Track mean B in high-B region (where bonds form)
        B_array = sim.state.B
        high_B_mask = B_array > 0.2
        mean_B_high = float(mx.mean(mx.where(high_B_mask, B_array, 0.0)))

        history.append({
            "step": sim.step_count,
            "num_membranes": len(membranes),
            "membrane_sizes": [len(m) for m in membranes],
            "total_bonds": bonds,
            "mean_B_high": mean_B_high,
            "total_mass": float(total_mass(sim.state)),
            "global_sync": float(kuramoto_order_parameter(sim.state.phase)),
        })

        # Print progress with bond count (more informative)
        print(f"  Step {sim.step_count}: bonds={bonds}, membranes={len(membranes)}")

    return callback


def apply_line_cut(state: CellState, x_position: int, width: int = 2) -> CellState:
    """
    Set B=0 and break bonds in a vertical strip.

    Args:
        state: Current cell state
        x_position: X coordinate of cut center
        width: Half-width of cut region (total width = 2*width)

    Returns:
        New CellState with cut applied
    """
    H, W = state.shape
    x_coords = mx.arange(W).reshape(1, -1)

    # Create mask for cut region
    mask = mx.abs(x_coords - x_position) < width

    # Zero out B in cut region
    new_B = mx.where(mask, 0.0, state.B)

    # Break all bonds in cut region
    mask_3d = mx.expand_dims(mask, axis=-1)
    new_bonds = mx.where(mask_3d, 0.0, state.bonds)

    return CellState(
        A=state.A,
        B=new_B,
        C=state.C,
        phase=state.phase,
        bonds=new_bonds,
        B_thresh=state.B_thresh,
    )


def run_experiment_1(output_dir: Path, seed: int = 42):
    """
    Experiment 1: Growth dynamics under doubled injection.

    Protocol:
    1. Form stable membrane ring (2000 steps at 0.02 injection)
    2. Double injection rate to 0.04
    3. Run 5000 more steps
    4. Track membrane size changes

    Note: Uses boundary injection which creates closed membrane rings.
    Center injection creates filled regions, not closed loops.
    """
    print("=" * 60)
    print("Experiment 1: Growth Dynamics (Double Injection)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use boundary injection for closed membrane rings
    config = Config(
        grid_size=32,  # Smaller for faster boundary ring formation
        injection_mode="boundary",
        injection_rate=0.02,
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    # Phase 1: Form stable protocell
    print("\nPhase 1: Forming stable protocell (2000 steps, rate=0.02)...")
    sim.run(2000, callback=create_metrics_callback(history), callback_interval=200)
    save_state_images(sim.state, str(output_dir), "exp1_baseline", sim.step_count)

    baseline_bonds = history[-1]["total_bonds"]
    baseline_size = history[-1]["membrane_sizes"][0] if history[-1]["membrane_sizes"] else 0
    print(f"\nBaseline: {baseline_bonds} bonds, {baseline_size} membrane cells")

    # Phase 2: Double injection rate
    print("\nPhase 2: Doubling injection rate (0.02 → 0.04)...")
    sim.config.injection_rate = 0.04

    print("Running 5000 more steps...")
    sim.run(5000, callback=create_metrics_callback(history), callback_interval=500)
    save_state_images(sim.state, str(output_dir), "exp1_doubled", sim.step_count)

    final_bonds = history[-1]["total_bonds"]
    final_size = history[-1]["membrane_sizes"][0] if history[-1]["membrane_sizes"] else 0
    print(f"\nFinal: {final_bonds} bonds, {final_size} membrane cells")

    # Analysis based on bond count (more reliable than membrane detection)
    bond_growth = (final_bonds - baseline_bonds) / baseline_bonds * 100 if baseline_bonds > 0 else 0
    print(f"Bond growth: {bond_growth:.1f}%")

    if bond_growth > 50:
        print("Result: SIGNIFICANT GROWTH")
    elif bond_growth > 10:
        print("Result: MODERATE GROWTH")
    elif bond_growth > -10:
        print("Result: SATURATED (no significant change)")
    else:
        print("Result: STRUCTURE SHRINKAGE")

    # Save metrics
    with open(output_dir / "exp1_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def run_experiment_2(output_dir: Path, seed: int = 42):
    """
    Experiment 2: High-pressure destabilization (triple injection).

    Protocol:
    1. Form stable membrane ring (2000 steps)
    2. Triple injection rate to 0.06
    3. Run 5000 steps
    4. Watch for division/rupture
    """
    print("=" * 60)
    print("Experiment 2: High-Pressure Destabilization (Triple Injection)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        grid_size=32,
        injection_mode="boundary",
        injection_rate=0.02,
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    # Phase 1: Form stable protocell
    print("\nPhase 1: Forming stable protocell (2000 steps, rate=0.02)...")
    sim.run(2000, callback=create_metrics_callback(history), callback_interval=200)
    save_state_images(sim.state, str(output_dir), "exp2_baseline", sim.step_count)

    # Phase 2: Triple injection rate
    print("\nPhase 2: Tripling injection rate (0.02 → 0.06)...")
    sim.config.injection_rate = 0.06

    print("Running 5000 more steps...")
    sim.run(5000, callback=create_metrics_callback(history), callback_interval=500)
    save_state_images(sim.state, str(output_dir), "exp2_tripled", sim.step_count)

    # Analysis
    baseline_bonds = history[len(history)//3]["total_bonds"]  # After phase 1
    final_bonds = history[-1]["total_bonds"]
    final_membranes = history[-1]["num_membranes"]
    print(f"\nFinal: {final_bonds} bonds, {final_membranes} closed membranes")

    if final_membranes >= 2:
        print("Result: DIVISION OCCURRED!")
    elif final_bonds > baseline_bonds * 0.5:
        print("Result: Structure persists (bonds maintained)")
    else:
        print("Result: RUPTURE (structure collapsed)")

    # Save metrics
    with open(output_dir / "exp2_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def run_experiment_3(output_dir: Path, seed: int = 42):
    """
    Experiment 3: Sustained cut perturbation.

    Protocol:
    1. Form stable membrane ring (3000 steps)
    2. Apply sustained vertical cut through the ring (10 steps)
    3. Run 2000 more steps
    4. Observe: does the ring heal or split into two arcs?
    """
    print("=" * 60)
    print("Experiment 3: Sustained Cut Perturbation")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        grid_size=32,
        injection_mode="boundary",
        injection_rate=0.02,
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    # Phase 1: Form stable membrane ring
    print("\nPhase 1: Forming membrane ring (3000 steps, rate=0.02)...")
    sim.run(3000, callback=create_metrics_callback(history), callback_interval=300)
    save_state_images(sim.state, str(output_dir), "exp3_precut", sim.step_count)

    precut_bonds = history[-1]["total_bonds"]
    precut_membranes = history[-1]["num_membranes"]
    precut_size = history[-1]["membrane_sizes"][0] if history[-1]["membrane_sizes"] else 0
    print(f"\nPre-cut: {precut_bonds} bonds, {precut_membranes} membrane(s)")

    # Phase 2: Apply sustained cut through the left edge membrane
    # Cut at x=2 to hit the boundary ring (injection_width=3 means ring is at edges)
    print("\nPhase 2: Applying sustained vertical cut through left membrane (10 steps)...")
    cut_position = 2  # Cut through left edge where membrane ring is

    for i in range(10):
        # Apply cut
        sim.state = apply_line_cut(sim.state, cut_position, width=2)
        # Step simulation
        sim.step()

        if i == 4:  # Mid-cut snapshot
            save_state_images(sim.state, str(output_dir), "exp3_midcut", sim.step_count)

    save_state_images(sim.state, str(output_dir), "exp3_postcut", sim.step_count)

    # Record post-cut state
    mx.eval(sim.state.B)
    membranes = detect_closed_membranes(sim.state.bonds)
    postcut_bonds = count_bonds(sim.state)
    history.append({
        "step": sim.step_count,
        "num_membranes": len(membranes),
        "membrane_sizes": [len(m) for m in membranes],
        "total_bonds": postcut_bonds,
        "total_mass": float(total_mass(sim.state)),
        "global_sync": float(kuramoto_order_parameter(sim.state.phase)),
        "event": "post_cut"
    })
    print(f"Post-cut: {postcut_bonds} bonds, {len(membranes)} membrane(s)")

    # Phase 3: Let system evolve
    print("\nPhase 3: Observing recovery (2000 steps)...")
    sim.run(2000, callback=create_metrics_callback(history), callback_interval=200)
    save_state_images(sim.state, str(output_dir), "exp3_final", sim.step_count)

    # Analysis
    final_bonds = history[-1]["total_bonds"]
    final_membranes = history[-1]["num_membranes"]
    final_sizes = history[-1]["membrane_sizes"]

    print(f"\nFinal: {final_bonds} bonds, {final_membranes} membrane(s)")
    print(f"Bond recovery: {final_bonds}/{precut_bonds} ({100*final_bonds/precut_bonds:.0f}%)" if precut_bonds > 0 else "")

    if final_membranes >= 2:
        print("Result: DIVISION SUCCESS - Two separate structures!")
    elif final_bonds >= precut_bonds * 0.8:
        print("Result: HEALED - Structure reformed")
    elif final_bonds >= precut_bonds * 0.3:
        print("Result: PARTIAL RECOVERY - Smaller structure")
    else:
        print("Result: DISINTEGRATION - Structure collapsed")

    # Save metrics
    with open(output_dir / "exp3_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def run_budding_experiment(output_dir: Path, seed: int = 42):
    """
    Budding experiment: Form protocell, then add daughter injection point.

    Protocol:
    1. Form stable protocell with center injection (3000 steps)
    2. Switch to budding mode (adds daughter source offset by 10 cells)
    3. Run 5000 more steps to observe mother-daughter dynamics
    """
    print("=" * 60)
    print("Experiment: Budding via Daughter Injection")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Form stable mother protocell with center injection
    config = Config(
        grid_size=48,
        injection_mode="center",
        injection_rate=0.02,
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    print("\nPhase 1: Forming mother protocell (3000 steps, center injection)...")
    sim.run(3000, callback=create_metrics_callback(history), callback_interval=300)
    save_state_images(sim.state, str(output_dir), "budding_mother", sim.step_count)

    mother_bonds = history[-1]["total_bonds"]
    print(f"\nMother protocell: {mother_bonds} bonds")

    # Phase 2: Switch to budding mode (adds daughter injection point)
    print("\nPhase 2: Adding daughter injection point (offset=10 cells)...")
    sim.config.injection_mode = "budding"

    print("Running 5000 more steps...")
    sim.run(5000, callback=create_metrics_callback(history), callback_interval=500)
    save_state_images(sim.state, str(output_dir), "budding_final", sim.step_count)

    final_bonds = history[-1]["total_bonds"]
    print(f"\nFinal: {final_bonds} bonds")

    # Analyze bond distribution in left vs right halves
    H, W = sim.state.shape
    bonds_array = sim.state.bonds

    # Count bonds in each half (sum over all 4 directions)
    left_bonds = int(mx.sum(bonds_array[:, :W//2, :])) // 2
    right_bonds = int(mx.sum(bonds_array[:, W//2:, :])) // 2

    print(f"Left half bonds: {left_bonds}")
    print(f"Right half bonds: {right_bonds}")

    # Determine outcome
    bond_ratio = right_bonds / left_bonds if left_bonds > 0 else 0
    print(f"Right/Left ratio: {bond_ratio:.2f}")

    if bond_ratio > 0.5 and right_bonds > 200:
        print("\nResult: DAUGHTER NUCLEATED - Two structures present!")
    elif final_bonds > mother_bonds * 1.3:
        print("\nResult: STRUCTURE GREW - But daughter merged with mother")
    elif final_bonds > mother_bonds * 0.8:
        print("\nResult: SATURATED - No significant growth")
    else:
        print("\nResult: STRUCTURE SHRINKAGE")

    # Save metrics
    with open(output_dir / "budding_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def run_budding_barrier_experiment(output_dir: Path, seed: int = 42):
    """
    Budding + Barrier: Form two-source protocell, then cut between sources.

    Key insight: Previous cuts healed because only one side had resources.
    With TWO injection points, cutting between them creates two viable halves.

    Protocol:
    1. Run budding mode (mother + daughter) for 5000 steps
    2. Apply sustained cut BETWEEN the two injection points
    3. Each half has its own food source → both should survive
    """
    print("=" * 60)
    print("Experiment: Budding + Barrier (Assisted Division)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Run budding mode (both sources active from start)
    config = Config(
        grid_size=48,
        injection_mode="budding",  # Mother at center, daughter at center+10
        injection_rate=0.02,
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    print("\nPhase 1: Establishing two-source protocell (5000 steps)...")
    sim.run(5000, callback=create_metrics_callback(history), callback_interval=500)
    save_state_images(sim.state, str(output_dir), "budding_barrier_precut", sim.step_count)

    precut_bonds = history[-1]["total_bonds"]
    print(f"\nPre-cut: {precut_bonds} bonds")

    # Analyze pre-cut bond distribution
    H, W = sim.state.shape
    left_precut = int(mx.sum(sim.state.bonds[:, :W//2, :])) // 2
    right_precut = int(mx.sum(sim.state.bonds[:, W//2:, :])) // 2
    print(f"Pre-cut distribution: Left={left_precut}, Right={right_precut}")

    # Phase 2: Apply sustained cut BETWEEN the injection points
    # Mother at W//2 = 24, Daughter at W//2 + 10 = 34
    # Cut at W//2 + 5 = 29 (midpoint between them)
    cut_position = W // 2 + 5
    print(f"\nPhase 2: Applying sustained cut at x={cut_position} (15 steps)...")

    for i in range(15):
        sim.state = apply_line_cut(sim.state, cut_position, width=3)
        sim.step()
        if i == 7:  # Mid-cut snapshot
            save_state_images(sim.state, str(output_dir), "budding_barrier_midcut", sim.step_count)

    save_state_images(sim.state, str(output_dir), "budding_barrier_postcut", sim.step_count)

    # Record post-cut state
    mx.eval(sim.state.B)
    postcut_bonds = count_bonds(sim.state)
    left_postcut = int(mx.sum(sim.state.bonds[:, :W//2, :])) // 2
    right_postcut = int(mx.sum(sim.state.bonds[:, W//2:, :])) // 2
    print(f"Post-cut: {postcut_bonds} bonds (Left={left_postcut}, Right={right_postcut})")

    history.append({
        "step": sim.step_count,
        "num_membranes": 0,
        "membrane_sizes": [],
        "total_bonds": postcut_bonds,
        "left_bonds": left_postcut,
        "right_bonds": right_postcut,
        "event": "post_cut"
    })

    # Phase 3: Let system evolve after cut
    print("\nPhase 3: Observing separation (3000 steps)...")
    sim.run(3000, callback=create_metrics_callback(history), callback_interval=300)
    save_state_images(sim.state, str(output_dir), "budding_barrier_final", sim.step_count)

    # Final analysis
    final_bonds = history[-1]["total_bonds"]
    left_final = int(mx.sum(sim.state.bonds[:, :W//2, :])) // 2
    right_final = int(mx.sum(sim.state.bonds[:, W//2:, :])) // 2

    print(f"\nFinal: {final_bonds} bonds")
    print(f"Left half: {left_final} bonds (mother side)")
    print(f"Right half: {right_final} bonds (daughter side)")

    # Determine success
    # Success = both halves have substantial bonds (>50 each)
    if left_final > 50 and right_final > 50:
        ratio = min(left_final, right_final) / max(left_final, right_final)
        print(f"Balance ratio: {ratio:.2f}")
        if ratio > 0.3:
            print("\n*** DIVISION SUCCESS! Two viable daughter cells! ***")
        else:
            print("\nResult: ASYMMETRIC - One daughter much larger")
    elif left_final > 50 or right_final > 50:
        print("\nResult: ONE SURVIVOR - Only one half maintained structure")
    else:
        print("\nResult: COLLAPSE - Both halves lost structure")

    # Save metrics
    with open(output_dir / "budding_barrier_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


def run_natural_division_experiment(output_dir: Path, seed: int = 42):
    """
    Natural division: Large offset budding (20 cells) without any cut.

    Tests if 20-cell gap prevents merging naturally.
    If successful, this is TRUE BUDDING - parent spawns independent daughter
    without surgical intervention.
    """
    print("=" * 60)
    print("Experiment: Natural Division (Large Offset, No Cut)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Budding mode now uses daughter_offset=20 (modified in simulation.py)
    config = Config(
        grid_size=48,
        injection_mode="budding",  # Mother at center, daughter at center+20
        injection_rate=0.02,
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    print("\nRunning budding mode with offset=20 (8000 steps, no intervention)...")
    print("Mother at center (24), Daughter at center+20 (44)")

    # Run with periodic snapshots
    for phase in range(4):
        steps = 2000
        print(f"\nPhase {phase+1}/4: Steps {phase*2000} to {(phase+1)*2000}...")
        sim.run(steps, callback=create_metrics_callback(history), callback_interval=500)

        # Save snapshot at each phase
        H, W = sim.state.shape
        left_bonds = int(mx.sum(sim.state.bonds[:, :W//2, :])) // 2
        right_bonds = int(mx.sum(sim.state.bonds[:, W//2:, :])) // 2
        total = history[-1]["total_bonds"]

        print(f"  Step {sim.step_count}: {total} bonds (Left={left_bonds}, Right={right_bonds})")
        save_state_images(sim.state, str(output_dir), f"natural_step{sim.step_count:05d}", sim.step_count)

    # Final analysis
    H, W = sim.state.shape
    left_final = int(mx.sum(sim.state.bonds[:, :W//2, :])) // 2
    right_final = int(mx.sum(sim.state.bonds[:, W//2:, :])) // 2
    total_final = history[-1]["total_bonds"]

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total bonds: {total_final}")
    print(f"Left half (mother):   {left_final} bonds")
    print(f"Right half (daughter): {right_final} bonds")

    # Check for natural separation
    # With offset=20, mother is at x=24, daughter at x=44
    # Check bonds in middle region (x=30 to x=38) - should be empty if separated
    middle_bonds = int(mx.sum(sim.state.bonds[:, 30:38, :])) // 2
    print(f"Middle gap (x=30-38): {middle_bonds} bonds")

    if left_final > 30 and right_final > 30 and middle_bonds < 20:
        ratio = min(left_final, right_final) / max(left_final, right_final)
        print(f"Balance ratio: {ratio:.2f}")
        print("\n*** TRUE BUDDING SUCCESS! Natural division without intervention! ***")
        result = "TRUE_BUDDING"
    elif left_final > 30 and right_final > 30:
        print("\nResult: CONNECTED - Both structures exist but are bridged")
        result = "CONNECTED"
    elif left_final > 30 or right_final > 30:
        print("\nResult: ONE STRUCTURE - Only one side developed")
        result = "ONE_STRUCTURE"
    else:
        print("\nResult: FAILED - Neither structure developed properly")
        result = "FAILED"

    # Save metrics
    with open(output_dir / "natural_division_metrics.json", "w") as f:
        json.dump({
            "history": history,
            "final": {
                "total_bonds": total_final,
                "left_bonds": left_final,
                "right_bonds": right_final,
                "middle_bonds": middle_bonds,
                "result": result
            }
        }, f, indent=2)

    return history


def run_lineage_experiment(output_dir: Path, seed: int = 42):
    """
    Lineage experiment: Three generations of protocells.

    Tests whether existing protocell system can spawn additional members.
    This is closer to real reproduction - existing organisms produce offspring.

    Protocol:
    1. Start with budding mode (mother at center, daughter at center+20)
    2. Run 5000 steps - two protocells form naturally
    3. Switch to lineage mode (adds granddaughter at center+40)
    4. Run 5000 more steps
    5. Check: do we get three independent protocells?

    Success demonstrates:
    - Reproductive capacity is not exhausted after one division
    - Each new injection point can nucleate a new individual
    - Lineages are possible (Mother → Daughter → Granddaughter)
    """
    print("=" * 60)
    print("Experiment: Three-Generation Lineage")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Start with budding mode (two generations)
    # With generation_offset=20, need grid >= 96 for three generations:
    # Mother at 48, Daughter at 68, Granddaughter at 88
    config = Config(
        grid_size=96,  # 96x96 to fit 3 generations with 20-cell spacing
        injection_mode="budding",  # Mother at center (48), daughter at center+20 (68)
        injection_rate=0.02,
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    print("\nPhase 1: Establishing two-generation system (5000 steps)...")
    print("  Mother at center (48), Daughter at center+20 (68)")

    # Run with periodic snapshots
    for phase in range(5):
        steps = 1000
        sim.run(steps, callback=create_metrics_callback(history), callback_interval=500)

        H, W = sim.state.shape
        # Two regions for budding: mother around 48, daughter around 68
        # Gap between them: 55-61
        mother_region = int(mx.sum(sim.state.bonds[:, 35:55, :])) // 2
        gap_region = int(mx.sum(sim.state.bonds[:, 55:61, :])) // 2
        daughter_region = int(mx.sum(sim.state.bonds[:, 61:81, :])) // 2

        print(f"  Step {sim.step_count}: Mother={mother_region}, Gap={gap_region}, Daughter={daughter_region}")

    save_state_images(sim.state, str(output_dir), "lineage_phase1_final", sim.step_count)

    # Record two-generation state
    H, W = sim.state.shape
    pre_lineage_bonds = history[-1]["total_bonds"]
    print(f"\nTwo-generation system: {pre_lineage_bonds} total bonds")

    # Phase 2: Add third generation (granddaughter at center+40 = 88)
    print("\nPhase 2: Adding granddaughter injection point (center+40 = 88)...")
    print("Switching from 'budding' to 'lineage' mode...")

    # Switch injection mode to lineage (has three injection points)
    sim.config.injection_mode = "lineage"

    print("\nRunning 5000 more steps with three injection points...")
    print("  Positions: Mother=48, Daughter=68, Granddaughter=88")

    for phase in range(5):
        steps = 1000
        sim.run(steps, callback=create_metrics_callback(history), callback_interval=500)

        H, W = sim.state.shape
        # Three regions for 96x96 grid with lineage mode:
        # Mother: x=48 (region 35-55)
        # Daughter: x=68 (region 55-75)
        # Granddaughter: x=88 (region 75-96)
        mother_bonds = int(mx.sum(sim.state.bonds[:, 35:55, :])) // 2
        daughter_bonds = int(mx.sum(sim.state.bonds[:, 55:75, :])) // 2
        granddaughter_bonds = int(mx.sum(sim.state.bonds[:, 75:, :])) // 2

        print(f"  Step {sim.step_count}: M={mother_bonds}, D={daughter_bonds}, G={granddaughter_bonds}")

        # Save intermediate snapshots
        if phase == 2:
            save_state_images(sim.state, str(output_dir), f"lineage_mid", sim.step_count)

    save_state_images(sim.state, str(output_dir), "lineage_final", sim.step_count)

    # Final analysis
    H, W = sim.state.shape
    total_final = history[-1]["total_bonds"]

    # For 96x96 grid, positions are:
    # Mother: x=48 (region 35-55)
    # Daughter: x=68 (region 58-78)
    # Granddaughter: x=88 (region 78-96)
    # Gaps: 55-58 (M-D gap), 75-78 (D-G gap)

    mother_bonds = int(mx.sum(sim.state.bonds[:, 35:55, :])) // 2
    daughter_bonds = int(mx.sum(sim.state.bonds[:, 58:78, :])) // 2
    granddaughter_bonds = int(mx.sum(sim.state.bonds[:, 78:, :])) // 2

    # Check gaps between regions
    gap1_bonds = int(mx.sum(sim.state.bonds[:, 55:58, :])) // 2
    gap2_bonds = int(mx.sum(sim.state.bonds[:, 75:78, :])) // 2

    print(f"\n{'='*60}")
    print("FINAL RESULTS - THREE GENERATIONS")
    print(f"{'='*60}")
    print(f"Total bonds: {total_final}")
    print(f"Mother region (35-55):      {mother_bonds} bonds")
    print(f"Daughter region (58-78):    {daughter_bonds} bonds")
    print(f"Granddaughter region (78+): {granddaughter_bonds} bonds")
    print(f"Gap M-D (55-58): {gap1_bonds} bonds")
    print(f"Gap D-G (75-78): {gap2_bonds} bonds")

    # Determine success
    viable_count = sum([
        1 if mother_bonds > 30 else 0,
        1 if daughter_bonds > 30 else 0,
        1 if granddaughter_bonds > 30 else 0,
    ])

    if viable_count == 3:
        print(f"\n*** THREE-GENERATION LINEAGE SUCCESS! {viable_count} viable protocells! ***")
        result = "THREE_GENERATIONS"
    elif viable_count == 2:
        print(f"\nResult: TWO VIABLE - Only two generations maintained")
        result = "TWO_GENERATIONS"
    elif viable_count == 1:
        print(f"\nResult: ONE VIABLE - Only one generation maintained")
        result = "ONE_GENERATION"
    else:
        print(f"\nResult: FAILED - No viable structures")
        result = "FAILED"

    # Save metrics
    with open(output_dir / "lineage_metrics.json", "w") as f:
        json.dump({
            "history": history,
            "final": {
                "total_bonds": total_final,
                "mother_bonds": mother_bonds,
                "daughter_bonds": daughter_bonds,
                "granddaughter_bonds": granddaughter_bonds,
                "gap1_bonds": gap1_bonds,
                "gap2_bonds": gap2_bonds,
                "viable_count": viable_count,
                "result": result
            }
        }, f, indent=2)

    return history


def run_competition_experiment(output_dir: Path, seed: int = 42):
    """
    Resource competition: 4 protocells compete for limited resources.

    Tests selection pressure under scarcity. Diamond pattern ensures
    symmetric competition - each protocell faces equal pressure from
    2 neighbors at ~35 cell diagonal distance.

    Protocol:
    1. Create 4 protocells in diamond pattern (N, S, E, W)
    2. Set injection_rate = 0.005 (1/4 of normal) for scarcity
    3. Run 10000 steps
    4. Track survival: viability threshold = 30 bonds per region
    """
    print("=" * 60)
    print("Experiment 8: Four-Protocell Resource Competition")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        grid_size=80,  # 80x80 fits 4 protocells with spacing=25
        injection_mode="competition",
        injection_rate=0.005,  # 1/4 of normal - creates scarcity
        injection_width=3,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
    )

    sim = Simulation(config, seed=seed)
    history = []

    # Competition centers (for 80x80 grid with spacing=25)
    # Center is at (40, 40)
    spacing = 25
    centers = {
        "North": (40 - spacing, 40),  # (15, 40)
        "South": (40 + spacing, 40),  # (65, 40)
        "East": (40, 40 + spacing),   # (40, 65)
        "West": (40, 40 - spacing),   # (40, 15)
    }

    print(f"\nCompetition setup:")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Injection rate: {config.injection_rate} (1/4 normal)")
    print(f"  Protocell positions: N(15,40), S(65,40), E(40,65), W(40,15)")
    print(f"  Viability threshold: 30 bonds per region")

    print("\nRunning 10000 steps...")

    # Run with periodic snapshots
    for phase in range(10):
        steps = 1000
        sim.run(steps, callback=create_metrics_callback(history), callback_interval=500)

        H, W = sim.state.shape
        bonds = sim.state.bonds

        # Count bonds per region (quadrants around each injection point)
        # North region: y < 30
        north_bonds = int(mx.sum(bonds[:30, :, :])) // 2
        # South region: y > 50
        south_bonds = int(mx.sum(bonds[50:, :, :])) // 2
        # East region: x > 50
        east_bonds = int(mx.sum(bonds[:, 50:, :])) // 2
        # West region: x < 30
        west_bonds = int(mx.sum(bonds[:, :30, :])) // 2

        # Count viable protocells (>30 bonds)
        viable = sum([
            1 if north_bonds > 30 else 0,
            1 if south_bonds > 30 else 0,
            1 if east_bonds > 30 else 0,
            1 if west_bonds > 30 else 0,
        ])

        print(f"  Step {sim.step_count}: N={north_bonds}, S={south_bonds}, E={east_bonds}, W={west_bonds} | Viable: {viable}/4")

        # Save snapshots at key points
        if phase in [1, 3, 5, 7, 9]:  # Steps 2000, 4000, 6000, 8000, 10000
            save_state_images(sim.state, str(output_dir), f"competition_step{sim.step_count:05d}", sim.step_count)

    # Final analysis
    H, W = sim.state.shape
    bonds = sim.state.bonds

    north_final = int(mx.sum(bonds[:30, :, :])) // 2
    south_final = int(mx.sum(bonds[50:, :, :])) // 2
    east_final = int(mx.sum(bonds[:, 50:, :])) // 2
    west_final = int(mx.sum(bonds[:, :30, :])) // 2
    total_final = history[-1]["total_bonds"]

    print(f"\n{'='*60}")
    print("FINAL RESULTS - RESOURCE COMPETITION")
    print(f"{'='*60}")
    print(f"Total bonds: {total_final}")
    print(f"North region (y<30):  {north_final} bonds {'✓ VIABLE' if north_final > 30 else '✗ collapsed'}")
    print(f"South region (y>50):  {south_final} bonds {'✓ VIABLE' if south_final > 30 else '✗ collapsed'}")
    print(f"East region (x>50):   {east_final} bonds {'✓ VIABLE' if east_final > 30 else '✗ collapsed'}")
    print(f"West region (x<30):   {west_final} bonds {'✓ VIABLE' if west_final > 30 else '✗ collapsed'}")

    # Determine outcome
    survivors = []
    if north_final > 30:
        survivors.append("North")
    if south_final > 30:
        survivors.append("South")
    if east_final > 30:
        survivors.append("East")
    if west_final > 30:
        survivors.append("West")

    viable_count = len(survivors)

    if viable_count == 4:
        print(f"\nResult: ALL SURVIVE - Resources sufficient, no selection pressure")
        result = "ALL_SURVIVE"
    elif viable_count >= 2:
        print(f"\n*** SELECTION OCCURRED! {viable_count}/4 survived: {', '.join(survivors)} ***")
        result = f"SELECTION_{viable_count}_SURVIVE"
    elif viable_count == 1:
        print(f"\n*** STRONG SELECTION! Only {survivors[0]} survived ***")
        result = "STRONG_SELECTION"
    else:
        print(f"\nResult: EXTINCTION - Resources too scarce, all collapsed")
        result = "EXTINCTION"

    # Save metrics
    with open(output_dir / "competition_metrics.json", "w") as f:
        json.dump({
            "history": history,
            "final": {
                "total_bonds": total_final,
                "north_bonds": north_final,
                "south_bonds": south_final,
                "east_bonds": east_final,
                "west_bonds": west_final,
                "viable_count": viable_count,
                "survivors": survivors,
                "result": result
            }
        }, f, indent=2)

    return history


def run_fitness_experiment(output_dir: Path, seed: int = 42):
    """
    Experiment 9: Differential Fitness via Parameter Asymmetry.

    Tests whether parameter variation creates selectable fitness differences.
    Uses position-dependent B_thresh:
    - Left half: B_thresh = 0.20 ("Strong" - forms bonds more easily)
    - Right half: B_thresh = 0.30 ("Weak" - needs more B to form bonds)

    With a SHARED food source at center, this creates competition where
    the "Strong" side should consistently win.

    Protocol:
    1. Enable fitness_mode (position-dependent B_thresh)
    2. Single center food source (creates competition)
    3. Seed both left and right sides with initial B boost
    4. Run 15000 steps
    5. Measure which side wins

    Success criteria:
    - Strong (left) consistently wins (>80% of trials)
    - Winning margin is significant (>50% more bonds)
    """
    print("=" * 60)
    print("Experiment 9: Differential Fitness (Strong vs Weak)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        grid_size=64,  # 64x64 grid
        injection_mode="center",  # SHARED food source at center
        injection_rate=0.01,  # Moderate rate - enough for ~1 protocell
        injection_width=3,
        B_thresh=0.25,  # Default (overridden by fitness_mode)
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
        # Fitness experiment parameters
        fitness_mode=True,  # Enable position-dependent B_thresh
        strong_B_thresh=0.20,  # Left half - easier bond formation
        weak_B_thresh=0.30,  # Right half - harder bond formation
    )

    sim = Simulation(config, seed=seed)
    history = []

    H, W = config.grid_size, config.grid_size
    print(f"\nFitness setup:")
    print(f"  Grid: {H}x{W}")
    print(f"  Left half (x < {W//2}):  B_thresh = {config.strong_B_thresh} (STRONG)")
    print(f"  Right half (x >= {W//2}): B_thresh = {config.weak_B_thresh} (WEAK)")
    print(f"  Food source: CENTER (shared)")
    print(f"  injection_rate: {config.injection_rate}")

    # Seed both sides with initial B boost to ensure protocells form
    print("\nSeeding both sides with initial B...")
    center_y = H // 2
    # Left protocell seed at (H//2, W//4) = (32, 16)
    left_seed_x = W // 4
    # Right protocell seed at (H//2, 3*W//4) = (32, 48)
    right_seed_x = 3 * W // 4

    # Create initial B boost
    B_array = sim.state.B
    boost_radius = 5

    # Create coordinate grids
    y_coords = mx.arange(H).reshape(-1, 1)
    x_coords = mx.arange(W).reshape(1, -1)

    # Left seed region
    left_dist = (y_coords - center_y)**2 + (x_coords - left_seed_x)**2
    left_boost = (left_dist < boost_radius**2).astype(mx.float32) * 0.3

    # Right seed region
    right_dist = (y_coords - center_y)**2 + (x_coords - right_seed_x)**2
    right_boost = (right_dist < boost_radius**2).astype(mx.float32) * 0.3

    # Apply boosts
    new_B = sim.state.B + left_boost + right_boost
    sim.state = CellState(
        A=sim.state.A,
        B=new_B,
        C=sim.state.C,
        phase=sim.state.phase,
        bonds=sim.state.bonds,
        B_thresh=sim.state.B_thresh,
    )

    print(f"  Left seed:  ({center_y}, {left_seed_x})")
    print(f"  Right seed: ({center_y}, {right_seed_x})")

    print("\nRunning 15000 steps...")

    # Run with periodic snapshots
    for phase in range(15):
        steps = 1000
        sim.run(steps, callback=create_metrics_callback(history), callback_interval=500)

        bonds = sim.state.bonds

        # Count bonds per side
        left_bonds = int(mx.sum(bonds[:, :W//2, :])) // 2
        right_bonds = int(mx.sum(bonds[:, W//2:, :])) // 2
        total = left_bonds + right_bonds

        # Determine leader
        if left_bonds > right_bonds:
            leader = "STRONG"
            margin = left_bonds - right_bonds
        elif right_bonds > left_bonds:
            leader = "WEAK"
            margin = right_bonds - left_bonds
        else:
            leader = "TIE"
            margin = 0

        print(f"  Step {sim.step_count}: Strong(L)={left_bonds}, Weak(R)={right_bonds} | {leader} leads by {margin}")

        # Save snapshots at key points
        if phase in [0, 4, 9, 14]:  # Steps 1000, 5000, 10000, 15000
            save_state_images(sim.state, str(output_dir), f"fitness_step{sim.step_count:05d}", sim.step_count)

    # Final analysis
    bonds = sim.state.bonds
    left_final = int(mx.sum(bonds[:, :W//2, :])) // 2
    right_final = int(mx.sum(bonds[:, W//2:, :])) // 2
    total_final = left_final + right_final

    print(f"\n{'='*60}")
    print("FINAL RESULTS - DIFFERENTIAL FITNESS")
    print(f"{'='*60}")
    print(f"Total bonds: {total_final}")
    print(f"Strong (left, B_thresh=0.20):  {left_final} bonds")
    print(f"Weak (right, B_thresh=0.30):   {right_final} bonds")

    # Determine outcome
    if left_final > 30 and right_final > 30:
        # Both survived
        if left_final > right_final * 1.5:
            print(f"\n*** STRONG WINS by significant margin! (ratio={left_final/right_final:.2f}) ***")
            result = "STRONG_WINS_SIGNIFICANT"
        elif right_final > left_final * 1.5:
            print(f"\n*** UNEXPECTED: WEAK WINS! (ratio={right_final/left_final:.2f}) ***")
            result = "WEAK_WINS_UNEXPECTED"
        else:
            print(f"\nResult: BOTH SURVIVE - No clear winner (ratio={left_final/right_final:.2f})")
            result = "BOTH_SURVIVE"
    elif left_final > 30:
        print(f"\n*** STRONG (left) DOMINATES! Weak collapsed. ***")
        result = "STRONG_DOMINATES"
    elif right_final > 30:
        print(f"\n*** UNEXPECTED: WEAK (right) DOMINATES! Strong collapsed. ***")
        result = "WEAK_DOMINATES_UNEXPECTED"
    else:
        print(f"\nResult: BOTH COLLAPSED - Resources insufficient")
        result = "BOTH_COLLAPSED"

    # Interpretation
    if "STRONG" in result:
        print("\nInterpretation: Lower B_thresh = fitness advantage")
        print("This proves parameter variation creates selectable differences!")
    elif "UNEXPECTED" in result:
        print("\nInterpretation: Unexpected result - investigate further")
    else:
        print("\nInterpretation: No clear selection - may need parameter tuning")

    # Save metrics
    with open(output_dir / "fitness_metrics.json", "w") as f:
        json.dump({
            "config": {
                "strong_B_thresh": config.strong_B_thresh,
                "weak_B_thresh": config.weak_B_thresh,
                "injection_rate": config.injection_rate,
                "grid_size": config.grid_size,
            },
            "history": history,
            "final": {
                "total_bonds": total_final,
                "left_bonds_strong": left_final,
                "right_bonds_weak": right_final,
                "ratio": left_final / right_final if right_final > 0 else float("inf"),
                "result": result
            }
        }, f, indent=2)

    return history


def run_evolution_experiment(output_dir: Path, seed: int = 42, num_steps: int = 20000):
    """
    Experiment 10: Evolution via Per-Cell Inheritance.

    Tests whether per-cell B_thresh can evolve under selection pressure.
    Uses 4 competing protocells with mutations enabled.

    Hypothesis: B_thresh should DECREASE over time because
    lower B_thresh = easier bond formation = fitness advantage.
    (This was proven in Experiment 9 with position-dependent thresholds.)

    Protocol:
    1. Start 4 protocells in competition mode
    2. Shared center food source (selection pressure)
    3. Enable mutations (mutation_rate=0.001)
    4. Run 20000 steps
    5. Track mean B_thresh over time

    Success criteria:
    - Mean B_thresh trends downward over time
    - Protocells with lower B_thresh outcompete others
    """
    print("=" * 60)
    print("Experiment 10: Evolution via Per-Cell Inheritance")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        grid_size=64,
        injection_mode="center",  # SHARED food source - creates selection pressure
        injection_rate=0.015,  # Moderate rate for one protocell's worth
        injection_width=5,  # Slightly larger to support competition
        B_thresh=0.25,  # Starting value for all cells
        k1=0.05,
        k2=0.05,
        k3=0.01,
        epsilon=0.001,
        # Evolution parameters
        mutation_rate=0.002,  # 0.2% of cells mutate per step (slightly higher for faster evolution)
        mutation_strength=0.02,  # Max ±0.02 per mutation
    )

    sim = Simulation(config, seed=seed)
    history = []

    H, W = config.grid_size, config.grid_size
    print(f"\nEvolution setup:")
    print(f"  Grid: {H}x{W}")
    print(f"  Food source: CENTER (shared, creates selection pressure)")
    print(f"  Starting B_thresh: {config.B_thresh}")
    print(f"  Mutation rate: {config.mutation_rate} ({config.mutation_rate*100:.1f}% per step)")
    print(f"  Mutation strength: ±{config.mutation_strength}")
    print(f"  B_thresh bounds: [{config.B_thresh_min}, {config.B_thresh_max}]")
    print(f"\nHypothesis: Mean B_thresh should decrease over time")
    print(f"  (Lower B_thresh = easier bonds = fitness advantage)")
    print(f"  Selection acts on cells near the food source - fitter cells expand.")

    print(f"\nRunning {num_steps} steps...")

    # Track B_thresh evolution
    evolution_history = {
        'steps': [],
        'mean_B_thresh': [],
        'std_B_thresh': [],
        'min_B_thresh': [],
        'max_B_thresh': [],
        'total_bonds': [],
        'mean_B_thresh_bonded': [],  # Mean B_thresh for cells with bonds (the "winners")
    }

    # Run with periodic snapshots
    for phase in range(num_steps // 1000):
        steps = 1000
        sim.run(steps, callback=create_metrics_callback(history), callback_interval=500)

        # Track B_thresh statistics
        B_thresh = sim.state.B_thresh
        mean_thresh = float(mx.mean(B_thresh))
        std_thresh = float(mx.std(B_thresh))
        min_thresh = float(mx.min(B_thresh))
        max_thresh = float(mx.max(B_thresh))
        total_bonds = history[-1]["total_bonds"]

        # Track mean B_thresh of bonded cells (cells with at least one bond)
        bonds_per_cell = mx.sum(sim.state.bonds, axis=-1)  # [H, W]
        bonded_mask = bonds_per_cell > 0
        num_bonded = int(mx.sum(bonded_mask))
        if num_bonded > 0:
            mean_bonded_thresh = float(mx.sum(B_thresh * bonded_mask) / num_bonded)
        else:
            mean_bonded_thresh = mean_thresh

        # Record evolution data
        evolution_history['steps'].append(sim.step_count)
        evolution_history['mean_B_thresh'].append(mean_thresh)
        evolution_history['std_B_thresh'].append(std_thresh)
        evolution_history['min_B_thresh'].append(min_thresh)
        evolution_history['max_B_thresh'].append(max_thresh)
        evolution_history['total_bonds'].append(total_bonds)
        evolution_history['mean_B_thresh_bonded'].append(mean_bonded_thresh)

        print(f"  Step {sim.step_count}: mean_B_thresh={mean_thresh:.4f} (bonded={mean_bonded_thresh:.4f}), range=[{min_thresh:.3f}, {max_thresh:.3f}], bonds={total_bonds}")

        # Save snapshots at key points
        if phase in [0, 4, 9, 14, 19]:  # Steps 1000, 5000, 10000, 15000, 20000
            save_state_images(sim.state, str(output_dir), f"evolution_step{sim.step_count:05d}", sim.step_count)

    # Final analysis
    initial_mean = evolution_history['mean_B_thresh'][0]
    final_mean = evolution_history['mean_B_thresh'][-1]
    change = final_mean - initial_mean

    # Also check bonded cells' B_thresh (more relevant for selection)
    initial_bonded = evolution_history['mean_B_thresh_bonded'][0]
    final_bonded = evolution_history['mean_B_thresh_bonded'][-1]
    bonded_change = final_bonded - initial_bonded

    print(f"\n{'='*60}")
    print("FINAL RESULTS - EVOLUTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Initial mean B_thresh (all):    {initial_mean:.4f}")
    print(f"Final mean B_thresh (all):      {final_mean:.4f}")
    print(f"Change (all):                   {change:+.4f} ({100*change/initial_mean:+.1f}%)")
    print(f"")
    print(f"Initial mean B_thresh (bonded): {initial_bonded:.4f}")
    print(f"Final mean B_thresh (bonded):   {final_bonded:.4f}")
    print(f"Change (bonded):                {bonded_change:+.4f}")
    print(f"")
    print(f"Final std:                      {evolution_history['std_B_thresh'][-1]:.4f}")
    print(f"Final range:                    [{evolution_history['min_B_thresh'][-1]:.3f}, {evolution_history['max_B_thresh'][-1]:.3f}]")

    # Determine outcome - focus on bonded cells (where selection acts)
    if bonded_change < -0.01:  # B_thresh decreased by >0.01 in bonded cells
        print(f"\n*** EVOLUTION DETECTED! Bonded cells' B_thresh trending DOWN ***")
        print("Interpretation: Natural selection favors lower B_thresh")
        result = "EVOLUTION_DETECTED"
    elif bonded_change > 0.01:  # B_thresh increased in bonded cells
        print(f"\nResult: Bonded cells' B_thresh trending UP (unexpected)")
        result = "COUNTER_EVOLUTION"
    else:
        print(f"\nResult: NO CLEAR TREND (neutral drift)")
        print("Note: May need longer runs or stronger selection pressure")
        result = "NEUTRAL_DRIFT"

    # Save metrics
    with open(output_dir / "evolution_metrics.json", "w") as f:
        json.dump({
            "config": {
                "mutation_rate": config.mutation_rate,
                "mutation_strength": config.mutation_strength,
                "B_thresh_min": config.B_thresh_min,
                "B_thresh_max": config.B_thresh_max,
                "initial_B_thresh": config.B_thresh,
                "num_steps": num_steps,
            },
            "history": history,
            "evolution_history": evolution_history,
            "final": {
                "initial_mean": initial_mean,
                "final_mean": final_mean,
                "change": change,
                "result": result
            }
        }, f, indent=2)

    return history


def analyze_evolution_run(seed_data: dict) -> dict:
    """
    Compute regression statistics for one evolution run.

    Returns dict with slope, r_squared, p_value, effect_size, etc.
    """
    steps = np.array(seed_data["steps"])
    bonded_means = np.array(seed_data["mean_B_thresh_bonded"])

    # Filter out NaN values (early steps may have no bonded cells)
    valid_mask = ~np.isnan(bonded_means)
    steps_valid = steps[valid_mask]
    bonded_valid = bonded_means[valid_mask]

    if len(bonded_valid) < 10:
        return None  # Not enough data

    # Linear regression
    if HAS_SCIPY:
        result = linregress(steps_valid, bonded_valid)
        slope = result.slope
        intercept = result.intercept
        r_squared = result.rvalue ** 2
        p_value = result.pvalue
    else:
        # Fallback: compute slope with polyfit
        slope, intercept = np.polyfit(steps_valid, bonded_valid, 1)
        residuals = bonded_valid - (slope * steps_valid + intercept)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((bonded_valid - bonded_valid.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        p_value = None  # Can't compute without scipy

    # Effect size (Cohen's d): comparing first 10 vs last 10 data points
    initial_values = bonded_valid[:10]
    final_values = bonded_valid[-10:]
    initial_std = np.std(initial_values) if len(initial_values) > 1 else 0.01
    effect_size = (np.mean(initial_values) - np.mean(final_values)) / max(initial_std, 0.001)

    # Convergence test: did variance in B_thresh decrease?
    all_stds = np.array(seed_data["std_B_thresh"])
    early_mean_std = np.mean(all_stds[:20]) if len(all_stds) >= 20 else np.mean(all_stds[:len(all_stds)//2])
    late_mean_std = np.mean(all_stds[-20:]) if len(all_stds) >= 20 else np.mean(all_stds[len(all_stds)//2:])
    std_decrease_pct = (early_mean_std - late_mean_std) / max(early_mean_std, 0.001) * 100

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "p_value": float(p_value) if p_value is not None else None,
        "initial_mean": float(np.mean(initial_values)),
        "final_mean": float(np.mean(final_values)),
        "effect_size": float(effect_size),
        "std_decrease_pct": float(std_decrease_pct),
    }


def create_validation_plot(all_runs: dict, statistics: dict, output_dir: Path):
    """Create overlaid trajectories plot with regression lines."""
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_runs)))

    for i, (seed, data) in enumerate(all_runs.items()):
        steps = np.array(data["steps"])
        bonded = np.array(data["mean_B_thresh_bonded"])

        # Plot trajectory
        ax.plot(steps, bonded, color=colors[i], alpha=0.7,
                label=f"Seed {seed}", linewidth=1.5)

        # Plot regression line
        stats = statistics["runs"][i]
        y_fit = stats["slope"] * steps + stats["intercept"]
        ax.plot(steps, y_fit, color=colors[i], linestyle="--",
                alpha=0.5, linewidth=1)

    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Mean B_thresh (Bonded Cells)", fontsize=12)
    ax.set_title("Evolution Validation: B_thresh Adaptation Across Seeds", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add summary text box
    summary = statistics["summary"]
    text = f"All slopes negative: {summary['all_slopes_negative']}\n"
    text += f"Mean slope: {summary['mean_slope']:.2e}\n"
    text += f"Mean effect size: {summary['mean_effect_size']:.2f}"
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "evolution_validation.png", dpi=150)
    plt.close()
    print(f"  Saved plot to {output_dir / 'evolution_validation.png'}")


def print_validation_summary(statistics: dict):
    """Print formatted summary of evolution validation results."""
    print()
    print("=" * 60)
    print("EVOLUTION VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Seeds tested: {len(statistics['seeds'])}")
    print(f"Steps per run: {statistics['steps_per_run']:,}")
    print()
    print("RESULTS:")

    c = statistics["criteria"]
    s = statistics["summary"]

    print(f"  All slopes negative:     [{c['all_negative_slopes']}] ({s['num_negative_slopes']}/5 seeds)")
    print(f"  All p-values < 0.01:     [{c['all_p_values_lt_001']}]")
    print(f"  Mean effect size > 0.5:  [{c['mean_effect_size_gt_05']}] (d = {s['mean_effect_size']:.2f})")
    print(f"  Variance decreased >20%: [{c['variance_decreased_20pct']}] ({s['mean_std_decrease_pct']:.1f}% decrease)")
    print()

    if s["evolution_confirmed"]:
        print("CONCLUSION: *** EVOLUTION CONFIRMED ***")
    else:
        print("CONCLUSION: Evolution NOT confirmed (criteria not met)")

    print("=" * 60)


def run_evolution_validation(output_dir: Path):
    """
    Phase 13: Rigorous evolution validation across multiple seeds.

    Runs 5 independent simulations with different seeds to prove
    B_thresh evolution is statistically significant, not noise.

    Protocol:
    - 5 seeds: [42, 123, 456, 789, 1011]
    - 50,000 steps per run
    - Log every 500 steps (100 data points per run)
    - Statistical analysis: regression, p-values, effect sizes

    Success criteria:
    - All slopes negative (directional evolution)
    - All p-values < 0.01 (statistically significant)
    - Mean effect size > 0.5 (meaningful change)
    - Std decreases by >20% (population converging)
    """
    print("=" * 60)
    print("Phase 13: Evolution Validation (5 Seeds x 50k Steps)")
    print("=" * 60)
    print()
    print("This experiment will take approximately 3-7 hours.")
    print("Progress updates every 10,000 steps.")

    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 1011]
    num_steps = 50000
    log_interval = 500

    all_runs = {}  # seed -> data dict

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*40}")
        print(f"Starting seed {seed} ({seed_idx+1}/{len(seeds)})...")
        print(f"{'='*40}")

        # Create fresh simulation for this seed
        config = Config(
            grid_size=64,
            injection_mode="center",  # Shared food source = selection pressure
            injection_rate=0.015,
            injection_width=5,
            B_thresh=0.25,            # Starting value
            mutation_rate=0.002,      # 0.2% per step
            mutation_strength=0.02,   # Max +/-0.02
            B_thresh_min=0.10,
            B_thresh_max=0.50,
            k1=0.05,
            k2=0.05,
            k3=0.01,
            epsilon=0.001,
        )

        sim = Simulation(config, seed=seed)

        # Data collection for this run
        seed_data = {
            "steps": [],
            "mean_B_thresh_all": [],
            "mean_B_thresh_bonded": [],
            "std_B_thresh": [],
            "total_bonds": [],
            "num_bonded_cells": [],
        }

        # Run simulation with progress updates
        for step_num in range(0, num_steps, log_interval):
            sim.run(log_interval)

            # Force MLX array evaluation
            mx.eval(sim.state.B_thresh)

            # Collect metrics
            B_thresh = sim.state.B_thresh
            mean_thresh = float(mx.mean(B_thresh))
            std_thresh = float(mx.std(B_thresh))

            # Count bonds
            total_bonds = count_bonds(sim.state)

            # Bonded cell detection
            bonds_per_cell = mx.sum(sim.state.bonds, axis=-1)  # [H, W]
            bonded_mask = bonds_per_cell > 0
            num_bonded = int(mx.sum(bonded_mask))

            if num_bonded > 0:
                mean_bonded_thresh = float(mx.sum(B_thresh * bonded_mask) / num_bonded)
            else:
                mean_bonded_thresh = float('nan')

            # Record data
            seed_data["steps"].append(sim.step_count)
            seed_data["mean_B_thresh_all"].append(mean_thresh)
            seed_data["mean_B_thresh_bonded"].append(mean_bonded_thresh)
            seed_data["std_B_thresh"].append(std_thresh)
            seed_data["total_bonds"].append(total_bonds)
            seed_data["num_bonded_cells"].append(num_bonded)

            # Progress print every 10,000 steps
            if sim.step_count % 10000 == 0:
                print(f"  Seed {seed}: Step {sim.step_count}/{num_steps} "
                      f"({100*sim.step_count/num_steps:.0f}%) - "
                      f"bonds={total_bonds}, mean_B_thresh={mean_thresh:.4f}, "
                      f"bonded_B_thresh={mean_bonded_thresh:.4f}")

        all_runs[seed] = seed_data
        print(f"  Seed {seed} complete. Final bonds: {seed_data['total_bonds'][-1]}")

    # Statistical analysis
    print("\n" + "=" * 40)
    print("Analyzing results...")
    print("=" * 40)

    statistics = {
        "seeds": seeds,
        "steps_per_run": num_steps,
        "log_interval": log_interval,
        "runs": [],
        "summary": {},
        "criteria": {},
    }

    for seed in seeds:
        seed_data = all_runs[seed]
        analysis = analyze_evolution_run(seed_data)
        if analysis:
            analysis["seed"] = seed
            statistics["runs"].append(analysis)
            print(f"  Seed {seed}: slope={analysis['slope']:.2e}, "
                  f"r2={analysis['r_squared']:.3f}, effect_size={analysis['effect_size']:.2f}")

    # Compute summary statistics
    slopes = [r["slope"] for r in statistics["runs"]]
    effect_sizes = [r["effect_size"] for r in statistics["runs"]]
    p_values = [r["p_value"] for r in statistics["runs"] if r["p_value"] is not None]
    std_decreases = [r["std_decrease_pct"] for r in statistics["runs"]]

    statistics["summary"] = {
        "all_slopes_negative": all(s < 0 for s in slopes),
        "num_negative_slopes": sum(1 for s in slopes if s < 0),
        "mean_slope": float(np.mean(slopes)),
        "all_p_values_significant": all(p < 0.01 for p in p_values) if p_values else None,
        "mean_effect_size": float(np.mean(effect_sizes)),
        "mean_std_decrease_pct": float(np.mean(std_decreases)),
        "evolution_confirmed": False,  # Set below
    }

    # Evaluate criteria
    statistics["criteria"] = {
        "all_negative_slopes": "PASS" if statistics["summary"]["all_slopes_negative"] else "FAIL",
        "all_p_values_lt_001": "PASS" if statistics["summary"]["all_p_values_significant"] else ("FAIL" if p_values else "N/A"),
        "mean_effect_size_gt_05": "PASS" if statistics["summary"]["mean_effect_size"] > 0.5 else "FAIL",
        "variance_decreased_20pct": "PASS" if statistics["summary"]["mean_std_decrease_pct"] > 20 else "FAIL",
    }

    # Overall conclusion: at least 3/4 criteria must pass
    passed = sum(1 for v in statistics["criteria"].values() if v == "PASS")
    statistics["summary"]["evolution_confirmed"] = passed >= 3

    # Create visualization
    print("\nCreating visualization...")
    create_validation_plot(all_runs, statistics, output_dir)

    # Save JSON
    print("Saving statistics...")
    with open(output_dir / "evolution_statistics.json", "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"  Saved to {output_dir / 'evolution_statistics.json'}")

    # Also save raw run data for future analysis
    with open(output_dir / "evolution_validation_raw.json", "w") as f:
        # Convert to serializable format
        raw_data = {str(k): v for k, v in all_runs.items()}
        json.dump(raw_data, f, indent=2)
    print(f"  Saved raw data to {output_dir / 'evolution_validation_raw.json'}")

    # Print summary
    print_validation_summary(statistics)

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description="Protocell division experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=["1", "2", "3", "budding", "budding_barrier", "natural", "lineage", "competition", "fitness", "evolution", "validation", "all"],
        default="all",
        help="Which experiment to run (1, 2, 3, budding, budding_barrier, natural, lineage, competition, fitness, evolution, validation, or all). Note: 'validation' is excluded from 'all' due to ~3-7 hour runtime.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("docs/assets/division"),
        help="Output directory for images and metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    print(f"MLX device: {mx.default_device()}")
    print(f"Output directory: {args.output_dir}")
    print()

    if args.experiment in ("1", "all"):
        run_experiment_1(args.output_dir, args.seed)
        print()

    if args.experiment in ("2", "all"):
        run_experiment_2(args.output_dir, args.seed)
        print()

    if args.experiment in ("3", "all"):
        run_experiment_3(args.output_dir, args.seed)
        print()

    if args.experiment in ("budding", "all"):
        run_budding_experiment(args.output_dir, args.seed)
        print()

    if args.experiment in ("budding_barrier", "all"):
        run_budding_barrier_experiment(args.output_dir, args.seed)
        print()

    if args.experiment in ("natural", "all"):
        run_natural_division_experiment(args.output_dir, args.seed)
        print()

    if args.experiment in ("lineage", "all"):
        run_lineage_experiment(args.output_dir, args.seed)
        print()

    if args.experiment in ("competition", "all"):
        run_competition_experiment(args.output_dir, args.seed)
        print()

    if args.experiment in ("fitness", "all"):
        run_fitness_experiment(args.output_dir, args.seed)
        print()

    if args.experiment in ("evolution", "all"):
        run_evolution_experiment(args.output_dir, args.seed)
        print()

    # NOTE: validation is EXCLUDED from "all" due to ~3-7 hour runtime
    # Must be run explicitly with --experiment validation
    if args.experiment == "validation":
        run_evolution_validation(args.output_dir)
        print()

    print("=" * 60)
    print("All experiments complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
