# Simulation Patterns for PCCS

Patterns for writing experiments based on `examples/division_experiments.py`.

## Experiment Template

```python
def run_experiment(output_dir: Path, seed: int = 42):
    """
    Experiment description.

    Protocol:
    1. Phase description
    2. Parameter change
    3. Expected outcome
    """
    print("=" * 60)
    print("Experiment Name")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Config with experiment-specific parameters
    config = Config(
        grid_size=32,  # Use 32-48 for basic, 96 for lineage
        injection_mode="boundary",  # or "center"
        injection_rate=0.02,
        B_thresh=0.25,
        k1=0.05,
        k2=0.05,
        k3=0.01,
    )

    # 2. Initialize Simulation with seed for reproducibility
    sim = Simulation(config, seed=seed)
    history = []

    # 3. Run phases with metrics callback
    print("\nPhase 1: Description...")
    sim.run(2000, callback=create_metrics_callback(history), callback_interval=200)
    save_state_images(sim.state, str(output_dir), "prefix_phase1", sim.step_count)

    # 4. Modify parameters between phases
    sim.config.injection_rate = 0.04  # Double rate

    print("\nPhase 2: Description...")
    sim.run(5000, callback=create_metrics_callback(history), callback_interval=200)
    save_state_images(sim.state, str(output_dir), "prefix_phase2", sim.step_count)

    # 5. Analyze and save metrics
    metrics = {
        "experiment": "name",
        "seed": seed,
        "final_bonds": history[-1]["total_bonds"],
        "history": history,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
```

## Metrics Callback Pattern

```python
def create_metrics_callback(history: list):
    """Create callback that logs membrane and bond stats."""
    def callback(sim: Simulation):
        # CRITICAL: Force MLX synchronization before metrics
        # This ensures GPU computations complete before reading values
        mx.synchronize(sim.state.B)

        membranes = detect_closed_membranes(sim.state.bonds)
        bonds = count_bonds(sim.state)

        history.append({
            "step": sim.step_count,
            "num_membranes": len(membranes),
            "membrane_sizes": [len(m) for m in membranes],
            "total_bonds": bonds,
            "total_mass": float(total_mass(sim.state)),
            "global_sync": float(kuramoto_order_parameter(sim.state.phase)),
        })

        print(f"  Step {sim.step_count}: bonds={bonds}")

    return callback
```

Note: In the actual codebase, `mx.eval(array)` is used for synchronization.
This is MLX's GPU sync function, not JavaScript's eval.

## Parameter Sweep Pattern

```python
def run_parameter_sweep(output_dir: Path, seed: int = 42):
    """Sweep over parameter values."""
    results = []

    for value in [0.001, 0.002, 0.005, 0.01, 0.02]:
        config = Config(
            grid_size=32,
            injection_rate=value,
        )
        sim = Simulation(config, seed=seed)
        sim.run(5000)

        # Collect metrics
        bonds = count_bonds(sim.state)
        results.append({
            "injection_rate": value,
            "final_bonds": bonds,
        })

        # Save snapshot
        save_state_images(
            sim.state,
            str(output_dir),
            f"sweep_{value:.3f}",
            sim.step_count
        )

    return results
```

## Multi-Seed Validation Pattern

```python
def run_validation(output_dir: Path, seeds: list[int] = None):
    """Run experiment across multiple seeds for statistical validation."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        metrics = run_single_experiment(output_dir, seed=seed)
        results.append(metrics)

    # Statistical analysis
    values = [r["final_bonds"] for r in results]
    mean_val = sum(values) / len(values)

    # Optional: p-value and effect size with scipy
    if HAS_SCIPY:
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(control_values, test_values)
        effect_size = (mean_test - mean_control) / pooled_std

    summary = {
        "n_seeds": len(seeds),
        "mean": mean_val,
        "std": np.std(values),
        "seeds": seeds,
        "results": results,
    }

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
```

## Key Guidelines

1. **Always use seeds** for reproducibility
2. **Sync GPU before metrics** to ensure computations complete
3. **Save JSON metrics** for all experiments
4. **Use appropriate grid sizes**: 32-48 for basic, 96 for lineage tracking
5. **Save images at key phases** with descriptive prefixes
6. **Document protocol** in docstrings with numbered steps
