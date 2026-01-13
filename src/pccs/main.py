"""
Command-line interface for PCCS simulation.

Usage:
    python -m pccs.main --help
    python -m pccs.main --steps 1000 --visualize
    python -m pccs.main --grid-size 128 --save-frames output/
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

from .config import Config
from .simulation import Simulation
from .visualization import Visualizer, save_state_images
from .metrics import compute_all_metrics, print_metrics_summary, total_mass
from .phase import kuramoto_order_parameter
from .bonds import detect_closed_membranes


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="PCCS - Phase-Coupled Catalytic Substrate Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Basic options
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of simulation steps"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    
    # Grid options
    parser.add_argument(
        "--grid-size", type=int, default=256,
        help="Grid width and height"
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show live visualization"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for visualization"
    )
    parser.add_argument(
        "--save-frames", type=str, default=None,
        help="Directory to save frame images"
    )
    parser.add_argument(
        "--save-interval", type=int, default=100,
        help="Save frame every N steps"
    )
    parser.add_argument(
        "--save-animation", type=str, default=None,
        help="Path to save animation (mp4 or gif)"
    )
    
    # Analysis options
    parser.add_argument(
        "--check-conservation", action="store_true",
        help="Check mass conservation"
    )
    parser.add_argument(
        "--measure-sync", action="store_true",
        help="Measure phase synchronization"
    )
    parser.add_argument(
        "--detect-membranes", action="store_true",
        help="Detect closed membranes"
    )
    parser.add_argument(
        "--print-metrics", action="store_true",
        help="Print all metrics at end"
    )
    parser.add_argument(
        "--save-metrics", type=str, default=None,
        help="Save metrics to JSON file"
    )
    
    # Config parameter overrides
    parser.add_argument("--D-base", type=float, default=None, dest="D_base")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--k1", type=float, default=None)
    parser.add_argument("--k2", type=float, default=None)
    parser.add_argument("--k3", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--omega-0", type=float, default=None, dest="omega_0")
    parser.add_argument("--K-phase", type=float, default=None, dest="K_phase")
    parser.add_argument("--chi", type=float, default=None)
    parser.add_argument("--B-thresh", type=float, default=None, dest="B_thresh")
    parser.add_argument("--cos-thresh", type=float, default=None, dest="cos_thresh")
    parser.add_argument(
        "--injection-mode", type=str, default=None,
        choices=["boundary", "uniform", "point_sources", "none"],
        dest="injection_mode"
    )
    parser.add_argument("--injection-rate", type=float, default=None, dest="injection_rate")
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Build config from defaults and overrides
    config_kwargs = {"grid_size": args.grid_size}
    
    # Add any specified overrides
    for param in [
        "D_base", "alpha", "k1", "k2", "k3", "kappa", "epsilon",
        "omega_0", "K_phase", "chi", "B_thresh", "cos_thresh",
        "injection_mode", "injection_rate"
    ]:
        value = getattr(args, param, None)
        if value is not None:
            config_kwargs[param] = value
    
    try:
        config = Config(**config_kwargs)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    
    print(f"PCCS Simulation")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {args.seed or 'random'}")
    print()
    
    # Check MLX device
    print(f"MLX device: {mx.default_device()}")
    print()
    
    # Create simulation
    sim = Simulation(config, seed=args.seed)
    
    # Visualization mode
    if args.visualize:
        viz = Visualizer(sim, mode="composite", fps=args.fps)
        viz.show_live(steps=args.steps)
        return 0
    
    # Save animation mode
    if args.save_animation:
        viz = Visualizer(sim, mode="composite", fps=args.fps)
        viz.save_animation(args.save_animation, steps=args.steps)
        return 0
    
    # Frame saving callback
    frame_callback = None
    if args.save_frames:
        output_dir = Path(args.save_frames)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def frame_callback(s: Simulation) -> None:
            save_state_images(s.state, str(output_dir), "frame", s.step_count)
            print(f"  Saved frame at step {s.step_count}")
    
    # Conservation checking callback
    if args.check_conservation:
        initial_mass = total_mass(sim.state)
        print(f"Initial mass: {initial_mass:.4f}")
        
        def conservation_callback(s: Simulation) -> None:
            current_mass = total_mass(s.state)
            loss_pct = (initial_mass - current_mass) / initial_mass * 100
            print(f"  Step {s.step_count}: mass={current_mass:.4f} (loss: {loss_pct:.2f}%)")
        
        frame_callback = conservation_callback
    
    # Sync measurement callback
    if args.measure_sync:
        def sync_callback(s: Simulation) -> None:
            R = kuramoto_order_parameter(s.state.phase)
            print(f"  Step {s.step_count}: R={R:.4f}")
        
        frame_callback = sync_callback
    
    # Run simulation
    print("Running simulation...")
    sim.run(
        args.steps,
        callback=frame_callback,
        callback_interval=args.save_interval,
        show_progress=True,
    )
    
    # Final analysis
    print()
    
    if args.detect_membranes:
        membranes = detect_closed_membranes(sim.state.bonds)
        print(f"Detected {len(membranes)} closed membrane(s)")
        for i, m in enumerate(membranes):
            print(f"  Membrane {i}: {len(m)} cells")
    
    if args.print_metrics:
        mx.eval(sim.state.A)  # Ensure evaluation
        metrics = compute_all_metrics(sim.state)
        print_metrics_summary(metrics)
    
    if args.save_metrics:
        mx.eval(sim.state.A)
        metrics = compute_all_metrics(sim.state)
        # Convert numpy types for JSON serialization
        metrics_json = json.loads(json.dumps(metrics, default=str))
        with open(args.save_metrics, "w") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"Metrics saved to {args.save_metrics}")
    
    print("Simulation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
