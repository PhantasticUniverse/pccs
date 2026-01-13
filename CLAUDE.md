# PCCS - Phase-Coupled Catalytic Substrate

A novel cellular automaton exploring emergent life-like complexity through phase-coupled chemistry.

## Quick Reference

```bash
# Setup (first time)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run simulation
python -m pccs.main

# Run tests
pytest

# Run specific test
pytest tests/test_dynamics.py -v

# Type checking
mypy src/pccs

# Format code
ruff format src tests
ruff check src tests --fix
```

## Tech Stack

- **Python 3.11+** (required for MLX)
- **MLX** - Apple's ML framework for Apple Silicon (primary compute backend)
- **NumPy** - Array operations and CPU fallback
- **Matplotlib** - Visualization
- **pytest** - Testing

## Project Structure

```
pccs/
├── src/pccs/
│   ├── __init__.py      # Package init, version
│   ├── state.py         # CellState dataclass, grid initialization
│   ├── config.py        # SimulationConfig with all parameters
│   ├── diffusion.py     # Membrane-gated diffusion dynamics
│   ├── reactions.py     # Phase-gated reaction system (A→B→C→A)
│   ├── phase.py         # Kuramoto oscillator dynamics
│   ├── bonds.py         # Bond formation/breaking logic
│   ├── simulation.py    # Main simulation loop, update orchestration
│   ├── visualization.py # Real-time and export visualization
│   ├── metrics.py       # Analysis: sync order, membrane detection
│   └── main.py          # CLI entry point
├── tests/               # Mirrors src structure
├── examples/            # Jupyter notebooks, demo scripts
└── docs/                # PRD, implementation guides
```

## Key Concepts

1. **Three Substrates**: A (precursor), B (structural/membrane), C (catalyst/energy)
2. **Phase Gating**: Reactions only fire when cell phase φ is near target values
3. **Dual-Condition Bonds**: Require both B concentration AND phase alignment
4. **Kuramoto Coupling**: Bonded cells synchronize phases more strongly

## Implementation Notes

- All grid operations use MLX arrays for GPU acceleration
- Diffusion implemented as convolution with bond-modulated kernel
- Phase wraps at 2π using `mx.remainder(phase, 2 * mx.pi)`
- Bonds are symmetric: if (i,j)→(i,j+1) bonded, so is reverse
- Use `mx.eval()` sparingly—only at visualization/logging points

## Verification Commands

```bash
# Check mass conservation (should be approximately constant)
python -m pccs.main --check-conservation

# Measure phase synchronization (Kuramoto order parameter)
python -m pccs.main --measure-sync

# Detect closed membranes
python -m pccs.main --detect-membranes
```

## Common Issues

- **Memory errors**: Reduce grid size from 256 to 128
- **Slow startup**: MLX compiles on first run, subsequent runs faster
- **No GPU**: Check `python -c "import mlx.core as mx; print(mx.default_device())"`
