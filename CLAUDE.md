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
│   ├── reactions.py     # Phase-gated reaction system (mass-conserving)
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
3. **B-Only Bonds**: Bonds form based on B concentration threshold (phase alignment removed - see FINDINGS.md)
4. **Kuramoto Coupling**: Bonded cells synchronize phases more strongly

## Reaction Stoichiometry (Mass-Conserving)

```
R1: 2A → 2B          (φ ≈ 0)      Dimerization
R2: 2B → A + C       (φ ≈ 2π/3)   Breakdown
R3: A + C → 2A       (φ ≈ 4π/3)   Autocatalysis
```

All reactions: 2 molecules in → 2 molecules out. Total mass conserved.

## Injection Modes

| Mode | Description |
|------|-------------|
| `boundary` | Inject A at grid edges → ring membrane |
| `center` | Inject A at center → single protocell |
| `dual` | Two points at 1/4 and 3/4 → two protocells |
| `competing` | Two points at 1/3 and 2/3 → close protocells |
| `point_sources` | Four corners |
| `uniform` | Small constant everywhere |
| `none` | No injection |

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

- **Memory errors**: Reduce grid size (32-48 works reliably for experiments, 64+ may hit GPU limits)
- **Slow startup**: MLX compiles on first run, subsequent runs faster
- **No GPU**: Check `python -c "import mlx.core as mx; print(mx.default_device())"`
- **No closed membranes**: Ensure using B-only bonds (current default), not phase-aligned bonds

## Working Parameters for Protocells

```python
Config(
    grid_size=48,
    injection_mode="center",  # or "competing", "dual"
    injection_rate=0.02,
    injection_width=3,
    B_thresh=0.25,
    k1=0.05, k2=0.05, k3=0.01,
    epsilon=0.001,
)
```

## Key Files

- `docs/FINDINGS.md` - Development insights and what worked/didn't work
- `docs/PRD.md` - Original design specification
- `docs/assets/` - Visualizations (spiral_waves.gif, protocell_center.png, etc.)
