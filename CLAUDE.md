# PCCS - Phase-Coupled Catalytic Substrate

A novel cellular automaton exploring emergent life-like complexity through phase-coupled chemistry.

## Quick Reference

```bash
# Setup (first time)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run simulation
python -m pccs.main

# Run division & evolution experiments
python examples/division_experiments.py --experiment natural      # Two-generation budding
python examples/division_experiments.py --experiment lineage      # Three-generation lineage
python examples/division_experiments.py --experiment competition  # Resource competition
python examples/division_experiments.py --experiment fitness      # Differential fitness
python examples/division_experiments.py --experiment evolution    # Per-cell B_thresh evolution

# Run tests
pytest

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
├── examples/
│   └── division_experiments.py  # Protocell division & lineage experiments
└── docs/
    ├── FINDINGS.md      # Development insights, experiment results
    ├── PRD.md           # Original design specification
    └── assets/          # Visualizations
```

## Key Concepts

1. **Three Substrates**: A (precursor), B (structural/membrane), C (catalyst/energy)
2. **Phase Gating**: Reactions only fire when cell phase φ is near target values
3. **B-Only Bonds**: Bonds form based on B concentration threshold (phase alignment removed - see FINDINGS.md)
4. **Kuramoto Coupling**: Bonded cells synchronize phases more strongly
5. **Per-Cell Parameters**: Each cell has its own `B_thresh` (bond threshold) that can mutate and be inherited

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
| `dual` | Two points at 1/4 and 3/4 → two protocells (far apart) |
| `competing` | Two points at 1/3 and 2/3 → two protocells (close) |
| `budding` | Mother at center + daughter at center+20 → natural division |
| `lineage` | Three generations: center, center+20, center+40 → lineage |
| `competition` | Four protocells in diamond pattern (N, S, E, W) → selection experiments |
| `point_sources` | Four corners |
| `uniform` | Small constant everywhere |
| `none` | No injection |

## Major Achievements

### Protocell Division (Experiment 6)
- **TRUE BUDDING**: Mother spawns independent daughter naturally
- Key: daughter_offset=20 cells (beyond diffusion range)
- No surgical intervention required

### Multi-Generational Lineage (Experiment 7)
- **THREE GENERATIONS**: Mother → Daughter → Granddaughter
- Each maintains ~130-145 bonds independently
- Unlimited reproductive capacity demonstrated

### Competitive Exclusion (Experiment 8)
- **SELECTION**: Shared resources → winner-take-all
- Independent food sources: all survive (just downsize)
- Shared food source: one winner, others collapse

### Differential Fitness (Experiment 9)
- **PARAMETER VARIATION = FITNESS ADVANTAGE**
- Strong (B_thresh=0.20) vs Weak (B_thresh=0.30)
- Strong wins **100%** of trials with 2.5-3.8x more bonds
- Proves: parameter differences create selectable fitness

### True Inheritance (Phase 12)
- **PER-CELL B_THRESH**: Each cell has its own bond formation threshold
- **MUTATION SYSTEM**: Stochastic mutations (±0.02) occur with configurable rate
- **IMPLICIT INHERITANCE**: Spatial continuity provides natural parent→offspring inheritance
- **SELECTION OBSERVED**: Bonded cells evolve lower B_thresh over time (fitness advantage)

### Path to Evolution - COMPLETE
1. ✅ **Reproduction** (budding, lineage)
2. ✅ **Selection** (competitive exclusion)
3. ✅ **Heritable variation** (per-cell B_thresh with mutation)
4. ✅ **True inheritance** (spatial continuity + mutation system)

## Division & Evolution Experiments

```bash
# Run specific experiment
python examples/division_experiments.py --experiment natural      # Two-generation budding
python examples/division_experiments.py --experiment lineage      # Three-generation lineage
python examples/division_experiments.py --experiment competition  # Resource competition
python examples/division_experiments.py --experiment fitness      # Differential fitness
python examples/division_experiments.py --experiment evolution    # Per-cell B_thresh evolution

# Run all experiments
python examples/division_experiments.py --experiment all
```

| Experiment | Description | Result |
|------------|-------------|--------|
| 1-3 | Pressure/cut tests | Homeostasis (no division) |
| 4 | Budding (offset=10) | Merged |
| 5 | Budding + barrier | Division (assisted) |
| 6 | Natural (offset=20) | **TRUE BUDDING** |
| 7 | Lineage (3 generations) | **THREE GENERATIONS** |
| 8 | Competition (shared food) | **SELECTION** |
| 9 | Fitness (B_thresh asymmetry) | **STRONG WINS 100%** |
| 10 | Evolution (per-cell B_thresh) | **SELECTION ON BONDED CELLS** |

## Implementation Notes

- All grid operations use MLX arrays for GPU acceleration
- Diffusion implemented as convolution with bond-modulated kernel
- Phase wraps at 2π using `mx.remainder(phase, 2 * mx.pi)`
- Bonds are symmetric: if (i,j)→(i,j+1) bonded, so is reverse
- Use MLX synchronization sparingly—only at visualization/logging points
- Division requires daughter_offset > diffusion_range (~20 cells)
- Per-cell B_thresh: bonds use average threshold of both cells `(B_thresh_i + B_thresh_j) / 2`
- Mutations applied each step after bond updates, clamped to [B_thresh_min, B_thresh_max]

## Common Issues

- **Memory errors**: Reduce grid size (48 for basic, 96 for lineage experiments)
- **Slow startup**: MLX compiles on first run, subsequent runs faster
- **No GPU**: Check `python -c "import mlx.core as mx; print(mx.default_device())"`
- **No closed membranes**: Ensure using B-only bonds (current default)
- **Structures merging**: Increase offset between injection points (≥20 cells)

## Working Parameters

```python
# Single protocell
Config(
    grid_size=48,
    injection_mode="center",
    injection_rate=0.02,
    injection_width=3,
    B_thresh=0.25,
    k1=0.05, k2=0.05, k3=0.01,
    epsilon=0.001,
)

# Natural division (budding)
Config(
    grid_size=48,
    injection_mode="budding",  # daughter_offset=20 in simulation.py
    ...
)

# Three-generation lineage
Config(
    grid_size=96,  # Larger grid for 3 generations
    injection_mode="lineage",
    ...
)

# Resource competition (4 protocells)
Config(
    grid_size=80,
    injection_mode="competition",
    injection_rate=0.01,  # Shared center source
    ...
)

# Differential fitness experiment
Config(
    grid_size=64,
    injection_mode="center",  # Shared food source
    injection_rate=0.01,
    fitness_mode=True,        # Enable position-dependent B_thresh
    strong_B_thresh=0.20,     # Left half (easier bond formation)
    weak_B_thresh=0.30,       # Right half (harder bond formation)
    ...
)

# Evolution experiment (per-cell B_thresh with mutation)
Config(
    grid_size=64,
    injection_mode="center",  # Shared food source (selection pressure)
    injection_rate=0.015,
    B_thresh=0.25,            # Starting value for all cells
    mutation_rate=0.002,      # 0.2% of cells mutate per step
    mutation_strength=0.02,   # Max ±0.02 per mutation
    B_thresh_min=0.10,        # Minimum allowed B_thresh
    B_thresh_max=0.50,        # Maximum allowed B_thresh
    ...
)
```

## Key Files

- `docs/FINDINGS.md` - Development insights, division experiment results
- `docs/PRD.md` - Original design specification
- `examples/division_experiments.py` - All division/lineage experiments
- `docs/assets/division/` - Division experiment visualizations
