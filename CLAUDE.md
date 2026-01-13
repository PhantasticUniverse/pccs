# PCCS - Phase-Coupled Catalytic Substrate

A cellular automaton demonstrating emergent life-like behavior: protocell formation, division, lineages, and **verified evolution**.

## Quick Reference

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run simulation
python -m pccs.main

# Run experiments
python examples/division_experiments.py --experiment evolution   # Recommended demo
python examples/division_experiments.py --experiment all         # All experiments

# Dev tools
pytest                        # Tests
mypy src/pccs                 # Type check
ruff format src tests         # Format
```

## Tech Stack

- **Python 3.11+**, **MLX** (Apple Silicon GPU), **NumPy**, **Matplotlib**, **pytest**

## Project Structure

```
src/pccs/           # Core simulation (state, config, diffusion, reactions, phase, bonds, simulation)
tests/              # Mirrors src
examples/           # division_experiments.py - all experiments
docs/               # FINDINGS.md (detailed results), PRD.md (original spec), assets/
```

## Key Concepts

1. **Three Substrates**: A (precursor) → B (membrane) → C (catalyst)
2. **Phase Gating**: Reactions fire at specific phases (Kuramoto oscillators)
3. **B-Only Bonds**: Bonds form where B concentration exceeds threshold
4. **Per-Cell Inheritance**: Each cell has mutable `B_thresh` passed to offspring

## Achievements

**Evolution demonstrated and statistically validated:**
- ✅ Reproduction (budding, 3-generation lineages)
- ✅ Selection (competitive exclusion under resource scarcity)
- ✅ Heritable variation (per-cell B_thresh with mutation)
- ✅ Validation (5 seeds, p < 10⁻⁴⁷, effect size d = 14.12)

See `docs/FINDINGS.md` for detailed experiment results and methodology.

## Implementation Notes

- MLX arrays for GPU acceleration; sync sparingly (visualization/logging only)
- Bonds are symmetric; use average threshold of both cells
- Division requires daughter_offset > diffusion_range (~20 cells)
- Mutations applied after bond updates, clamped to [B_thresh_min, B_thresh_max]

## Common Issues

| Issue | Solution |
|-------|----------|
| Memory errors | Reduce grid_size (48 basic, 96 lineage) |
| Slow startup | Normal - MLX compiles on first run |
| Structures merging | Increase offset between injection points (≥20) |

## Key Files

- `docs/FINDINGS.md` - All experiment results and insights
- `docs/PRD.md` - Original design specification
- `examples/division_experiments.py` - Experiment implementations
