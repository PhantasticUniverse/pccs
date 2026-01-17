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

## Code Standards

### Type Hints
- All public functions must have type hints
- Use `CellState`, `Config`, `mx.array` types from the codebase
- Return types are required

### Docstrings
- Public functions need docstrings with Args/Returns
- Experiment functions should document Protocol steps
- Keep docstrings concise but complete

### Style
- Prefer composition over inheritance
- Use dataclasses for state containers
- Keep functions pure where possible (return new state, don't mutate)

## MLX Rules

### When to Sync GPU
- Before reading array values in callbacks (use MLX sync functions)
- Before visualization/saving images
- Before JSON metrics collection
- **Never** sync in hot loops

### GPU-Friendly Patterns
```python
# Use mx.where for conditional updates
result = mx.where(mask, new_value, old_array)

# Use mx.roll for neighbor access
north = mx.roll(field, shift=-1, axis=0)

# Use [N, C, H, W] for convolutions
field_4d = field.reshape(1, 1, H, W)
```

### Anti-Patterns to Avoid
- Direct indexing: `arr[y, x] = value` (breaks GPU graph)
- Unseeded random: `mx.random.uniform(shape=...)` (non-reproducible)
- Sync in loops: Forcing GPU sync after every operation (kills performance)

## Testing Standards

### Grid Sizes
- Unit tests: Use `small_config` fixture (grid_size=16)
- Integration tests: Use `default_config` fixture (grid_size=32)
- Never use grid_size > 32 in regular tests

### Assertions
```python
# Float comparison
assert mx.allclose(actual, expected)

# Boolean checks
assert mx.all(state.B >= 0.0)
```

### Fixtures
Use fixtures from `tests/conftest.py`:
- `default_config` - standard config (32x32)
- `small_config` - fast tests (16x16)
- `uniform_state` - controlled initial state
- `rng_key` - deterministic random key
