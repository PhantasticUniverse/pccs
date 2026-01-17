# Run PCCS Experiment

Run protocell division experiments.

## Usage

- `/experiment evolution` - Run evolution experiment (recommended demo)
- `/experiment validation` - Run full 5-seed validation suite
- `/experiment all` - Run all experiments
- `/experiment <name>` - Run specific experiment by name

## Available Experiments

| Name | Description |
|------|-------------|
| `1` | Growth dynamics under doubled injection |
| `2` | Triple injection experiment |
| `3` | Mechanical cut division |
| `budding` | Budding division mode |
| `budding_barrier` | Budding with barrier |
| `natural` | Natural division dynamics |
| `lineage` | 3-generation lineage tracking |
| `competition` | Resource competition |
| `fitness` | Fitness selection |
| `evolution` | Full evolution demo (recommended) |
| `multiparameter` | Multi-parameter evolution |
| `k1_attractor` | K1 attractor dynamics |
| `scarcity` | Resource scarcity experiment |
| `validation` | 5-seed statistical validation |
| `all` | All standard experiments (excludes validation, k1_attractor, scarcity) |

## Command

```bash
python examples/division_experiments.py --experiment $ARGUMENTS --output-dir docs/assets/division/
```

## Output

Results are saved to `docs/assets/division/`:
- PNG images: `{experiment}_{phase}_{step}_{type}.png`
- Metrics: `{experiment}_metrics.json`

## Examples

Run the evolution experiment:
```bash
python examples/division_experiments.py --experiment evolution --output-dir docs/assets/division/
```

Run validation across 5 seeds:
```bash
python examples/division_experiments.py --experiment validation --output-dir docs/assets/division/
```
