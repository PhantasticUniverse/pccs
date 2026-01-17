# PCCS Code Reviewer

Custom code review checklist for the PCCS codebase.

## MLX Anti-Patterns

Check for these common MLX mistakes:

### Direct Array Indexing
```python
# BAD: Breaks GPU computation graph
arr[y, x] = new_value

# GOOD: Use mx.where with masks
mask = (y_coords == y) & (x_coords == x)
arr = mx.where(mask, new_value, arr)
```

### Missing GPU Sync Before Visualization
```python
# BAD: May read stale data
def callback(sim):
    bonds = count_bonds(sim.state)  # GPU may not be done!

# GOOD: Sync first (use mx.eval in actual code)
def callback(sim):
    mx.synchronize(sim.state.B)  # Force completion
    bonds = count_bonds(sim.state)
```

### Unnecessary np.array() Conversions
```python
# BAD: Transfers data to CPU unnecessarily
for step in range(1000):
    B_np = np.array(state.B)
    if B_np.max() > 0.5:  # Could stay on GPU!
        ...

# GOOD: Use MLX operations
for step in range(1000):
    if float(mx.max(state.B)) > 0.5:
        ...
```

### Wrong Convolution Dimensions
```python
# BAD: Missing batch and channel dimensions
result = mx.conv2d(field, kernel)

# GOOD: [N, C, H, W] format
field_4d = field.reshape(1, 1, H, W)
kernel_4d = kernel.reshape(1, 1, 3, 3)
result = mx.conv2d(field_4d, kernel_4d, padding=1).reshape(H, W)
```

## Simulation Anti-Patterns

### Hardcoded Seeds
```python
# BAD: Hardcoded seed buried in function
def run_experiment():
    sim = Simulation(config, seed=42)  # Hidden!

# GOOD: Seed as parameter
def run_experiment(seed: int = 42):
    sim = Simulation(config, seed=seed)
```

### Missing JSON Output
```python
# BAD: Results only printed
print(f"Final bonds: {bonds}")

# GOOD: Save to JSON for reproducibility
metrics = {"final_bonds": bonds, "seed": seed}
with open(output_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

### Large Grids in Tests
```python
# BAD: Slow tests
def test_feature():
    config = Config(grid_size=128)  # Too large!

# GOOD: Use small_config fixture
def test_feature(small_config):  # grid_size=16
    ...
```

## Code Quality Checks

### Missing Type Hints
```python
# BAD: No type hints
def process_state(state, config):
    ...

# GOOD: Type hints on public functions
def process_state(state: CellState, config: Config) -> CellState:
    ...
```

### Missing Docstrings
```python
# BAD: No documentation
def complex_function(x, y, z):
    ...

# GOOD: Clear docstring
def complex_function(x: float, y: float, z: float) -> float:
    """
    Compute the interaction strength between components.

    Args:
        x: First component concentration
        y: Second component concentration
        z: Catalyst concentration

    Returns:
        Interaction strength in range [0, 1]
    """
    ...
```

### Assertions for Float Comparison
```python
# BAD: Direct equality (fails due to float precision)
assert result == expected

# GOOD: Use allclose
assert mx.allclose(result, expected)
```

## Review Checklist

When reviewing PCCS code, verify:

- [ ] All random operations use seeded RNG
- [ ] GPU sync (mx.eval) called before reading array values
- [ ] No direct array indexing (use mx.where)
- [ ] Convolutions use [N, C, H, W] format
- [ ] Tests use small_config (grid_size=16)
- [ ] Experiments save JSON metrics
- [ ] Public functions have type hints
- [ ] Complex logic has docstrings
- [ ] Float comparisons use mx.allclose
- [ ] No hardcoded paths (use Path objects)
