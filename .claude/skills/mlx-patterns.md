# MLX Patterns for PCCS

Best practices for working with MLX arrays in this codebase.

## Array Creation

**Do:**
```python
# Seeded random for reproducibility
key = mx.random.key(seed)
key, subkey = mx.random.split(key)
values = mx.random.uniform(subkey, shape=(H, W))
```

**Don't:**
```python
# Unseeded random - non-reproducible
values = mx.random.uniform(shape=(H, W))
```

## GPU-Friendly Operations

**Do:**
```python
# Use mx.where() with coordinate masks
x_coords = mx.arange(W).reshape(1, -1)
y_coords = mx.arange(H).reshape(-1, 1)
mask = (x_coords > 10) & (y_coords < 20)
result = mx.where(mask, new_value, old_array)
```

**Don't:**
```python
# Direct indexing breaks GPU graph
arr[y, x] = new_value  # Avoid!
```

## Neighbor Access (Toroidal Boundaries)

**Do:**
```python
# Use mx.roll for neighbor access
north = mx.roll(field, shift=-1, axis=0)
south = mx.roll(field, shift=1, axis=0)
east = mx.roll(field, shift=-1, axis=1)
west = mx.roll(field, shift=1, axis=1)
```

**Don't:**
```python
# Manual indexing with wraparound
north = field[(y-1) % H, x]  # Slow and breaks GPU
```

## Convolutions

**Do:**
```python
# Correct dimension order: [N, C, H, W]
field_4d = field.reshape(1, 1, H, W)
kernel_4d = kernel.reshape(1, 1, 3, 3)
result = mx.conv2d(field_4d, kernel_4d, padding=1)
result = result.reshape(H, W)
```

**Don't:**
```python
# Wrong dimension order causes errors
result = mx.conv2d(field, kernel)  # Missing batch/channel dims
```

## GPU Synchronization

**Do:**
```python
# Sync only before callbacks/visualization using mx synchronization
def callback(sim: Simulation):
    mx.synchronize(sim.state.B)  # Force GPU computation
    membranes = detect_closed_membranes(sim.state.bonds)
    # ... compute metrics
```

Note: In PCCS, we use `mx.eval(array)` to force synchronization before
reading values. This is MLX's synchronization primitive, not JavaScript eval.

**Don't:**
```python
# Sync after every operation (kills performance)
for step in range(1000):
    state = simulation_step(state)
    mx.synchronize(state.B)  # Unnecessary sync!
```

## CPU Transfer

**Do:**
```python
# Transfer only for visualization/saving
def save_state_images(state, ...):
    B_np = np.array(state.B)  # Once, for matplotlib
    plt.imshow(B_np)
```

**Don't:**
```python
# Unnecessary conversions in hot loops
for step in range(1000):
    B_np = np.array(state.B)  # Kills performance
    # ... do nothing with it
```

## Common Patterns in PCCS

### Creating Masks
```python
# High-B region mask
high_B_mask = state.B > threshold
count = mx.sum(high_B_mask.astype(mx.float32))
```

### Applying Perturbations
```python
# Line cut example
x_coords = mx.arange(W).reshape(1, -1)
mask = mx.abs(x_coords - x_position) < width
new_B = mx.where(mask, 0.0, state.B)
```

### Reducing Over Grids
```python
# Total mass
total = mx.sum(state.A + state.B + state.C)

# Mean in masked region
mean_B_high = mx.mean(mx.where(high_B_mask, state.B, 0.0))
```

## Key MLX Functions

| Function | Purpose |
|----------|---------|
| `mx.where(cond, x, y)` | Conditional selection |
| `mx.roll(arr, shift, axis)` | Neighbor access |
| `mx.conv2d()` | Convolution (diffusion) |
| `mx.sum()`, `mx.mean()` | Reductions |
| `mx.random.key(seed)` | Seeded RNG |
| `np.array(mx_arr)` | GPU to CPU transfer |
