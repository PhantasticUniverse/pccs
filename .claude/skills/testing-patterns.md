# Testing Patterns for PCCS

Patterns for writing tests based on `tests/conftest.py` and existing tests.

## Available Fixtures

From `tests/conftest.py`:

```python
@pytest.fixture
def default_config() -> Config:
    """Default configuration for tests (grid_size=32)."""
    return Config(grid_size=32)

@pytest.fixture
def small_config() -> Config:
    """Small grid for fast tests (grid_size=16)."""
    return Config(grid_size=16)

@pytest.fixture
def default_state(default_config: Config) -> CellState:
    """Default initial state with seed=42."""
    return create_initial_state(default_config, seed=42)

@pytest.fixture
def uniform_state(default_config: Config) -> CellState:
    """Uniform state for diffusion tests (A=0.5, B=0.1, C=0.1)."""
    H = W = default_config.grid_size
    return CellState(
        A=mx.ones((H, W)) * 0.5,
        B=mx.ones((H, W)) * 0.1,
        C=mx.ones((H, W)) * 0.1,
        phase=mx.zeros((H, W)),
        bonds=mx.zeros((H, W, 4)),
    )

@pytest.fixture
def rng_key() -> mx.array:
    """Deterministic random key for stochastic tests."""
    return mx.random.key(12345)
```

## Test Structure

```python
def test_feature_behavior(default_config, small_config):
    """Test specific behavior with clear assertion."""
    # Arrange
    state = create_initial_state(small_config, seed=42)

    # Act
    result = function_under_test(state, small_config)

    # Assert
    assert mx.allclose(result.B, expected_B)
```

## MLX Assertions

**Floating-point comparison:**
```python
# Use allclose for float comparisons (default atol=1e-5, rtol=1e-5)
assert mx.allclose(actual, expected)

# With custom tolerance
assert mx.allclose(actual, expected, atol=1e-3)
```

**Element-wise boolean:**
```python
# All elements satisfy condition
assert mx.all(state.B >= 0.0)
assert mx.all(state.B <= 1.0)

# Any element satisfies condition
assert mx.any(state.bonds > 0)
```

**Shape checks:**
```python
assert state.B.shape == (H, W)
assert state.bonds.shape == (H, W, 4)
```

## Validation Testing

```python
def test_invalid_config_raises():
    """Test that invalid config raises appropriate error."""
    with pytest.raises(ValueError, match="grid_size must be positive"):
        Config(grid_size=-1)
```

## Determinism Testing

```python
def test_simulation_determinism(small_config):
    """Verify same seed produces same result."""
    sim1 = Simulation(small_config, seed=42)
    sim1.run(100)

    sim2 = Simulation(small_config, seed=42)
    sim2.run(100)

    assert mx.allclose(sim1.state.B, sim2.state.B)
```

## Mass Conservation Testing

```python
def test_mass_conservation(small_config):
    """Verify total mass is conserved."""
    state = create_initial_state(small_config, seed=42)
    initial_mass = total_mass(state)

    # Run simulation steps
    for _ in range(100):
        state = simulation_step(state, small_config)

    final_mass = total_mass(state)

    # Allow small numerical error
    assert mx.allclose(initial_mass, final_mass, atol=1e-4)
```

## Performance Testing

```python
@pytest.mark.slow
def test_large_grid_performance(default_config):
    """Test with larger grid - marked slow."""
    config = Config(grid_size=64)
    sim = Simulation(config, seed=42)
    sim.run(1000)

    # Just verify it completes without error
    assert sim.step_count == 1000
```

## Guidelines

1. **Use small grids (16)** for unit tests - faster execution
2. **Use default grids (32)** for integration tests
3. **Never use grids > 32** in regular tests - too slow
4. **Always use `mx.allclose`** for float comparisons
5. **Always use seeds** for reproducible tests
6. **Mark slow tests** with `@pytest.mark.slow`
7. **Test edge cases**: empty arrays, zero values, boundary conditions
