# Testing Strategy

This document outlines the testing approach for PCCS, organized by verification type.

---

## Unit Tests

### Config Tests (`tests/test_config.py`)

```python
def test_config_defaults():
    """Config initializes with PRD-specified defaults."""
    config = Config()
    assert config.D_base == 0.1
    assert config.alpha == 0.9
    # ... all defaults

def test_config_validation():
    """Config rejects invalid parameter ranges."""
    with pytest.raises(ValueError):
        Config(alpha=1.5)  # Must be < 1
    with pytest.raises(ValueError):
        Config(D_base=-0.1)  # Must be > 0

def test_config_serialization():
    """Config round-trips through dict."""
    config = Config(grid_size=128, D_base=0.2)
    restored = Config.from_dict(config.to_dict())
    assert config == restored
```

### State Tests (`tests/test_state.py`)

```python
def test_state_shapes():
    """State arrays have correct shapes."""
    config = Config(grid_size=64)
    state = create_initial_state(config)
    assert state.A.shape == (64, 64)
    assert state.bonds.shape == (64, 64, 4)

def test_state_on_gpu():
    """State arrays are on GPU."""
    state = create_initial_state(Config())
    # MLX arrays are on GPU by default
    assert state.A.dtype == mx.float32

def test_state_random_initialization():
    """Random init produces values in expected ranges."""
    state = create_initial_state(Config(), seed=42)
    assert mx.all(state.A >= 0) and mx.all(state.A <= 1)
    assert mx.all(state.phase >= 0) and mx.all(state.phase < 2 * mx.pi)

def test_state_seeded_initialization():
    """Seeded region has elevated concentrations."""
    config = Config(grid_size=64)
    state = create_initial_state(config, seed=42, with_seed_region=True)
    center = state.B[28:36, 28:36]
    edge = state.B[0:8, 0:8]
    assert mx.mean(center) > mx.mean(edge)
```

### Diffusion Tests (`tests/test_diffusion.py`)

```python
def test_diffusion_uniform_field():
    """Uniform field has zero net diffusion."""
    config = Config(grid_size=32)
    state = CellState(
        A=mx.ones((32, 32)) * 0.5,
        B=mx.zeros((32, 32)),
        C=mx.zeros((32, 32)),
        phase=mx.zeros((32, 32)),
        bonds=mx.zeros((32, 32, 4))
    )
    dA, dB, dC = compute_diffusion(state, config)
    assert mx.allclose(dA, mx.zeros_like(dA), atol=1e-6)

def test_diffusion_gradient():
    """Gradient diffuses toward uniformity."""
    config = Config(grid_size=32, D_base=0.1)
    # Create left-right gradient
    A = mx.broadcast_to(mx.arange(32).reshape(1, 32) / 32, (32, 32))
    state = CellState(A=A, B=mx.zeros((32, 32)), ...)
    
    dA, _, _ = compute_diffusion(state, config)
    # Left side should increase, right side decrease
    assert mx.mean(dA[:, :16]) > 0
    assert mx.mean(dA[:, 16:]) < 0

def test_diffusion_membrane_blocking():
    """Bonds reduce diffusion rate."""
    config = Config(grid_size=32, D_base=0.1, alpha=0.9)
    
    # Create gradient with vertical wall of bonds
    A = mx.broadcast_to(mx.arange(32).reshape(1, 32) / 32, (32, 32))
    bonds = mx.zeros((32, 32, 4))
    # Add East-West bonds at column 16
    bonds = bonds.at[:, 15, 1].set(1)  # East bonds
    bonds = bonds.at[:, 16, 3].set(1)  # West bonds
    
    state_no_bonds = CellState(A=A, bonds=mx.zeros((32, 32, 4)), ...)
    state_with_bonds = CellState(A=A, bonds=bonds, ...)
    
    dA_no_bonds, _, _ = compute_diffusion(state_no_bonds, config)
    dA_with_bonds, _, _ = compute_diffusion(state_with_bonds, config)
    
    # Flux at boundary should be reduced
    flux_no_bonds = mx.abs(mx.mean(dA_no_bonds[:, 15:17]))
    flux_with_bonds = mx.abs(mx.mean(dA_with_bonds[:, 15:17]))
    assert flux_with_bonds < flux_no_bonds

def test_diffusion_mass_conservation():
    """Total mass is conserved during diffusion."""
    config = Config(grid_size=32)
    state = create_initial_state(config, seed=42)
    
    mass_before = mx.sum(state.A) + mx.sum(state.B) + mx.sum(state.C)
    dA, dB, dC = compute_diffusion(state, config)
    mass_change = mx.sum(dA) + mx.sum(dB) + mx.sum(dC)
    
    assert mx.abs(mass_change) < 1e-6
```

### Reaction Tests (`tests/test_reactions.py`)

```python
def test_phase_gate_peak():
    """Phase gate peaks at target phase."""
    phase = mx.linspace(0, 2 * mx.pi, 100)
    target = mx.pi
    kappa = 2.0
    
    gate = phase_gate(phase, target, kappa)
    peak_idx = mx.argmax(gate)
    
    # Peak should be near target
    assert mx.abs(phase[peak_idx] - target) < 0.1

def test_phase_gate_width():
    """Higher kappa gives sharper gate."""
    phase = mx.linspace(0, 2 * mx.pi, 100)
    target = mx.pi
    
    gate_sharp = phase_gate(phase, target, kappa=5.0)
    gate_broad = phase_gate(phase, target, kappa=1.0)
    
    # Sharp gate should have smaller width
    width_sharp = mx.sum(gate_sharp > 0.5)
    width_broad = mx.sum(gate_broad > 0.5)
    assert width_sharp < width_broad

def test_reaction_stoichiometry():
    """Reactions conserve elements (approximately, with dissipation)."""
    config = Config(epsilon=0.0)  # Disable dissipation for this test
    
    state = CellState(
        A=mx.ones((32, 32)) * 0.5,
        B=mx.ones((32, 32)) * 0.3,
        C=mx.ones((32, 32)) * 0.2,
        phase=mx.zeros((32, 32)),  # All at phase 0 (favors reaction 1)
        bonds=mx.zeros((32, 32, 4))
    )
    
    dA, dB, dC = compute_reactions(state, config)
    
    # Without dissipation, total should be conserved
    # (accounting for stoichiometric coefficients)
    total_change = mx.sum(dA) + mx.sum(dB) + mx.sum(dC)
    assert mx.abs(total_change) < 1e-4

def test_reaction_phase_sensitivity():
    """Reactions only fire near target phase."""
    config = Config(kappa=3.0)
    
    # At target phase
    state_on = CellState(phase=mx.zeros((32, 32)), ...)  # φ = 0
    # Far from target
    state_off = CellState(phase=mx.ones((32, 32)) * mx.pi, ...)  # φ = π
    
    dA_on, dB_on, dC_on = compute_reactions(state_on, config)
    dA_off, dB_off, dC_off = compute_reactions(state_off, config)
    
    # Reaction 1 (φ ≈ 0) should be stronger in state_on
    assert mx.sum(mx.abs(dA_on)) > mx.sum(mx.abs(dA_off)) * 2
```

### Phase Tests (`tests/test_phase.py`)

```python
def test_natural_frequency():
    """Isolated cells oscillate at ω₀."""
    config = Config(omega_0=0.1, K_phase=0.0)  # No coupling
    state = CellState(phase=mx.zeros((32, 32)), ...)
    
    dphase = compute_phase_update(state, config)
    
    assert mx.allclose(dphase, mx.ones_like(dphase) * config.omega_0, atol=1e-6)

def test_kuramoto_synchronization():
    """Coupled oscillators synchronize."""
    config = Config(omega_0=0.1, K_phase=0.5)
    
    # Start with random phases
    state = create_initial_state(config, seed=42)
    
    # Run for many steps
    for _ in range(1000):
        dphase = compute_phase_update(state, config)
        state.phase = (state.phase + dphase) % (2 * mx.pi)
    
    # Measure synchronization
    R = kuramoto_order_parameter(state.phase)
    assert R > 0.5  # Should be somewhat synchronized

def test_bond_enhanced_coupling():
    """Bonded cells couple more strongly."""
    config = Config(K_phase=0.5)
    
    # Two cells with phase difference
    # One pair bonded, one pair not
    # ... construct test case
    
    # Bonded pair should synchronize faster

def test_phase_wrapping():
    """Phase correctly wraps at 2π."""
    config = Config(omega_0=0.5)
    state = CellState(phase=mx.ones((32, 32)) * (2 * mx.pi - 0.1), ...)
    
    dphase = compute_phase_update(state, config)
    new_phase = (state.phase + dphase) % (2 * mx.pi)
    
    assert mx.all(new_phase >= 0)
    assert mx.all(new_phase < 2 * mx.pi)
```

### Bond Tests (`tests/test_bonds.py`)

```python
def test_bond_formation_conditions():
    """Bonds form when B high AND phase aligned."""
    config = Config(B_thresh=0.3, cos_thresh=0.7)
    
    # High B, aligned phase - should form
    state_form = CellState(
        B=mx.ones((32, 32)) * 0.5,
        phase=mx.zeros((32, 32)),  # All aligned
        ...
    )
    
    # Low B - should not form
    state_low_B = CellState(
        B=mx.ones((32, 32)) * 0.1,
        phase=mx.zeros((32, 32)),
        ...
    )
    
    # Misaligned phase - should not form
    phase_random = mx.random.uniform(0, 2 * mx.pi, (32, 32))
    state_misaligned = CellState(
        B=mx.ones((32, 32)) * 0.5,
        phase=phase_random,
        ...
    )
    
    # Test formation probabilities
    ...

def test_bond_symmetry():
    """Bonds are always symmetric."""
    config = Config()
    state = create_initial_state(config)
    
    # Run some steps
    for _ in range(100):
        new_bonds = compute_bond_updates(state, config, rng_key)
        state.bonds = new_bonds
    
    # Check symmetry: bond[i,j,E] == bond[i,j+1,W]
    east_bonds = state.bonds[:, :-1, 1]  # East bonds
    west_bonds = state.bonds[:, 1:, 3]   # West bonds of neighbors
    assert mx.all(east_bonds == west_bonds)
```

---

## Integration Tests

### Conservation Laws (`tests/test_conservation.py`)

```python
def test_mass_conservation_short_run():
    """Mass is approximately conserved over short runs."""
    config = Config(epsilon=0.001, grid_size=64)  # Small dissipation
    sim = Simulation(config, seed=42)
    
    mass_initial = total_mass(sim.state)
    sim.run(100)
    mass_final = total_mass(sim.state)
    
    # Allow for small dissipation
    expected_loss = config.epsilon * 100 * 64 * 64  # Rough estimate
    actual_loss = mass_initial - mass_final
    
    assert actual_loss < expected_loss * 2  # Within factor of 2

def test_no_nan_or_inf():
    """Simulation doesn't produce NaN or Inf."""
    config = Config(grid_size=64)
    sim = Simulation(config, seed=42)
    
    sim.run(1000)
    
    assert not mx.any(mx.isnan(sim.state.A))
    assert not mx.any(mx.isinf(sim.state.A))
    # ... check all state components
```

### Emergence Tests (`tests/test_emergence.py`)

```python
def test_phase_domains_emerge():
    """Phase synchronization domains emerge from random initial conditions."""
    config = Config(grid_size=128, K_phase=0.5)
    sim = Simulation(config, seed=42)
    
    R_initial = kuramoto_order_parameter(sim.state.phase)
    sim.run(500)
    R_final = kuramoto_order_parameter(sim.state.phase)
    
    # Global sync may not occur, but local domains should
    # Measure local sync instead
    assert R_final > R_initial  # Some increase in order

def test_bonds_form_and_persist():
    """Bond clusters form and persist over time."""
    config = Config(grid_size=64)
    sim = Simulation(config, seed=42)
    
    # Run until bonds form
    sim.run(500)
    bonds_mid = mx.sum(sim.state.bonds)
    
    # Run more
    sim.run(500)
    bonds_final = mx.sum(sim.state.bonds)
    
    assert bonds_mid > 0  # Bonds formed
    assert bonds_final > bonds_mid * 0.5  # Most persist
```

---

## Performance Tests

### Benchmark (`tests/test_performance.py`)

```python
import time

def test_step_performance():
    """Single step completes in reasonable time."""
    config = Config(grid_size=256)
    sim = Simulation(config)
    
    # Warm up (JIT compilation)
    sim.step()
    mx.eval(sim.state.A)  # Force evaluation
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        sim.step()
    mx.eval(sim.state.A)
    elapsed = time.time() - start
    
    steps_per_second = 100 / elapsed
    print(f"Steps/second: {steps_per_second:.1f}")
    
    assert steps_per_second > 50  # At least 50 steps/s for 256x256

def test_memory_stability():
    """Memory usage doesn't grow over time."""
    import tracemalloc
    
    config = Config(grid_size=128)
    sim = Simulation(config)
    
    # Warm up
    sim.run(100)
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    sim.run(1000)
    
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare memory
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_growth = sum(stat.size_diff for stat in stats)
    
    # Allow some growth but not unbounded
    assert total_growth < 10 * 1024 * 1024  # Less than 10 MB growth
```

---

## Visual Regression Tests

### Snapshot Tests (`tests/test_visualization.py`)

```python
def test_rgb_visualization():
    """RGB visualization produces valid image."""
    config = Config(grid_size=64)
    state = create_initial_state(config, seed=42)
    
    image = state_to_rgb(state)
    
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert np.all(image >= 0) and np.all(image <= 255)

def test_phase_colormap_cyclic():
    """Phase colormap is cyclic (0 ≈ 2π)."""
    config = Config(grid_size=64)
    
    state_0 = CellState(phase=mx.zeros((64, 64)), ...)
    state_2pi = CellState(phase=mx.ones((64, 64)) * (2 * mx.pi - 0.01), ...)
    
    image_0 = state_to_phase_image(state_0)
    image_2pi = state_to_phase_image(state_2pi)
    
    # Should be very similar colors
    diff = np.mean(np.abs(image_0.astype(float) - image_2pi.astype(float)))
    assert diff < 10  # Small color difference
```

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=pccs --cov-report=html

# Specific category
pytest tests/test_diffusion.py -v

# Performance tests only (may be slow)
pytest tests/test_performance.py -v

# Skip slow tests
pytest -m "not slow"
```

## Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-14  # Apple Silicon runner
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=pccs
```
