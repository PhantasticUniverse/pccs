# Implementation Phases

This document provides a phased implementation plan for PCCS, designed to be followed by Claude Code or a human developer. Each phase builds on the previous one, with clear verification criteria.

---

## Phase 1: Foundation

**Goal**: Basic project structure, configuration, and grid state management.

### Files to Create

1. **`src/pccs/__init__.py`**
   ```python
   __version__ = "0.1.0"
   from .config import Config
   from .state import CellState, create_initial_state
   from .simulation import Simulation
   ```

2. **`src/pccs/config.py`**
   - Dataclass `Config` with all parameters from PRD Appendix A
   - Default values as specified
   - Validation methods (e.g., ensure 0 < alpha < 1)
   - `to_dict()` and `from_dict()` for serialization

3. **`src/pccs/state.py`**
   - Dataclass `CellState` containing:
     - `A`: mx.array [H, W] - Precursor concentration
     - `B`: mx.array [H, W] - Structural concentration
     - `C`: mx.array [H, W] - Catalyst concentration
     - `phase`: mx.array [H, W] - Oscillator phase
     - `bonds`: mx.array [H, W, 4] - Bond states (N, E, S, W)
   - `create_initial_state(config: Config, seed: int = None) -> CellState`
     - Random initialization as per PRD
     - Optional seed region for reproducibility

### Verification

```bash
pytest tests/test_config.py -v
pytest tests/test_state.py -v
```

- [ ] Config loads with defaults
- [ ] State creates with correct shapes
- [ ] All arrays are on GPU (`mx.default_device()`)
- [ ] Seed initialization creates localized high-concentration region

---

## Phase 2: Diffusion Dynamics

**Goal**: Implement membrane-gated diffusion using convolution.

### Files to Create/Modify

1. **`src/pccs/diffusion.py`**
   ```python
   def compute_diffusion(state: CellState, config: Config) -> tuple[mx.array, mx.array, mx.array]:
       """
       Compute diffusion fluxes for A, B, C.
       
       Returns:
           Tuple of (dA, dB, dC) changes to apply
       """
   ```

### Implementation Details

- Use `mx.conv2d` for neighbor averaging
- Create diffusion kernel that accounts for bond states:
  ```python
  # For each neighbor direction, weight = (1 - alpha * bond_state)
  # Then apply standard Laplacian diffusion
  ```
- Handle toroidal boundary with padding mode

### Key Equations

```
ΔSᵢ = D_base × Σⱼ (1 - α×βᵢⱼ) × (Sⱼ - Sᵢ) / |N|
```

### Verification

```bash
pytest tests/test_diffusion.py -v
```

- [ ] Uniform field remains uniform (no spurious flux)
- [ ] Gradient field diffuses toward uniformity
- [ ] Bonds reduce diffusion rate (compare with/without bonds)
- [ ] Total mass is conserved (within numerical tolerance)

---

## Phase 3: Phase-Gated Reactions

**Goal**: Implement the three-reaction system with phase gating.

### Files to Create/Modify

1. **`src/pccs/reactions.py`**
   ```python
   def phase_gate(phase: mx.array, target: float, kappa: float) -> mx.array:
       """Compute phase gating function G(φ, φ_target)"""
       return mx.exp(-kappa * (1 - mx.cos(phase - target)))
   
   def compute_reactions(state: CellState, config: Config) -> tuple[mx.array, mx.array, mx.array]:
       """
       Compute reaction fluxes for A, B, C.
       
       Reactions:
           2A + C → B + ε    (φ ≈ 0)
           2B → C + ε        (φ ≈ 2π/3)
           C + A → 2A        (φ ≈ 4π/3)
       
       Returns:
           Tuple of (dA, dB, dC) changes to apply
       """
   ```

### Implementation Details

- Phase targets: 0, 2π/3, 4π/3
- Reaction rates follow mass action kinetics
- Apply energy dissipation term ε

### Verification

```bash
pytest tests/test_reactions.py -v
```

- [ ] Reactions fire maximally at target phase
- [ ] Reactions suppressed away from target phase
- [ ] Stoichiometry is correct (check element balance)
- [ ] Energy dissipation reduces total mass over time

---

## Phase 4: Kuramoto Phase Dynamics

**Goal**: Implement coupled oscillator dynamics.

### Files to Create/Modify

1. **`src/pccs/phase.py`**
   ```python
   def compute_phase_update(state: CellState, config: Config) -> mx.array:
       """
       Compute phase velocity for each cell.
       
       dφ/dt = ω₀ + K_phase × Σⱼ wᵢⱼ × sin(φⱼ - φᵢ) + χ × (C - C̄)
       
       Returns:
           dphase: mx.array [H, W] - Phase change to apply
       """
   ```

### Implementation Details

- Coupling weight wᵢⱼ = 1 + bonds[i,j] (stronger for bonded pairs)
- Use convolution to compute neighbor phase differences
- Chemical coupling: faster oscillation where C is above average

### Verification

```bash
pytest tests/test_phase.py -v
```

- [ ] Isolated cells oscillate at natural frequency ω₀
- [ ] Coupled cells synchronize (measure order parameter)
- [ ] Bonded cells synchronize faster than unbonded
- [ ] Phase wraps correctly at 2π

---

## Phase 5: Bond Formation and Breaking

**Goal**: Implement dual-condition bond dynamics.

### Files to Create/Modify

1. **`src/pccs/bonds.py`**
   ```python
   def compute_bond_updates(state: CellState, config: Config, rng_key: mx.array) -> mx.array:
       """
       Compute probabilistic bond updates.
       
       P = σ(θ_B × (Bᵢ + Bⱼ - 2×B_thresh)) × σ(θ_φ × (cos(φᵢ - φⱼ) - cos_thresh))
       
       Returns:
           new_bonds: mx.array [H, W, 4] - Updated bond states
       """
   ```

### Implementation Details

- Sigmoid function: `mx.sigmoid(x)`
- Check both B concentration AND phase alignment
- Bonds must be symmetric (update both directions)
- Handle boundary wrapping for edge bonds

### Verification

```bash
pytest tests/test_bonds.py -v
```

- [ ] Bonds form when B high AND phase aligned
- [ ] Bonds break when either condition fails
- [ ] Bonds are always symmetric
- [ ] Bond count increases then stabilizes over time

---

## Phase 6: Simulation Integration

**Goal**: Combine all components into coherent simulation loop.

### Files to Create/Modify

1. **`src/pccs/simulation.py`**
   ```python
   class Simulation:
       def __init__(self, config: Config, seed: int = None):
           self.config = config
           self.state = create_initial_state(config, seed)
           self.step_count = 0
           self.rng_key = mx.random.key(seed or 42)
       
       def step(self) -> None:
           """Advance simulation by one timestep."""
           # 1. Compute reaction fluxes
           # 2. Apply reactions
           # 3. Compute diffusion fluxes
           # 4. Apply diffusion
           # 5. Update phase
           # 6. Update bonds
           # 7. Clamp and wrap
           self.step_count += 1
       
       def run(self, steps: int, callback: Callable = None) -> None:
           """Run simulation for multiple steps."""
   ```

### Update Order (Critical)

1. Compute reaction rates (phase-gated)
2. Apply reactions to concentrations
3. Compute diffusion fluxes
4. Apply diffusion
5. Update phase dynamics
6. Update bond states
7. Clamp concentrations to [0, 1]
8. Wrap phase to [0, 2π)

### Verification

```bash
pytest tests/test_simulation.py -v
```

- [ ] Simulation runs without errors for 1000 steps
- [ ] State shapes remain constant
- [ ] No NaN or Inf values appear
- [ ] Memory usage is stable (no leaks)

---

## Phase 7: Visualization

**Goal**: Real-time and export visualization.

### Files to Create/Modify

1. **`src/pccs/visualization.py`**
   ```python
   def state_to_rgb(state: CellState) -> np.ndarray:
       """Convert state to RGB image (A=blue, B=green, C=red)."""
   
   def state_to_phase_image(state: CellState) -> np.ndarray:
       """Convert phase to HSV cyclic colormap."""
   
   def overlay_bonds(image: np.ndarray, bonds: mx.array) -> np.ndarray:
       """Draw bond edges on image."""
   
   class Visualizer:
       def __init__(self, sim: Simulation, fps: int = 30):
           ...
       
       def show_live(self) -> None:
           """Display real-time visualization."""
       
       def save_frame(self, path: str) -> None:
           """Save current frame as image."""
   ```

### Visualization Modes

1. **Concentration RGB**: R=C, G=B, B=A
2. **Phase HSV**: Hue from phase, saturation from total concentration
3. **Bond overlay**: White edges where bonds exist
4. **Composite**: Side-by-side concentration and phase

### Verification

- [ ] Images render without errors
- [ ] Phase colormap is cyclic (0 and 2π same color)
- [ ] Bond overlay correctly shows connections

---

## Phase 8: Metrics and Analysis

**Goal**: Quantitative analysis tools.

### Files to Create/Modify

1. **`src/pccs/metrics.py`**
   ```python
   def kuramoto_order_parameter(phase: mx.array) -> float:
       """
       Compute global phase synchronization.
       R = |1/N × Σⱼ exp(i×φⱼ)|
       R = 1 means perfect sync, R ≈ 0 means random.
       """
   
   def detect_closed_membranes(bonds: mx.array) -> list[set[tuple[int, int]]]:
       """Find closed loops in bond graph."""
   
   def measure_compartment_stats(state: CellState, membranes: list) -> dict:
       """Compute internal vs external concentrations for each membrane."""
   
   def total_mass(state: CellState) -> float:
       """Sum of all A + B + C."""
   ```

### Verification

```bash
pytest tests/test_metrics.py -v
```

- [ ] Order parameter = 1 for uniform phase
- [ ] Order parameter ≈ 0 for random phase
- [ ] Closed membrane detection finds obvious loops
- [ ] Mass calculation matches expected conservation

---

## Phase 9: CLI and Main Entry Point

**Goal**: Command-line interface for running simulations.

### Files to Create/Modify

1. **`src/pccs/main.py`**
   ```python
   def main():
       parser = argparse.ArgumentParser(description="PCCS Simulation")
       parser.add_argument("--grid-size", type=int, default=256)
       parser.add_argument("--steps", type=int, default=10000)
       parser.add_argument("--visualize", action="store_true")
       parser.add_argument("--fps", type=int, default=30)
       parser.add_argument("--save-frames", type=str, default=None)
       parser.add_argument("--seed", type=int, default=None)
       parser.add_argument("--check-conservation", action="store_true")
       parser.add_argument("--measure-sync", action="store_true")
       parser.add_argument("--detect-membranes", action="store_true")
       # ... all config parameters as CLI args
       
       args = parser.parse_args()
       config = Config.from_args(args)
       sim = Simulation(config, seed=args.seed)
       
       if args.visualize:
           viz = Visualizer(sim, fps=args.fps)
           viz.show_live()
       else:
           sim.run(args.steps)
   
   if __name__ == "__main__":
       main()
   ```

### Verification

```bash
python -m pccs.main --help
python -m pccs.main --steps 100 --grid-size 64
```

- [ ] Help text displays all options
- [ ] Simulation runs with CLI arguments
- [ ] Visualization launches when requested

---

## Phase 10: Resource Injection (Environmental Dynamics)

**Goal**: Add resource replenishment to prevent system death.

### Addition to Config

```python
@dataclass
class Config:
    # ... existing params ...
    
    # Resource injection
    injection_mode: str = "boundary"  # "boundary", "uniform", "point_sources"
    injection_rate: float = 0.01     # Rate of A injection
    injection_width: int = 5         # Width of boundary injection zone
```

### Implementation Options

1. **Boundary injection**: A flows in from edges
2. **Uniform rain**: Small constant A addition everywhere
3. **Point sources**: Localized nutrient wells

### Add to Simulation Step

```python
def _inject_resources(self) -> None:
    """Replenish environmental A."""
    if self.config.injection_mode == "boundary":
        # Add A to edge cells
        ...
```

### Verification

- [ ] System survives indefinitely with injection
- [ ] Protocells migrate toward resource sources
- [ ] Competition emerges when resources are scarce

---

## Completion Checklist

### Tier 1: Basic Functionality
- [ ] Config and State management
- [ ] Diffusion with membrane gating
- [ ] Phase-gated reactions
- [ ] Kuramoto phase dynamics
- [ ] Bond formation/breaking
- [ ] Simulation loop runs

### Tier 2: Quality
- [ ] Visualization working
- [ ] Metrics implemented
- [ ] CLI functional
- [ ] All tests passing
- [ ] Type hints complete

### Tier 3: Phenomena
- [ ] Phase synchronization domains emerge
- [ ] Closed membranes form
- [ ] Protocells persist > 500 steps
- [ ] Internal/external concentration difference

### Tier 4: Advanced (Aspirational)
- [ ] Protocell growth observed
- [ ] Division event captured
- [ ] Ecological dynamics with multiple protocells
