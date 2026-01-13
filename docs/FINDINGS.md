# PCCS Development Findings

This document summarizes the key insights discovered during the development of the Phase-Coupled Catalytic Substrate system.

## Summary

PCCS successfully demonstrates emergent protocell formation through the interplay of phase-gated chemistry and B-concentration-based membrane dynamics. The system produces spiral waves, closed membranes, and multi-protocell coexistence.

---

## What Worked

### 1. Mass-Conserving Reaction Stoichiometry

The original PRD proposed reactions that destroyed mass over time. The working system uses strictly mass-conserving stoichiometry:

```
R1: 2A → 2B          (φ ≈ 0)      Dimerization
R2: 2B → A + C       (φ ≈ 2π/3)   Breakdown
R3: A + C → 2A       (φ ≈ 4π/3)   Autocatalysis
```

**Key insight**: Every reaction converts 2 molecules in → 2 molecules out. This maintains total mass (A + B + C = constant) except for the small epsilon dissipation term.

### 2. B-Only Bond Formation

The original design required both B concentration AND phase alignment for bonds:

```python
# Original (didn't create closed membranes)
P_bond = sigmoid(θ_B × (B_i + B_j - 2×B_thresh)) × sigmoid(θ_φ × (cos(Δφ) - cos_thresh))
```

The working system uses B concentration only:

```python
# Working (creates closed membranes)
P_bond = sigmoid(θ_B × (B_i + B_j - 2×B_thresh))
```

**Key insight**: Phase-aligned bonds fill domain interiors because bonds form where phases MATCH. B-only bonds create boundaries where B concentration is HIGH, which follows the resource injection pattern.

### 3. Point Source Resource Injection

Protocells form around localized injection points. Different injection modes produce different structures:

| Mode | Result |
|------|--------|
| `boundary` | Ring membrane at grid edge |
| `center` | Single protocell in center |
| `dual` | Two independent protocells (far apart) |
| `competing` | Two protocells (close together, coexisting) |

---

## What Didn't Work

### 1. Original PRD Stoichiometry

```
R1: 2A + C → B + ε    (destroys 2 molecules)
R2: 2B → C + ε        (destroys 1 molecule)
R3: C + A → 2A        (creates 1 molecule)
```

**Problem**: Net mass loss per cycle. System collapses to empty state within ~1000 steps.

### 2. Phase-Aligned Bonds for Membrane Formation

With dual-condition bonds (B AND phase), bonds formed INSIDE synchronized domains, not at boundaries.

**Analysis**: Correlation between bond density and phase gradient was r = -0.79 (strong negative). Bonds form where phase gradient is LOW (inside domains), not HIGH (at boundaries).

### 3. Multi-Seed Domain Separation

Initializing with multiple seeds at different phases (e.g., 0, π/2, π, 3π/2) did not create persistent separate domains.

**Problem**: Kuramoto coupling (K_phase = 0.5) pulls phases toward synchronization. Within ~1000 steps, all seeds merge into a single spiral pattern.

---

## Key Insights

### 1. Bonds Form WHERE Their Condition Is Met

This seems obvious but has profound implications:

- **Phase-aligned bonds** → form inside synchronized domains (cytoplasm, not membrane)
- **B-concentration bonds** → form where B is high (follows B production patterns)
- **If you want boundaries**, the bond condition must be HIGH at boundaries

### 2. Membrane Pattern = Injection Pattern

With B-only bonds, membranes form around high-B regions. Since B is produced from A (R1: 2A → 2B), and A is injected, the membrane pattern mirrors the injection pattern:

- Boundary injection → ring membrane at boundary
- Center injection → membrane surrounding center
- Two point sources → two separate membranes

### 3. Phase and Chemistry Are Decoupled in Membrane Formation

The spiral wave dynamics (phase) are essentially independent of the membrane dynamics (bonds). The phase field shows a single coupled spiral even when there are two separate protocells. The protocells' identity comes from the B distribution, not the phase distribution.

### 4. Protocells Can Coexist at Close Range

Even when injection points are at 1/3 and 2/3 across the grid (competing mode), both protocells maintain enclosure. They share the global phase dynamics but maintain separate chemical identities through their B-based membranes.

---

## Visualizations

| Image | Description |
|-------|-------------|
| ![Spiral Waves](assets/spiral_waves.gif) | Emergent spiral wave dynamics from phase-gated reactions |
| ![Bond Correlation](assets/bond_spiral_correlation.png) | Bond density vs phase gradient (r = -0.79) |
| ![Membrane Ring](assets/membrane_ring_closure.png) | First closed membrane (boundary injection) |
| ![Protocell](assets/protocell_center.png) | Single protocell with center injection |
| ![Two Protocells](assets/two_protocells.png) | Two independent protocells (dual injection) |
| ![Competing](assets/competing_protocells.png) | Close protocells coexisting (competing injection) |

---

## Open Questions

1. **Can protocells divide?** What would cause a single protocell to split into two?

2. **Resource competition**: With limited total A, do protocells compete for resources? Does one grow while another shrinks?

3. **Membrane dynamics**: Can membranes exchange material? Merge? Break?

4. **Phase-membrane coupling**: The current design has decoupled phase and membrane. Could a different design couple them productively?

5. **Scaling**: What happens with many (10+) protocells? Do they tile the space? Form hierarchies?

---

## Parameter Summary

Working parameter set for protocell formation:

```python
Config(
    grid_size=48,
    D_base=0.1,
    alpha=0.9,
    k1=0.05,           # Reduced from PRD 0.1
    k2=0.05,           # Reduced from PRD 0.1
    k3=0.01,           # Reduced from PRD 0.1
    kappa=2.0,
    epsilon=0.001,     # Reduced from PRD 0.01
    omega_0=0.1,
    K_phase=0.5,
    chi=0.2,
    B_thresh=0.25,     # Lowered for easier membrane formation
    injection_mode="center",  # or "competing", "dual"
    injection_rate=0.02,
    injection_width=3,
)
```

---

## Conclusion

PCCS demonstrates that simple rules can produce complex, life-like structures. The key breakthrough was recognizing that bond formation conditions determine WHERE structures form, not just WHETHER they form. By removing the phase requirement from bonds, we enabled membranes to form at B-concentration boundaries rather than inside phase domains.

The system now reliably produces:
- Spiral wave dynamics
- Closed membrane structures (protocells)
- Multi-protocell coexistence

Future work could explore protocell division, competition, and more complex injection patterns.
