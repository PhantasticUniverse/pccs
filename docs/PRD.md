# Product Requirements Document

# Phase-Coupled Catalytic Substrate (PCCS)

**A Novel Cellular Automaton for Emergent Life-Like Complexity**

Version 1.0 • January 2026  
Designed by Claude (Anthropic)

---

## Executive Summary

This document specifies the Phase-Coupled Catalytic Substrate (PCCS), a novel cellular automaton designed to exhibit emergent life-like complexity through the interplay of three organizational axes: spatial structure (membranes and compartments), chemical dynamics (reactions and catalysis), and temporal organization (phase-coupled oscillations).

**The Core Insight:** Life is not just chemistry—it is *orchestrated* chemistry. Real metabolism operates through precisely timed reaction cascades, not random diffusion. PCCS captures this by making temporal phase relationships fundamental to the dynamics, creating conditions where sustained metabolism *requires* the spontaneous emergence of phase-locked compartmentalized structures.

PCCS represents a novel contribution to the artificial life literature by introducing phase dynamics as a primary organizational mechanism, distinct from prior approaches based purely on reaction-diffusion systems, continuous state cellular automata, or abstract artificial chemistries.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background and Motivation](#background-and-motivation)
3. [Design Philosophy](#design-philosophy)
4. [Technical Specification](#technical-specification)
5. [Implementation Guidelines](#implementation-guidelines)
6. [Expected Behaviors and Phenomena](#expected-behaviors-and-phenomena)
7. [Research Questions](#research-questions)
8. [Success Metrics](#success-metrics)
9. [Appendices](#appendices)

---

## Background and Motivation

### The Challenge of Emergent Life

Creating artificial systems that exhibit life-like properties remains one of the central challenges of artificial life research. While numerous approaches have produced remarkable results—from von Neumann's self-replicating automata to Langton's loops, from Lenia's continuous creatures to combinatory chemistry's emergent metabolisms—a significant gap remains between these systems and the layered, hierarchical complexity characteristic of biological life.

Existing approaches typically excel along one axis while underemphasizing others:

- **Discrete CA (Game of Life, Langton loops):** Excellent at producing computational structures and self-replication, but require carefully designed initial conditions and lack thermodynamic realism.

- **Continuous CA (Lenia, Flow-Lenia):** Produce beautiful, creature-like patterns with mass conservation, but lack genuine metabolic structure—the "creatures" are patterns, not processors.

- **Artificial Chemistry (Combinatory chemistry, Ono-Ikegami):** Can produce autocatalytic metabolisms and even protocells, but often lack spatial structure or produce either chaos or boring equilibrium.

### The Missing Ingredient: Temporal Organization

Real biological systems exhibit a crucial property that these approaches underemphasize: temporal coordination. Metabolism is not just a network of reactions—it is a system of oscillating processes that must be synchronized in time to function. Consider:

- Circadian rhythms coordinate cellular processes across 24-hour cycles
- The cell cycle organizes replication into discrete, ordered phases
- Calcium oscillations encode information through frequency modulation
- Glycolytic oscillations emerge spontaneously in yeast populations
- Neural oscillations coordinate information processing across brain regions

These are not epiphenomena—they are functional. The timing creates the organization. PCCS makes this insight central: reactions are gated by phase, and phase dynamics are coupled to chemistry and spatial structure.

### State of the Art

Recent advances that inform this design include:

- **Flow-Lenia (Plantec et al., 2025):** Introduced mass conservation to continuous CA, enabling genuine resource constraints and creature-like emergent dynamics.

- **Outlier Rule (2025):** Discovered through genetic programming, exhibits hierarchical self-replication across multiple spatial scales in binary CA.

- **Hierarchical NCA:** Demonstrated that explicit hierarchical organization improves evolvability for morphogenesis tasks.

- **Ono-Ikegami Protocells:** Showed that self-maintaining, self-reproducing membrane-bounded structures can emerge spontaneously from resource scarcity.

PCCS synthesizes insights from these approaches while adding phase-coupled dynamics as a novel organizational mechanism.

---

## Design Philosophy

### Guiding Principles

**1. Emergence Over Specification:** The system should not hard-code life-like properties. Membranes, metabolism, and organization should emerge from simple local rules, not be specified a priori.

**2. Thermodynamic Realism:** Resources must be conserved and finite. There must be genuine selection pressure—structures that waste resources should dissipate.

**3. Minimal Sufficient Complexity:** The rule set should be the simplest that admits the phenomena of interest. Every parameter should have a clear physical interpretation.

**4. Hierarchical Potential:** The dynamics should support emergence at multiple scales—patterns forming substrates for higher-order patterns.

**5. Temporal-Spatial Coupling:** Spatial and temporal organization should be mutually reinforcing—phase dynamics should affect spatial structure and vice versa.

### The Three Axes of Organization

PCCS is built around three coupled organizational axes:

| Axis | Mechanism | Emergent Structure |
|------|-----------|-------------------|
| Spatial | Bonds, diffusion gating | Membranes, compartments |
| Chemical | Reactions, catalysis | Metabolism, autocatalysis |
| Temporal | Phase oscillations, coupling | Synchronized domains, timing |

The key insight is that these axes are not independent—they form a mutually reinforcing triad. Membranes create compartments where phase can synchronize. Synchronized phase enables efficient metabolism. Efficient metabolism produces the structural molecules that form membranes.

---

## Technical Specification

### Grid Structure

PCCS operates on a 2D toroidal grid with von Neumann neighborhood (4-connectivity for bonds) and Moore neighborhood (8-connectivity for diffusion and reaction).

| Parameter | Value |
|-----------|-------|
| Grid Size | 256×256 (default), scalable |
| Topology | Toroidal (wrapping boundaries) |
| Bond Neighborhood | Von Neumann (N, E, S, W) |
| Diffusion Neighborhood | Moore (8 neighbors) |

### Cell State

Each cell maintains the following state variables:

#### Substrate Concentrations S = [A, B, C]

| Variable | Name | Physical Interpretation |
|----------|------|------------------------|
| A ∈ [0,1] | Precursor | Environmental resource, raw material for metabolism |
| B ∈ [0,1] | Structural | Membrane-forming molecule, intermediate metabolite |
| C ∈ [0,1] | Catalyst/Energy | Energy carrier, reaction catalyst, autocatalytic agent |

#### Phase φ ∈ [0, 2π)

Internal oscillator phase. Represents the cell's position in an abstract "metabolic cycle." Phase is continuous and wraps at 2π.

#### Bond States β = [βN, βE, βS, βW]

Binary bond states for each cardinal direction. βd ∈ {0, 1} where 1 indicates a bond exists with the neighbor in direction d. Bonds are symmetric—if cell (i,j) is bonded to cell (i,j+1), then cell (i,j+1) is also bonded to cell (i,j).

### Dynamics

The system evolves through four coupled processes applied each timestep. All updates use the current state to compute the next state (synchronous update).

#### 1. Diffusion

Substances diffuse between neighboring cells. The diffusion rate is modulated by bond state:

```
ΔSᵢ = D_base × Σⱼ (1 - α×βᵢⱼ) × (Sⱼ - Sᵢ) / |N|
```

where D_base is the base diffusion coefficient, α ∈ [0,1] is the membrane impermeability factor, βᵢⱼ is the bond state between cells i and j, and |N| is the neighborhood size.

| Parameter | Default | Physical Meaning |
|-----------|---------|------------------|
| D_base | 0.1 | Base diffusion rate |
| α (alpha) | 0.9 | Membrane impermeability |

#### 2. Reactions (Phase-Gated)

Reactions only occur efficiently when the local phase is near specific target values. The phase-gating function is:

```
G(φ, φ_target) = exp(-κ × (1 - cos(φ - φ_target)))
```

This creates Gaussian-like windows around target phases. The reaction set is:

| Reaction | Phase Gate | Rate | Interpretation |
|----------|------------|------|----------------|
| 2A + C → B + ε | φ ≈ 0 | r₁ = k₁·A²·C | Anabolism |
| 2B → C + ε | φ ≈ 2π/3 | r₂ = k₂·B² | Catabolism |
| C + A → 2A | φ ≈ 4π/3 | r₃ = k₃·C·A | Autocatalysis |

The ε terms represent small energy dissipation to prevent perpetual motion. The reaction cycle forms a closed loop: A → B → C → A, with C acting as both energy carrier and autocatalyst.

#### 3. Phase Dynamics

Each cell's phase evolves according to coupled oscillator dynamics:

```
dφ/dt = ω₀ + K_phase × Σⱼ wᵢⱼ × sin(φⱼ - φᵢ) + χ × (C - C̄)
```

where:

- ω₀ is the natural oscillation frequency
- K_phase is the coupling strength
- wᵢⱼ is the coupling weight (stronger for bonded neighbors)
- χ is the chemical-phase coupling coefficient
- C̄ is the mean catalyst concentration

This creates Kuramoto-style synchronization, with the twist that bonded cells couple more strongly. The chemical term allows local chemistry to modulate oscillation frequency.

#### 4. Bond Formation and Breaking

Bonds form when structural conditions and phase alignment are both satisfied:

```
P_form(i,j) = σ(θ_B × (Bᵢ + Bⱼ - 2×B_thresh)) × σ(θ_φ × (cos(φᵢ - φⱼ) - cos_thresh))
```

where σ is the sigmoid function. Bonds form probabilistically with probability P_form when currently unbonded.

Bonds break when either B concentration drops below threshold or phase desynchronizes:

```
P_break(i,j) = 1 - σ(θ_B × (Bᵢ + Bⱼ - 2×B_thresh)) × σ(θ_φ × (cos(φᵢ - φⱼ) - cos_thresh))
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| B_thresh | 0.3 | Minimum B for bond formation |
| cos_thresh | 0.7 | Minimum phase alignment for bonds |
| θ_B, θ_φ | 10.0 | Sigmoid steepness (sharpness of transitions) |

---

## Implementation Guidelines

### Computational Architecture

PCCS is well-suited to GPU acceleration due to its local update rules. The recommended implementation approach:

1. Store state as GPU tensors: [H, W, 3] for concentrations, [H, W] for phase, [H, W, 4] for bonds
2. Implement updates as convolution operations where possible
3. Use JAX or PyTorch for autodiff capability (enables parameter optimization)
4. Batch multiple simulations for parameter sweeps

### Update Order

Each timestep, apply updates in this order:

1. Compute reaction rates (phase-gated)
2. Apply reactions to concentrations
3. Compute diffusion fluxes
4. Apply diffusion
5. Update phase dynamics
6. Update bond states
7. Clamp concentrations to [0, 1]
8. Wrap phase to [0, 2π)

### Initialization Strategies

#### Random Initialization (Exploration)

For exploring emergent dynamics:

- A: Uniform random in [0.1, 0.5]
- B: Uniform random in [0.0, 0.2]
- C: Uniform random in [0.0, 0.2]
- φ: Uniform random in [0, 2π)
- β: All zeros (no initial bonds)

#### Seeded Initialization (Reproducibility)

For studying specific phenomena, seed a small region with elevated concentrations:

- Background: Low uniform A (0.1), zero B and C
- Seed region (e.g., 10×10 cells): Higher A (0.3), B (0.2), C (0.2)
- Phase: Random or spatially coherent in seed region

### Visualization

Recommended visualization mappings:

| Channel | RGB Mapping | Rationale |
|---------|-------------|-----------|
| A (Precursor) | Blue channel | Environmental resource |
| B (Structural) | Green channel | Membrane material |
| C (Catalyst) | Red channel | Metabolic activity |
| φ (Phase) | HSV hue (cyclic colormap) | Temporal organization |
| Bonds | Edge overlay or cell outline | Spatial structure |

Consider providing multiple visualization modes, including composite views and individual channel displays.

---

## Expected Behaviors and Phenomena

### Phase 1: Transient Dynamics (t < 100)

Initial random configurations should exhibit:

- Rapid diffusive mixing of concentrations
- Sporadic reaction activity at random phase alignments
- Occasional transient bond formation and breaking
- No persistent structure

### Phase 2: Local Organization (100 < t < 1000)

As the system evolves:

- Phase synchronization domains should emerge—regions of coherent oscillation
- Within synchronized domains, sustained reaction cycling becomes possible
- B accumulation in active regions enables bond formation
- Small membrane fragments should appear and disappear

### Phase 3: Protocell Emergence (t > 1000)

If parameters are tuned correctly:

- Closed membrane structures should form spontaneously
- Interior phase should synchronize strongly (protected from external noise)
- Sustained metabolic cycling should occur within compartments
- Resource depletion should create selection pressure

### Advanced Phenomena (Aspirational)

With appropriate parameters, the system may exhibit:

- **Protocell Growth:** Membrane expansion as metabolism produces structural molecules.

- **Protocell Division:** Large membranes becoming unstable and pinching off.

- **Proto-organelles:** Sub-domains within cells with distinct phase relationships.

- **Metabolic Specialization:** Different compartments specializing in different parts of the reaction cycle.

- **Ecological Dynamics:** Competition between protocells for limited environmental resources.

---

## Research Questions

PCCS is designed to enable investigation of fundamental questions about the emergence of life-like organization:

### Minimal Conditions

- What is the minimal parameter regime that supports stable protocell formation?
- How does the phase-gating mechanism affect the probability of autocatalytic emergence?
- Is there a critical density of initial resources below which organization cannot emerge?

### Hierarchical Organization

- Do proto-organelles emerge spontaneously, or do they require additional mechanisms?
- What determines the typical size of emergent compartments?
- Can nested levels of phase organization emerge (oscillations within oscillations)?

### Evolutionary Dynamics

- Does variation exist between protocells that could support selection?
- Can protocells "inherit" characteristics through division?
- Do we observe lineages with increasing fitness over time?

### Information and Computation

- Can phase patterns encode information that persists across metabolic cycles?
- Is the system capable of universal computation (in principle)?
- What is the relationship between compartment complexity and computational capacity?

---

## Success Metrics

### Tier 1: Basic Functionality

| Metric | Target |
|--------|--------|
| Phase synchronization | Observable coherent domains emerge |
| Reaction cycling | Sustained A→B→C→A flux in synchronized regions |
| Bond formation | Stable bond clusters form and persist |

### Tier 2: Emergent Structure

| Metric | Target |
|--------|--------|
| Closed membranes | At least one closed bond loop per 256×256 grid |
| Compartment persistence | Closed structures survive >500 timesteps |
| Internal homeostasis | Distinct internal vs external concentration profiles |

### Tier 3: Life-Like Dynamics

| Metric | Target |
|--------|--------|
| Protocell growth | Observable membrane expansion |
| Protocell division | At least one division event observed |
| Population dynamics | Multiple protocells coexisting/competing |

---

## Appendices

### Appendix A: Complete Parameter Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| D_base | 0.1 | [0.01, 0.5] | Base diffusion rate |
| α | 0.9 | [0.5, 0.99] | Membrane impermeability |
| k₁, k₂, k₃ | 0.1 | [0.01, 1.0] | Reaction rate constants |
| κ | 2.0 | [0.5, 5.0] | Phase gate sharpness |
| ω₀ | 0.1 | [0.01, 0.5] | Natural frequency |
| K_phase | 0.5 | [0.1, 2.0] | Phase coupling strength |
| χ | 0.2 | [0.0, 1.0] | Chemical-phase coupling |
| B_thresh | 0.3 | [0.1, 0.6] | Bond formation threshold |
| cos_thresh | 0.7 | [0.3, 0.95] | Phase alignment threshold |
| ε (epsilon) | 0.01 | [0.001, 0.1] | Energy dissipation rate |

### Appendix B: Pseudocode

Complete update step pseudocode:

```python
def update_step(state):
    A, B, C, phase, bonds = state

    # 1. Phase-gated reactions
    G1 = exp(-κ * (1 - cos(phase - 0)))
    G2 = exp(-κ * (1 - cos(phase - 2π/3)))
    G3 = exp(-κ * (1 - cos(phase - 4π/3)))

    r1 = k1 * A² * C * G1    # 2A + C → B
    r2 = k2 * B² * G2        # 2B → C
    r3 = k3 * C * A * G3     # C + A → 2A

    dA = -2*r1 + r3
    dB = r1 - 2*r2
    dC = r2 - r1 - r3 - ε*(A+B+C)

    # 2. Diffusion with membrane gating
    for each neighbor j:
        w = 1 - α * bonds[i,j]
        dA += D_base * w * (A[j] - A[i])
        dB += D_base * w * (B[j] - B[i])
        dC += D_base * w * (C[j] - C[i])

    # 3. Phase dynamics
    dphase = ω₀
    for each neighbor j:
        w = 1 + bonds[i,j]  # stronger coupling if bonded
        dphase += K_phase * w * sin(phase[j] - phase[i])
    dphase += χ * (C[i] - mean(C))

    # 4. Bond updates
    for each neighbor j:
        P = sigmoid(θ_B * (B[i] + B[j] - 2*B_thresh)) \
          * sigmoid(θ_φ * (cos(phase[i] - phase[j]) - cos_thresh))
        if bonds[i,j] == 0 and random() < P:
            bonds[i,j] = bonds[j,i] = 1
        elif bonds[i,j] == 1 and random() > P:
            bonds[i,j] = bonds[j,i] = 0

    # 5. Apply updates
    A = clamp(A + dA, 0, 1)
    B = clamp(B + dB, 0, 1)
    C = clamp(C + dC, 0, 1)
    phase = (phase + dphase) % (2π)

    return A, B, C, phase, bonds
```

### Appendix C: Novelty Statement

PCCS represents a novel contribution to the artificial life literature through the following innovations:

**1. Phase-Gated Chemistry:** Unlike prior artificial chemistry approaches, PCCS makes temporal phase a primary determinant of reaction rates. This is not merely an addition of oscillatory dynamics—it fundamentally changes the conditions for sustained metabolism.

**2. Dual-Condition Membrane Formation:** Bonds require both chemical conditions (B concentration) AND temporal conditions (phase alignment). This creates more robust membranes that naturally partition oscillating domains.

**3. Coupled Hierarchies:** The mutual reinforcement between spatial organization (membranes) and temporal organization (phase synchronization) creates conditions for hierarchical emergence without explicit specification.

**4. Thermodynamic Grounding:** The explicit energy dissipation term (ε) ensures that the system has genuine selection pressure—structures that waste energy dissipate, while efficient metabolizers persist.

This design draws inspiration from real biological systems (oscillatory metabolism, membrane compartmentalization, autocatalysis) while remaining simple enough for systematic analysis and efficient simulation.

---

*— End of Document —*
