"""
Phase-gated reaction system for PCCS.

The reaction cycle forms a closed loop: A → B → C → A
Each reaction is gated by phase, only firing efficiently near its target phase.
"""

import math

import mlx.core as mx

from .config import Config
from .state import CellState


def phase_gate(phase: mx.array, target: float, kappa: float) -> mx.array:
    """
    Compute phase gating function.
    
    G(φ, φ_target) = exp(-κ × (1 - cos(φ - φ_target)))
    
    This creates a Gaussian-like window around the target phase,
    with width controlled by kappa.
    
    Args:
        phase: Array of phase values [H, W]
        target: Target phase for this reaction
        kappa: Gate sharpness (higher = narrower window)
    
    Returns:
        Gate values in [0, 1] peaking at target phase
    """
    phase_diff = phase - target
    return mx.exp(-kappa * (1.0 - mx.cos(phase_diff)))


def compute_reactions(
    state: CellState, config: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Compute reaction fluxes for all substrates.
    
    Reaction system:
        R1: 2A + C → B + ε    (φ ≈ 0)        Anabolism
        R2: 2B → C + ε        (φ ≈ 2π/3)     Catabolism
        R3: C + A → 2A        (φ ≈ 4π/3)     Autocatalysis
    
    The ε term represents energy dissipation.
    
    Args:
        state: Current cell state
        config: Simulation configuration
    
    Returns:
        Tuple of (dA, dB, dC) concentration changes
    """
    # Get phase targets
    phi_1, phi_2, phi_3 = config.reaction_phases  # 0, 2π/3, 4π/3
    
    # Compute phase gates for each reaction
    G1 = phase_gate(state.phase, phi_1, config.kappa)
    G2 = phase_gate(state.phase, phi_2, config.kappa)
    G3 = phase_gate(state.phase, phi_3, config.kappa)
    
    # Compute reaction rates (mass action kinetics with phase gating)
    # R1: 2A + C → B
    r1 = config.k1 * state.A * state.A * state.C * G1
    
    # R2: 2B → C
    r2 = config.k2 * state.B * state.B * G2
    
    # R3: C + A → 2A (autocatalysis)
    r3 = config.k3 * state.C * state.A * G3
    
    # Compute concentration changes from stoichiometry
    # dA/dt from reactions:
    #   -2 from R1 (consumes 2A)
    #   +2 from R3 (produces 2A)
    #   -1 from R3 (consumes 1A)
    # Net: dA = -2*r1 + r3
    dA = -2.0 * r1 + r3
    
    # dB/dt from reactions:
    #   +1 from R1 (produces 1B)
    #   -2 from R2 (consumes 2B)
    dB = r1 - 2.0 * r2
    
    # dC/dt from reactions:
    #   -1 from R1 (consumes 1C)
    #   +1 from R2 (produces 1C)
    #   -1 from R3 (consumes 1C)
    dC = -r1 + r2 - r3
    
    # Apply energy dissipation (small loss from total mass)
    total = state.A + state.B + state.C
    dissipation = config.epsilon * total
    
    # Distribute dissipation proportionally
    dA = dA - config.epsilon * state.A
    dB = dB - config.epsilon * state.B
    dC = dC - config.epsilon * state.C
    
    return dA, dB, dC


def compute_reaction_rates(
    state: CellState, config: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Compute individual reaction rates (for analysis/visualization).
    
    Args:
        state: Current cell state
        config: Simulation configuration
    
    Returns:
        Tuple of (r1, r2, r3) reaction rates
    """
    phi_1, phi_2, phi_3 = config.reaction_phases
    
    G1 = phase_gate(state.phase, phi_1, config.kappa)
    G2 = phase_gate(state.phase, phi_2, config.kappa)
    G3 = phase_gate(state.phase, phi_3, config.kappa)
    
    r1 = config.k1 * state.A * state.A * state.C * G1
    r2 = config.k2 * state.B * state.B * G2
    r3 = config.k3 * state.C * state.A * G3
    
    return r1, r2, r3
