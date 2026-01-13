"""
Phase-gated reaction system for PCCS.

Mass-conserving reaction cycle:
  R1: 2A → 2B       (dimerization)
  R2: 2B → A + C    (breakdown, releases catalyst)
  R3: A + C → 2A    (autocatalysis)

Each reaction is gated by phase, only firing efficiently near its target phase.
Total mass (A + B + C) is conserved by the reactions themselves.
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

    Mass-conserving reaction system:
        R1: 2A → 2B          (φ ≈ 0)        Dimerization
        R2: 2B → A + C       (φ ≈ 2π/3)    Breakdown (releases catalyst)
        R3: A + C → 2A       (φ ≈ 4π/3)    Autocatalysis

    Mass is conserved: dA + dB + dC = 0 (before dissipation).
    The epsilon term adds optional dissipation for thermodynamic realism.

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
    # R1: 2A → 2B (dimerization) - uses per-cell k1 for evolution
    r1 = state.k1 * state.A * state.A * G1

    # R2: 2B → A + C (breakdown)
    r2 = config.k2 * state.B * state.B * G2

    # R3: A + C → 2A (autocatalysis)
    r3 = config.k3 * state.A * state.C * G3

    # Compute concentration changes from stoichiometry
    # R1: 2A → 2B        => dA = -2, dB = +2
    # R2: 2B → A + C     => dB = -2, dA = +1, dC = +1
    # R3: A + C → 2A     => dA = +1 (net), dC = -1
    dA = -2.0 * r1 + r2 + r3
    dB = 2.0 * r1 - 2.0 * r2
    dC = r2 - r3

    # Apply energy dissipation (small loss from total mass)
    # This is the only source of mass loss
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

    # R1: 2A → 2B - uses per-cell k1 for evolution
    r1 = state.k1 * state.A * state.A * G1
    # R2: 2B → A + C
    r2 = config.k2 * state.B * state.B * G2
    # R3: A + C → 2A
    r3 = config.k3 * state.A * state.C * G3

    return r1, r2, r3
