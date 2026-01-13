"""
Kuramoto coupled oscillator dynamics for PCCS.

Phase dynamics create synchronized domains, with bonded cells coupling more strongly.
"""

import mlx.core as mx

from .config import Config
from .state import CellState, NORTH, EAST, SOUTH, WEST


def compute_phase_update(state: CellState, config: Config) -> mx.array:
    """
    Compute phase velocity for each cell.
    
    dφ/dt = ω₀ + K_phase × Σⱼ wᵢⱼ × sin(φⱼ - φᵢ) + χ × (C - C̄)
    
    where:
        - ω₀ is the natural oscillation frequency
        - K_phase is the coupling strength
        - wᵢⱼ = 1 + bonds[i,j] (stronger coupling for bonded pairs)
        - χ is the chemical-phase coupling coefficient
        - C̄ is the mean catalyst concentration
    
    Args:
        state: Current cell state
        config: Simulation configuration
    
    Returns:
        dphase: Phase velocity for each cell [H, W]
    """
    H, W = state.shape
    
    # Start with natural frequency
    dphase = mx.ones((H, W)) * config.omega_0
    
    # Get neighbor phases using roll (toroidal boundary)
    neighbor_phases = {
        NORTH: mx.roll(state.phase, shift=-1, axis=0),
        SOUTH: mx.roll(state.phase, shift=1, axis=0),
        EAST: mx.roll(state.phase, shift=-1, axis=1),
        WEST: mx.roll(state.phase, shift=1, axis=1),
    }
    
    # Compute Kuramoto coupling term
    coupling = mx.zeros((H, W))
    
    for direction, phi_neighbor in neighbor_phases.items():
        # Coupling weight: 1 for unbonded, 2 for bonded
        bond_strength = state.bonds[:, :, direction]
        weight = 1.0 + bond_strength
        
        # Kuramoto coupling: sin(φⱼ - φᵢ)
        phase_diff = phi_neighbor - state.phase
        coupling = coupling + weight * mx.sin(phase_diff)
    
    # Add coupling term (scaled by K_phase and normalized)
    dphase = dphase + config.K_phase * coupling / 4.0
    
    # Chemical-phase coupling: cells with high C oscillate faster
    C_mean = mx.mean(state.C)
    dphase = dphase + config.chi * (state.C - C_mean)
    
    return dphase


def wrap_phase(phase: mx.array) -> mx.array:
    """
    Wrap phase values to [0, 2π).
    
    Args:
        phase: Phase array (may have values outside [0, 2π))
    
    Returns:
        Phase wrapped to [0, 2π)
    """
    two_pi = 2.0 * mx.pi
    return mx.mod(phase, two_pi)


def kuramoto_order_parameter(phase: mx.array) -> float:
    """
    Compute the Kuramoto order parameter for global synchronization.
    
    R = |1/N × Σⱼ exp(i×φⱼ)|
    
    R = 1 means perfect synchronization
    R ≈ 0 means random/incoherent phases
    
    Args:
        phase: Phase array [H, W]
    
    Returns:
        Order parameter R in [0, 1]
    """
    # Compute complex order parameter
    exp_i_phi = mx.exp(1j * phase.astype(mx.complex64))
    mean_exp = mx.mean(exp_i_phi)
    
    # Return magnitude
    R = mx.abs(mean_exp)
    return float(R)


def local_order_parameter(phase: mx.array, window_size: int = 5) -> mx.array:
    """
    Compute local Kuramoto order parameter for each cell.
    
    This measures synchronization in a local neighborhood.
    
    Args:
        phase: Phase array [H, W]
        window_size: Size of local window
    
    Returns:
        Local order parameter at each cell [H, W]
    """
    # Convert to complex exponential
    exp_i_phi = mx.exp(1j * phase.astype(mx.complex64))
    
    # Create averaging kernel
    kernel_size = window_size
    kernel = mx.ones((1, 1, kernel_size, kernel_size), dtype=mx.complex64)
    kernel = kernel / (kernel_size * kernel_size)
    
    # Pad for convolution
    pad = kernel_size // 2
    exp_padded = mx.pad(
        exp_i_phi.reshape(1, 1, *exp_i_phi.shape),
        [(0, 0), (0, 0), (pad, pad), (pad, pad)],
        mode="wrap"
    )
    
    # Compute local mean
    local_mean = mx.conv2d(exp_padded, kernel, padding=0)
    local_mean = local_mean.reshape(phase.shape)
    
    # Return magnitude
    return mx.abs(local_mean).astype(mx.float32)
