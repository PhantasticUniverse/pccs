"""
Membrane-gated diffusion dynamics for PCCS.

Diffusion is modulated by bond states: bonded cell pairs have reduced diffusion,
simulating membrane impermeability.
"""

import mlx.core as mx

from .config import Config
from .state import CellState, NORTH, EAST, SOUTH, WEST


def compute_diffusion(state: CellState, config: Config) -> tuple[mx.array, mx.array, mx.array]:
    """
    Compute diffusion fluxes for all substrates.
    
    The diffusion equation with membrane gating:
        ΔSᵢ = D_base × Σⱼ (1 - α×βᵢⱼ) × (Sⱼ - Sᵢ) / |N|
    
    where:
        - D_base is the base diffusion coefficient
        - α is the membrane impermeability factor
        - βᵢⱼ is the bond state between cells i and j
        - |N| is the neighborhood size (4 for von Neumann)
    
    Args:
        state: Current cell state
        config: Simulation configuration
    
    Returns:
        Tuple of (dA, dB, dC) concentration changes
    """
    H, W = state.shape
    
    # Get neighbor values using roll (toroidal boundary)
    def get_neighbors(arr: mx.array) -> dict[int, mx.array]:
        """Get neighbor arrays for each direction."""
        return {
            NORTH: mx.roll(arr, shift=-1, axis=0),  # Cell above
            SOUTH: mx.roll(arr, shift=1, axis=0),   # Cell below
            EAST: mx.roll(arr, shift=-1, axis=1),   # Cell right
            WEST: mx.roll(arr, shift=1, axis=1),    # Cell left
        }
    
    # Compute diffusion for a single substrate
    def diffuse_substrate(S: mx.array) -> mx.array:
        """Compute diffusion flux for substrate S."""
        neighbors = get_neighbors(S)
        
        # Initialize flux accumulator
        flux = mx.zeros_like(S)
        
        # Sum over all neighbors with bond-modulated weights
        for direction, S_neighbor in neighbors.items():
            # Weight is reduced where bonds exist
            bond_strength = state.bonds[:, :, direction]
            weight = 1.0 - config.alpha * bond_strength
            
            # Flux from this neighbor
            flux = flux + weight * (S_neighbor - S)
        
        # Scale by diffusion coefficient and normalize by neighborhood size
        return config.D_base * flux / 4.0
    
    # Compute fluxes for each substrate
    dA = diffuse_substrate(state.A)
    dB = diffuse_substrate(state.B)
    dC = diffuse_substrate(state.C)
    
    return dA, dB, dC


def compute_diffusion_convolution(
    state: CellState, config: Config
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Alternative implementation using convolution (may be faster for large grids).
    
    This uses a Laplacian kernel for diffusion, but requires more complex
    handling of the bond-modulated weights.
    
    Args:
        state: Current cell state
        config: Simulation configuration
    
    Returns:
        Tuple of (dA, dB, dC) concentration changes
    """
    # Standard Laplacian kernel (without bond modulation)
    # This is a simplified version - full implementation needs per-direction weights
    laplacian = mx.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=mx.float32).reshape(1, 1, 3, 3)
    
    # For convolution, we need [N, C, H, W] format
    def apply_laplacian(S: mx.array) -> mx.array:
        S_4d = S.reshape(1, 1, *S.shape)
        # Pad for toroidal boundary
        S_padded = mx.pad(S_4d, [(0, 0), (0, 0), (1, 1), (1, 1)], mode="wrap")
        result = mx.conv2d(S_padded, laplacian, padding=0)
        return result.reshape(S.shape)
    
    # Note: This simple version doesn't handle bond modulation
    # Full implementation would need directional kernels weighted by bonds
    
    dA = config.D_base * apply_laplacian(state.A) / 4.0
    dB = config.D_base * apply_laplacian(state.B) / 4.0
    dC = config.D_base * apply_laplacian(state.C) / 4.0
    
    return dA, dB, dC
