"""
Cell state representation and initialization for PCCS.

The state consists of:
- Three substrate concentrations (A, B, C)
- Oscillator phase (φ)
- Bond states (4 per cell, one for each cardinal direction)
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .config import Config


@dataclass
class CellState:
    """
    Complete state of the cellular automaton grid.
    
    All arrays have shape [H, W] except bonds which is [H, W, 4].
    The bond indices are: 0=North, 1=East, 2=South, 3=West.
    
    Attributes:
        A: Precursor concentration [0, 1]
        B: Structural/membrane concentration [0, 1]
        C: Catalyst/energy concentration [0, 1]
        phase: Oscillator phase [0, 2π)
        bonds: Bond states {0, 1} for each direction [H, W, 4]
    """
    
    A: mx.array
    B: mx.array
    C: mx.array
    phase: mx.array
    bonds: mx.array
    
    @property
    def shape(self) -> tuple[int, int]:
        """Grid dimensions (H, W)."""
        return self.A.shape
    
    @property
    def total_concentration(self) -> mx.array:
        """Sum of all concentrations at each cell."""
        return self.A + self.B + self.C
    
    def clone(self) -> "CellState":
        """Create a deep copy of the state."""
        return CellState(
            A=mx.array(self.A),
            B=mx.array(self.B),
            C=mx.array(self.C),
            phase=mx.array(self.phase),
            bonds=mx.array(self.bonds),
        )


# Bond direction indices
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Opposite direction mapping
OPPOSITE = {NORTH: SOUTH, EAST: WEST, SOUTH: NORTH, WEST: EAST}


def create_initial_state(
    config: Config,
    seed: Optional[int] = None,
    with_seed_region: bool = False,
) -> CellState:
    """
    Create initial state for simulation.
    
    Args:
        config: Simulation configuration
        seed: Random seed for reproducibility
        with_seed_region: If True, create a localized high-concentration seed
    
    Returns:
        Initialized CellState with random values in configured ranges
    """
    H = W = config.grid_size
    
    # Set random seed
    if seed is not None:
        key = mx.random.key(seed)
    else:
        key = mx.random.key(42)
    
    # Split key for each random operation
    keys = mx.random.split(key, 4)
    
    # Initialize concentrations with uniform random values
    A = mx.random.uniform(
        config.init_A_range[0],
        config.init_A_range[1],
        shape=(H, W),
        key=keys[0],
    )
    
    B = mx.random.uniform(
        config.init_B_range[0],
        config.init_B_range[1],
        shape=(H, W),
        key=keys[1],
    )
    
    C = mx.random.uniform(
        config.init_C_range[0],
        config.init_C_range[1],
        shape=(H, W),
        key=keys[2],
    )
    
    # Initialize phase uniformly in [0, 2π)
    phase = mx.random.uniform(
        0.0,
        2.0 * mx.pi,
        shape=(H, W),
        key=keys[3],
    )
    
    # Initialize bonds to zero (no initial bonds)
    bonds = mx.zeros((H, W, 4), dtype=mx.float32)
    
    # Optionally add a seed region with elevated concentrations
    if with_seed_region:
        center = H // 2
        half_size = config.seed_region_size // 2
        
        # Create index ranges for seed region
        y_start = max(0, center - half_size)
        y_end = min(H, center + half_size)
        x_start = max(0, center - half_size)
        x_end = min(W, center + half_size)
        
        # Set seed region values
        # Using array indexing with mx.where for GPU compatibility
        y_coords = mx.arange(H).reshape(-1, 1)
        x_coords = mx.arange(W).reshape(1, -1)
        
        in_seed = (
            (y_coords >= y_start) & (y_coords < y_end) &
            (x_coords >= x_start) & (x_coords < x_end)
        )
        
        A = mx.where(in_seed, config.seed_A, A)
        B = mx.where(in_seed, config.seed_B, B)
        C = mx.where(in_seed, config.seed_C, C)
        
        # Optionally synchronize phase in seed region
        phase = mx.where(in_seed, 0.0, phase)
    
    return CellState(A=A, B=B, C=C, phase=phase, bonds=bonds)


def create_uniform_state(
    config: Config,
    A_val: float = 0.3,
    B_val: float = 0.1,
    C_val: float = 0.1,
    phase_val: float = 0.0,
) -> CellState:
    """
    Create a uniform state (useful for testing).
    
    Args:
        config: Simulation configuration
        A_val: Uniform A concentration
        B_val: Uniform B concentration
        C_val: Uniform C concentration
        phase_val: Uniform phase value
    
    Returns:
        CellState with uniform values
    """
    H = W = config.grid_size
    
    return CellState(
        A=mx.ones((H, W)) * A_val,
        B=mx.ones((H, W)) * B_val,
        C=mx.ones((H, W)) * C_val,
        phase=mx.ones((H, W)) * phase_val,
        bonds=mx.zeros((H, W, 4)),
    )


def ensure_symmetric_bonds(bonds: mx.array) -> mx.array:
    """
    Ensure bond symmetry: if (i,j) bonded to (i,j+1), reverse must also be true.
    
    This handles the constraint that bonds are bidirectional.
    
    Args:
        bonds: Bond array [H, W, 4]
    
    Returns:
        Symmetrized bond array
    """
    H, W, _ = bonds.shape
    
    # North-South symmetry
    # If cell (i,j) has North bond, cell (i-1,j) must have South bond
    north_bonds = bonds[:, :, NORTH]
    south_bonds = bonds[:, :, SOUTH]
    
    # Roll to align neighbors
    north_shifted = mx.roll(north_bonds, shift=1, axis=0)  # Shift down
    south_shifted = mx.roll(south_bonds, shift=-1, axis=0)  # Shift up
    
    # Symmetrize: bond exists if either direction says so
    new_north = mx.maximum(north_bonds, south_shifted)
    new_south = mx.maximum(south_bonds, north_shifted)
    
    # East-West symmetry
    east_bonds = bonds[:, :, EAST]
    west_bonds = bonds[:, :, WEST]
    
    east_shifted = mx.roll(east_bonds, shift=1, axis=1)  # Shift right
    west_shifted = mx.roll(west_bonds, shift=-1, axis=1)  # Shift left
    
    new_east = mx.maximum(east_bonds, west_shifted)
    new_west = mx.maximum(west_bonds, east_shifted)
    
    # Stack back into bond array
    return mx.stack([new_north, new_east, new_south, new_west], axis=-1)
