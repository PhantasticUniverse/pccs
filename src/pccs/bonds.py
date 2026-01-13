"""
Bond formation and breaking dynamics for PCCS.

Bonds require dual conditions: sufficient B concentration AND phase alignment.
This creates membranes that naturally partition synchronized domains.
"""

from typing import Optional, Union

import mlx.core as mx
import numpy as np

from .config import Config
from .state import CellState, NORTH, EAST, SOUTH, WEST, ensure_symmetric_bonds


def sigmoid(x: mx.array) -> mx.array:
    """Standard sigmoid function."""
    return 1.0 / (1.0 + mx.exp(-x))


def compute_bond_probability(
    B_i: mx.array,
    B_j: mx.array,
    phase_i: mx.array,
    phase_j: mx.array,
    config: Config,
    B_thresh_override: Optional[Union[mx.array, float]] = None,
) -> mx.array:
    """
    Compute bond formation/persistence probability.

    EXPERIMENTAL: B-only bond formation.
    P = σ(θ_B × (Bᵢ + Bⱼ - 2×B_thresh))

    Bonds form based solely on B concentration, independent of phase.
    This allows bonds to span phase boundaries.

    Args:
        B_i: B concentration at cell i
        B_j: B concentration at neighbor j
        phase_i: Phase at cell i (unused in B-only mode)
        phase_j: Phase at neighbor j (unused in B-only mode)
        config: Simulation configuration
        B_thresh_override: Optional position-dependent threshold array or scalar.
                          If provided, overrides config.B_thresh.
                          Used for fitness experiments with asymmetric protocells.

    Returns:
        Bond probability at each cell
    """
    # Use override if provided, otherwise global config
    B_thresh = B_thresh_override if B_thresh_override is not None else config.B_thresh

    # B concentration condition ONLY
    B_sum = B_i + B_j
    B_condition = sigmoid(config.theta_B * (B_sum - 2.0 * B_thresh))

    # Phase alignment requirement REMOVED
    # Bonds can now form across phase boundaries
    return B_condition


def compute_bond_updates(
    state: CellState,
    config: Config,
    rng_key: mx.array,
) -> mx.array:
    """
    Compute probabilistic bond updates.

    Bonds form/persist when both cells have high B concentration.
    Uses per-cell B_thresh from state.B_thresh for evolution experiments.

    For bonds between neighboring cells, the effective threshold is the
    average of both cells' B_thresh values.

    Args:
        state: Current cell state (includes per-cell B_thresh)
        config: Simulation configuration
        rng_key: Random key for stochastic updates

    Returns:
        Updated bond states [H, W, 4]
    """
    H, W = state.shape

    # Split key for each direction
    keys = mx.random.split(rng_key, 4)

    # Get neighbor values for B concentration
    neighbor_B = {
        NORTH: mx.roll(state.B, shift=-1, axis=0),
        SOUTH: mx.roll(state.B, shift=1, axis=0),
        EAST: mx.roll(state.B, shift=-1, axis=1),
        WEST: mx.roll(state.B, shift=1, axis=1),
    }

    neighbor_phase = {
        NORTH: mx.roll(state.phase, shift=-1, axis=0),
        SOUTH: mx.roll(state.phase, shift=1, axis=0),
        EAST: mx.roll(state.phase, shift=-1, axis=1),
        WEST: mx.roll(state.phase, shift=1, axis=1),
    }

    # Get neighbor B_thresh values for per-cell threshold averaging
    neighbor_B_thresh = {
        NORTH: mx.roll(state.B_thresh, shift=-1, axis=0),
        SOUTH: mx.roll(state.B_thresh, shift=1, axis=0),
        EAST: mx.roll(state.B_thresh, shift=-1, axis=1),
        WEST: mx.roll(state.B_thresh, shift=1, axis=1),
    }

    # Compute bond probabilities for each direction
    new_bonds = []

    for i, direction in enumerate([NORTH, EAST, SOUTH, WEST]):
        # Use average B_thresh of both cells for this bond
        local_thresh = (state.B_thresh + neighbor_B_thresh[direction]) / 2.0

        # Get probability for this direction
        P = compute_bond_probability(
            state.B,
            neighbor_B[direction],
            state.phase,
            neighbor_phase[direction],
            config,
            B_thresh_override=local_thresh,
        )

        # Stochastic update: bond exists if random < P
        random_vals = mx.random.uniform(0, 1, shape=(H, W), key=keys[i])
        bond = (random_vals < P).astype(mx.float32)

        new_bonds.append(bond)

    # Stack into [H, W, 4] array
    bonds = mx.stack(new_bonds, axis=-1)

    # Ensure symmetry
    bonds = ensure_symmetric_bonds(bonds)

    return bonds


def count_bonds(state: CellState) -> int:
    """Count total number of bonds in the grid."""
    # Each bond is counted twice (once per cell), so divide by 2
    return int(mx.sum(state.bonds)) // 2


def find_bond_clusters(bonds: mx.array) -> mx.array:
    """
    Find connected components in the bond graph.
    
    Uses iterative label propagation on GPU.
    
    Args:
        bonds: Bond array [H, W, 4]
    
    Returns:
        Cluster labels for each cell [H, W]
    """
    H, W, _ = bonds.shape
    
    # Initialize each cell with unique label
    labels = mx.arange(H * W).reshape(H, W).astype(mx.int32)
    
    # Iteratively propagate minimum label through bonds
    for _ in range(max(H, W)):  # Worst case iterations
        old_labels = labels
        
        # Get neighbor labels
        neighbor_labels = {
            NORTH: mx.roll(labels, shift=-1, axis=0),
            SOUTH: mx.roll(labels, shift=1, axis=0),
            EAST: mx.roll(labels, shift=-1, axis=1),
            WEST: mx.roll(labels, shift=1, axis=1),
        }
        
        # For each direction, propagate minimum label through bond
        for direction, n_labels in neighbor_labels.items():
            has_bond = bonds[:, :, direction] > 0.5
            # Take minimum of own label and bonded neighbor's label
            labels = mx.where(
                has_bond,
                mx.minimum(labels, n_labels),
                labels
            )
        
        # Check for convergence
        if mx.all(labels == old_labels):
            break
    
    return labels


def detect_closed_membranes(bonds: mx.array) -> list[set[tuple[int, int]]]:
    """
    Detect closed membrane loops in the bond graph.
    
    A closed membrane is a connected component of bonds that forms a cycle.
    
    Args:
        bonds: Bond array [H, W, 4]
    
    Returns:
        List of sets, each containing (y, x) coordinates of cells in a closed membrane
    """
    # Find connected components
    labels = find_bond_clusters(bonds)

    # Get unique labels (convert to numpy since MLX doesn't have unique)
    labels_np = np.array(labels.tolist())
    unique_labels = np.unique(labels_np)

    closed_membranes = []

    # For each component, check if it forms a closed loop
    # A simple heuristic: if all cells in the component have exactly 2 bonds,
    # and the component has at least 4 cells, it's likely closed

    for label in unique_labels:
        mask = labels == label
        if mx.sum(mask) < 4:
            continue
        
        # Count bonds per cell in this component
        bonds_per_cell = mx.sum(bonds, axis=-1) * mask
        
        # Check if all cells have exactly 2 bonds (part of a chain/loop)
        cells_in_component = mask.astype(mx.float32)
        total_bonds_in_component = mx.sum(bonds_per_cell)
        num_cells = mx.sum(cells_in_component)
        
        # For a closed loop, total_bonds = 2 * num_cells (each cell has 2 bonds)
        if mx.abs(total_bonds_in_component - 2 * num_cells) < 0.1:
            # This might be a closed membrane
            coords = set()
            mask_np = mask.tolist()
            for y in range(len(mask_np)):
                for x in range(len(mask_np[0])):
                    if mask_np[y][x]:
                        coords.add((y, x))
            if len(coords) >= 4:
                closed_membranes.append(coords)
    
    return closed_membranes
