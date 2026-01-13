"""
Main simulation loop for PCCS.

Orchestrates the four coupled dynamics: diffusion, reactions, phase, and bonds.
"""

from typing import Callable, Optional

import mlx.core as mx
from tqdm import tqdm

from .config import Config
from .state import CellState, create_initial_state
from .diffusion import compute_diffusion
from .reactions import compute_reactions
from .phase import compute_phase_update, wrap_phase
from .bonds import compute_bond_updates


class Simulation:
    """
    PCCS simulation manager.
    
    Handles state evolution and provides hooks for visualization/analysis.
    
    Attributes:
        config: Simulation configuration
        state: Current cell state
        step_count: Number of steps executed
        rng_key: Random key for stochastic operations
    """
    
    def __init__(
        self,
        config: Config,
        seed: Optional[int] = None,
        initial_state: Optional[CellState] = None,
    ):
        """
        Initialize simulation.
        
        Args:
            config: Simulation configuration
            seed: Random seed for reproducibility
            initial_state: Optional pre-initialized state
        """
        self.config = config
        self.step_count = 0
        self.seed = seed if seed is not None else 42
        self.rng_key = mx.random.key(self.seed)
        
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = create_initial_state(
                config,
                seed=self.seed,
                with_seed_region=True,
            )
    
    def step(self) -> None:
        """
        Advance simulation by one timestep.
        
        Update order (critical for correct dynamics):
            1. Compute reaction fluxes (phase-gated)
            2. Apply reactions to concentrations
            3. Compute diffusion fluxes
            4. Apply diffusion
            5. Inject resources (if enabled)
            6. Update phase dynamics
            7. Update bond states
            8. Clamp concentrations to [0, 1]
            9. Wrap phase to [0, 2π)
        """
        # Split RNG key for this step
        self.rng_key, step_key = mx.random.split(self.rng_key)
        
        # 1-2. Reactions
        dA_react, dB_react, dC_react = compute_reactions(self.state, self.config)
        
        A = self.state.A + dA_react
        B = self.state.B + dB_react
        C = self.state.C + dC_react
        
        # Update state with new concentrations (temporary)
        self.state = CellState(
            A=A, B=B, C=C,
            phase=self.state.phase,
            bonds=self.state.bonds,
            B_thresh=self.state.B_thresh,
            k1=self.state.k1,
        )
        
        # 3-4. Diffusion
        dA_diff, dB_diff, dC_diff = compute_diffusion(self.state, self.config)
        
        A = self.state.A + dA_diff
        B = self.state.B + dB_diff
        C = self.state.C + dC_diff
        
        # 5. Resource injection
        A = self._inject_resources(A)

        # 6. Phase dynamics
        # Use post-diffusion concentrations for chemical-phase coupling
        diffused_state = CellState(
            A=A, B=B, C=C,
            phase=self.state.phase,
            bonds=self.state.bonds,
            B_thresh=self.state.B_thresh,
            k1=self.state.k1,
        )
        dphase = compute_phase_update(diffused_state, self.config)
        phase = self.state.phase + dphase
        
        # 7. Bond updates
        # Use updated concentrations and phase for bond calculation
        temp_state = CellState(
            A=A, B=B, C=C,
            phase=phase,
            bonds=self.state.bonds,
            B_thresh=self.state.B_thresh,
            k1=self.state.k1,
        )
        bonds = compute_bond_updates(temp_state, self.config, step_key)

        # 8. Apply mutations to B_thresh and k1 (if enabled)
        B_thresh = self.state.B_thresh
        k1 = self.state.k1
        if self.config.mutation_rate > 0:
            self.rng_key, b_mutation_key = mx.random.split(self.rng_key)
            B_thresh = self._apply_mutations(B_thresh, b_mutation_key)
            self.rng_key, k1_mutation_key = mx.random.split(self.rng_key)
            k1 = self._apply_k1_mutations(k1, k1_mutation_key)

        # 9. Clamp concentrations
        A = mx.clip(A, 0.0, 1.0)
        B = mx.clip(B, 0.0, 1.0)
        C = mx.clip(C, 0.0, 1.0)

        # 10. Wrap phase
        phase = wrap_phase(phase)

        # Final state update
        self.state = CellState(A=A, B=B, C=C, phase=phase, bonds=bonds, B_thresh=B_thresh, k1=k1)
        self.step_count += 1
    
    def _inject_resources(self, A: mx.array) -> mx.array:
        """
        Replenish environmental resources.
        
        Args:
            A: Current precursor concentration
        
        Returns:
            Updated A with injected resources
        """
        if self.config.injection_mode == "none":
            return A
        
        H, W = A.shape
        rate = self.config.injection_rate
        
        if self.config.injection_mode == "boundary":
            # Inject A at grid boundaries
            width = self.config.injection_width
            
            # Create boundary mask
            y_coords = mx.arange(H).reshape(-1, 1)
            x_coords = mx.arange(W).reshape(1, -1)
            
            boundary = (
                (y_coords < width) | (y_coords >= H - width) |
                (x_coords < width) | (x_coords >= W - width)
            )
            
            # Add resources at boundary
            A = A + rate * boundary.astype(mx.float32)
            
        elif self.config.injection_mode == "uniform":
            # Small constant addition everywhere
            A = A + rate
            
        elif self.config.injection_mode == "point_sources":
            # Localized nutrient wells at corners
            width = self.config.injection_width

            y_coords = mx.arange(H).reshape(-1, 1)
            x_coords = mx.arange(W).reshape(1, -1)

            # Four corners
            corners = (
                ((y_coords < width) & (x_coords < width)) |
                ((y_coords < width) & (x_coords >= W - width)) |
                ((y_coords >= H - width) & (x_coords < width)) |
                ((y_coords >= H - width) & (x_coords >= W - width))
            )

            A = A + rate * 5.0 * corners.astype(mx.float32)

        elif self.config.injection_mode == "center":
            # Inject A at grid center - creates protocell
            width = self.config.injection_width
            center_y, center_x = H // 2, W // 2

            y_coords = mx.arange(H).reshape(-1, 1)
            x_coords = mx.arange(W).reshape(1, -1)

            # Circular region at center
            dist_sq = (y_coords - center_y)**2 + (x_coords - center_x)**2
            center_region = dist_sq < width**2

            A = A + rate * 5.0 * center_region.astype(mx.float32)

        elif self.config.injection_mode == "competing":
            # Two injection points at 1/3 and 2/3 across grid (closer than dual)
            # Tests protocell interaction when close together
            width = self.config.injection_width

            y_coords = mx.arange(H).reshape(-1, 1)
            x_coords = mx.arange(W).reshape(1, -1)

            # Left center at 1/3
            left_y, left_x = H // 2, W // 3
            dist_left = (y_coords - left_y)**2 + (x_coords - left_x)**2
            left_region = dist_left < width**2

            # Right center at 2/3
            right_y, right_x = H // 2, 2 * W // 3
            dist_right = (y_coords - right_y)**2 + (x_coords - right_x)**2
            right_region = dist_right < width**2

            A = A + rate * 5.0 * (left_region | right_region).astype(mx.float32)

        elif self.config.injection_mode == "budding":
            # Mother at center, daughter offset to the right
            # Used for protocell division experiments
            width = self.config.injection_width
            center_y, center_x = H // 2, W // 2
            daughter_offset = 20  # Cells to the right - beyond diffusion range

            y_coords = mx.arange(H).reshape(-1, 1)
            x_coords = mx.arange(W).reshape(1, -1)

            # Mother region (center)
            dist_mother = (y_coords - center_y)**2 + (x_coords - center_x)**2
            mother_region = dist_mother < width**2

            # Daughter region (offset from center)
            daughter_x = center_x + daughter_offset
            dist_daughter = (y_coords - center_y)**2 + (x_coords - daughter_x)**2
            daughter_region = dist_daughter < width**2

            A = A + rate * 5.0 * (mother_region | daughter_region).astype(mx.float32)

        elif self.config.injection_mode == "lineage":
            # Three generations: mother, daughter, granddaughter
            # Tests multi-generational protocell formation
            width = self.config.injection_width
            center_y, center_x = H // 2, W // 2
            generation_offset = 20  # Distance between generations

            y_coords = mx.arange(H).reshape(-1, 1)
            x_coords = mx.arange(W).reshape(1, -1)

            # Generation 0: Mother (center)
            dist_mother = (y_coords - center_y)**2 + (x_coords - center_x)**2
            mother_region = dist_mother < width**2

            # Generation 1: Daughter (center + 20)
            daughter_x = center_x + generation_offset
            dist_daughter = (y_coords - center_y)**2 + (x_coords - daughter_x)**2
            daughter_region = dist_daughter < width**2

            # Generation 2: Granddaughter (center + 40)
            granddaughter_x = center_x + 2 * generation_offset
            dist_granddaughter = (y_coords - center_y)**2 + (x_coords - granddaughter_x)**2
            granddaughter_region = dist_granddaughter < width**2

            A = A + rate * 5.0 * (mother_region | daughter_region | granddaughter_region).astype(mx.float32)

        elif self.config.injection_mode == "competition":
            # Four protocells in diamond pattern for symmetric competition
            # Tests resource scarcity and selection pressure
            width = self.config.injection_width
            center_y, center_x = H // 2, W // 2
            spacing = 25  # Distance from center to each protocell

            y_coords = mx.arange(H).reshape(-1, 1)
            x_coords = mx.arange(W).reshape(1, -1)

            # North protocell
            dist_n = (y_coords - (center_y - spacing))**2 + (x_coords - center_x)**2
            north_region = dist_n < width**2

            # South protocell
            dist_s = (y_coords - (center_y + spacing))**2 + (x_coords - center_x)**2
            south_region = dist_s < width**2

            # East protocell
            dist_e = (y_coords - center_y)**2 + (x_coords - (center_x + spacing))**2
            east_region = dist_e < width**2

            # West protocell
            dist_w = (y_coords - center_y)**2 + (x_coords - (center_x - spacing))**2
            west_region = dist_w < width**2

            A = A + rate * 5.0 * (north_region | south_region | east_region | west_region).astype(mx.float32)

        return A

    def _apply_mutations(self, B_thresh: mx.array, rng_key: mx.array) -> mx.array:
        """
        Apply stochastic mutations to B_thresh array.

        Each cell has a probability (mutation_rate) of mutating its B_thresh
        by a random amount in [-mutation_strength, +mutation_strength].

        Args:
            B_thresh: Current per-cell B_thresh values [H, W]
            rng_key: Random key for stochastic operations

        Returns:
            Updated B_thresh array (clamped to valid range)
        """
        H, W = B_thresh.shape

        # Generate mutation mask (which cells mutate this step)
        rng_key, mask_key = mx.random.split(rng_key)
        mutate_mask = mx.random.uniform(shape=(H, W), key=mask_key) < self.config.mutation_rate

        # Generate mutation values (uniform in [-strength, +strength])
        rng_key, delta_key = mx.random.split(rng_key)
        delta = mx.random.uniform(
            low=-self.config.mutation_strength,
            high=self.config.mutation_strength,
            shape=(H, W),
            key=delta_key,
        )

        # Apply mutations where mask is True
        B_thresh = B_thresh + mutate_mask * delta

        # Clamp to valid range
        B_thresh = mx.clip(B_thresh, self.config.B_thresh_min, self.config.B_thresh_max)

        return B_thresh

    def _apply_k1_mutations(self, k1: mx.array, rng_key: mx.array) -> mx.array:
        """
        Apply stochastic mutations to k1 (dimerization rate) array.

        Each cell has a probability (mutation_rate) of mutating its k1
        by a random amount in [-k1_mutation_strength, +k1_mutation_strength].

        Args:
            k1: Current per-cell k1 values [H, W]
            rng_key: Random key for stochastic operations

        Returns:
            Updated k1 array (clamped to valid range)
        """
        H, W = k1.shape

        # Generate mutation mask (which cells mutate this step)
        rng_key, mask_key = mx.random.split(rng_key)
        mutate_mask = mx.random.uniform(shape=(H, W), key=mask_key) < self.config.mutation_rate

        # Generate mutation values (uniform in [-strength, +strength])
        rng_key, delta_key = mx.random.split(rng_key)
        delta = mx.random.uniform(
            low=-self.config.k1_mutation_strength,
            high=self.config.k1_mutation_strength,
            shape=(H, W),
            key=delta_key,
        )

        # Apply mutations where mask is True
        k1 = k1 + mutate_mask * delta

        # Clamp to valid range
        k1 = mx.clip(k1, self.config.k1_min, self.config.k1_max)

        return k1

    def run(
        self,
        steps: int,
        callback: Optional[Callable[["Simulation"], None]] = None,
        callback_interval: int = 100,
        show_progress: bool = True,
    ) -> None:
        """
        Run simulation for multiple steps.
        
        Args:
            steps: Number of steps to run
            callback: Optional function called periodically
            callback_interval: How often to call callback
            show_progress: Whether to show progress bar
        """
        iterator = range(steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating")
        
        for i in iterator:
            self.step()
            
            if callback is not None and (i + 1) % callback_interval == 0:
                mx.eval(self.state.A)  # Force evaluation before callback
                callback(self)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset simulation to initial state.
        
        Args:
            seed: New random seed (uses original if not provided)
        """
        if seed is not None:
            self.seed = seed
        
        self.rng_key = mx.random.key(self.seed)
        self.state = create_initial_state(
            self.config,
            seed=self.seed,
            with_seed_region=True,
        )
        self.step_count = 0
    
    def get_state_dict(self) -> dict:
        """Get serializable state dictionary."""
        return {
            "step_count": self.step_count,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "state": {
                "A": self.state.A.tolist(),
                "B": self.state.B.tolist(),
                "C": self.state.C.tolist(),
                "phase": self.state.phase.tolist(),
                "bonds": self.state.bonds.tolist(),
            }
        }
