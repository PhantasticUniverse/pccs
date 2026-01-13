"""
Configuration dataclass for PCCS simulation parameters.

All parameters have physical interpretations as described in the PRD.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import math


@dataclass
class Config:
    """
    Complete configuration for PCCS simulation.
    
    All default values are from the PRD specification.
    
    Attributes:
        grid_size: Width and height of the simulation grid
        
        # Diffusion parameters
        D_base: Base diffusion rate for all substances
        alpha: Membrane impermeability factor (0 = transparent, 1 = opaque)
        
        # Reaction parameters
        k1: Rate constant for anabolism (2A + C → B)
        k2: Rate constant for catabolism (2B → C)
        k3: Rate constant for autocatalysis (C + A → 2A)
        kappa: Phase gate sharpness (higher = narrower reaction windows)
        epsilon: Energy dissipation rate
        
        # Phase dynamics parameters
        omega_0: Natural oscillation frequency
        K_phase: Phase coupling strength (Kuramoto coupling)
        chi: Chemical-phase coupling coefficient
        
        # Bond parameters
        B_thresh: Minimum B concentration for bond formation
        cos_thresh: Minimum phase alignment (cos(Δφ)) for bond formation
        theta_B: Sigmoid steepness for B condition
        theta_phi: Sigmoid steepness for phase condition
        
        # Resource injection
        injection_mode: How resources are replenished ("boundary", "uniform", "point_sources", "none")
        injection_rate: Rate of A injection
        injection_width: Width of boundary injection zone
    """
    
    # Grid
    grid_size: int = 256
    
    # Diffusion
    D_base: float = 0.1
    alpha: float = 0.9
    
    # Reactions
    k1: float = 0.1
    k2: float = 0.1
    k3: float = 0.1
    kappa: float = 2.0
    epsilon: float = 0.01
    
    # Phase dynamics
    omega_0: float = 0.1
    K_phase: float = 0.5
    chi: float = 0.2
    
    # Bonds
    B_thresh: float = 0.3
    cos_thresh: float = 0.7
    theta_B: float = 10.0
    theta_phi: float = 10.0
    
    # Resource injection
    injection_mode: str = "boundary"
    injection_rate: float = 0.01
    injection_width: int = 5
    
    # Initialization
    init_A_range: tuple[float, float] = (0.1, 0.5)
    init_B_range: tuple[float, float] = (0.0, 0.2)
    init_C_range: tuple[float, float] = (0.0, 0.2)
    
    # Seed region (optional localized initialization)
    seed_region_size: int = 10
    seed_A: float = 0.3
    seed_B: float = 0.2
    seed_C: float = 0.2
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate()
    
    def _validate(self) -> None:
        """Check that all parameters are in valid ranges."""
        if self.grid_size < 8:
            raise ValueError(f"grid_size must be >= 8, got {self.grid_size}")
        
        if not 0 < self.D_base <= 1:
            raise ValueError(f"D_base must be in (0, 1], got {self.D_base}")
        
        if not 0 <= self.alpha < 1:
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")
        
        if self.kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {self.kappa}")
        
        if not 0 <= self.epsilon < 1:
            raise ValueError(f"epsilon must be in [0, 1), got {self.epsilon}")
        
        if not 0 < self.B_thresh < 1:
            raise ValueError(f"B_thresh must be in (0, 1), got {self.B_thresh}")
        
        if not -1 < self.cos_thresh < 1:
            raise ValueError(f"cos_thresh must be in (-1, 1), got {self.cos_thresh}")
        
        valid_modes = {"boundary", "uniform", "point_sources", "none"}
        if self.injection_mode not in valid_modes:
            raise ValueError(f"injection_mode must be one of {valid_modes}, got {self.injection_mode}")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Handle tuple conversion for ranges
        if "init_A_range" in d and isinstance(d["init_A_range"], list):
            d["init_A_range"] = tuple(d["init_A_range"])
        if "init_B_range" in d and isinstance(d["init_B_range"], list):
            d["init_B_range"] = tuple(d["init_B_range"])
        if "init_C_range" in d and isinstance(d["init_C_range"], list):
            d["init_C_range"] = tuple(d["init_C_range"])
        return cls(**d)
    
    @classmethod
    def from_args(cls, args: Any) -> "Config":
        """Create config from argparse namespace."""
        # Extract only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        config_dict = {k: v for k, v in vars(args).items() if k in known_fields and v is not None}
        return cls(**config_dict)
    
    @property
    def reaction_phases(self) -> tuple[float, float, float]:
        """Target phases for the three reactions."""
        return (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  grid_size={self.grid_size},\n"
            f"  D_base={self.D_base}, alpha={self.alpha},\n"
            f"  k1={self.k1}, k2={self.k2}, k3={self.k3}, kappa={self.kappa},\n"
            f"  omega_0={self.omega_0}, K_phase={self.K_phase}, chi={self.chi},\n"
            f"  B_thresh={self.B_thresh}, cos_thresh={self.cos_thresh}\n"
            f")"
        )
