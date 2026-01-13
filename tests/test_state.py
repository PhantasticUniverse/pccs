"""
Tests for PCCS cell state.
"""

import pytest
import mlx.core as mx

from pccs.config import Config
from pccs.state import CellState, create_initial_state, create_uniform_state


class TestCellState:
    """Tests for CellState dataclass."""
    
    def test_shape_property(self, default_state):
        """State reports correct shape."""
        assert default_state.shape == (32, 32)
    
    def test_total_concentration(self, default_state):
        """Total concentration computed correctly."""
        total = default_state.total_concentration
        
        expected = default_state.A + default_state.B + default_state.C
        assert mx.allclose(total, expected)
    
    def test_clone(self, default_state):
        """Clone creates independent copy."""
        cloned = default_state.clone()
        
        # Shapes should match
        assert cloned.shape == default_state.shape
        
        # Values should match
        assert mx.allclose(cloned.A, default_state.A)


class TestCreateInitialState:
    """Tests for state initialization."""
    
    def test_shapes(self, default_config):
        """State arrays have correct shapes."""
        state = create_initial_state(default_config, seed=42)
        H = W = default_config.grid_size
        
        assert state.A.shape == (H, W)
        assert state.B.shape == (H, W)
        assert state.C.shape == (H, W)
        assert state.phase.shape == (H, W)
        assert state.bonds.shape == (H, W, 4)
    
    def test_concentration_ranges(self, default_config):
        """Concentrations are in expected ranges."""
        state = create_initial_state(default_config, seed=42)
        
        assert mx.all(state.A >= 0) and mx.all(state.A <= 1)
        assert mx.all(state.B >= 0) and mx.all(state.B <= 1)
        assert mx.all(state.C >= 0) and mx.all(state.C <= 1)
    
    def test_phase_range(self, default_config):
        """Phase is in [0, 2π)."""
        state = create_initial_state(default_config, seed=42)
        
        assert mx.all(state.phase >= 0)
        assert mx.all(state.phase < 2 * mx.pi)
    
    def test_no_initial_bonds(self, default_config):
        """Initial bonds are all zero."""
        state = create_initial_state(default_config, seed=42)
        
        assert mx.all(state.bonds == 0)
    
    def test_reproducibility(self, default_config):
        """Same seed produces same state."""
        state1 = create_initial_state(default_config, seed=42)
        state2 = create_initial_state(default_config, seed=42)
        
        assert mx.allclose(state1.A, state2.A)
        assert mx.allclose(state1.phase, state2.phase)
    
    def test_different_seeds(self, default_config):
        """Different seeds produce different states."""
        state1 = create_initial_state(default_config, seed=42)
        state2 = create_initial_state(default_config, seed=123)
        
        assert not mx.allclose(state1.A, state2.A)
    
    def test_seed_region(self, default_config):
        """Seed region has elevated concentrations."""
        state = create_initial_state(default_config, seed=42, with_seed_region=True)
        
        H = W = default_config.grid_size
        center = H // 2
        half_size = default_config.seed_region_size // 2
        
        # Compare center to corner
        center_B = state.B[center-half_size:center+half_size, 
                          center-half_size:center+half_size]
        corner_B = state.B[0:5, 0:5]
        
        # Seed region should have higher B (set to seed_B)
        assert float(mx.mean(center_B)) >= float(mx.mean(corner_B))


class TestCreateUniformState:
    """Tests for uniform state creation."""
    
    def test_uniform_values(self, default_config):
        """Uniform state has specified values."""
        state = create_uniform_state(
            default_config,
            A_val=0.3,
            B_val=0.2,
            C_val=0.1,
            phase_val=1.0,
        )
        
        assert mx.all(state.A == 0.3)
        assert mx.all(state.B == 0.2)
        assert mx.all(state.C == 0.1)
        assert mx.all(state.phase == 1.0)
        assert mx.all(state.bonds == 0)
