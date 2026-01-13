"""
Tests for PCCS configuration.
"""

import pytest
from pccs.config import Config


class TestConfig:
    """Tests for Config dataclass."""
    
    def test_defaults(self):
        """Config initializes with PRD-specified defaults."""
        config = Config()
        
        assert config.grid_size == 256
        assert config.D_base == 0.1
        assert config.alpha == 0.9
        assert config.kappa == 2.0
        assert config.K_phase == 0.5
        assert config.B_thresh == 0.3
        assert config.cos_thresh == 0.7
    
    def test_custom_values(self):
        """Config accepts custom values."""
        config = Config(
            grid_size=128,
            D_base=0.2,
            alpha=0.8,
        )
        
        assert config.grid_size == 128
        assert config.D_base == 0.2
        assert config.alpha == 0.8
    
    def test_invalid_alpha_high(self):
        """Config rejects alpha >= 1."""
        with pytest.raises(ValueError, match="alpha"):
            Config(alpha=1.0)
    
    def test_invalid_alpha_negative(self):
        """Config rejects alpha < 0."""
        with pytest.raises(ValueError, match="alpha"):
            Config(alpha=-0.1)
    
    def test_invalid_D_base(self):
        """Config rejects non-positive D_base."""
        with pytest.raises(ValueError, match="D_base"):
            Config(D_base=0.0)
        with pytest.raises(ValueError, match="D_base"):
            Config(D_base=-0.1)
    
    def test_invalid_grid_size(self):
        """Config rejects grid_size < 8."""
        with pytest.raises(ValueError, match="grid_size"):
            Config(grid_size=4)
    
    def test_invalid_injection_mode(self):
        """Config rejects invalid injection_mode."""
        with pytest.raises(ValueError, match="injection_mode"):
            Config(injection_mode="invalid")
    
    def test_serialization_roundtrip(self):
        """Config serializes and deserializes correctly."""
        config = Config(
            grid_size=64,
            D_base=0.15,
            alpha=0.85,
        )
        
        config_dict = config.to_dict()
        restored = Config.from_dict(config_dict)
        
        assert config.grid_size == restored.grid_size
        assert config.D_base == restored.D_base
        assert config.alpha == restored.alpha
    
    def test_reaction_phases(self):
        """Reaction phases are at correct values."""
        config = Config()
        phi1, phi2, phi3 = config.reaction_phases
        
        import math
        assert abs(phi1 - 0.0) < 1e-6
        assert abs(phi2 - 2 * math.pi / 3) < 1e-6
        assert abs(phi3 - 4 * math.pi / 3) < 1e-6
