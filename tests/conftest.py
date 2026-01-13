"""
Pytest configuration and fixtures for PCCS tests.
"""

import pytest
import mlx.core as mx

from pccs.config import Config
from pccs.state import CellState, create_initial_state


@pytest.fixture
def default_config() -> Config:
    """Default configuration for tests."""
    return Config(grid_size=32)


@pytest.fixture
def small_config() -> Config:
    """Small grid for fast tests."""
    return Config(grid_size=16)


@pytest.fixture
def default_state(default_config: Config) -> CellState:
    """Default initial state for tests."""
    return create_initial_state(default_config, seed=42)


@pytest.fixture
def uniform_state(default_config: Config) -> CellState:
    """Uniform state for diffusion tests."""
    H = W = default_config.grid_size
    return CellState(
        A=mx.ones((H, W)) * 0.5,
        B=mx.ones((H, W)) * 0.1,
        C=mx.ones((H, W)) * 0.1,
        phase=mx.zeros((H, W)),
        bonds=mx.zeros((H, W, 4)),
    )


@pytest.fixture
def rng_key() -> mx.array:
    """Random key for stochastic tests."""
    return mx.random.key(12345)
