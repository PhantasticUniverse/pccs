"""
PCCS - Phase-Coupled Catalytic Substrate

A novel cellular automaton for emergent life-like complexity.
"""

__version__ = "0.1.0"

from .config import Config
from .state import CellState, create_initial_state

__all__ = [
    "Config",
    "CellState",
    "create_initial_state",
    "__version__",
]
