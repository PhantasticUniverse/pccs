"""
Visualization utilities for PCCS.

Provides real-time display and image export for simulation state.
"""

from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from .state import CellState, NORTH, EAST, SOUTH, WEST


def state_to_rgb(state: CellState, normalize: bool = True) -> np.ndarray:
    """
    Convert state concentrations to RGB image.
    
    Mapping: R = C (catalyst), G = B (structural), B = A (precursor)
    
    Args:
        state: Cell state
        normalize: Whether to normalize to [0, 255]
    
    Returns:
        RGB image [H, W, 3] as uint8
    """
    # Convert to numpy
    A = np.array(state.A)
    B = np.array(state.B)
    C = np.array(state.C)
    
    # Stack as RGB
    rgb = np.stack([C, B, A], axis=-1)
    
    if normalize:
        # Clip and scale to [0, 255]
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def state_to_phase_image(state: CellState) -> np.ndarray:
    """
    Convert phase to HSV cyclic colormap image.
    
    Hue = phase (cyclic)
    Saturation = 1
    Value = total concentration (brightness indicates activity)
    
    Args:
        state: Cell state
    
    Returns:
        RGB image [H, W, 3] as uint8
    """
    # Convert to numpy
    phase = np.array(state.phase)
    total = np.array(state.A + state.B + state.C)
    
    # Normalize phase to [0, 1] for hue
    hue = phase / (2 * np.pi)
    
    # Saturation fixed at 1
    saturation = np.ones_like(hue)
    
    # Value from total concentration
    value = np.clip(total / 1.5, 0.3, 1.0)  # Min brightness 0.3
    
    # Stack HSV
    hsv = np.stack([hue, saturation, value], axis=-1)
    
    # Convert to RGB
    rgb = mcolors.hsv_to_rgb(hsv)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb


def overlay_bonds(image: np.ndarray, bonds: mx.array, color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Draw bond edges on image.
    
    Args:
        image: Base image [H, W, 3]
        bonds: Bond array [H, W, 4]
        color: RGB color for bond lines
    
    Returns:
        Image with bond overlay
    """
    bonds_np = np.array(bonds)
    H, W, _ = bonds_np.shape
    
    # Make copy to avoid modifying original
    result = image.copy()
    
    # Draw horizontal bonds (East-West)
    for y in range(H):
        for x in range(W):
            # East bond
            if bonds_np[y, x, EAST] > 0.5:
                x_next = (x + 1) % W
                # Draw line between cells
                result[y, x, :] = color
                result[y, x_next, :] = color
            
            # South bond  
            if bonds_np[y, x, SOUTH] > 0.5:
                y_next = (y + 1) % H
                result[y, x, :] = color
                result[y_next, x, :] = color
    
    return result


def create_composite_image(state: CellState, show_bonds: bool = True) -> np.ndarray:
    """
    Create side-by-side concentration and phase visualization.
    
    Args:
        state: Cell state
        show_bonds: Whether to overlay bonds
    
    Returns:
        Composite image [H, 2*W, 3]
    """
    rgb = state_to_rgb(state)
    phase_img = state_to_phase_image(state)
    
    if show_bonds:
        rgb = overlay_bonds(rgb, state.bonds)
        phase_img = overlay_bonds(phase_img, state.bonds)
    
    # Stack horizontally
    composite = np.concatenate([rgb, phase_img], axis=1)
    
    return composite


class Visualizer:
    """
    Real-time visualization manager.
    
    Provides live display of simulation state using matplotlib.
    """
    
    def __init__(
        self,
        simulation: "Simulation",  # Forward reference
        mode: str = "composite",
        fps: int = 30,
        show_bonds: bool = True,
    ):
        """
        Initialize visualizer.
        
        Args:
            simulation: Simulation to visualize
            mode: Display mode ("rgb", "phase", "composite")
            fps: Target frames per second
            show_bonds: Whether to show bond overlay
        """
        self.sim = simulation
        self.mode = mode
        self.fps = fps
        self.show_bonds = show_bonds
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 5) if mode == "composite" else (6, 6))
        self.ax.set_axis_off()
        
        # Initialize image
        self._update_image()
        self.im = self.ax.imshow(self.current_image)
        
        # Add step counter text
        self.text = self.ax.text(
            0.02, 0.98, f"Step: {self.sim.step_count}",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5)
        )
    
    def _update_image(self) -> None:
        """Update current image from simulation state."""
        if self.mode == "rgb":
            self.current_image = state_to_rgb(self.sim.state)
            if self.show_bonds:
                self.current_image = overlay_bonds(self.current_image, self.sim.state.bonds)
        elif self.mode == "phase":
            self.current_image = state_to_phase_image(self.sim.state)
            if self.show_bonds:
                self.current_image = overlay_bonds(self.current_image, self.sim.state.bonds)
        else:  # composite
            self.current_image = create_composite_image(self.sim.state, self.show_bonds)
    
    def _animation_update(self, frame: int) -> list:
        """Update function for animation."""
        # Run simulation step
        self.sim.step()
        mx.eval(self.sim.state.A)  # Force evaluation
        
        # Update image
        self._update_image()
        self.im.set_array(self.current_image)
        self.text.set_text(f"Step: {self.sim.step_count}")
        
        return [self.im, self.text]
    
    def show_live(self, steps: Optional[int] = None) -> None:
        """
        Display live animation.
        
        Args:
            steps: Number of steps to run (None for infinite)
        """
        frames = steps if steps is not None else 10000
        interval = 1000 // self.fps
        
        anim = FuncAnimation(
            self.fig,
            self._animation_update,
            frames=frames,
            interval=interval,
            blit=True,
        )
        
        plt.show()
    
    def save_frame(self, path: str) -> None:
        """
        Save current frame as image.
        
        Args:
            path: Output file path
        """
        self._update_image()
        plt.imsave(path, self.current_image)
    
    def save_animation(
        self,
        path: str,
        steps: int = 500,
        fps: Optional[int] = None,
    ) -> None:
        """
        Save animation to file.
        
        Args:
            path: Output file path (mp4, gif, etc.)
            steps: Number of frames
            fps: Frames per second (uses self.fps if not provided)
        """
        if fps is None:
            fps = self.fps
        
        interval = 1000 // fps
        
        anim = FuncAnimation(
            self.fig,
            self._animation_update,
            frames=steps,
            interval=interval,
            blit=True,
        )
        
        # Determine writer from extension
        suffix = Path(path).suffix.lower()
        if suffix == ".gif":
            anim.save(path, writer="pillow", fps=fps)
        else:
            anim.save(path, writer="ffmpeg", fps=fps)
        
        print(f"Animation saved to {path}")


def save_state_images(
    state: CellState,
    output_dir: str,
    prefix: str = "frame",
    step: int = 0,
) -> None:
    """
    Save all visualization modes as separate images.
    
    Args:
        state: Cell state
        output_dir: Output directory
        prefix: Filename prefix
        step: Step number for filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # RGB concentration
    rgb = state_to_rgb(state)
    plt.imsave(output_path / f"{prefix}_{step:06d}_rgb.png", rgb)
    
    # Phase
    phase_img = state_to_phase_image(state)
    plt.imsave(output_path / f"{prefix}_{step:06d}_phase.png", phase_img)
    
    # Composite
    composite = create_composite_image(state)
    plt.imsave(output_path / f"{prefix}_{step:06d}_composite.png", composite)
