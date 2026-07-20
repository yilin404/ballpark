"""Configuration for sphere decomposition."""

from __future__ import annotations

from enum import Enum

import jax_dataclasses as jdc


class SpherePreset(Enum):
    """Preset configurations for sphere decomposition.

    SURFACE: Tight fit to mesh surface. Best for visualization and
             precise collision bounds. May under-approximate concavities.

    BALANCED: Default. Balanced between coverage and efficiency.
              Good general-purpose setting for most robots.

    CONSERVATIVE: Over-approximates to ensure full coverage.
                  Larger spheres, safer for collision checking but
                  less precise. Good when false negatives are costly.
    """

    SURFACE = "surface"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@jdc.pytree_dataclass
class SpherizeParams:
    """Parameters for the adaptive splitting algorithm."""

    target_tightness: float = 1.2
    """Max acceptable sphere_vol/hull_vol ratio before splitting."""

    aspect_threshold: float = 1.3
    """Max acceptable aspect ratio before splitting."""

    n_samples: int = 5000
    """Number of surface samples to use."""

    padding: float = 1.02
    """Radius multiplier for safety margin (1.02 = 2% larger)."""

    percentile: float = 98.0
    """Percentile of distances to use for radius (handles outliers)."""

    max_radius_ratio: float = 0.5
    """Cap radius relative to bounding box diagonal."""

    uniform_radius: bool = False
    """If True, post-process to make radii more uniform."""

    axis_mode: str = "aligned"
    """Split direction: 'aligned' (axis with max variance), 'pca' (arbitrary principal axis)."""

    symmetry_mode: str = "auto"
    """Symmetry handling:
    - 'auto': Detect symmetry; if found, spherize one half and mirror.
    - 'force': Always assume symmetric; spherize one half and mirror.
    - 'off': No symmetry handling; process entire mesh normally.
    """

    symmetry_tolerance: float = 0.05
    """Tolerance for approximate symmetry detection (0.0 = perfect, 0.1 = 10% deviation)."""

    odd_budget_mode: str = "round_up"
    """How to handle odd sphere budgets: 'round_up' (add +1) or 'center' (place one on plane)."""


@jdc.pytree_dataclass
class RefineParams:
    """Parameters for independent per-link Warp/Torch refinement."""

    # Optimization params
    n_iters: int = 100
    """Maximum number of optimization iterations."""

    min_radius: float = 1e-4
    """Minimum allowed sphere radius."""

    n_samples: int = 5000
    """Points to sample per link for loss computation."""

    n_sphere_surface_samples: int = 128
    """Fixed surface directions sampled per optimized sphere."""

    center_lr: float = 0.005
    """AdamW learning rate for link-normalized sphere centers."""

    radius_lr: float = 0.001
    """AdamW learning rate for unconstrained softplus radius parameters."""

    grad_clip_norm: float = 1.0
    """Independent gradient norm limit for center and radius tensors."""

    lambda_coverage: float = 1000.0
    """Weight for collision-mesh coverage loss."""

    lambda_protrusion: float = 10.0
    """Weight for positive mesh-SDF sphere-surface loss."""

    lambda_tangency: float = 1.0
    """Weight for sphere-to-mesh tangency loss."""

    lambda_overlap: float = 0.1
    """Weight for pairwise sphere overlap loss."""

    lambda_clip_plane: float = 1000.0
    """Weight for per-link soft half-plane penetration."""

    clip_plane_buffer: float = 0.02
    """Required sphere-to-clip-plane clearance in meters."""


@jdc.pytree_dataclass
class BallparkConfig:
    """Unified configuration for sphere decomposition.

    Combines spherize and refine parameters with optional preset support.

    Usage:
        # Use a preset
        config = BallparkConfig.from_preset(SpherePreset.CONSERVATIVE)

        # Customize from preset
        config = BallparkConfig.from_preset(SpherePreset.BALANCED)
        config = jdc.replace(config, spherize=jdc.replace(config.spherize, padding=1.03))

        # Fully custom
        config = BallparkConfig(
            spherize=SpherizeParams(target_tightness=1.1),
            refine=RefineParams(n_iters=200),
        )
    """

    spherize: SpherizeParams = jdc.field(default_factory=SpherizeParams)
    """Parameters for adaptive splitting algorithm."""

    refine: RefineParams = jdc.field(default_factory=RefineParams)
    """Parameters for gradient-based refinement."""

    @classmethod
    def from_preset(cls, preset: SpherePreset) -> "BallparkConfig":
        """Create config from a preset.

        Args:
            preset: Base preset to use

        Returns:
            BallparkConfig with preset values
        """
        return jdc.replace(_PRESET_CONFIGS[preset])


# Preset definitions
_PRESET_CONFIGS: dict[SpherePreset, BallparkConfig] = {
    SpherePreset.SURFACE: BallparkConfig(
        spherize=SpherizeParams(
            target_tightness=1.1,  # Tighter splits
            aspect_threshold=1.2,  # Split elongated shapes more aggressively
            padding=1.01,  # Minimal padding
            percentile=99.0,  # Use more of the points
            max_radius_ratio=0.4,  # Smaller max spheres
            uniform_radius=False,
        ),
        refine=RefineParams(
            lambda_coverage=1000.0,
            lambda_protrusion=10.0,
            n_iters=300,  # More iterations for precision
        ),
    ),
    SpherePreset.BALANCED: BallparkConfig(
        spherize=SpherizeParams(
            target_tightness=1.2,
            aspect_threshold=1.3,
            padding=1.02,
            percentile=98.0,
            max_radius_ratio=0.5,
            uniform_radius=False,
        ),
        refine=RefineParams(
            lambda_coverage=1000.0,
            lambda_protrusion=10.0,
            n_iters=200,
        ),
    ),
    SpherePreset.CONSERVATIVE: BallparkConfig(
        spherize=SpherizeParams(
            target_tightness=1.4,  # Looser splits - fewer, larger spheres
            aspect_threshold=1.5,  # More tolerant of elongation
            padding=1.05,  # More padding for safety
            percentile=95.0,  # Ignore more outliers
            max_radius_ratio=0.6,  # Allow larger spheres
            uniform_radius=True,  # More uniform sizes
        ),
        refine=RefineParams(
            lambda_coverage=1000.0,
            lambda_protrusion=10.0,
            n_iters=160,  # Fewer iterations needed
        ),
    ),
}
