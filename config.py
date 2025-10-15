"""
Configuration settings and constants for 3D Gaussian Splatting with NeRF datasets.
"""
import warp as wp
import numpy as np
import random

SEED = 42
random.seed(SEED)

# Warp data types and constants (keep capitalized as they are types)
WP_FLOAT16 = wp.float16
WP_FLOAT32 = wp.float32
WP_INT = wp.int32
WP_VEC2 = wp.vec2
WP_VEC2H = wp.vec2h
VEC6 = wp.types.vector(length=6, dtype=WP_FLOAT32)

# DEVICE = "cpu" # Use "cpu" or "cuda"
DEVICE = "cuda"
TILE_M = wp.constant(16)
TILE_N = wp.constant(16)
TILE_THREADS = wp.constant(256)


class GaussianParams:
    """Parameters for 3D Gaussian Splatting."""

    # === TRAINING CONFIGURATION ===
    num_iterations = 7000  # Default number of training iterations
    num_points = 5000      # Initial number of Gaussian points
    save_interval = 500    # Save checkpoint every N iterations

    # === LEARNING RATE SCHEDULE ===
    use_lr_scheduler = True
    lr_scheduler_config = {
        'lr_pos': 1e-2,      # Initial learning rate for positions
        'lr_scale': 5e-3,    # Initial learning rate for scales  
        'lr_rot': 5e-3,      # Initial learning rate for rotations
        'lr_sh': 2e-3,       # Initial learning rate for spherical harmonics
        'lr_opac': 5e-3,     # Initial learning rate for opacities
        'final_lr_factor': 0.01  # Final LR will be 1% of initial LR
    }

    # === ADAM OPTIMIZER ===
    adam_beta1 = 0.9       # Adam optimizer beta1 parameter (momentum)
    adam_beta2 = 0.999     # Adam optimizer beta2 parameter (RMSprop)
    adam_epsilon = 1e-8    # Adam optimizer epsilon parameter (numerical stability)

    # === DENSIFICATION & PRUNING ===
    densification_interval = 100      # Perform densification every N iterations
    pruning_interval = 100            # Perform pruning every N iterations
    opacity_reset_interval = 3000    # Reset opacities every N iterations
    densify_grad_threshold = 0.0002  # Gradient threshold for densification
    cull_opacity_threshold = 0.005   # Opacity threshold for pruning
    start_prune_iter = 500           # Start pruning at this iteration
    end_prune_iter = 15000           # Stop pruning at this iteration
    percent_dense = 0.01             # Percentage of scene extent for densification
    max_allowed_prune_ratio = 1.0    # Maximum pruning ratio (1.0 = no limit)

    # === GAUSSIAN PARAMETERS ===
    initial_scale = 0.1    # Initial scale for Gaussian points
    scale_modifier = 1.0   # Scaling factor for Gaussian splats
    sh_degree = 3          # Spherical harmonics degree (0=diffuse, 3=view-dependent)

    # === SCENE & RENDERING ===
    scene_scale = 1.0                     # Scale factor for the scene
    background_color = [0.0, 0.0, 0.0]   # Background color (black for NeRF synthetic)
    near = 0.01                           # Near clipping plane
    far = 100.0                           # Far clipping plane

    # === LOSS FUNCTION ===
    lambda_dssim = 0.0     # Weight for SSIM loss (1.0=only SSIM, 0.0=only L1)

    @classmethod
    def update(cls, **kwargs):
        """Update parameters with new values."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    @classmethod
    def get_config_dict(cls):
        """Get parameters as a dictionary."""
        return {
            'num_iterations': cls.num_iterations,
            'num_points': cls.num_points,
            'save_interval': cls.save_interval,
            'use_lr_scheduler': cls.use_lr_scheduler,
            'lr_scheduler_config': cls.lr_scheduler_config,
            'adam_beta1': cls.adam_beta1,
            'adam_beta2': cls.adam_beta2,
            'adam_epsilon': cls.adam_epsilon,
            'densification_interval': cls.densification_interval,
            'pruning_interval': cls.pruning_interval,
            'opacity_reset_interval': cls.opacity_reset_interval,
            'densify_grad_threshold': cls.densify_grad_threshold,
            'cull_opacity_threshold': cls.cull_opacity_threshold,
            'start_prune_iter': cls.start_prune_iter,
            'end_prune_iter': cls.end_prune_iter,
            'percent_dense': cls.percent_dense,
            'max_allowed_prune_ratio': cls.max_allowed_prune_ratio,
            'initial_scale': cls.initial_scale,
            'scale_modifier': cls.scale_modifier,
            'sh_degree': cls.sh_degree,
            'scene_scale': cls.scene_scale,
            'background_color': cls.background_color,
            'near': cls.near,
            'far': cls.far,
            'lambda_dssim': cls.lambda_dssim,
        }

