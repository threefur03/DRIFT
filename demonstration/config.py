"""
PDE Risk Field Configuration
Advection-Diffusion-Telegrapher Model for Traffic Risk Propagation
"""
import numpy as np


class Config:
    """Configuration for the PDE risk field solver."""
    
    # ============ Spatial Domain ============
    x_min, x_max = -30, 100   # Longitudinal range [m]
    y_min, y_max = -15, 15    # Lateral range [m]
    nx, ny = 150, 70          # Grid resolution
    
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # ============ PDE Parameters ============
    # Diffusion
    D0 = 8.0              # Base diffusion coefficient [m²/s]
    D_occ = 15.0          # Additional diffusion in occluded regions
    
    # Decay
    lambda_decay = 0.1    # Risk decay rate [1/s]
    
    # Telegrapher (wave-like propagation)
    tau = 0.3             # Inertia/reaction time [s]
    
    # ============ Source Parameters ============
    # GVF-style Gaussian kernel
    sigma_x = 12.0        # Longitudinal spread [m]
    sigma_y = 3.0         # Lateral spread [m]
    
    # ============ Merge Zone ============
    merge_x_start = 30.0  # Merge zone start [m]
    merge_x_end = 70.0    # Merge zone end (gore point) [m]
    merge_y_ramp = 6.0    # Ramp lane y-position [m]
    
    # ============ Lane Geometry ============
    lane_width = 4.0
    lane_centers = [-4.0, 0.0, 4.0, 8.0]  # Lane center y-positions
    
    # ============ Vehicle Dimensions ============
    car_length = 5.0
    car_width = 2.0
    truck_length = 12.0
    truck_width = 2.5
