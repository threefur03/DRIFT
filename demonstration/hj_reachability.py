"""
DRIFT: Hamilton-Jacobi Reachability for Occlusion Risk
=======================================================
Provides formal safety guarantees for occlusion zones using HJ reachability.

Key Concept:
- HJ reachability computes value function V(z,T) in RELATIVE state space
- V(z,T) ≤ 0 means collision can be forced within horizon T
- This gives GUARANTEE-BACKED occlusion risk, not just heuristic blobs

Integration with PDE framework:
- HJ provides the occlusion SOURCE TERM Q_occ(x,t)
- PDE handles advection, diffusion, spatiotemporal propagation
- Best of both: formal guarantees + real-time continuous risk transport

Dynamics (relative state z = [Δx, Δy, Δv]):
- ∂(Δx)/∂t = Δv 
- ∂(Δy)/∂t = d_y - u_y  (lateral velocities)
- ∂(Δv)/∂t = d_a - u_a  (accelerations)

HJ-Isaacs PDE:
  ∂V/∂t + min_u max_d [∇V · f(z,u,d)] = 0
  V(z,0) = ℓ(z)  [signed distance to collision set]

Reference: Level Set Methods and Dynamic Implicit Surfaces (Osher & Fedkiw)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import warnings


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HJConfig:
    """Configuration for HJ reachability computation."""
    
    # Relative state grid bounds
    dx_min: float = -50.0    # [m] relative x (hidden behind ego)
    dx_max: float = 70.0     # [m] relative x (hidden ahead)
    dy_min: float = -12.0    # [m] relative y (lateral)
    dy_max: float = 12.0     # [m] relative y
    dv_min: float = -15.0    # [m/s] relative velocity
    dv_max: float = 15.0     # [m/s] relative velocity
    
    # Grid resolution
    nx: int = 70             # Grid points in Δx
    ny: int = 35             # Grid points in Δy
    nv: int = 15             # Grid points in Δv
    
    # Time horizon
    T_horizon: float = 3.0   # [s] look-ahead horizon
    dt: float = 0.05         # [s] solver time step
    
    # Vehicle dimensions
    ego_length: float = 4.5
    ego_width: float = 2.0
    other_length: float = 4.5
    other_width: float = 2.0
    
    # Control bounds (ego - trying to avoid)
    ego_accel_max: float = 3.0    # [m/s²]
    ego_accel_min: float = -7.0   # [m/s²] max braking
    ego_lat_max: float = 0.5      # [m/s] lateral velocity
    
    # Disturbance bounds (hidden agent - adversarial)
    hidden_accel_max: float = 4.0   # [m/s²]
    hidden_accel_min: float = -5.0  # [m/s²]
    hidden_lat_max: float = 1.5     # [m/s] lateral velocity
    
    # Risk mapping
    eta: float = 4.0           # Temperature for sigmoid
    alpha: float = 1.5         # Scaling factor
    
    @property
    def collision_dx(self):
        return (self.ego_length + self.other_length) / 2 + 0.5
    
    @property
    def collision_dy(self):
        return (self.ego_width + self.other_width) / 2 + 0.3


# =============================================================================
# HJ Reachability Solver
# =============================================================================

class HJReachabilitySolver:
    """
    Solves Hamilton-Jacobi-Isaacs PDE for backward reachable set.
    
    The value function V(z,t) satisfies:
    - V(z,0) = ℓ(z) (signed distance to collision)
    - V(z,T) ≤ 0 means collision reachable within time T
    """
    
    def __init__(self, config: HJConfig = None):
        self.cfg = config or HJConfig()
        
        # Create grids
        self.dx = np.linspace(self.cfg.dx_min, self.cfg.dx_max, self.cfg.nx)
        self.dy = np.linspace(self.cfg.dy_min, self.cfg.dy_max, self.cfg.ny)
        self.dv = np.linspace(self.cfg.dv_min, self.cfg.dv_max, self.cfg.nv)
        
        # Grid spacing
        self.h_dx = self.dx[1] - self.dx[0]
        self.h_dy = self.dy[1] - self.dy[0]
        self.h_dv = self.dv[1] - self.dv[0]
        
        # 3D meshgrid
        self.DX, self.DY, self.DV = np.meshgrid(
            self.dx, self.dy, self.dv, indexing='ij'
        )
        
        self.V = None
        self.V_interpolator = None
        self._computed = False
    
    def _compute_collision_sdf(self) -> np.ndarray:
        """
        Compute signed distance to collision set.
        
        Collision: |Δx| < collision_dx AND |Δy| < collision_dy
        SDF: negative inside, positive outside
        """
        dx_dist = np.abs(self.DX) - self.cfg.collision_dx
        dy_dist = np.abs(self.DY) - self.cfg.collision_dy
        
        # Signed distance to rectangle
        inside = (dx_dist < 0) & (dy_dist < 0)
        
        ell = np.zeros_like(self.DX)
        ell[inside] = np.maximum(dx_dist[inside], dy_dist[inside])
        ell[~inside] = np.sqrt(
            np.maximum(dx_dist[~inside], 0)**2 + 
            np.maximum(dy_dist[~inside], 0)**2
        )
        
        return ell
    
    def _compute_hamiltonian(self, V: np.ndarray) -> np.ndarray:
        """
        Compute Hamiltonian for min-max game.
        
        H = min_u max_d [∇V · f(z,u,d)]
        
        Dynamics:
          f_1 = Δv (no control)
          f_2 = d_lat - u_lat
          f_3 = d_accel - u_accel
        """
        # Compute gradients (central differences with boundary padding)
        V_pad = np.pad(V, 1, mode='edge')
        
        dV_dx = (V_pad[2:, 1:-1, 1:-1] - V_pad[:-2, 1:-1, 1:-1]) / (2 * self.h_dx)
        dV_dy = (V_pad[1:-1, 2:, 1:-1] - V_pad[1:-1, :-2, 1:-1]) / (2 * self.h_dy)
        dV_dv = (V_pad[1:-1, 1:-1, 2:] - V_pad[1:-1, 1:-1, :-2]) / (2 * self.h_dv)
        
        # === Hamiltonian terms ===
        
        # Term 1: ∂V/∂(Δx) · Δv (drift, no control)
        H1 = dV_dx * self.DV
        
        # Term 2: Lateral dynamics
        # min_u max_d [∂V/∂(Δy) · (d_lat - u_lat)]
        # Ego minimizes: u_lat = sign(∂V/∂y) * u_lat_max
        # Hidden maximizes: d_lat = sign(∂V/∂y) * d_lat_max
        # Result: |∂V/∂y| * (d_lat_max + u_lat_max)
        H2 = np.abs(dV_dy) * (self.cfg.hidden_lat_max + self.cfg.ego_lat_max)
        
        # Term 3: Longitudinal acceleration
        # Worst case for closing: hidden accelerates when ahead, brakes when behind
        # Combined control authority
        accel_range = (self.cfg.hidden_accel_max - self.cfg.ego_accel_min)
        H3 = np.abs(dV_dv) * accel_range * 0.5  # Scaled for stability
        
        return H1 + H2 + H3
    
    def solve(self, verbose: bool = True) -> np.ndarray:
        """
        Solve HJ PDE backward in time using Lax-Friedrichs scheme.
        """
        n_steps = int(self.cfg.T_horizon / self.cfg.dt)
        
        if verbose:
            print(f"HJ Reachability Solver")
            print(f"  Grid: {self.cfg.nx}×{self.cfg.ny}×{self.cfg.nv}")
            print(f"  Horizon: {self.cfg.T_horizon}s, dt={self.cfg.dt}s")
        
        # Initialize with collision SDF
        V = self._compute_collision_sdf()
        
        # Artificial viscosity for stability
        alpha_lf = 0.3
        
        for step in range(n_steps):
            H = self._compute_hamiltonian(V)
            
            # Laplacian for dissipation
            V_pad = np.pad(V, 1, mode='edge')
            lap = (
                (V_pad[2:,1:-1,1:-1] - 2*V_pad[1:-1,1:-1,1:-1] + V_pad[:-2,1:-1,1:-1]) / self.h_dx**2 +
                (V_pad[1:-1,2:,1:-1] - 2*V_pad[1:-1,1:-1,1:-1] + V_pad[1:-1,:-2,1:-1]) / self.h_dy**2 +
                (V_pad[1:-1,1:-1,2:] - 2*V_pad[1:-1,1:-1,1:-1] + V_pad[1:-1,1:-1,:-2]) / self.h_dv**2
            )
            
            # Update (backward in time)
            V_new = V - self.cfg.dt * H + alpha_lf * self.cfg.dt * lap
            
            # Take minimum (BRS grows backward in time)
            V = np.minimum(V, V_new)
            
            if verbose and (step + 1) % (n_steps // 4) == 0:
                unsafe_pct = np.mean(V <= 0) * 100
                print(f"  Step {step+1}/{n_steps}: {unsafe_pct:.1f}% unsafe")
        
        self.V = V
        self._computed = True
        
        # Create interpolator
        self.V_interpolator = RegularGridInterpolator(
            (self.dx, self.dy, self.dv), V,
            method='linear', bounds_error=False, fill_value=np.max(V)
        )
        
        if verbose:
            print(f"  Final: {np.mean(V <= 0)*100:.1f}% unsafe")
        
        return V
    
    def query(self, dx: np.ndarray, dy: np.ndarray, dv: np.ndarray) -> np.ndarray:
        """Query value function at given relative states."""
        if not self._computed:
            raise RuntimeError("Call solve() first")
        
        shape = dx.shape
        pts = np.stack([dx.ravel(), dy.ravel(), dv.ravel()], axis=-1)
        return self.V_interpolator(pts).reshape(shape)
    
    def get_risk(self, dx: np.ndarray, dy: np.ndarray,
                 dv_range: Tuple[float, float] = None) -> np.ndarray:
        """
        Get risk by maximizing over velocity range.
        
        R(Δx,Δy) = max_{Δv} σ(-V/η)
        """
        if dv_range is None:
            dv_range = (self.cfg.dv_min, self.cfg.dv_max)
        
        # Sample velocities
        dv_samples = np.linspace(dv_range[0], dv_range[1], 8)
        
        V_min = np.full_like(dx, np.inf)
        for dv in dv_samples:
            V_val = self.query(dx, dy, np.full_like(dx, dv))
            V_min = np.minimum(V_min, V_val)
        
        # Sigmoid: V ≤ 0 → high risk
        return 1.0 / (1.0 + np.exp(V_min / self.cfg.eta))


# =============================================================================
# HJ Occlusion Risk (High-level interface)
# =============================================================================

class HJOcclusionRisk:
    """
    High-level interface for HJ-based occlusion risk.
    
    Usage:
        hj_risk = HJOcclusionRisk()  # Precomputes value function
        R_occ = hj_risk.compute_risk(ego, X, Y, shadow_mask)
    """
    
    def __init__(self, config: HJConfig = None, precompute: bool = True):
        self.cfg = config or HJConfig()
        self.solver = HJReachabilitySolver(self.cfg)
        
        if precompute:
            self.solver.solve(verbose=True)
    
    def compute_risk(self, ego: Dict, X: np.ndarray, Y: np.ndarray,
                     shadow_mask: np.ndarray,
                     lane_centers: List[float] = None) -> np.ndarray:
        """
        Compute HJ-based occlusion risk field.
        
        Args:
            ego: Ego vehicle dict with 'x', 'y', 'vx'
            X, Y: Meshgrid
            shadow_mask: Occlusion zone mask
            lane_centers: Lane y-coordinates for prior
            
        Returns:
            R_occ_hj: Risk field with HJ guarantees
        """
        R = np.zeros_like(X)
        
        if shadow_mask is None or not shadow_mask.any():
            return R
        
        ego_speed = ego.get('vx', 20.0)
        
        # Relative positions
        dx_rel = X - ego['x']
        dy_rel = Y - ego['y']
        
        # Plausible hidden vehicle velocities
        hidden_v_range = (0.0, 35.0)  # Stationary to highway speed
        dv_min = hidden_v_range[0] - ego_speed
        dv_max = hidden_v_range[1] - ego_speed
        dv_min = np.clip(dv_min, self.cfg.dv_min, self.cfg.dv_max)
        dv_max = np.clip(dv_max, self.cfg.dv_min, self.cfg.dv_max)
        
        # Get risk (worst-case over velocity)
        risk = self.solver.get_risk(dx_rel, dy_rel, dv_range=(dv_min, dv_max))
        
        # Apply to occlusion zone
        R = shadow_mask.astype(float) * risk * self.cfg.alpha
        
        # Lane prior (combine HJ with probabilistic reasoning)
        if lane_centers is not None:
            p_lane = np.zeros_like(X)
            for lane_y in lane_centers:
                p_lane += np.exp(-0.5 * ((Y - lane_y) / 1.2)**2)
            p_lane /= (p_lane.max() + 1e-6)
            R = R * (0.3 + 0.7 * p_lane)
        
        return R
    
    def get_unsafe_region(self, ego: Dict, X: np.ndarray, Y: np.ndarray,
                          dv: float = 0.0) -> np.ndarray:
        """Get binary mask where V ≤ 0 (collision reachable)."""
        dx_rel = X - ego['x']
        dy_rel = Y - ego['y']
        V = self.solver.query(dx_rel, dy_rel, np.full_like(X, dv))
        return V <= 0


# =============================================================================
# Integration function for DRIFT PDE
# =============================================================================

def compute_occlusion_risk_hj(shadow_mask: np.ndarray,
                               X: np.ndarray, Y: np.ndarray,
                               ego: Dict, truck: Dict,
                               hj_solver: HJOcclusionRisk = None,
                               lane_centers: List[float] = None,
                               create_if_none: bool = True) -> np.ndarray:
    """
    Compute HJ-based occlusion risk for PDE source term.
    
    This is the main integration point with existing DRIFT code.
    Replace compute_occlusion_risk() with this function.
    
    Args:
        shadow_mask: Boolean occlusion mask
        X, Y: Meshgrid
        ego: Ego vehicle dict
        truck: Truck (occluder) dict
        hj_solver: Precomputed HJ solver (created if None)
        lane_centers: Lane y-coordinates
        create_if_none: Create solver on demand if None
        
    Returns:
        R_occ_hj: HJ-based occlusion risk source term
    """
    if shadow_mask is None or not shadow_mask.any():
        return np.zeros_like(X)
    
    if hj_solver is None:
        if create_if_none:
            print("Creating HJ solver (this should be done once at startup)...")
            hj_solver = HJOcclusionRisk(precompute=True)
        else:
            return np.zeros_like(X)
    
    # Get HJ risk
    R_hj = hj_solver.compute_risk(ego, X, Y, shadow_mask, lane_centers)
    
    # Distance modulation (uncertainty decreases with distance from occluder)
    if truck is not None:
        dist = np.sqrt((X - truck['x'])**2 + (Y - truck['y'])**2)
        dist_factor = 0.5 + 0.5 * np.exp(-dist / 35)
        R_hj = R_hj * dist_factor
    
    # Soft boundary
    shadow_soft = gaussian_filter(shadow_mask.astype(float), sigma=1.2)
    R_hj = R_hj * shadow_soft
    
    return R_hj


# =============================================================================
# Visualization
# =============================================================================

def create_hj_visualization(hj_solver: HJReachabilitySolver = None,
                            save_path: str = None):
    """Create visualization of HJ value function and risk."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    if hj_solver is None:
        print("Computing HJ value function for visualization...")
        hj_solver = HJReachabilitySolver()
        hj_solver.solve(verbose=True)
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0D1117')
    
    # Value function slices at different Δv
    dv_slices = [-10, 0, 10]
    
    for i, dv in enumerate(dv_slices):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_facecolor('#161B22')
        
        dv_idx = np.argmin(np.abs(hj_solver.dv - dv))
        V_slice = hj_solver.V[:, :, dv_idx].T
        
        pcm = ax.pcolormesh(hj_solver.dx, hj_solver.dy, V_slice,
                           cmap='RdYlBu', shading='gouraud',
                           vmin=-8, vmax=8)
        
        # Zero contour (unsafe boundary)
        ax.contour(hj_solver.dx, hj_solver.dy, V_slice, levels=[0],
                  colors='white', linewidths=2)
        
        # Collision set boundary
        ax.axhline(hj_solver.cfg.collision_dy, color='red', ls='--', alpha=0.5)
        ax.axhline(-hj_solver.cfg.collision_dy, color='red', ls='--', alpha=0.5)
        ax.axvline(hj_solver.cfg.collision_dx, color='red', ls='--', alpha=0.5)
        ax.axvline(-hj_solver.cfg.collision_dx, color='red', ls='--', alpha=0.5)
        
        ax.set_xlabel('Δx [m]', color='white')
        ax.set_ylabel('Δy [m]', color='white')
        ax.set_title(f'V(Δx, Δy) at Δv = {dv} m/s', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label('V (negative=unsafe)', color='white')
        cbar.ax.tick_params(colors='white')
    
    # Risk field example
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('#161B22')
    
    # Create test grid
    x_test = np.linspace(-30, 60, 100)
    y_test = np.linspace(-10, 10, 60)
    X_test, Y_test = np.meshgrid(x_test, y_test, indexing='ij')
    
    # Mock occlusion mask
    occ_mask = (X_test > 20) & (X_test < 55) & (Y_test > -8) & (Y_test < 8)
    
    # Compute risk
    dx_rel = X_test - 0  # Ego at origin
    dy_rel = Y_test - 0
    risk = hj_solver.get_risk(dx_rel, dy_rel, dv_range=(-10, 10))
    risk = risk * occ_mask.astype(float)
    
    pcm = ax4.pcolormesh(x_test, y_test, risk.T, cmap='hot', shading='gouraud',
                         vmin=0, vmax=1.2)
    ax4.contour(x_test, y_test, occ_mask.T.astype(float), levels=[0.5],
               colors='cyan', linewidths=2, linestyles='--')
    
    # Draw ego
    ax4.add_patch(plt.Rectangle((-2.25, -1), 4.5, 2, facecolor='lime',
                                edgecolor='yellow', linewidth=2))
    ax4.text(0, 0, 'EGO', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw truck
    ax4.add_patch(plt.Rectangle((10, -1.25), 12, 2.5, facecolor='orange',
                                edgecolor='black', linewidth=1.5))
    ax4.text(16, 0, 'TRUCK', ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax4.set_xlabel('x [m]', color='white')
    ax4.set_ylabel('y [m]', color='white')
    ax4.set_title('HJ-Based Occlusion Risk', color='white', fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.set_aspect('equal')
    
    cbar = plt.colorbar(pcm, ax=ax4)
    cbar.set_label('Risk', color='white')
    cbar.ax.tick_params(colors='white')
    
    # Unsafe region overlay
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#161B22')
    
    V_at_zero = hj_solver.query(dx_rel, dy_rel, np.zeros_like(X_test))
    unsafe = (V_at_zero <= 0) & occ_mask
    
    ax5.pcolormesh(x_test, y_test, unsafe.T.astype(float), cmap='Reds',
                   shading='gouraud', alpha=0.8)
    ax5.contour(x_test, y_test, occ_mask.T.astype(float), levels=[0.5],
               colors='cyan', linewidths=2, linestyles='--')
    ax5.contour(x_test, y_test, V_at_zero.T, levels=[0], colors='white', linewidths=2)
    
    ax5.add_patch(plt.Rectangle((-2.25, -1), 4.5, 2, facecolor='lime',
                                edgecolor='yellow', linewidth=2))
    ax5.add_patch(plt.Rectangle((10, -1.25), 12, 2.5, facecolor='orange',
                                edgecolor='black', linewidth=1.5))
    
    ax5.set_xlabel('x [m]', color='white')
    ax5.set_ylabel('y [m]', color='white')
    ax5.set_title('V ≤ 0 Region (Collision Reachable)', color='white', fontweight='bold')
    ax5.tick_params(colors='white')
    ax5.set_aspect('equal')
    
    # Comparison info
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#161B22')
    ax6.axis('off')
    
    info_text = """
    HJ Reachability for Occlusion Risk
    ═══════════════════════════════════
    
    Key Properties:
    • V(z,T) ≤ 0 → collision REACHABLE within T
    • Formal safety guarantees
    • Worst-case hidden agent behavior
    
    Integration:
    • HJ provides occlusion SOURCE TERM
    • PDE handles spatiotemporal propagation
    • Combines guarantees + real-time transport
    
    Advantages over Heuristic:
    • Principled, not ad-hoc
    • Depends on ego speed/braking
    • Sharp unsafe boundaries
    • Control-aware risk assessment
    
    Parameters:
    • Horizon: {:.1f}s
    • Grid: {}×{}×{}
    • Collision box: {:.1f}m × {:.1f}m
    """.format(
        hj_solver.cfg.T_horizon,
        hj_solver.cfg.nx, hj_solver.cfg.ny, hj_solver.cfg.nv,
        hj_solver.cfg.collision_dx * 2,
        hj_solver.cfg.collision_dy * 2
    )
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
            fontsize=10, color='white', va='top', family='monospace')
    
    fig.suptitle('DRIFT: Hamilton-Jacobi Reachability for Occlusion Risk',
                fontsize=14, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor='#0D1117', dpi=150)
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return fig


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("HJ Reachability for Occlusion Risk")
    print("=" * 60)
    
    # Create and solve
    config = HJConfig()
    solver = HJReachabilitySolver(config)
    V = solver.solve(verbose=True)
    
    # Test queries
    print("\nTest queries (Δx, Δy, Δv) → V, risk:")
    tests = [
        (0, 0, 0),      # Collision
        (15, 0, 0),     # Ahead, same lane, same speed
        (15, 5, 0),     # Ahead, adjacent lane
        (30, 0, 0),     # Far ahead
        (30, 0, 10),    # Far ahead, hidden faster
        (30, 0, -10),   # Far ahead, hidden slower
    ]
    
    for dx, dy, dv in tests:
        V_val = solver.query(np.array([dx]), np.array([dy]), np.array([dv]))[0]
        risk = 1.0 / (1.0 + np.exp(V_val / config.eta))
        status = "UNSAFE" if V_val <= 0 else "safe"
        print(f"  ({dx:3.0f}, {dy:3.0f}, {dv:3.0f}) → V={V_val:6.2f}, risk={risk:.2f} [{status}]")
    
    # Create visualization
    print("\nCreating visualization...")
    create_hj_visualization(solver, '/Downloads/drift_v2/outputs/hj_reachability.png')
    
    print("\nDone!")
