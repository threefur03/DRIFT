"""
DRIFT: Dynamic Risk Inference via Field Transport
=================================================
CORRECTED visualization of Advection and Diffusion effects:

1. Advection: Risk field moves WITH traffic flow, merging vehicle introduces 
   risk that propagates downstream with the traffic
   
2. Diffusion: Risk spreads LOCALLY within/around occlusion zone (uncertainty 
   about hidden vehicles), NOT spreading everywhere

Based on AD4CHE occlusion visualization style.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import copy

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration for DRIFT visualization."""
    # Domain (follows vehicles)
    x_min, x_max = 0, 200
    y_min, y_max = -2, 22
    nx, ny = 250, 60
    
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    
    # PDE parameters
    D0 = 2.0              # Base diffusion (LOW - risk stays with sources)
    D_occ = 12.0          # Enhanced diffusion in occlusion only
    lambda_decay = 0.02   # Very slow decay
    
    # GVF kernel
    sigma_x = 10.0
    sigma_y = 2.0
    
    # Lane geometry
    lane_width = 3.7
    lane_centers = [2, 5.7, 9.4, 13.1, 16.8]  # 5 lanes
    
    # Merge zone
    merge_x_start = 100
    merge_x_end = 160
    
    # Colors
    BG_DARK = '#0D1117'
    BG_PANEL = '#161B22'
    SHADOW_COLOR = '#4A4A4A'
    ROAD_COLOR = '#1F2937'


cfg = Config()


def get_grid(x_offset=0):
    """Get meshgrid with optional x offset (for following vehicles)."""
    x = np.linspace(cfg.x_min + x_offset, cfg.x_max + x_offset, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    return np.meshgrid(x, y)


# =============================================================================
# Vehicle Functions
# =============================================================================

def create_vehicle(vid, x, y, vx, vy, vclass='car', length=None, width=None):
    """Create vehicle dictionary."""
    if length is None:
        length = 16.5 if vclass == 'truck_trailer' else (10.0 if vclass == 'truck' else 4.5)
    if width is None:
        width = 2.5 if 'truck' in vclass else 2.0
    heading = np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0
    return {
        'id': vid, 'x': x, 'y': y, 'vx': vx, 'vy': vy,
        'heading': heading, 'class': vclass, 'length': length, 'width': width
    }


def move_vehicle(v, dt, ax=0, ay=0):
    """Move vehicle forward in time."""
    if v is None:
        return None
    new_x = v['x'] + v['vx'] * dt
    new_y = v['y'] + v['vy'] * dt
    new_vx = max(10, v['vx'] + ax * dt)
    new_vy = v['vy'] + ay * dt
    new_y = np.clip(new_y, cfg.lane_centers[0] - 1, cfg.lane_centers[-1] + 1)
    return create_vehicle(v['id'], new_x, new_y, new_vx, new_vy,
                         v['class'], v['length'], v['width'])


def compute_shadow(ego, truck, X, Y):
    """Compute occlusion shadow polygon and mask."""
    dx = truck['x'] - ego['x']
    dy = truck['y'] - ego['y']
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 5 or dx < 0:
        return None, np.zeros_like(X, dtype=bool)
    
    L, W = truck['length'], truck['width']
    corners_local = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
    
    h = truck['heading']
    cos_h, sin_h = np.cos(h), np.sin(h)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (rot @ corners_local.T).T + np.array([truck['x'], truck['y']])
    
    angles = np.arctan2(corners[:, 1] - ego['y'], corners[:, 0] - ego['x'])
    left_idx, right_idx = np.argmax(angles), np.argmin(angles)
    left_corner, right_corner = corners[left_idx], corners[right_idx]
    
    shadow_length = 60
    left_dir = (left_corner - np.array([ego['x'], ego['y']]))
    left_dir = left_dir / (np.linalg.norm(left_dir) + 1e-6)
    right_dir = (right_corner - np.array([ego['x'], ego['y']]))
    right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-6)
    
    left_far = left_corner + left_dir * shadow_length
    right_far = right_corner + right_dir * shadow_length
    
    shadow_polygon = np.array([left_corner, left_far, right_far, right_corner])
    
    path = MplPath(shadow_polygon)
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask = path.contains_points(points).reshape(X.shape)
    
    return shadow_polygon, mask


# =============================================================================
# Risk Field Computation (Moves with Vehicles)
# =============================================================================

def compute_risk_from_vehicles(vehicles, X, Y, ego_id=0):
    """
    Compute risk field that is ATTACHED to vehicles.
    Risk is centered on each vehicle and moves with them.
    """
    R = np.zeros_like(X)
    
    for v in vehicles:
        if v is None:
            continue
        
        # Risk Gaussian centered on vehicle
        weight = 1.5 if 'truck' in v['class'] else 1.0
        
        # Anisotropic kernel aligned with vehicle
        h = v['heading']
        cos_h, sin_h = np.cos(h), np.sin(h)
        dX = X - v['x']
        dY = Y - v['y']
        dX_rot = cos_h * dX + sin_h * dY
        dY_rot = -sin_h * dX + cos_h * dY
        
        # Elongated along heading
        sigma_x = cfg.sigma_x * (1 + v['length'] / 10)
        sigma_y = cfg.sigma_y * (1 + v['width'] / 4)
        
        gaussian = np.exp(-0.5 * (dX_rot**2 / sigma_x**2 + dY_rot**2 / sigma_y**2))
        R += weight * gaussian
    
    return R


def compute_merge_risk(vehicles, X, Y):
    """Compute merge conflict risk from merging vehicles."""
    R_merge = np.zeros_like(X)
    
    for v in vehicles:
        if v is None:
            continue
        
        # Check if vehicle is in merge zone
        in_ramp = abs(v['y'] - cfg.lane_centers[-1]) < cfg.lane_width
        in_merge = cfg.merge_x_start < v['x'] < cfg.merge_x_end
        
        if in_ramp and in_merge:
            # Conflict zone: extends ahead and toward mainline
            dX = X - v['x']
            dY = Y - v['y']
            
            # Risk extends ahead and laterally toward mainline
            ahead = np.exp(-0.5 * ((dX - 15)**2 / 200))  # Peak ahead of vehicle
            lateral = np.exp(-0.5 * ((dY + 2)**2 / 20))  # Toward mainline
            
            # Urgency increases toward gore point
            urgency = 1 + 2 * (v['x'] - cfg.merge_x_start) / (cfg.merge_x_end - cfg.merge_x_start)
            
            R_merge += 1.2 * urgency * ahead * lateral
    
    return R_merge


def compute_occlusion_risk(shadow_mask, X, Y, truck):
    """
    Compute risk IN the occlusion zone only.
    This represents uncertainty about hidden vehicles.
    """
    R_occ = np.zeros_like(X)
    
    if shadow_mask is None or not shadow_mask.any():
        return R_occ
    
    # Risk only within shadow, concentrated at lane centers
    dX = X - truck['x']
    dist = np.sqrt(dX**2 + (Y - truck['y'])**2)
    
    # Emergence probability at lane centers
    p_lane = np.zeros_like(X)
    for lane_y in cfg.lane_centers:
        p_lane += np.exp(-0.5 * ((Y - lane_y) / 1.5)**2)
    p_lane = p_lane / (p_lane.max() + 1e-6)
    
    # Risk peaks at moderate distance from truck (where vehicles might be hiding)
    p_dist = np.exp(-((dist - 25)**2) / 400)
    
    R_occ = shadow_mask.astype(float) * 0.8 * p_lane * p_dist
    
    return R_occ


# =============================================================================
# Drawing Functions
# =============================================================================

def draw_road(ax, x_view_min, x_view_max):
    """Draw road with lanes."""
    ax.fill_between([x_view_min, x_view_max],
                   [cfg.lane_centers[0] - cfg.lane_width/2] * 2,
                   [cfg.lane_centers[-1] + cfg.lane_width/2] * 2,
                   color=cfg.ROAD_COLOR, alpha=0.3, zorder=0)
    
    for i, lane_y in enumerate(cfg.lane_centers):
        if i == 0:
            ax.axhline(y=lane_y - cfg.lane_width/2, color='white', 
                      linestyle='-', linewidth=2, alpha=0.8)
        if i == len(cfg.lane_centers) - 1:
            ax.axhline(y=lane_y + cfg.lane_width/2, color='white',
                      linestyle='-', linewidth=2, alpha=0.8)
        elif i < len(cfg.lane_centers) - 1:
            ax.axhline(y=lane_y + cfg.lane_width/2, color='white',
                      linestyle='--', linewidth=1.2, alpha=0.6)
    
    # Merge zone
    if cfg.merge_x_start < x_view_max and cfg.merge_x_end > x_view_min:
        ax.axvline(x=cfg.merge_x_start, color='yellow', linestyle=':', lw=1.5, alpha=0.5)
        ax.axvline(x=cfg.merge_x_end, color='red', linestyle=':', lw=2, alpha=0.7)
        
        merge_y_top = cfg.lane_centers[-1] + cfg.lane_width/2
        merge_y_bot = cfg.lane_centers[-2]
        ax.fill_between([max(cfg.merge_x_start, x_view_min), min(cfg.merge_x_end, x_view_max)],
                       [merge_y_bot] * 2, [merge_y_top] * 2,
                       color='orange', alpha=0.1, zorder=0)


def draw_shadow(ax, shadow_polygon, alpha=0.5):
    """Draw occlusion shadow."""
    if shadow_polygon is None:
        return
    patch = plt.Polygon(shadow_polygon, facecolor=cfg.SHADOW_COLOR, alpha=alpha,
                       edgecolor='red', linewidth=2.5, linestyle='--', zorder=1)
    ax.add_patch(patch)
    
    center = shadow_polygon.mean(axis=0)
    ax.text(center[0], center[1], 'OCCLUSION\nZONE',
           ha='center', va='center', fontsize=10, color='red',
           fontweight='bold', alpha=0.9,
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))


def draw_vehicle(ax, v, is_ego=False, is_merging=False):
    """Draw vehicle."""
    if v is None:
        return
    
    x, y = v['x'], v['y']
    L, W = v['length'], v['width']
    
    if 'truck' in v['class']:
        cab_len = 3.5
        trailer_len = L - cab_len - 1
        trailer = mpatches.Rectangle((x - L/2, y - W/2), trailer_len, W,
                                     facecolor='orange', edgecolor='black',
                                     linewidth=1.5, zorder=10)
        ax.add_patch(trailer)
        cab = mpatches.Rectangle((x - L/2 + trailer_len + 1, y - W/2), cab_len, W,
                                facecolor='darkblue', edgecolor='black',
                                linewidth=1.5, zorder=11)
        ax.add_patch(cab)
    else:
        if is_ego:
            color = '#2ECC71'
            ec = 'yellow'
            lw = 3
        elif is_merging:
            color = '#F39C12'
            ec = 'white'
            lw = 2
        else:
            color = '#3498DB'
            ec = 'white'
            lw = 1.2
        
        rect = mpatches.FancyBboxPatch((x - L/2, y - W/2), L, W,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor=ec,
                                       linewidth=lw, zorder=10)
        ax.add_patch(rect)
    
    speed = np.sqrt(v['vx']**2 + v['vy']**2)
    label = 'EGO' if is_ego else ('MERGE' if is_merging else f"{speed:.0f} m/s")
    ax.text(x, y - W/2 - 1.5, label, ha='center', va='top',
           fontsize=8, color='yellow' if is_ego else 'white', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7))


# =============================================================================
# ADVECTION Animation - Risk Moves WITH Traffic
# =============================================================================

def create_advection_animation(output_path='./output', n_frames=60, fps=8):
    """
    ADVECTION: Risk field moves WITH traffic flow.
    - Risk is attached to vehicles
    - As vehicles move downstream, risk moves with them
    - Merging vehicle introduces risk that flows with traffic
    """
    print("\n" + "=" * 70)
    print("DRIFT: ADVECTION Effect")
    print("Risk field MOVES WITH traffic flow")
    print("=" * 70)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Create vehicles - they will move and risk follows
    vehicles = [
        create_vehicle(0, 40, cfg.lane_centers[1], 20, 0, 'car'),  # Ego
        create_vehicle(1, 80, cfg.lane_centers[1], 18, 0, 'truck_trailer'),  # Truck
        create_vehicle(2, 50, cfg.lane_centers[0], 24, 0, 'car'),
        create_vehicle(3, 100, cfg.lane_centers[0], 22, 0, 'car'),
        create_vehicle(4, 70, cfg.lane_centers[2], 19, 0, 'car'),
        create_vehicle(5, 110, cfg.lane_centers[4], 18, -0.3, 'car'),  # Merging!
    ]
    
    # Track which is merging
    merging_id = 5
    
    dt = 0.15
    frames = []
    
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(cfg.BG_DARK)
    
    print(f"Generating {n_frames} frames...")
    
    for step in range(n_frames):
        if step % 10 == 0:
            print(f"  Frame {step}/{n_frames}")
        
        fig.clf()
        
        ego = vehicles[0]
        truck = vehicles[1]
        merging = vehicles[merging_id] if merging_id < len(vehicles) else None
        
        # View follows ego
        view_center = ego['x'] + 40
        x_view_min = view_center - 60
        x_view_max = view_center + 80
        
        X, Y = get_grid(x_offset=x_view_min)
        
        # Compute risk field ATTACHED to vehicles (moves with them)
        R_vehicles = compute_risk_from_vehicles(vehicles, X, Y, ego_id=0)
        
        # Add merge conflict risk
        R_merge = compute_merge_risk(vehicles, X, Y)
        
        # Shadow for visualization
        shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)
        
        # Occlusion risk (in shadow zone)
        R_occ = compute_occlusion_risk(shadow_mask, X, Y, truck)
        
        # Total risk
        R_total = R_vehicles + R_merge + R_occ
        R_total = gaussian_filter(R_total, sigma=1.0)
        
        # ---- Main visualization ----
        ax = fig.add_subplot(111)
        ax.set_facecolor(cfg.BG_PANEL)
        
        # Risk field
        vmax = 2.0
        pcm = ax.pcolormesh(X, Y, R_total, cmap='jet', shading='gouraud',
                           vmin=0, vmax=vmax)
        
        # Contours
        if R_total.max() > 0.2:
            levels = np.linspace(0.3, min(R_total.max(), vmax) * 0.9, 5)
            ax.contour(X, Y, R_total, levels=levels, colors='white',
                      linewidths=1, alpha=0.6)
        
        # Draw road and shadow
        draw_road(ax, x_view_min, x_view_max)
        draw_shadow(ax, shadow_poly, alpha=0.4)
        
        # Draw vehicles
        for v in vehicles:
            if v is None:
                continue
            is_ego = (v['id'] == 0)
            is_merging = (v['id'] == merging_id)
            draw_vehicle(ax, v, is_ego=is_ego, is_merging=is_merging)
        
        # Velocity arrows showing traffic flow direction
        skip = 15
        flow_vx = 20 * np.ones_like(X)  # Traffic flows right
        flow_vy = np.zeros_like(Y)
        
        # Add merge drift in merge zone
        in_merge_zone = (X > cfg.merge_x_start) & (X < cfg.merge_x_end) & (Y > cfg.lane_centers[-2])
        flow_vy = np.where(in_merge_zone, -3, 0)
        
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                 flow_vx[::skip, ::skip], flow_vy[::skip, ::skip],
                 color='cyan', scale=250, alpha=0.4, width=0.003, zorder=2)
        
        ax.set_xlim(x_view_min, x_view_max)
        ax.set_ylim(cfg.y_min, cfg.y_max)
        ax.set_aspect('equal')
        
        time = step * dt
        ax.set_title(f'ADVECTION: Risk Moves WITH Traffic | t = {time:.1f}s | Step {step+1}\n'
                    f'Risk field attached to vehicles, propagates downstream with flow',
                    fontsize=13, fontweight='bold', color='white')
        ax.set_xlabel('x [m]', color='white', fontsize=11)
        ax.set_ylabel('y [m]', color='white', fontsize=11)
        ax.tick_params(colors='white')
        
        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Risk Value', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')
        
        # Info text
        info = f"Traffic speed: ~20 m/s\nMerge vehicle moving toward mainline"
        ax.text(x_view_min + 5, cfg.y_max - 1, info,
               fontsize=9, color='white', va='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Capture frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())
        
        # Move ALL vehicles (risk moves with them!)
        new_vehicles = []
        for v in vehicles:
            if v is None:
                new_vehicles.append(None)
                continue
            
            if v['id'] == merging_id:
                # Merging vehicle: slow down and move laterally
                new_v = move_vehicle(v, dt, ax=-0.3, ay=-1.5 if v['y'] > cfg.lane_centers[-2] else 0)
                if new_v['y'] < cfg.lane_centers[-2]:
                    new_v['vy'] = 0
            else:
                new_v = move_vehicle(v, dt, ax=0, ay=0)
            
            new_vehicles.append(new_v)
        
        vehicles = new_vehicles
    
    plt.close()
    
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, 'drift_advection_correct.gif')
        print(f"\nSaving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"Saved: {gif_path}")
        return gif_path
    
    return None


# =============================================================================
# DIFFUSION Animation - Risk Spreads LOCALLY in Occlusion
# =============================================================================

def create_diffusion_animation(output_path='./output', n_frames=60, fps=8):
    """
    DIFFUSION: Risk spreads LOCALLY, especially in occlusion zone.
    - Start with localized risk in occlusion zone
    - Show how diffusion spreads it within that region
    - Enhanced D in occlusion means faster spreading THERE
    """
    print("\n" + "=" * 70)
    print("DRIFT: DIFFUSION Effect")
    print("Risk spreads LOCALLY within occlusion zone")
    print("=" * 70)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Static vehicles (focus on diffusion, not motion)
    vehicles = [
        create_vehicle(0, 30, cfg.lane_centers[1], 20, 0, 'car'),  # Ego
        create_vehicle(1, 70, cfg.lane_centers[1], 18, 0, 'truck_trailer'),  # Truck
        create_vehicle(2, 45, cfg.lane_centers[0], 22, 0, 'car'),
        create_vehicle(3, 60, cfg.lane_centers[2], 19, 0, 'car'),
    ]
    
    ego = vehicles[0]
    truck = vehicles[1]
    
    # Fixed grid
    X, Y = get_grid(x_offset=0)
    
    # Compute shadow
    shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)
    
    # Diffusion coefficient: D0 outside, D0+D_occ inside shadow
    D = cfg.D0 * np.ones_like(X)
    D[shadow_mask] += cfg.D_occ
    D = gaussian_filter(D, sigma=1.5)
    
    # Initialize risk: LOCALIZED sources in/near occlusion zone
    # This represents initial uncertainty about hidden vehicles
    R = np.zeros_like(X)
    
    # Put initial risk at lane centers INSIDE the shadow
    for lane_y in cfg.lane_centers[1:3]:  # Middle lanes
        # Point source at moderate distance in shadow
        source_x = truck['x'] + 25
        source_y = lane_y
        R += 1.5 * np.exp(-((X - source_x)**2 / 30 + (Y - source_y)**2 / 8))
    
    # Also some risk near truck edges (where cut-ins happen)
    R += 1.0 * np.exp(-((X - truck['x'] - 15)**2 / 40 + (Y - truck['y'] - 3)**2 / 6))
    R += 1.0 * np.exp(-((X - truck['x'] - 15)**2 / 40 + (Y - truck['y'] + 3)**2 / 6))
    
    # Mask to keep risk primarily in shadow region
    R = R * (0.2 + 0.8 * shadow_mask.astype(float))
    
    dt = 0.08
    frames = []
    
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(cfg.BG_DARK)
    
    print(f"Generating {n_frames} frames...")
    
    for step in range(n_frames):
        if step % 10 == 0:
            print(f"  Frame {step}/{n_frames}")
        
        fig.clf()
        gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2,
                     height_ratios=[1, 1.2])
        
        # ---- Panel 1: Diffusion coefficient D(x) ----
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(cfg.BG_PANEL)
        
        pcm1 = ax1.pcolormesh(X, Y, D, cmap='Greens', shading='gouraud',
                             vmin=cfg.D0, vmax=cfg.D0 + cfg.D_occ)
        
        # Shadow boundary
        ax1.contour(X, Y, shadow_mask.astype(float), levels=[0.5],
                   colors='red', linewidths=3, linestyles='--')
        
        draw_road(ax1, X.min(), X.max())
        for v in vehicles:
            draw_vehicle(ax1, v, is_ego=(v['id'] == 0))
        
        ax1.set_xlim(20, 150)
        ax1.set_ylim(cfg.y_min, cfg.y_max)
        ax1.set_aspect('equal')
        ax1.set_title('Diffusion Coefficient $D(x)$\nEnhanced in occlusion (red boundary)',
                     fontsize=11, fontweight='bold', color='white')
        ax1.tick_params(colors='white')
        cbar1 = plt.colorbar(pcm1, ax=ax1, shrink=0.7)
        cbar1.set_label('D [m²/s]', color='white')
        cbar1.ax.tick_params(colors='white')
        
        # Annotation
        ax1.annotate(f'$D_0$ = {cfg.D0}', xy=(30, 15), fontsize=10,
                    color='white', fontweight='bold')
        ax1.annotate(f'$D_0 + D_{{occ}}$ = {cfg.D0 + cfg.D_occ}', 
                    xy=(90, 8), fontsize=10, color='lime', fontweight='bold')
        
        # ---- Panel 2: Shadow zone detail ----
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(cfg.BG_PANEL)
        
        occ_cmap = LinearSegmentedColormap.from_list('occ', 
            [(0, '#161B22'), (0.3, '#2D2D4A'), (0.7, '#6B4A8A'), (1, '#EC4899')])
        pcm2 = ax2.pcolormesh(X, Y, shadow_mask.astype(float), 
                             cmap=occ_cmap, shading='gouraud', alpha=0.8)
        
        draw_road(ax2, X.min(), X.max())
        draw_shadow(ax2, shadow_poly, alpha=0.6)
        for v in vehicles:
            draw_vehicle(ax2, v, is_ego=(v['id'] == 0))
        
        ax2.set_xlim(20, 150)
        ax2.set_ylim(cfg.y_min, cfg.y_max)
        ax2.set_aspect('equal')
        ax2.set_title('Occlusion Shadow $\\Omega_{occ}$\nSensor blind zone behind truck',
                     fontsize=11, fontweight='bold', color='white')
        ax2.tick_params(colors='white')
        
        # ---- Panel 3: Risk field with diffusion ----
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor(cfg.BG_PANEL)
        
        R_vis = gaussian_filter(R, sigma=0.5)
        vmax = 1.5
        pcm3 = ax3.pcolormesh(X, Y, R_vis, cmap='jet', shading='gouraud',
                             vmin=0, vmax=vmax)
        
        # Contours
        if R_vis.max() > 0.1:
            levels = np.linspace(0.15, min(R_vis.max(), vmax) * 0.9, 6)
            ax3.contour(X, Y, R_vis, levels=levels, colors='white',
                       linewidths=0.8, alpha=0.6)
        
        # Show enhanced D boundary
        ax3.contour(X, Y, D, levels=[(cfg.D0 + cfg.D_occ/2)],
                   colors='lime', linewidths=2.5, linestyles='-')
        
        draw_road(ax3, X.min(), X.max())
        draw_shadow(ax3, shadow_poly, alpha=0.3)
        for v in vehicles:
            draw_vehicle(ax3, v, is_ego=(v['id'] == 0))
        
        ax3.set_xlim(20, 150)
        ax3.set_ylim(cfg.y_min, cfg.y_max)
        ax3.set_aspect('equal')
        
        time = step * dt
        ax3.set_title(f'Risk Spreading via DIFFUSION | t = {time:.1f}s | Step {step+1}\n'
                     f'$\\partial R/\\partial t = \\nabla \\cdot (D \\nabla R)$ — '
                     f'Risk spreads FASTER inside occlusion (green boundary)',
                     fontsize=12, fontweight='bold', color='white')
        ax3.set_xlabel('x [m]', color='white', fontsize=11)
        ax3.set_ylabel('y [m]', color='white', fontsize=11)
        ax3.tick_params(colors='white')
        
        cbar3 = plt.colorbar(pcm3, ax=ax3, shrink=0.6, pad=0.02)
        cbar3.set_label('Risk (Uncertainty)', color='white', fontsize=10)
        cbar3.ax.tick_params(colors='white')
        
        # Info
        ax3.text(22, cfg.y_max - 1, 
                'Diffusion represents\nuncertainty spreading\nabout hidden vehicles',
                fontsize=9, color='white', va='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        fig.suptitle('DRIFT: Diffusion Effect — Risk Spreads LOCALLY in Occlusion\n'
                    'Enhanced diffusion $D_{occ}$ in shadow zone represents greater uncertainty',
                    fontsize=14, fontweight='bold', color='white', y=0.98)
        
        # Capture frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())
        
        # Evolve diffusion (risk spreads within shadow faster)
        # Laplacian with variable D
        laplacian = np.zeros_like(R)
        laplacian[1:-1, 1:-1] = D[1:-1, 1:-1] * (
            (R[1:-1, 2:] - 2*R[1:-1, 1:-1] + R[1:-1, :-2]) / cfg.dx**2 +
            (R[2:, 1:-1] - 2*R[1:-1, 1:-1] + R[:-2, 1:-1]) / cfg.dy**2
        )
        
        R = R + dt * laplacian
        
        # Very slow decay
        R = R * (1 - cfg.lambda_decay * dt)
        
        # Keep some risk in shadow (continuous uncertainty)
        R_source = 0.02 * shadow_mask.astype(float) * np.exp(-((X - truck['x'] - 20)**2 / 200))
        R = R + R_source
        
        R = np.clip(R, 0, 3)
        
        # Boundary conditions
        R[0, :] = R[1, :]
        R[-1, :] = R[-2, :]
        R[:, 0] = R[:, 1]
        R[:, -1] = R[:, -2]
    
    plt.close()
    
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, 'drift_diffusion_correct.gif')
        print(f"\nSaving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"Saved: {gif_path}")
        return gif_path
    
    return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DRIFT: Corrected Advection vs Diffusion')
    parser.add_argument('-o', '--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames')
    parser.add_argument('--fps', type=int, default=8, help='Frames per second')
    
    args = parser.parse_args()
    
    create_advection_animation(args.output, n_frames=args.frames, fps=args.fps)
    create_diffusion_animation(args.output, n_frames=args.frames, fps=args.fps)
    
    print("\n" + "=" * 70)
    print("Done!")
    print(f"Output saved to: {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
