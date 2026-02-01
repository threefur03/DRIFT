"""
DRIFT: Dynamic Risk Inference via Field Transport
-------------------------------------------------
This version uses the actual PDE solver (advection–diffusion–telegrapher)
from `pde_solver.PDESolver` instead of synthetic, hand-smoothed fields.

High-level flow per frame:
1) Build source term Q from vehicles, merge zone, and occlusion shadow.
2) Build spatially varying diffusion D (higher inside occlusion).
3) Build velocity field v(x,y) from surrounding traffic (GVF-style).
4) Advance the PDE with several sub-steps for stability.
5) Render solver.R as the risk field.
"""

import os
import copy
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec
import numpy.random as npr
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from config import Config as cfg
from pde_solver import PDESolver, compute_velocity_field

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio")


# =============================================================================
# Visualization colors (kept local)
# =============================================================================

BG_DARK = "#0D1117"
BG_PANEL = "#161B22"
SHADOW_COLOR = "#4A4A4A"
ROAD_COLOR = "#1F2937"


def get_grid():
    """Use the global grid from config (shared with PDE solver)."""
    return cfg.X, cfg.Y


def build_diffusion_field(
    occ_mask,
    base_scale=0.10,
    occ_scale=1.8,
    skirt_gain=0.6,
    skirt_sigma=3.0,
    smooth_sigma=0.65,
):
    """
    Visualization-tuned diffusion:
    - Low background diffusion to keep fields from washing out.
    - Boosted diffusion confined to occlusion to make that plume visible.
    - A soft 'skirt' around the occlusion so risk can leak out smoothly.
    """
    D = cfg.D0 * base_scale * np.ones_like(cfg.X)
    if occ_mask is not None and occ_mask.any():
        core = gaussian_filter(occ_mask.astype(float), sigma=1.2)
        skirt = gaussian_filter(occ_mask.astype(float), sigma=skirt_sigma)
        D += cfg.D_occ * occ_scale * core
        D += cfg.D_occ * skirt_gain * skirt
    return gaussian_filter(D, sigma=smooth_sigma)


# =============================================================================
# Vehicle utilities (parameterized by cfg values)
# =============================================================================

def create_vehicle(vid, x, y, vx, vy, vclass="car", length=None, width=None):
    """Create vehicle dictionary."""
    if length is None:
        if vclass == "truck":
            length = cfg.truck_length
        elif "truck_trailer" in vclass or "trailer" in vclass:
            length = cfg.truck_length * 1.5  # longer than single-unit truck
        else:
            length = cfg.car_length
    if width is None:
        width = cfg.truck_width if "truck" in vclass else cfg.car_width
    heading = np.arctan2(vy, vx) if (vx != 0 or vy != 0) else 0
    return {
        "id": vid,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "heading": heading,
        "class": vclass,
        "length": length,
        "width": width,
    }


def move_vehicle(v, dt, ax=0, ay=0):
    """Move vehicle forward in time; clamp to roadway bounds."""
    if v is None:
        return None
    new_x = v["x"] + v["vx"] * dt + 0.5 * ax * dt**2
    new_y = v["y"] + v["vy"] * dt + 0.5 * ay * dt**2
    new_vx = v["vx"] + ax * dt
    new_vy = v["vy"] + ay * dt

    # Keep within lane band
    new_y = np.clip(new_y, cfg.lane_centers[0] - 1, cfg.lane_centers[-1] + 1)

    # Simple wrap so cars stay in view
    if new_x > cfg.x_max - 2:
        new_x = cfg.x_min + 5
    if new_x < cfg.x_min + 1:
        new_x = cfg.x_min + 5

    return create_vehicle(v["id"], new_x, new_y, new_vx, new_vy, v["class"], v["length"], v["width"])


# =============================================================================
# Occlusion geometry
# =============================================================================

def compute_shadow(ego, truck, X, Y):
    """Compute occlusion shadow polygon and mask."""
    if truck is None or ego is None:
        return None, np.zeros_like(X, dtype=bool)

    dx = truck["x"] - ego["x"]
    dy = truck["y"] - ego["y"]
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 3 or dx < 0:
        return None, np.zeros_like(X, dtype=bool)

    L, W = truck["length"], truck["width"]
    corners_local = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])

    h = truck["heading"]
    cos_h, sin_h = np.cos(h), np.sin(h)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (rot @ corners_local.T).T + np.array([truck["x"], truck["y"]])

    angles = np.arctan2(corners[:, 1] - ego["y"], corners[:, 0] - ego["x"])
    left_idx, right_idx = np.argmax(angles), np.argmin(angles)
    left_corner, right_corner = corners[left_idx], corners[right_idx]

    shadow_length = 40
    left_dir = (left_corner - np.array([ego["x"], ego["y"]])) / (np.linalg.norm(left_corner - np.array([ego["x"], ego["y"]])) + 1e-6)
    right_dir = (right_corner - np.array([ego["x"], ego["y"]])) / (np.linalg.norm(right_corner - np.array([ego["x"], ego["y"]])) + 1e-6)

    left_far = left_corner + left_dir * shadow_length
    right_far = right_corner + right_dir * shadow_length

    shadow_polygon = np.array([left_corner, left_far, right_far, right_corner])

    path = MplPath(shadow_polygon)
    points = np.column_stack([X.ravel(), Y.ravel()])
    mask = path.contains_points(points).reshape(X.shape)

    return shadow_polygon, mask


# =============================================================================
# Source terms (become Q for PDE)
# =============================================================================

def compute_risk_from_vehicles(vehicles, X, Y):
    """Risk Gaussian centered on each vehicle; moves with them."""
    R = np.zeros_like(X)
    for v in vehicles:
        if v is None:
            continue
        weight = 1.5 if "truck" in v["class"] else 1.0
        h = v["heading"]
        cos_h, sin_h = np.cos(h), np.sin(h)
        dX = X - v["x"]
        dY = Y - v["y"]
        dX_rot = cos_h * dX + sin_h * dY
        dY_rot = -sin_h * dX + cos_h * dY
        sigma_x = cfg.sigma_x * (1 + v["length"] / 12)
        sigma_y = cfg.sigma_y * 0.6 * (1 + v["width"] / 6)
        gaussian = np.exp(-0.5 * (dX_rot**2 / sigma_x**2 + dY_rot**2 / sigma_y**2))
        R += weight * gaussian
    return R


def _smoothstep(s):
    s_clamped = np.clip(s, 0, 1)
    return 3 * s_clamped**2 - 2 * s_clamped**3


def compute_merge_risk(vehicles, X, Y, merging_id=None):
    """
    Merge conflict source localized to ramp/mainline interface.

    Enhancements:
    - Smooth onset (smoothstep) as the merge car progresses.
    - Stronger intensity near gore; exponential decay after merge completes.
    - Gaussian smoothing at the end to blend into main field.
    """
    R_merge = np.zeros_like(X)
    merge_len = cfg.merge_x_end - cfg.merge_x_start
    decay_len = 40.0  # meters past gore to fade/hand off

    for v in vehicles:
        if v is None:
            continue

        is_merge_car = merging_id is not None and v.get("id") == merging_id
        in_ramp_band = abs(v["y"] - cfg.lane_centers[-1]) < cfg.lane_width
        in_target_band = abs(v["y"] - cfg.lane_centers[-2]) < cfg.lane_width
        in_merge_x = (cfg.merge_x_start - 5) < v["x"] < (cfg.merge_x_end + decay_len)
        if not ((in_ramp_band or (is_merge_car and in_target_band)) and in_merge_x):
            continue

        dX = X - v["x"]
        dY = Y - v["y"]

        # Longitudinal envelope ahead of vehicle
        ahead = np.exp(-0.5 * ((dX - 8) ** 2 / 140))

        # Lateral envelope aiming toward mainline center lane
        target_y = cfg.lane_centers[-2]
        # Blend toward mainline as x approaches gore, then fully to mainline after
        mix = np.clip((v["x"] - (cfg.merge_x_end - 8)) / 12.0, 0, 1)
        aim_y = (1 - mix) * v["y"] + mix * target_y
        lateral = np.exp(-0.5 * ((Y - aim_y) ** 2 / 14))

        # Progress-based intensity
        progress = (v["x"] - cfg.merge_x_start) / merge_len
        progress = np.clip(progress, 0, 1.2)  # allow slight overshoot for decay trigger
        rise = _smoothstep(progress)

        if merging_id is not None and v.get("id") == merging_id:
            # Highlight merge car: sharper rise, visible decay after gore
            if v["x"] <= cfg.merge_x_end:
                decay = 1.0
            else:
                over = v["x"] - cfg.merge_x_end
                decay = np.exp(-over / decay_len)
            gain = 1.6 * rise * decay
        else:
            gain = 0.6 * (0.4 + 0.6 * rise)

        # Hand-off kernel to smear into mainline once past gore
        handoff = 1.0
        if v["x"] > cfg.merge_x_end:
            over = v["x"] - cfg.merge_x_end
            handoff = np.exp(-over / (decay_len * 0.6))

        R_merge += gain * ahead * lateral * handoff

    # Smooth the merge field so it blends into the main risk
    if R_merge.max() > 0:
        R_merge = gaussian_filter(R_merge, sigma=1.6)
    return R_merge


def compute_occlusion_risk(shadow_mask, X, Y, truck):
    """Risk only within shadow, concentrated near lane centers."""
    R_occ = np.zeros_like(X)
    if shadow_mask is None or not shadow_mask.any() or truck is None:
        return R_occ

    dist = np.sqrt((X - truck["x"]) ** 2 + (Y - truck["y"]) ** 2)

    # Lane prior to keep risk centered on likely hidden vehicles
    p_lane = np.zeros_like(X)
    for lane_y in cfg.lane_centers:
        p_lane += np.exp(-0.5 * ((Y - lane_y) / 1.0) ** 2)
    p_lane = p_lane / (p_lane.max() + 1e-6)

    # Distance prior: peak a bit behind the truck, taper outward
    p_dist = np.exp(-((dist - 18) ** 2) / 220)

    core = shadow_mask.astype(float) * 1.1 * p_lane * p_dist

    # Soft halo around occlusion to allow smooth propagation outside
    halo = gaussian_filter(shadow_mask.astype(float), sigma=2.5) * 0.6 * p_lane

    R_occ = core + halo
    return R_occ


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_road(ax):
    ax.fill_between([cfg.x_min, cfg.x_max], [cfg.lane_centers[0] - cfg.lane_width / 2] * 2, [cfg.lane_centers[-1] + cfg.lane_width / 2] * 2, color=ROAD_COLOR, alpha=0.3, zorder=0)
    for i, lane_y in enumerate(cfg.lane_centers):
        if i == 0:
            ax.axhline(y=lane_y - cfg.lane_width / 2, color="white", linestyle="-", linewidth=2, alpha=0.8)
        if i == len(cfg.lane_centers) - 1:
            ax.axhline(y=lane_y + cfg.lane_width / 2, color="white", linestyle="-", linewidth=2, alpha=0.8)
        elif i < len(cfg.lane_centers) - 1:
            ax.axhline(y=lane_y + cfg.lane_width / 2, color="white", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(x=cfg.merge_x_start, color="yellow", linestyle=":", lw=1.5, alpha=0.5)
    ax.axvline(x=cfg.merge_x_end, color="red", linestyle=":", lw=2, alpha=0.7)


def draw_shadow(ax, shadow_polygon, alpha=0.45):
    if shadow_polygon is None:
        return
    patch = plt.Polygon(shadow_polygon, facecolor=SHADOW_COLOR, alpha=alpha, edgecolor="red", linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(patch)


def draw_vehicle(ax, v, is_ego=False, is_merging=False):
    if v is None:
        return
    x, y = v["x"], v["y"]
    L, W = v["length"], v["width"]
    if "truck" in v["class"]:
        gap = 0.8
        cab_len = min(4.5, max(3.0, 0.32 * L))
        trailer_len = max(0.6 * L, L - cab_len - gap)
        trailer = mpatches.Rectangle((x - L / 2, y - W / 2), trailer_len, W, facecolor="orange", edgecolor="black", linewidth=1.5, zorder=10)
        ax.add_patch(trailer)
        cab = mpatches.Rectangle((x - L / 2 + trailer_len + gap, y - W / 2), cab_len, W, facecolor="darkblue", edgecolor="black", linewidth=1.5, zorder=11)
        ax.add_patch(cab)
    else:
        if is_ego:
            color = "#2ECC71"
            ec = "yellow"
            lw = 2.2
        elif is_merging:
            color = "#F39C12"
            ec = "white"
            lw = 1.6
        else:
            color = "#3498DB"
            ec = "white"
            lw = 1.0
        rect = mpatches.FancyBboxPatch((x - L / 2, y - W / 2), L, W, boxstyle="round,pad=0.1", facecolor=color, edgecolor=ec, linewidth=lw, zorder=10)
        ax.add_patch(rect)
    speed = np.sqrt(v["vx"] ** 2 + v["vy"] ** 2)
    label = "EGO" if is_ego else ("MERGE" if is_merging else f"{speed:.0f} m/s")
    ax.text(x, y - W / 2 - 1.2, label, ha="center", va="top", fontsize=7, color="yellow" if is_ego else "white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65))


def enforce_min_gap(vehicles, min_gap=10.0):
    """
    Push following vehicles back to maintain a longitudinal gap in each lane band.
    """
    lanes = cfg.lane_centers
    half = cfg.lane_width / 2
    for lane_y in lanes:
        lane_veh = [v for v in vehicles if v and abs(v["y"] - lane_y) <= half]
        # sort by x descending (front to back)
        lane_veh.sort(key=lambda v: v["x"], reverse=True)
        for ahead, behind in zip(lane_veh, lane_veh[1:]):
            if behind["x"] > ahead["x"] - min_gap:
                new_x = ahead["x"] - min_gap
                behind["x"] = new_x
                behind["vx"] = min(behind["vx"], ahead["vx"])


def maybe_random_lane_change(v):
    """
    Occasionally nudge non-ego/non-truck vehicles laterally to mimic random lane changes.
    """
    if v is None or v["id"] == 0 or "truck" in v["class"]:
        return 0.0
    if npr.rand() < 0.02:
        return npr.choice([-1.6, 1.6])
    return 0.0


# =============================================================================
# Animations powered by PDE solver
# =============================================================================

def create_advection_animation(output_path="./output", n_frames=50, fps=8):
    """
    Advection + diffusion using the PDE solver.
    """
    print("\n" + "=" * 70)
    print("DRIFT: ADVECTION (PDE-driven)")
    print("=" * 70)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vehicles = [
        create_vehicle(0, -10, cfg.lane_centers[1], 20, 0, "car"),  # Ego
        create_vehicle(1, 10, cfg.lane_centers[1], 18, 0, "truck_trailer"),
        create_vehicle(2, -20, cfg.lane_centers[0], 22, 0, "car"),
        create_vehicle(3, 25, cfg.lane_centers[0], 21, 0, "car"),
        create_vehicle(4, 5, cfg.lane_centers[2], 19, 0, "car"),
        create_vehicle(5, 40, cfg.lane_centers[-1], 17, -0.2, "car"),  # Merging
    ]
    merging_id = 5

    solver = PDESolver()
    dt_frame = 0.12
    substeps = 3
    frames = []

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(BG_DARK)

    X, Y = get_grid()

    # ------- Warm start so risk is present at t=0 -------
    ego = vehicles[0]
    truck = vehicles[1]
    shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)
    Q0 = (
        compute_risk_from_vehicles(vehicles, X, Y)
        + compute_merge_risk(vehicles, X, Y, merging_id=merging_id)
        + compute_occlusion_risk(shadow_mask, X, Y, truck)
    )
    # Seed with slightly stronger occlusion component so it is visible at t=0
    solver.R = gaussian_filter(Q0, sigma=0.8) + 0.4 * gaussian_filter(shadow_mask.astype(float), sigma=2.0)

    for step in range(n_frames):
        fig.clf()
        ego = vehicles[0]
        truck = vehicles[1]
        merging = vehicles[merging_id] if merging_id < len(vehicles) else None

        R_veh = compute_risk_from_vehicles(vehicles, X, Y)
        R_merge = compute_merge_risk(vehicles, X, Y, merging_id=merging_id)
        shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)
        R_occ = compute_occlusion_risk(shadow_mask, X, Y, truck)

        # Blend sources; keep occlusion visible but not overwhelming
        Q_total = 0.9 * R_merge + gaussian_filter(R_veh + 1.05 * R_occ, sigma=0.20)

        D = build_diffusion_field(shadow_mask, base_scale=0.10, occ_scale=1.4, skirt_gain=0.5, smooth_sigma=0.6)
        vx, vy, *_ = compute_velocity_field(vehicles, ego, X, Y)
        vy = vy * 0.5  # damp lateral advection to keep plume centered on traffic group

        sub_dt = dt_frame / substeps
        for _ in range(substeps):
            solver.step(Q_total, D, vx, vy, dt=sub_dt)

        R_display = gaussian_filter(solver.R, sigma=0.40)

        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_PANEL)

        vmax = 2.0
        pcm = ax.pcolormesh(X, Y, R_display, cmap="jet", shading="gouraud", vmin=0, vmax=vmax)

        max_val = R_display.max()
        lower = 0.25
        upper = min(max_val, vmax) * 0.9
        if max_val > lower and upper > lower:
            levels = np.linspace(lower, upper, 5)
            ax.contour(X, Y, R_display, levels=levels, colors="white", linewidths=1, alpha=0.6)

        draw_road(ax)
        draw_shadow(ax, shadow_poly, alpha=0.4)
        for v in vehicles:
            if v is None:
                continue
            draw_vehicle(ax, v, is_ego=(v["id"] == 0), is_merging=(v["id"] == merging_id))

        skip = 10
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], vx[::skip, ::skip], vy[::skip, ::skip], color="cyan", scale=250, alpha=0.35, width=0.003, zorder=2)

        ax.set_xlim(cfg.x_min, cfg.x_max)
        ax.set_ylim(cfg.y_min, cfg.y_max)
        ax.set_aspect("equal")
        ax.set_title(f"ADVECTION (PDE) | t = {step*dt_frame:.2f}s | Step {step+1}", fontsize=13, fontweight="bold", color="white")
        ax.set_xlabel("x [m]", color="white", fontsize=11)
        ax.set_ylabel("y [m]", color="white", fontsize=11)
        ax.tick_params(colors="white")

        if merging is not None:
            merge_len = cfg.merge_x_end - cfg.merge_x_start
            merge_progress = np.clip((merging["x"] - cfg.merge_x_start) / merge_len, 0, 1.2)
            phase = "approaching" if merging["x"] < cfg.merge_x_end else "merged"
            ax.text(cfg.x_min + 3, cfg.y_max - 1.2, f"Merge progress: {merge_progress*100:4.0f}% ({phase})", fontsize=9, color="white", bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

        cbar = plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Risk Value", color="white", fontsize=10)
        cbar.ax.tick_params(colors="white")

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())

        new_vehicles = []
        for v in vehicles:
            if v is None:
                new_vehicles.append(None)
                continue
            if v["id"] == merging_id:
                ay = -1.0 if v["y"] > cfg.lane_centers[-2] else 0
                new_v = move_vehicle(v, dt_frame, ax=-0.3, ay=ay)
            else:
                ay = maybe_random_lane_change(v)
                new_v = move_vehicle(v, dt_frame, ax=0, ay=ay)
            new_vehicles.append(new_v)
        enforce_min_gap(new_vehicles, min_gap=10.0)
        vehicles = new_vehicles

    plt.close()
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, "drift_advection_pde.gif")
        print(f"\nSaving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        return gif_path
    return None


def create_diffusion_animation(output_path="./output", n_frames=50, fps=8):
    """
    Diffusion-focused animation driven by PDE solver (no vehicle motion).
    """
    print("\n" + "=" * 70)
    print("DRIFT: DIFFUSION (PDE-driven)")
    print("=" * 70)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vehicles = [
        create_vehicle(0, -15, cfg.lane_centers[1], 20, 0, "car"),  # Ego
        create_vehicle(1, 15, cfg.lane_centers[1], 18, 0, "truck_trailer"),
        create_vehicle(2, -5, cfg.lane_centers[0], 22, 0, "car"),
        create_vehicle(3, 5, cfg.lane_centers[2], 19, 0, "car"),
    ]
    ego = vehicles[0]
    truck = vehicles[1]

    X, Y = get_grid()
    shadow_poly, shadow_mask = compute_shadow(ego, truck, X, Y)

    solver = PDESolver()
    dt_frame = 0.08
    substeps = 4
    frames = []

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(BG_DARK)

    # Seed the field so occlusion plume is present at t=0 (emphasize occlusion)
    R_veh0 = compute_risk_from_vehicles(vehicles, X, Y)
    R_occ0 = compute_occlusion_risk(shadow_mask, X, Y, truck)
    R_merge0 = compute_merge_risk(vehicles, X, Y, merging_id=None)
    solver.R = gaussian_filter(2.0 * R_occ0 + 0.2 * R_veh0 + 0.8 * R_merge0, sigma=0.65)

    for step in range(n_frames):
        fig.clf()

        R_veh = compute_risk_from_vehicles(vehicles, X, Y)
        R_occ = compute_occlusion_risk(shadow_mask, X, Y, truck)
        R_merge = compute_merge_risk(vehicles, X, Y, merging_id=None)
        # Keep occlusion-driven plume dominant in diffusion demo
        Q_total = 1.6 * R_occ + 1.05 * R_merge + 0.2 * gaussian_filter(R_veh, sigma=0.25)

        D = build_diffusion_field(shadow_mask, base_scale=0.10, occ_scale=1.8, skirt_gain=0.6, smooth_sigma=0.6)

        vx, vy, *_ = compute_velocity_field(vehicles, ego, X, Y)

        sub_dt = dt_frame / substeps
        for _ in range(substeps):
            solver.step(Q_total, D, vx, vy, dt=sub_dt)

        R_vis = gaussian_filter(solver.R, sigma=0.35)
        vmax = 1.1

        gs = matplotlib.gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.25, height_ratios=[1, 1.2])

        # Panel 1: D field
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(BG_PANEL)
        pcm1 = ax1.pcolormesh(X, Y, D, cmap="Greens", shading="gouraud", vmin=cfg.D0, vmax=cfg.D0 + cfg.D_occ)
        ax1.contour(X, Y, shadow_mask.astype(float), levels=[0.5], colors="red", linewidths=2, linestyles="--")
        draw_road(ax1)
        for v in vehicles:
            draw_vehicle(ax1, v, is_ego=(v["id"] == 0))
        ax1.set_xlim(cfg.x_min, cfg.x_max)
        ax1.set_ylim(cfg.y_min, cfg.y_max)
        ax1.set_aspect("equal")
        ax1.set_title("Diffusion Coefficient D(x)", fontsize=11, fontweight="bold", color="white")
        ax1.tick_params(colors="white")
        cbar1 = plt.colorbar(pcm1, ax=ax1, shrink=0.7)
        cbar1.set_label("D [m²/s]", color="white")
        cbar1.ax.tick_params(colors="white")

        # Panel 2: Shadow visualization
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(BG_PANEL)
        occ_cmap = LinearSegmentedColormap.from_list("occ", [(0, "#161B22"), (0.3, "#2D2D4A"), (0.7, "#6B4A8A"), (1, "#EC4899")])
        ax2.pcolormesh(X, Y, shadow_mask.astype(float), cmap=occ_cmap, shading="gouraud", alpha=0.8)
        draw_road(ax2)
        draw_shadow(ax2, shadow_poly, alpha=0.55)
        for v in vehicles:
            draw_vehicle(ax2, v, is_ego=(v["id"] == 0))
        ax2.set_xlim(cfg.x_min, cfg.x_max)
        ax2.set_ylim(cfg.y_min, cfg.y_max)
        ax2.set_aspect("equal")
        ax2.set_title("Occlusion Shadow", fontsize=11, fontweight="bold", color="white")
        ax2.tick_params(colors="white")

        # Panel 3: Risk field (PDE)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor(BG_PANEL)
        pcm3 = ax3.pcolormesh(X, Y, R_vis, cmap="jet", shading="gouraud", vmin=0, vmax=vmax)
        max_val = R_vis.max()
        lower = 0.15
        upper = min(max_val, vmax) * 0.9
        if max_val > lower and upper > lower:
            levels = np.linspace(lower, upper, 6)
            ax3.contour(X, Y, R_vis, levels=levels, colors="white", linewidths=0.8, alpha=0.6)
        # Highlight occlusion boundary to show where diffusion is enhanced
        ax3.contour(X, Y, shadow_mask.astype(float), levels=[0.25], colors="lime", linewidths=2.2, linestyles="--", alpha=0.9)
        draw_road(ax3)
        draw_shadow(ax3, shadow_poly, alpha=0.35)
        for v in vehicles:
            draw_vehicle(ax3, v, is_ego=(v["id"] == 0))
        ax3.set_xlim(cfg.x_min, cfg.x_max)
        ax3.set_ylim(cfg.y_min, cfg.y_max)
        ax3.set_aspect("equal")
        ax3.set_title(f"Risk via PDE Diffusion | t = {step*dt_frame:.2f}s | Step {step+1}", fontsize=12, fontweight="bold", color="white")
        ax3.set_xlabel("x [m]", color="white", fontsize=11)
        ax3.set_ylabel("y [m]", color="white", fontsize=11)
        ax3.tick_params(colors="white")
        cbar3 = plt.colorbar(pcm3, ax=ax3, shrink=0.65, pad=0.02)
        cbar3.set_label("Risk (Uncertainty)", color="white", fontsize=10)
        cbar3.ax.tick_params(colors="white")

        fig.suptitle("DRIFT: PDE-driven Diffusion (Enhanced in Occlusion)", fontsize=14, fontweight="bold", color="white", y=0.98)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())

    plt.close()
    if HAS_IMAGEIO:
        gif_path = os.path.join(output_path, "drift_diffusion_pde.gif")
        print(f"\nSaving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        return gif_path
    return None


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="DRIFT: PDE-based Advection/Diffusion")
    parser.add_argument("-o", "--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")

    args = parser.parse_args()

    create_advection_animation(args.output, n_frames=args.frames, fps=args.fps)
    create_diffusion_animation(args.output, n_frames=args.frames, fps=args.fps)

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Output saved to: {os.path.abspath(args.output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
