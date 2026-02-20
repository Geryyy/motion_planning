import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geom import plot_scene
from geom.spline_opt import optimize_bspline_path, yaw_deg_to_quat
from geom.utils import quat_to_rot
from scenarios import build_scenario, list_scenarios


DEMO_CONFIG = {
    "n_vias": 2,
    "safety_margin": 0.0,
    "preferred_safety_margin": 0.02,
    "relax_preferred_final_fraction": 0.25,
    "approach_only_clearance": 0.015,
    "contact_window_fraction": 0.08,
    "n_yaw_vias": 2,
    "combined_4d": True,
    "approach_fraction": 0.25,
    "w_via_dev": 0.06,
    "w_yaw_monotonic": 80.0,
    "yaw_goal_reach_u": 0.5,
    "goal_approach_window_fraction": 0.12,
    "init_offset_scale": 0.7,
    "method": "Powell",
    "goal_clearance_target": 0.0,
    "w_len": 5.0,
    "n_samples_curve": 101,
    "collision_check_subsample": 1,
    "w_curv": 0.12,
    "w_yaw_smooth": 0.008,
    "w_safe": 380.0,
    "w_safe_preferred": 24.0,
    "w_approach_rebound": 280.0,
    "w_goal_clearance": 35.0,
    "w_goal_clearance_target": 260.0,
    "w_approach_clearance": 420.0,
    "w_approach_collision": 1400.0,
    "w_yaw_dev": 0.05,
    "w_yaw_schedule": 55.0,
    "w_goal_approach_normal": 80.0,
    "options": {"maxiter": 160, "xtol": 1e-3, "ftol": 1e-3},
}


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n < eps:
        return np.zeros_like(arr)
    return arr / n


def _approach_alignment_vectors(
    curve: np.ndarray,
    goal_normals: np.ndarray,
    terminal_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tail_n = max(3, int(np.ceil(float(terminal_fraction) * curve.shape[0])))
    v_approach = _normalize(np.sum(np.diff(curve[-tail_n:], axis=0), axis=0))

    normals = np.asarray(goal_normals, dtype=float).reshape(-1, 3)
    if normals.size == 0:
        summed_normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        summed_normal = _normalize(np.sum(normals, axis=0))
        if not np.any(summed_normal):
            summed_normal = _normalize(normals[0])
    desired_approach = -summed_normal
    return v_approach, summed_normal, desired_approach


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="front",
        help="Scenario name to run.",
    )
    parser.add_argument(
        "--scenarios-file",
        default=str(Path(__file__).with_name("scenarios.yaml")),
        help="Path to scenarios YAML file.",
    )
    args = parser.parse_args()
    scenario_names = list_scenarios(args.scenarios_file)
    if args.scenario not in scenario_names:
        available = ", ".join(scenario_names)
        raise ValueError(f"Unknown scenario '{args.scenario}'. Available: {available}")

    scenario = build_scenario(args.scenario, scenarios_file=args.scenarios_file)
    scene = scenario.scene
    start = scenario.start
    goal = scenario.goal
    moving_block_size = scenario.moving_block_size
    start_yaw_deg = scenario.start_yaw_deg
    goal_yaw_deg = scenario.goal_yaw_deg
    goal_normals = np.asarray(scenario.goal_normals, dtype=float)

    planner_cfg = DEMO_CONFIG

    # Single-stage optimization.
    t_start = time.time()
    S, vias_opt, info = optimize_bspline_path(
        scene=scene,
        start=start,
        goal=goal,
        moving_block_size=moving_block_size,
        start_yaw_deg=start_yaw_deg,
        goal_yaw_deg=goal_yaw_deg,
        goal_approach_normals=goal_normals,
        **planner_cfg,
    )
    opt_duration = time.time() - t_start
    print(f"Optimization took {opt_duration:.2f} seconds")

    # Sample and plot
    u = np.linspace(0, 1, 250)
    curve = S(u)
    v_approach, summed_normal, desired_approach = _approach_alignment_vectors(
        curve=curve,
        goal_normals=goal_normals,
        terminal_fraction=DEMO_CONFIG["goal_approach_window_fraction"],
    )
    align_cos = float(np.clip(np.dot(v_approach, desired_approach), -1.0, 1.0))
    align_angle_deg = float(np.degrees(np.arccos(align_cos)))
    print(
        f"Approach alignment angle: {align_angle_deg:.2f} deg "
        f"(0 deg means perfectly aligned with -summed surface normals)"
    )

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax = plot_scene(scene, ax=ax, start=start, goal=goal)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "k-", lw=2, label="C2 B-spline (4D + Powell)")

    for i, vp in enumerate(vias_opt):
        ax.scatter(*vp, s=30, label=f"v{i + 1} (opt)")

    normal_len = 0.35 * max(float(np.linalg.norm(np.asarray(moving_block_size, dtype=float))), 1e-6)
    g = np.asarray(goal, dtype=float)
    for n in goal_normals:
        nn = _normalize(np.asarray(n, dtype=float))
        ax.quiver(g[0], g[1], g[2], nn[0], nn[1], nn[2], length=normal_len, color="deepskyblue", linewidth=2.0)
    ax.quiver(
        g[0], g[1], g[2],
        summed_normal[0], summed_normal[1], summed_normal[2],
        length=normal_len, color="magenta", linewidth=2.5,
    )
    ax.quiver(
        g[0], g[1], g[2],
        v_approach[0], v_approach[1], v_approach[2],
        length=normal_len, color="red", linewidth=2.5,
    )
    # Proxy handles so quiver arrows appear in legend.
    ax.plot([], [], [], color="deepskyblue", lw=2, label="surface normals @ goal")
    ax.plot([], [], [], color="magenta", lw=2, label="resultant normal (sum normals)")
    ax.plot([], [], [], color="red", lw=2, label="actual approach direction")

    # Animate moving payload block along the spline and color by clearance.
    def _box_vertices(center, size, yaw_deg):
        cx, cy, cz = center
        sx, sy, sz = size
        hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
        local = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ],
            dtype=float,
        )
        R = quat_to_rot(yaw_deg_to_quat(yaw_deg))
        center_vec = np.array([cx, cy, cz], dtype=float)
        return (R @ local.T).T + center_vec

    def _box_faces(vertices):
        return [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[4], vertices[7], vertices[3], vertices[0]],
        ]

    anim_u = np.linspace(0.0, 1.0, 180)
    anim_pts = S(anim_u)
    yaw_fn = info["yaw_fn"]
    anim_yaw = np.asarray(yaw_fn(anim_u), dtype=float)
    anim_dists = np.array(
        [
            scene.signed_distance_block(size=moving_block_size, position=p, quat=yaw_deg_to_quat(anim_yaw[i]))
            for i, p in enumerate(anim_pts)
        ],
        dtype=float,
    )
    print(f"Min sampled clearance along animation path: {anim_dists.min():+.3f} m")

    # Second subplot: clearance profile along path parameter u.
    ax_clear = fig.add_subplot(1, 2, 2)
    ax_clear.plot(anim_u, anim_dists, "b-", lw=2, label="signed distance")
    ax_clear.axhline(0.0, color="r", lw=1, ls="--", label="collision boundary")
    ax_clear.axhline(info["preferred_clearance"], color="orange", lw=1, ls="--", label="preferred clearance")
    if info.get("approach_only_clearance") is not None:
        ax_clear.axhline(info["approach_only_clearance"], color="green", lw=1, ls="--", label="approach clearance")
    clear_marker, = ax_clear.plot([anim_u[0]], [anim_dists[0]], "ko", ms=6)
    ax_clear.set_xlabel("path parameter u")
    ax_clear.set_ylabel("signed distance [m]")
    ax_clear.set_title("Block Clearance Along Path")
    ax_clear.grid(True, alpha=0.3)
    ax_clear.legend(loc="best")

    v0 = _box_vertices(anim_pts[0], moving_block_size, anim_yaw[0])
    moving_poly = Poly3DCollection(_box_faces(v0), alpha=0.25, facecolor="limegreen", edgecolor="k", linewidths=0.8)
    ax.add_collection3d(moving_poly)
    moving_center = ax.scatter([anim_pts[0, 0]], [anim_pts[0, 1]], [anim_pts[0, 2]], s=40, c="k", label="moving block")
    dist_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def _frame_color(dist):
        if dist < 0.0:
            return "crimson"
        if dist < 0.03:
            return "darkorange"
        return "limegreen"

    def _update(frame_idx):
        p = anim_pts[frame_idx]
        dist = float(anim_dists[frame_idx])
        vv = _box_vertices(p, moving_block_size, float(anim_yaw[frame_idx]))
        moving_poly.set_verts(_box_faces(vv))
        moving_poly.set_facecolor(_frame_color(dist))
        moving_center._offsets3d = ([p[0]], [p[1]], [p[2]])
        dist_text.set_text(f"clearance: {dist:+.3f} m, yaw: {anim_yaw[frame_idx]:+.1f} deg")
        clear_marker.set_data([anim_u[frame_idx]], [dist])
        return moving_poly, moving_center, dist_text, clear_marker

    anim = FuncAnimation(fig=ax.figure, func=_update, frames=len(anim_pts), interval=50, blit=False, repeat=True)

    ax.legend(loc="upper right")
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.show()

    print("Optimize success:", info["success"], "-", info["message"])
    print(
        f"Total cost: {info['fun']:.6f}, length: {info['length']:.6f}, "
        f"curvature: {info['curvature_cost']:.6e}, yaw_smooth: {info['yaw_smoothness_cost']:.6e}, "
        f"safety_cost: {info['safety_cost']:.6e}, pref_safety_cost: {info['preferred_safety_cost']:.6e}, "
        f"approach_rebound: {info['approach_rebound_cost']:.6e}, goal_clear: {info['goal_clearance_cost']:.6e}, "
        f"goal_target: {info['goal_clearance_target_cost']:.6e}, "
        f"approach_clear: {info['approach_clearance_cost']:.6e}, "
        f"approach_col: {info['approach_collision_cost']:.6e}, "
        f"via_dev: {info['via_deviation_cost']:.6e}, yaw_dev: {info['yaw_deviation_cost']:.6e}, "
        f"yaw_mono: {info['yaw_monotonic_cost']:.6e}, yaw_sched: {info['yaw_schedule_cost']:.6e}, "
        f"goal_normal: {info['goal_approach_normal_cost']:.6e}, "
        f"iterations: {info['nit']}"
    )
    print(
        f"mean_turn_angle_deg: {info['turn_angle_mean_deg']:.4f}, "
        f"min_clearance: {info['min_clearance']:+.4f} m, "
        f"mean_clearance: {info['mean_clearance']:+.4f} m, "
        f"required_clearance: {info['required_clearance']:+.4f} m, "
        f"preferred_clearance: {info['preferred_clearance']:+.4f} m"
    )
    straight_len = float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)))
    print(
        f"path_efficiency: {info['length'] / max(straight_len, 1e-9):.3f}x "
        f"(length={info['length']:.3f} m, straight={straight_len:.3f} m)"
    )


if __name__ == "__main__":
    main()
