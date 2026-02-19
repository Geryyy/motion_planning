import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geom import Scene, plot_scene
from geom.spline_opt import optimize_bspline_path, yaw_deg_to_quat
from geom.utils import quat_to_rot


def main():
    scene = Scene()
    # Add base objects with explicit IDs
    table_id = scene.add_block(size=(2.0, 2, 0.1), position=(0.0, 0.0, 0.05), quat=(0.0, 0.0, 0.0, 1.0), object_id="table")
    scene.add_block(size=(0.1, 2, 2), position=(1, 0.0, 1), quat=(0.0, 0.0, 0.0, 1.0), object_id="wall")

    # Compute a point to place a cube on top of the table, centered
    new_size = (0.6, 0.9, 0.6)
    pos_top = scene.get_top_point(table_id, new_size, gap=0.0, xy_offset=(0.0, -0.5))
    scene.add_block(size=new_size, position=tuple(pos_top), quat=(0.0, 0.0, 0.0, 1.0), object_id="cube_top")

    # Payload block dimensions: x=0.9, y=0.6, z=0.6 [m]
    moving_block_size = (0.9, 0.6, 0.6)
    start_yaw_deg = 0.0
    goal_yaw_deg = 90.0

    # Goal placement uses the payload dimensions at final yaw=90 deg.
    moving_block_size_goal = (moving_block_size[1], moving_block_size[0], moving_block_size[2])
    goal_t = scene.get_front_point(
        base="cube_top",
        new_size=moving_block_size_goal,
        gap=0.00,
        xz_offset=(0.0, 0.0),
    )

    start = (-0.4, -0.2, 1.50)
    goal = goal_t

    n_vias = 3

    # Two-stage optimization: fast coarse solve, then warm-started refine.
    t_start = time.time()
    S1, vias1, info1 = optimize_bspline_path(
        scene,
        start,
        goal,
        n_vias=n_vias,
        moving_block_size=moving_block_size,
        safety_margin=0.0,
        preferred_safety_margin=0.03,
        relax_preferred_final_fraction=0.25,
        approach_only_clearance=0.025,
        contact_window_fraction=0.08,
        n_samples_curve=61,
        collision_check_subsample=3,
        start_yaw_deg=start_yaw_deg,
        goal_yaw_deg=goal_yaw_deg,
        n_yaw_vias=n_vias,
        combined_4d=True,
        w_len=1.0,
        w_curv=0.20,
        w_yaw_smooth=0.006,
        w_safe=300.0,
        w_safe_preferred=25.0,
        w_approach_rebound=180.0,
        w_goal_clearance=20.0,
        goal_clearance_target=0.0,
        w_goal_clearance_target=120.0,
        w_approach_clearance=280.0,
        w_approach_collision=900.0,
        approach_fraction=0.25,
        w_via_dev=0.4,
        w_yaw_dev=0.08,
        w_yaw_monotonic=150.0,
        yaw_goal_reach_u=0.5,
        w_yaw_schedule=35.0,
        init_offset_scale=0.7,
        method="Powell",
        options={"maxiter": 80, "xtol": 3e-3, "ftol": 3e-3},
    )

    S, vias_opt, info = optimize_bspline_path(
        scene,
        start,
        goal,
        n_vias=n_vias,
        moving_block_size=moving_block_size,
        safety_margin=0.0,
        preferred_safety_margin=0.03,
        relax_preferred_final_fraction=0.25,
        approach_only_clearance=0.025,
        contact_window_fraction=0.08,
        n_samples_curve=101,
        collision_check_subsample=1,
        start_yaw_deg=start_yaw_deg,
        goal_yaw_deg=goal_yaw_deg,
        n_yaw_vias=n_vias,
        combined_4d=True,
        w_len=1.0,
        w_curv=0.25,
        w_yaw_smooth=0.008,
        w_safe=380.0,
        w_safe_preferred=40.0,
        w_approach_rebound=280.0,
        w_goal_clearance=35.0,
        goal_clearance_target=0.0,
        w_goal_clearance_target=260.0,
        w_approach_clearance=420.0,
        w_approach_collision=1400.0,
        approach_fraction=0.25,
        w_via_dev=0.4,
        w_yaw_dev=0.10,
        w_yaw_monotonic=150.0,
        yaw_goal_reach_u=0.5,
        w_yaw_schedule=55.0,
        init_vias=vias1,
        init_yaw_vias_deg=np.asarray(info1["yaw_ctrl_deg"], dtype=float)[1:-1],
        init_offset_scale=0.7,
        method="Powell",
        options={"maxiter": 120, "xtol": 1e-3, "ftol": 1e-3},
    )
    opt_duration = time.time() - t_start
    print(f"Two-stage optimization took {opt_duration:.2f} seconds")

    # Sample and plot
    u = np.linspace(0, 1, 250)
    curve = S(u)

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax = plot_scene(scene, ax=ax, start=start, goal=goal)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "k-", lw=2, label="C2 B-spline (4D + Powell)")

    for i, vp in enumerate(vias_opt):
        ax.scatter(*vp, s=30, label=f"v{i + 1} (opt)")

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
        f"iterations: {info['nit']}"
    )
    print(
        f"mean_turn_angle_deg: {info['turn_angle_mean_deg']:.4f}, "
        f"min_clearance: {info['min_clearance']:+.4f} m, "
        f"mean_clearance: {info['mean_clearance']:+.4f} m, "
        f"required_clearance: {info['required_clearance']:+.4f} m, "
        f"preferred_clearance: {info['preferred_clearance']:+.4f} m"
    )

    # SDF slice demo
    # z_slice = 0.10
    # xs = np.linspace(-2, 2, 120)
    # ys = np.linspace(-2, 2, 120)
    # sdf_slice = np.zeros((len(xs), len(ys)))
    # for i, x in enumerate(xs):
    #     for j, y in enumerate(ys):
    #         sdf_slice[i, j] = scene.signed_distance(np.array([x, y, z_slice], dtype=float))

    # plt.figure()
    # plt.imshow(
    #     sdf_slice.T,
    #     origin="lower",
    #     extent=(xs[0], xs[-1], ys[0], ys[-1]),
    #     cmap="RdBu",
    #     vmin=-0,
    #     vmax=1,
    #     aspect="equal",
    # )
    # plt.colorbar(label="Signed distance (m)")
    # plt.title(f"SDF slice at z={z_slice:.2f} m")
    # plt.xlabel("X (m)")
    # plt.ylabel("Y (m)")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
