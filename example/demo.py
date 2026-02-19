import numpy as np
import matplotlib.pyplot as plt
import time
from geom import Scene, plot_scene
from geom import goal_and_via_for_placement
from geom.spline_opt import optimize_bspline_two_vias

def main():
    scene = Scene()
    # Add base objects with explicit IDs
    table_id = scene.add_block(size=(2.0, 2, 0.1), position=(0.0, 0.0, 0.05), quat=(0.0, 0.0, 0.0, 1.0), object_id="table")
    wall_id  = scene.add_block(size=(0.1, 2, 2), position=(1, 0.0, 1), quat=(0.0, 0.0, 0.0, 1.0), object_id="wall")

    # Compute a point to place a cube on top of the table, centered
    new_size = (0.6, 0.9, 0.6)
    pos_top = scene.get_top_point(table_id, new_size, gap=0.0, xy_offset=(0.0, -0.5))
    cube_top_id = scene.add_block(size=new_size, position=tuple(pos_top), quat=(0.0, 0.0, 0.0, 1.0), object_id="cube_top")

    # Place another cube in front of the wall with a 2 cm gap, shifted +5 cm in local x
    # pos_front = scene.get_front_point(cube_top_id, new_size, gap=0.02, xz_offset=(0.0, 0.0))
    # cube_front_id = scene.add_block(size=new_size, position=tuple(pos_front), quat=(0.0, 0.0, 0.0, 1.0), object_id="cube_front")

    # goal_point = scene.get_top_point(cube_front_id, new_size, gap=0.02)

    # Top placement with 5 cm interlock along local +x; via +30 cm along local z
    goal_t, via_t = goal_and_via_for_placement(
        scene, base_id="cube_top", direction="front",
        gap=0.02, lateral_offset=(0.0, 0.0), interlock_x=0.05,
        via_offset_local=(0.0, 0.0, 0.30)
    )

    # Visualize
    start = (-0.4, -0.2, 1.50)
    via   = via_t
    goal  = goal_t

    # Optimize via points for a smooth path
    t_start = time.time()
    S, v1, v2, info = optimize_bspline_two_vias(
        scene, start, via, goal,
        tool_half_extents=(0.03, 0.03, 0.10),
        safety_margin=0.01,
        n_samples_curve=121,
        w_len=1.0, w_curv=0.05, w_safe=100.0,  # stronger penalty on safety
        init_offset_scale=0.5,                 # initial guess spread
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 5e-4, "fatol": 5e-4}
    )
    opt_duration = time.time() - t_start
    print(f"Optimization took {opt_duration:.2f} seconds")

    # Sample and plot
    u = np.linspace(0, 1, 200)
    curve = S(u)

    ax = plot_scene(scene, start=start, via=via, goal=goal)
    ax.plot(curve[:,0], curve[:,1], curve[:,2], 'k-', lw=2, label='C2 B-spline')
    ax.scatter(*v1, c='cyan', s=40, label='v1 (opt)')
    ax.scatter(*v2, c='magenta', s=40, label='v2 (opt)')
    ax.legend(loc='upper right')
    ax.view_init(elev=25, azim=45)
    plt.tight_layout(); plt.show()

    print("Optimize success:", info["success"], "-", info["message"])
    print(f"Total cost: {info['fun']:.3f}, length: {info['length']:.3f}, curvature: {info['curvature_cost']:.3f}, safety_cost: {info['safety_cost']:.3f}, iterations: {info['nit']}")


    # ax = plot_scene(scene, start=start, via=via, goal=goal)
    # ax.view_init(elev=25, azim=45)
    # plt.tight_layout()
    # plt.show()

    # SDF slice demo
    z_slice = 0.10
    xs = np.linspace(-2, 2, 120)
    ys = np.linspace(-2, 2, 120)
    sdf_slice = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            sdf_slice[i, j] = scene.signed_distance(np.array([x, y, z_slice], dtype=float))

    plt.figure()
    plt.imshow(
        sdf_slice.T,
        origin='lower',
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
        cmap='RdBu',
        vmin=-0, vmax=1,
        aspect='equal'
    )
    plt.colorbar(label='Signed distance (m)')
    plt.title(f'SDF slice at z={z_slice:.2f} m')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()