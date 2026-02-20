#!/usr/bin/env python3
"""Standalone scenario visualizer for generated scenarios YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


Vec3 = Tuple[float, float, float]


def _box_vertices(center: Vec3, size: Vec3) -> np.ndarray:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    return np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=float,
    )


def _box_faces(vertices: np.ndarray) -> List[np.ndarray]:
    return [
        vertices[[0, 1, 2, 3]],
        vertices[[4, 5, 6, 7]],
        vertices[[0, 1, 5, 4]],
        vertices[[2, 3, 7, 6]],
        vertices[[1, 2, 6, 5]],
        vertices[[4, 7, 3, 0]],
    ]


def _normalize(v: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(v), dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return np.zeros(3, dtype=float)
    return arr / n


def _goal_from_face(base_center: Vec3, base_size: Vec3, moving_size: Vec3, face: str) -> Vec3:
    bx, by, bz = base_center
    sx, _, sz = base_size
    mx, _, mz = moving_size
    f = face.lower()
    if f == "front":
        return (bx + 0.5 * sx + 0.5 * mx, by, bz)
    if f == "back":
        return (bx - 0.5 * sx - 0.5 * mx, by, bz)
    if f == "top":
        return (bx, by, bz + 0.5 * sz + 0.5 * mz)
    raise ValueError(f"Unsupported face '{face}'")


def _parse_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "scenarios" not in data:
        raise ValueError(f"Invalid scenarios file: {path}")
    if not isinstance(data["scenarios"], dict):
        raise ValueError("'scenarios' must be a mapping")
    return data


def _collect_limits(points: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.vstack(points)
    return stack.min(axis=0), stack.max(axis=0)


def _setup_equal_axes(ax, pmin: np.ndarray, pmax: np.ndarray) -> None:
    center = 0.5 * (pmin + pmax)
    radius = 0.5 * float(np.max(pmax - pmin))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_box_aspect((1, 1, 0.8))


def _draw_scenario(ax, name: str, cfg: Dict) -> None:
    blocks = cfg.get("blocks", [])
    moving = cfg["moving_block"]

    scene_points: List[np.ndarray] = []
    block_centers: Dict[str, Vec3] = {}
    block_sizes: Dict[str, Vec3] = {}

    for b in blocks:
        bid = str(b["id"])
        c = tuple(float(v) for v in b["position"])
        s = tuple(float(v) for v in b["size"])
        block_centers[bid] = c
        block_sizes[bid] = s

        verts = _box_vertices(c, s)
        scene_points.append(verts)
        color = "lightgray" if bid == "table" else "steelblue"
        alpha = 0.22 if bid == "table" else 0.35
        poly = Poly3DCollection(_box_faces(verts), facecolor=color, edgecolor="k", linewidths=0.7, alpha=alpha)
        ax.add_collection3d(poly)

    move_size = tuple(float(v) for v in moving.get("size", [0.9, 0.6, 0.6]))
    start = tuple(float(v) for v in moving["start"])
    goal_cfg = moving["goal"]

    if str(goal_cfg.get("type", "")).lower() != "face":
        raise ValueError(f"{name}: only goal.type=face is supported in this visualizer")
    base_id = str(goal_cfg["base"])
    goal = _goal_from_face(
        base_center=block_centers[base_id],
        base_size=block_sizes[base_id],
        moving_size=move_size,
        face=str(goal_cfg["face"]),
    )

    start_verts = _box_vertices(start, move_size)
    goal_verts = _box_vertices(goal, move_size)
    scene_points.extend([start_verts, goal_verts, np.asarray([start, goal], dtype=float)])

    start_poly = Poly3DCollection(_box_faces(start_verts), facecolor="orange", edgecolor="k", linewidths=0.8, alpha=0.45)
    goal_poly = Poly3DCollection(_box_faces(goal_verts), facecolor="limegreen", edgecolor="k", linewidths=0.8, alpha=0.45)
    ax.add_collection3d(start_poly)
    ax.add_collection3d(goal_poly)

    ax.scatter([start[0]], [start[1]], [start[2]], s=25, c="orange")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], s=25, c="green")
    ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], "k--", lw=1.0, alpha=0.7)

    arrow_scale = max(float(np.linalg.norm(np.asarray(move_size, dtype=float))), 1e-6) * 0.8
    for n in moving.get("goal_normals", []):
        nn = _normalize(n)
        ax.quiver(goal[0], goal[1], goal[2], nn[0], nn[1], nn[2], length=arrow_scale, color="deepskyblue", linewidth=2.0)

    if "approach_direction" in moving:
        ad = _normalize(moving["approach_direction"])
        ax.quiver(goal[0], goal[1], goal[2], ad[0], ad[1], ad[2], length=arrow_scale, color="crimson", linewidth=2.2)

    pmin, pmax = _collect_limits(scene_points)
    _setup_equal_axes(ax, pmin, pmax)
    ax.set_title(name)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize scenarios from YAML.")
    parser.add_argument(
        "--scenarios-file",
        default=str(Path(__file__).with_name("generated_scenarios.yaml")),
        help="Path to scenarios YAML file.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario name; if omitted, visualize all scenarios in a grid.",
    )
    args = parser.parse_args()

    payload = _parse_yaml(Path(args.scenarios_file))
    scenarios: Dict[str, Dict] = payload["scenarios"]
    if not scenarios:
        raise ValueError("No scenarios found")

    if args.scenario is not None:
        key = str(args.scenario)
        if key not in scenarios:
            available = ", ".join(sorted(scenarios.keys()))
            raise ValueError(f"Unknown scenario '{key}'. Available: {available}")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        _draw_scenario(ax, key, scenarios[key])
        plt.tight_layout()
        plt.show()
        return

    names = sorted(scenarios.keys())
    n = len(names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(5.5 * ncols, 4.8 * nrows))
    for idx, name in enumerate(names, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        _draw_scenario(ax, name, scenarios[name])
    fig.suptitle(f"Scenarios from {args.scenarios_file}", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
