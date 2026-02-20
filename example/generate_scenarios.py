#!/usr/bin/env python3
"""Generate wall-building scenarios and write them to YAML.

Scenarios follow explicit construction steps on a block-size-aligned grid:
1) place first block on ground plane,
2) place second block in front of first,
3) place third block on top,
4) place one block between two existing blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

BLOCK_SIZE = (0.9, 0.6, 0.6)  # x, y, z (depth, width, height) in meters
TABLE_SIZE = (4.0, 4.0, 0.1)
TABLE_POS = (0.0, 0.0, 0.05)

GX, GY, GZ = BLOCK_SIZE
TABLE_TOP_Z = TABLE_POS[2] + 0.5 * TABLE_SIZE[2]
GROUND_BLOCK_Z = TABLE_TOP_Z + 0.5 * GZ


def _unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = v
    n = (x * x + y * y + z * z) ** 0.5
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (x / n, y / n, z / n)


def _clean(v: float, ndigits: int = 6) -> float:
    out = round(float(v), ndigits)
    if abs(out) < 1e-12:
        return 0.0
    return out


def _goal_center_from_face(
    base_center: Tuple[float, float, float],
    base_size: Tuple[float, float, float],
    moving_size: Tuple[float, float, float],
    face: str,
) -> Tuple[float, float, float]:
    bx, by, bz = base_center
    sx, sy, sz = base_size
    mx, my, mz = moving_size
    if face == "front":
        return (bx + 0.5 * sx + 0.5 * mx, by, bz)
    if face == "back":
        return (bx - 0.5 * sx - 0.5 * mx, by, bz)
    if face == "top":
        return (bx, by, bz + 0.5 * sz + 0.5 * mz)
    raise ValueError(f"Unsupported face: {face}")


def _normal_from_geometry(
    base_center: Tuple[float, float, float],
    goal_center: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    dx = goal_center[0] - base_center[0]
    dy = goal_center[1] - base_center[1]
    dz = goal_center[2] - base_center[2]
    axis = max([(abs(dx), 0), (abs(dy), 1), (abs(dz), 2)], key=lambda t: t[0])[1]
    if axis == 0:
        return _unit((1.0 if dx >= 0.0 else -1.0, 0.0, 0.0))
    if axis == 1:
        return _unit((0.0, 1.0 if dy >= 0.0 else -1.0, 0.0))
    return _unit((0.0, 0.0, 1.0 if dz >= 0.0 else -1.0))


def _infer_approach_direction(
    normals: List[Tuple[float, float, float]],
    fallback: Tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> Tuple[float, float, float]:
    sx = sum(n[0] for n in normals)
    sy = sum(n[1] for n in normals)
    sz = sum(n[2] for n in normals)
    summed = _unit((sx, sy, sz))
    if summed == (0.0, 0.0, 0.0):
        return _unit(fallback)
    nx, ny, nz = summed
    return _unit((-nx, -ny, -nz))


def _contains_xy(block_center: Tuple[float, float, float], block_size: Tuple[float, float, float], x: float, y: float) -> bool:
    bx, by, _ = block_center
    sx, sy, _ = block_size
    return (abs(x - bx) <= 0.5 * sx + 1e-9) and (abs(y - by) <= 0.5 * sy + 1e-9)


def _support_normal(
    blocks: List[Dict],
    goal_center: Tuple[float, float, float],
    moving_size: Tuple[float, float, float],
) -> Tuple[float, float, float] | None:
    # Detect support surface directly beneath moving block bottom at goal.
    gx, gy, gz = goal_center
    _, _, mz = moving_size
    target_top = gz - 0.5 * mz
    best_gap = float("inf")
    found = False
    for b in blocks:
        c = tuple(float(v) for v in b["position"])
        s = tuple(float(v) for v in b["size"])
        if not _contains_xy(c, s, gx, gy):
            continue
        top = c[2] + 0.5 * s[2]
        gap = abs(top - target_top)
        if gap < best_gap:
            best_gap = gap
            found = True
    if found and best_gap <= 1e-6:
        return (0.0, 0.0, 1.0)
    return None


def _goal_normals_with_support(
    base_normals: List[Tuple[float, float, float]],
    blocks: List[Dict],
    goal_center: Tuple[float, float, float],
    moving_size: Tuple[float, float, float],
) -> List[Tuple[float, float, float]]:
    out = [_unit(n) for n in base_normals]
    support_n = _support_normal(blocks=blocks, goal_center=goal_center, moving_size=moving_size)
    if support_n is not None:
        sn = _unit(support_n)
        is_new = all(abs(sn[0] * n[0] + sn[1] * n[1] + sn[2] * n[2]) < 0.999 for n in out)
        if is_new:
            out.append(sn)
    return out


def _block(block_id: str, center: Tuple[float, float, float]) -> Dict:
    return {
        "id": block_id,
        "size": list(BLOCK_SIZE),
        "position": [_clean(v) for v in center],
    }


def _start_from_goal(goal: Tuple[float, float, float], approach_dir: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # Keep starts above the scene and offset opposite the approach vector.
    return (
        goal[0] - 1.2 * approach_dir[0],
        goal[1] - 1.2 * approach_dir[1],
        max(goal[2] + 0.9, 1.25),
    )


def _scenario_payload() -> Dict:
    scenarios: Dict[str, Dict] = {}

    table = {"id": "table", "size": list(TABLE_SIZE), "position": list(TABLE_POS)}

    c1 = (0.0, 0.0, GROUND_BLOCK_Z)
    c2_front = (GX, 0.0, GROUND_BLOCK_Z)
    c3_top = (GX, 0.0, GROUND_BLOCK_Z + GZ)
    c_left = (-GX, 0.0, GROUND_BLOCK_Z)
    c_right = (GX, 0.0, GROUND_BLOCK_Z)

    # Step 1: place first block on the ground plane.
    blocks1 = [table]
    n1 = _goal_normals_with_support([(0.0, 0.0, 1.0)], blocks=blocks1, goal_center=(0.0, 0.0, GROUND_BLOCK_Z), moving_size=BLOCK_SIZE)
    a1 = _infer_approach_direction(n1)
    g1 = (0.0, 0.0, GROUND_BLOCK_Z)
    scenarios["step_01_first_on_ground"] = {
        "blocks": blocks1,
        "moving_block": {
            "size": list(BLOCK_SIZE),
            "start": [_clean(v) for v in _start_from_goal(g1, a1)],
            "start_yaw_deg": 0.0,
            "goal_yaw_deg": 0.0,
            "goal": {
                "type": "face",
                "base": "table",
                "face": "top",
                "gap": 0.0,
                "tangential_offset": [0.0, 0.0],
                "size": list(BLOCK_SIZE),
            },
            "goal_normals": [[_clean(v) for v in n] for n in n1],
            "approach_direction": [_clean(v) for v in a1],
        },
    }

    # Step 2: place second block in front of first.
    blocks2 = [table, _block("block_1", c1)]
    g2 = _goal_center_from_face(c1, BLOCK_SIZE, BLOCK_SIZE, "front")
    n2 = _goal_normals_with_support([_normal_from_geometry(c1, g2)], blocks=blocks2, goal_center=g2, moving_size=BLOCK_SIZE)
    a2 = _infer_approach_direction(n2)
    scenarios["step_02_second_in_front"] = {
        "blocks": blocks2,
        "moving_block": {
            "size": list(BLOCK_SIZE),
            "start": [_clean(v) for v in _start_from_goal(g2, a2)],
            "start_yaw_deg": 0.0,
            "goal_yaw_deg": 0.0,
            "goal": {
                "type": "face",
                "base": "block_1",
                "face": "front",
                "gap": 0.0,
                "tangential_offset": [0.0, 0.0],
                "size": list(BLOCK_SIZE),
            },
            "goal_normals": [[_clean(v) for v in n] for n in n2],
            "approach_direction": [_clean(v) for v in a2],
        },
    }

    # Step 3: place third block on top (on top of second/front block).
    blocks3 = [table, _block("block_1", c1), _block("block_2", c2_front)]
    g3 = _goal_center_from_face(c2_front, BLOCK_SIZE, BLOCK_SIZE, "top")
    n3 = _goal_normals_with_support([_normal_from_geometry(c2_front, g3)], blocks=blocks3, goal_center=g3, moving_size=BLOCK_SIZE)
    a3 = _infer_approach_direction(n3)
    scenarios["step_03_third_on_top"] = {
        "blocks": blocks3,
        "moving_block": {
            "size": list(BLOCK_SIZE),
            "start": [_clean(v) for v in _start_from_goal(g3, a3)],
            "start_yaw_deg": 0.0,
            "goal_yaw_deg": 0.0,
            "goal": {
                "type": "face",
                "base": "block_2",
                "face": "top",
                "gap": 0.0,
                "tangential_offset": [0.0, 0.0],
                "size": list(BLOCK_SIZE),
            },
            "goal_normals": [[_clean(v) for v in n] for n in n3],
            "approach_direction": [_clean(v) for v in a3],
        },
    }

    # Step 4: place block between two existing blocks.
    blocks4 = [table, _block("left_block", c_left), _block("right_block", c_right)]
    g4 = (0.0, 0.0, GROUND_BLOCK_Z)
    n4 = _goal_normals_with_support([(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)], blocks=blocks4, goal_center=g4, moving_size=BLOCK_SIZE)
    a4 = _infer_approach_direction(n4, fallback=(0.0, -1.0, 0.0))
    scenarios["step_04_between_two_blocks"] = {
        "blocks": blocks4,
        "moving_block": {
            "size": list(BLOCK_SIZE),
            "start": [_clean(v) for v in _start_from_goal(g4, a4)],
            "start_yaw_deg": 0.0,
            "goal_yaw_deg": 0.0,
            "goal": {
                "type": "between",
                "ids": ["left_block", "right_block"],
                "position": [None, None, _clean(GROUND_BLOCK_Z)],
            },
            "goal_normals": [[_clean(v) for v in n] for n in n4],
            "approach_direction": [_clean(v) for v in a4],
        },
    }

    return {
        "defaults": {"base_size": list(BLOCK_SIZE)},
        "scenarios": scenarios,
    }


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value and all(ch.isalnum() or ch in "._-/" for ch in value):
            return value
        return '"' + value.replace('"', '\\"') + '"'
    raise TypeError(f"Unsupported scalar type: {type(value)}")


def _emit_yaml(node, indent: int = 0) -> List[str]:
    pad = " " * indent
    lines: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_emit_yaml(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(value)}")
        return lines
    if isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_emit_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    lines.append(f"{pad}{_yaml_scalar(node)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate placement scenarios YAML.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("generated_scenarios.yaml")),
        help="Output YAML path.",
    )
    args = parser.parse_args()

    payload = _scenario_payload()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = "\n".join(_emit_yaml(payload)) + "\n"
    with out.open("w", encoding="utf-8") as fh:
        fh.write(yaml_text)
    print(f"Wrote {len(payload['scenarios'])} scenarios to {out}")


if __name__ == "__main__":
    main()
