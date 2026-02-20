#!/usr/bin/env python3
"""Generate placement scenarios and write them to YAML.

Scenarios include:
- a ground plane ("table"),
- 1 to 3 fixed blocks on the table,
- one moving block whose goal is placed on the front/top/back face of block_1.

For each scenario this script computes:
- goal_normals: outward contact normal(s) at goal,
- approach_direction: inferred approach vector = -summed(goal_normals).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

BLOCK_SIZE = (0.9, 0.6, 0.6)  # x, y, z (depth, width, height) in meters
TABLE_SIZE = (4.0, 4.0, 0.1)
TABLE_POS = (0.0, 0.0, 0.05)

# Deterministic support block centers resting on the table.
SUPPORT_POSITIONS = [
    (-0.8, -0.4, 0.4),
    (0.0, 0.45, 0.4),
    (0.8, -0.2, 0.4),
]


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


def _infer_approach_direction(normals: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    sx = sum(n[0] for n in normals)
    sy = sum(n[1] for n in normals)
    sz = sum(n[2] for n in normals)
    nx, ny, nz = _unit((sx, sy, sz))
    return _unit((-nx, -ny, -nz))


def _scenario_payload() -> Dict:
    scenarios: Dict[str, Dict] = {}
    faces = ("front", "top", "back")

    for n_blocks in (1, 2, 3):
        blocks = [
            {"id": "table", "size": list(TABLE_SIZE), "position": list(TABLE_POS)},
        ]
        for i in range(n_blocks):
            blocks.append(
                {
                    "id": f"block_{i + 1}",
                    "size": list(BLOCK_SIZE),
                    "position": list(SUPPORT_POSITIONS[i]),
                }
            )

        base_center = SUPPORT_POSITIONS[0]
        for face in faces:
            goal_center = _goal_center_from_face(
                base_center=base_center,
                base_size=BLOCK_SIZE,
                moving_size=BLOCK_SIZE,
                face=face,
            )
            normal = _normal_from_geometry(base_center=base_center, goal_center=goal_center)
            approach_dir = _infer_approach_direction([normal])
            start = (
                goal_center[0] - 1.2 * approach_dir[0],
                goal_center[1] - 1.2 * approach_dir[1] + 0.35,
                max(goal_center[2] + 0.85, 1.25),
            )
            name = f"n{n_blocks}_{face}"
            scenarios[name] = {
                "blocks": blocks,
                "moving_block": {
                    "size": list(BLOCK_SIZE),
                    "start": [_clean(v) for v in start],
                    "start_yaw_deg": 0.0,
                    "goal_yaw_deg": 0.0,
                    "goal": {
                        "type": "face",
                        "base": "block_1",
                        "face": face,
                        "gap": 0.0,
                        "tangential_offset": [0.0, 0.0],
                        "size": list(BLOCK_SIZE),
                    },
                    "goal_normals": [[_clean(v) for v in normal]],
                    "approach_direction": [_clean(v) for v in approach_dir],
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
