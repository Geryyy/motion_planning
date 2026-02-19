from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from geom import Scene
from geom.utils import quat_to_rot


@dataclass(frozen=True)
class ScenarioConfig:
    scene: Scene
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    moving_block_size: Tuple[float, float, float]
    start_yaw_deg: float
    goal_yaw_deg: float
    goal_normals: Tuple[Tuple[float, float, float], ...]


DEFAULT_SCENARIOS_FILE = Path(__file__).with_name("scenarios.yaml")


class WorldModel:
    def __init__(self, scenarios_file: Path | str = DEFAULT_SCENARIOS_FILE):
        self.scenarios_file = Path(scenarios_file)
        self.payload = _load_yaml_payload(self.scenarios_file)

    def list_scenarios(self) -> List[str]:
        return sorted(self.payload["scenarios"].keys())

    def build_scenario(self, name: str) -> ScenarioConfig:
        scenarios = self.payload["scenarios"]
        key = str(name).strip().lower()
        if key not in scenarios:
            available = ", ".join(sorted(scenarios.keys()))
            raise ValueError(f"Unknown scenario '{name}'. Available: {available}")

        defaults = self.payload.get("defaults", {})
        base_size = tuple(float(v) for v in defaults.get("base_size", [0.6, 0.9, 0.6]))
        cfg = scenarios[key]

        scene = Scene()
        _load_blocks(scene, cfg.get("blocks", []), base_size)

        moving_cfg = cfg["moving_block"]
        moving_size = tuple(float(v) for v in moving_cfg.get("size", base_size))
        start = tuple(float(v) for v in moving_cfg["start"])
        start_yaw_deg = float(moving_cfg.get("start_yaw_deg", 0.0))
        goal_yaw_deg = float(moving_cfg.get("goal_yaw_deg", 0.0))

        goal = _resolve_goal(scene, moving_size, moving_cfg["goal"])
        normal_cfg = moving_cfg.get("normal_query", {})
        goal_normals = query_goal_normals(
            scene=scene,
            goal=np.asarray(goal, dtype=float),
            moving_block_size=moving_size,
            goal_yaw_deg=goal_yaw_deg,
            search_radius=float(normal_cfg.get("search_radius", 0.15)),
            tangential_margin=float(normal_cfg.get("tangential_margin", 0.03)),
            max_normals=int(normal_cfg.get("max_normals", 4)),
        )

        return ScenarioConfig(
            scene=scene,
            start=start,
            goal=goal,
            moving_block_size=moving_size,
            start_yaw_deg=start_yaw_deg,
            goal_yaw_deg=goal_yaw_deg,
            goal_normals=goal_normals,
        )


def list_scenarios(scenarios_file: Path | str = DEFAULT_SCENARIOS_FILE) -> List[str]:
    return WorldModel(scenarios_file=scenarios_file).list_scenarios()


def build_scenario(name: str, scenarios_file: Path | str = DEFAULT_SCENARIOS_FILE) -> ScenarioConfig:
    return WorldModel(scenarios_file=scenarios_file).build_scenario(name)


def query_goal_normals(
    scene: Scene,
    goal: np.ndarray,
    moving_block_size: Tuple[float, float, float],
    goal_yaw_deg: float,
    search_radius: float = 0.15,
    tangential_margin: float = 0.03,
    max_normals: int = 4,
) -> Tuple[Tuple[float, float, float], ...]:
    """Collect outward face normals from scene faces near moving-block contact at goal."""
    g = np.asarray(goal, dtype=float).reshape(3)
    moving_half = 0.5 * np.asarray(moving_block_size, dtype=float)
    R_goal = _yaw_rot(goal_yaw_deg)

    def moving_support(n_world: np.ndarray) -> float:
        n = _normalize(n_world)
        if not np.any(n):
            return 0.0
        # Box support extent along world direction n.
        return float(
            abs(float(np.dot(n, R_goal[:, 0]))) * moving_half[0]
            + abs(float(np.dot(n, R_goal[:, 1]))) * moving_half[1]
            + abs(float(np.dot(n, R_goal[:, 2]))) * moving_half[2]
        )

    candidates: List[Tuple[float, np.ndarray]] = []
    for block in scene.blocks:
        R = quat_to_rot(block.quat)
        c = np.asarray(block.position, dtype=float)
        h = 0.5 * np.asarray(block.size, dtype=float)
        p_local = R.T @ (g - c)

        for axis in range(3):
            tang = [i for i in range(3) if i != axis]
            for sign in (-1.0, 1.0):
                n_world = R[:, axis] * sign
                # Only keep faces whose outward normal points toward the goal center.
                if float(np.dot(n_world, g - c)) <= 0.0:
                    continue
                support_n = moving_support(n_world)
                plane_delta = abs(p_local[axis] - sign * h[axis])
                contact_delta = max(0.0, float(plane_delta - support_n))

                inside_tangent = (
                    abs(p_local[tang[0]]) <= h[tang[0]] + tangential_margin
                    and abs(p_local[tang[1]]) <= h[tang[1]] + tangential_margin
                )
                if contact_delta <= search_radius and inside_tangent:
                    n_oriented = _orient_toward_goal(n_world=n_world, block_center=c, goal=g)
                    candidates.append((float(contact_delta), n_oriented))

    # Fallback to closest face if query radius misses everything.
    if not candidates:
        best = (float("inf"), np.array([0.0, 0.0, 1.0], dtype=float))
        for block in scene.blocks:
            R = quat_to_rot(block.quat)
            c = np.asarray(block.position, dtype=float)
            h = 0.5 * np.asarray(block.size, dtype=float)
            p_local = R.T @ (g - c)
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    n_world = R[:, axis] * sign
                    if float(np.dot(n_world, g - c)) <= 0.0:
                        continue
                    plane_delta = abs(p_local[axis] - sign * h[axis])
                    contact_delta = max(0.0, float(plane_delta - moving_support(n_world)))
                    if contact_delta < best[0]:
                        n_oriented = _orient_toward_goal(n_world=n_world, block_center=c, goal=g)
                        best = (float(contact_delta), n_oriented)
        candidates = [best]

    candidates.sort(key=lambda item: item[0])
    unique: List[np.ndarray] = []
    for _, normal in candidates:
        n = _normalize(normal)
        if not np.any(n):
            continue
        is_new = all(abs(float(np.dot(n, u))) < 0.995 for u in unique)
        if is_new:
            unique.append(n)
        if len(unique) >= max(1, max_normals):
            break

    if not unique:
        unique = [np.array([0.0, 0.0, 1.0], dtype=float)]

    return tuple(tuple(float(v) for v in n.tolist()) for n in unique)


def _load_yaml_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict) or "scenarios" not in payload:
        raise ValueError(f"Invalid scenarios YAML: {path}")
    if not isinstance(payload["scenarios"], dict):
        raise ValueError("'scenarios' must be a mapping")
    return payload


def _load_blocks(scene: Scene, blocks: List[Dict[str, Any]], base_size: Tuple[float, float, float]) -> None:
    for item in blocks:
        size = tuple(float(v) for v in item.get("size", base_size))
        position = tuple(float(v) for v in item["position"])
        quat = tuple(float(v) for v in item.get("quat", [0.0, 0.0, 0.0, 1.0]))
        object_id = str(item["id"])
        scene.add_block(size=size, position=position, quat=quat, object_id=object_id)


def _resolve_goal(scene: Scene, moving_size: Tuple[float, float, float], goal_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    goal_type = str(goal_cfg.get("type", "point")).lower()

    if goal_type == "point":
        return tuple(float(v) for v in goal_cfg["position"])

    if goal_type == "face":
        base = goal_cfg["base"]
        face = str(goal_cfg["face"]).lower()
        gap = float(goal_cfg.get("gap", 0.0))
        tangential_offset = tuple(float(v) for v in goal_cfg.get("tangential_offset", [0.0, 0.0]))
        size_for_goal = tuple(float(v) for v in goal_cfg.get("size", moving_size))
        pos = scene.get_stack_point_on_face(
            base=base,
            new_size=size_for_goal,
            face=face,
            gap=gap,
            tangential_offset=tangential_offset,
        )
        return tuple(float(v) for v in pos.tolist())

    if goal_type == "between":
        ids = goal_cfg["ids"]
        p0 = np.asarray(scene.get_block(ids[0]).position, dtype=float)
        p1 = np.asarray(scene.get_block(ids[1]).position, dtype=float)
        mid = 0.5 * (p0 + p1)
        if "position" in goal_cfg:
            # Use provided coordinates with 'null' values meaning midpoint component.
            out = []
            provided = goal_cfg["position"]
            for i in range(3):
                out.append(float(mid[i]) if provided[i] is None else float(provided[i]))
            return tuple(out)
        return tuple(float(v) for v in mid.tolist())

    raise ValueError(f"Unknown goal type: {goal_type}")


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n < eps:
        return np.zeros_like(arr)
    return arr / n


def _yaw_rot(yaw_deg: float) -> np.ndarray:
    y = np.deg2rad(float(yaw_deg))
    c = float(np.cos(y))
    s = float(np.sin(y))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _orient_toward_goal(n_world: np.ndarray, block_center: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Flip normal so it points from block center toward the goal region."""
    n = _normalize(n_world)
    if not np.any(n):
        return n
    to_goal = np.asarray(goal, dtype=float).reshape(3) - np.asarray(block_center, dtype=float).reshape(3)
    if float(np.dot(n, to_goal)) < 0.0:
        return -n
    return n
