from dataclasses import dataclass
from typing import Callable, Dict, Tuple

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


base_size = (0.6, 0.9, 0.6)

def _common_scene() -> Scene:
    scene = Scene()
    scene.add_block(
        size=(2.0, 2.0, 0.1),
        position=(0.0, 0.0, 0.05),
        quat=(0.0, 0.0, 0.0, 1.0),
        object_id="table",
    )
    scene.add_block(
        size=(0.1, 2.0, 2.0),
        position=(1.0, 0.0, 1.0),
        quat=(0.0, 0.0, 0.0, 1.0),
        object_id="wall",
    )
    return scene


def scenario_front_of_block() -> ScenarioConfig:
    """Original demo: place payload in front of an existing block."""
    scene = _common_scene()
    
    pos_top = scene.get_top_point("table", base_size, gap=0.0, xy_offset=(0.0, -0.5))
    scene.add_block(size=base_size, position=tuple(pos_top), quat=(0.0, 0.0, 0.0, 1.0), object_id="cube_top")

    moving_block_size = base_size
    start_yaw_deg = 0.0
    goal_yaw_deg = 90.0
    moving_block_size_goal = (moving_block_size[1], moving_block_size[0], moving_block_size[2])

    goal = scene.get_front_point(
        base="cube_top",
        new_size=moving_block_size_goal,
        gap=0.0,
        xz_offset=(0.0, 0.0),
    )
    start = (-0.4, -0.2, 1.50)

    cube_top = scene.get_block("cube_top")
    front_normal = tuple(quat_to_rot(cube_top.quat)[:, 1].tolist())
    table = scene.get_block("table")
    up_normal = tuple(quat_to_rot(table.quat)[:, 2].tolist())

    return ScenarioConfig(
        scene=scene,
        start=start,
        goal=tuple(goal),
        moving_block_size=moving_block_size,
        start_yaw_deg=start_yaw_deg,
        goal_yaw_deg=goal_yaw_deg,
        goal_normals=(front_normal, up_normal),
    )


def scenario_block_on_top() -> ScenarioConfig:
    """Place payload on top of an existing block."""
    scene = _common_scene()

    scene.add_block(
        size=base_size,
        position=(0.05, -0.2, 0.35),
        quat=(0.0, 0.0, 0.0, 1.0),
        object_id="base_block",
    )

    moving_block_size = base_size
    start_yaw_deg = 10.0
    goal_yaw_deg = 0.0

    goal = scene.get_top_point(
        base="base_block",
        new_size=moving_block_size,
        gap=0.0,
        xy_offset=(0.0, 0.0),
    )
    start = (-0.9, 0.6, 1.35)

    base = scene.get_block("base_block")
    up_normal = tuple(quat_to_rot(base.quat)[:, 2].tolist())
    return ScenarioConfig(
        scene=scene,
        start=start,
        goal=tuple(goal),
        moving_block_size=moving_block_size,
        start_yaw_deg=start_yaw_deg,
        goal_yaw_deg=goal_yaw_deg,
        goal_normals=(up_normal,),
    )


def scenario_between_two_blocks() -> ScenarioConfig:
    """Place payload centered between two existing blocks."""
    scene = _common_scene()

    left_id = scene.add_block(
        size=base_size,
        position=(-0.6, 0.0, 0.35),
        quat=(0.0, 0.0, 0.0, 1.0),
        object_id="left_block",
    )
    right_id = scene.add_block(
        size=base_size,
        position=(0.6, 0.0, 0.35),
        quat=(0.0, 0.0, 0.0, 1.0),
        object_id="right_block",
    )

    moving_block_size = base_size
    start_yaw_deg = -15.0
    goal_yaw_deg = 0.0

    left_pos = scene.get_block(left_id).position
    right_pos = scene.get_block(right_id).position
    goal = (
        0.5 * (left_pos[0] + right_pos[0]),
        0.5 * (left_pos[1] + right_pos[1]),
        0.35,
    )
    start = (-2, 0.0, 1.5)

    table = scene.get_block("table")
    up_normal = tuple(quat_to_rot(table.quat)[:, 2].tolist())
    return ScenarioConfig(
        scene=scene,
        start=start,
        goal=goal,
        moving_block_size=moving_block_size,
        start_yaw_deg=start_yaw_deg,
        goal_yaw_deg=goal_yaw_deg,
        goal_normals=(up_normal,),
    )


SCENARIO_BUILDERS: Dict[str, Callable[[], ScenarioConfig]] = {
    "front": scenario_front_of_block,
    "on_top": scenario_block_on_top,
    "between": scenario_between_two_blocks,
}


def build_scenario(name: str) -> ScenarioConfig:
    key = name.strip().lower()
    if key not in SCENARIO_BUILDERS:
        available = ", ".join(sorted(SCENARIO_BUILDERS.keys()))
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}")
    return SCENARIO_BUILDERS[key]()
