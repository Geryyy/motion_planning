import numpy as np
from typing import Tuple, Callable, Dict, Optional, List

try:
    from scipy.interpolate import make_interp_spline
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError("Please install SciPy: pip install scipy") from e


def build_cubic_bspline(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a cubic (C2) B-spline interpolant through the given waypoints.
    points: (N,3) with N >= 4 for cubic interpolation.
    Returns S(u) with u in [0,1].
    """
    n = points.shape[0]
    if n < 4:
        raise ValueError("Cubic B-spline interpolation requires at least 4 waypoints.")
    u = np.linspace(0.0, 1.0, n)
    spline = make_interp_spline(u, points, k=3, axis=0)
    return lambda uq: spline(np.asarray(uq, dtype=float))


def build_scalar_bspline(values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Build a smooth scalar spline with automatic degree selection."""
    y = np.asarray(values, dtype=float).reshape(-1)
    n = y.size
    if n < 2:
        raise ValueError("Scalar spline requires at least 2 control points.")
    u = np.linspace(0.0, 1.0, n)
    k = min(3, n - 1)
    spline = make_interp_spline(u, y, k=k, axis=0)
    return lambda uq: np.asarray(spline(np.asarray(uq, dtype=float)), dtype=float)


def yaw_deg_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion [x,y,z,w] for a pure yaw rotation about +z."""
    half = 0.5 * np.deg2rad(float(yaw_deg))
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


def sample_curve(S: Callable, n: int = 101) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample curve and first differences for length/curvature approximations.
    Returns (P, dP) with shapes (n,3) and (n-1,3)
    """
    us = np.linspace(0.0, 1.0, n)
    P = S(us)
    dP = np.diff(P, axis=0)
    return P, dP


def path_length(P: np.ndarray) -> float:
    """
    Discrete path length of P (n,3)
    """
    dP = np.diff(P, axis=0)
    seg = np.linalg.norm(dP, axis=1)
    return float(np.sum(seg))


def curvature_cost(P: np.ndarray) -> float:
    """
    Discrete bending energy approximation: integral(kappa^2 ds).
    Low value -> smoother, less sharply curved paths.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 3:
        return 0.0
    du = 1.0 / float(n - 1)
    d1 = np.gradient(P, du, axis=0)
    d2 = np.gradient(d1, du, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = np.linalg.norm(np.cross(d1, d2), axis=1)
    eps = 1e-9
    kappa = cross / np.maximum(speed, eps) ** 3
    return float(np.sum((kappa * kappa) * speed) * du)


def mean_turn_angle_deg(P: np.ndarray, eps: float = 1e-12) -> float:
    """Mean turning angle between consecutive segments (degrees)."""
    dP = np.diff(P, axis=0)
    if dP.shape[0] < 2:
        return 0.0
    a = dP[:-1]
    b = dP[1:]
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    valid = (an > eps) & (bn > eps)
    if not np.any(valid):
        return 0.0
    cosang = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(np.degrees(np.mean(ang)))


def yaw_smoothness_cost(yaw_deg_samples: np.ndarray) -> float:
    """Smoothness penalty for yaw profile."""
    y = np.asarray(yaw_deg_samples, dtype=float).reshape(-1)
    if y.size < 3:
        return 0.0
    D2 = y[:-2] - 2.0 * y[1:-1] + y[2:]
    return float(np.sum(D2 * D2))


def _path_distances(
    scene,
    P: np.ndarray,
    moving_block_size: Optional[Tuple[float, float, float]] = None,
    moving_block_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    moving_block_quats: Optional[np.ndarray] = None,
    ignore_ids: Optional[List[str]] = None,
) -> np.ndarray:
    """Distance profile for points or a moving block sampled along P."""
    if moving_block_size is None:
        return np.array([scene.signed_distance(p) for p in P], dtype=float)
    if moving_block_quats is not None:
        Q = np.asarray(moving_block_quats, dtype=float)
        if Q.shape != (P.shape[0], 4):
            raise ValueError("moving_block_quats must have shape (len(P), 4)")
        return np.array(
            [
                scene.signed_distance_block(
                    size=moving_block_size,
                    position=p,
                    quat=tuple(Q[i].tolist()),
                    ignore_ids=ignore_ids,
                )
                for i, p in enumerate(P)
            ],
            dtype=float,
        )
    return np.array(
        [
            scene.signed_distance_block(
                size=moving_block_size,
                position=p,
                quat=moving_block_quat,
                ignore_ids=ignore_ids,
            )
            for p in P
        ],
        dtype=float,
    )


def safety_cost(
    scene,
    P: np.ndarray,
    required_clearance: float,
    moving_block_size: Optional[Tuple[float, float, float]] = None,
    moving_block_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    moving_block_quats: Optional[np.ndarray] = None,
    ignore_ids: Optional[List[str]] = None,
) -> float:
    """
    Safety penalty from sampled signed distances along the path.
    cost = sum(max(0, required_clearance - d_i)^2)
    """
    dists = _path_distances(
        scene,
        P,
        moving_block_size=moving_block_size,
        moving_block_quat=moving_block_quat,
        moving_block_quats=moving_block_quats,
        ignore_ids=ignore_ids,
    )
    deficit = np.maximum(0.0, float(required_clearance) - dists)
    return float(np.sum(deficit * deficit))


def _default_via_initialization(start: np.ndarray, goal: np.ndarray, n_vias: int) -> np.ndarray:
    """Place vias uniformly on the line from start to goal."""
    if n_vias <= 0:
        return np.empty((0, 3), dtype=float)
    t = np.linspace(1.0 / (n_vias + 1), n_vias / (n_vias + 1), n_vias)
    return start[None, :] + t[:, None] * (goal - start)[None, :]


def _cem_optimize(
    objective: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    sigma0: np.ndarray,
    population_size: int = 96,
    elite_frac: float = 0.2,
    max_iter: int = 120,
    tol: float = 1e-3,
    alpha: float = 0.7,
    min_sigma: float = 1e-3,
    seed: Optional[int] = None,
):
    """Simple Cross-Entropy Method optimizer for continuous variables."""
    rng = np.random.default_rng(seed)
    mu = x0.astype(float).copy()
    sigma = np.maximum(np.asarray(sigma0, dtype=float), min_sigma)

    n_dim = mu.size
    elite_count = max(2, int(np.ceil(population_size * elite_frac)))

    best_x = mu.copy()
    best_val = float("inf")
    prev_best = float("inf")

    nit = 0
    for it in range(max_iter):
        nit = it + 1
        eps = rng.normal(size=(population_size, n_dim))
        candidates = mu[None, :] + eps * sigma[None, :]
        vals = objective(candidates)

        best_idx = int(np.argmin(vals))
        if vals[best_idx] < best_val:
            best_val = float(vals[best_idx])
            best_x = candidates[best_idx].copy()

        elite_idx = np.argpartition(vals, elite_count - 1)[:elite_count]
        elites = candidates[elite_idx]

        mu_new = np.mean(elites, axis=0)
        sigma_new = np.std(elites, axis=0) + min_sigma

        mu = alpha * mu + (1.0 - alpha) * mu_new
        sigma = np.maximum(alpha * sigma + (1.0 - alpha) * sigma_new, min_sigma)

        if np.abs(prev_best - best_val) < tol and float(np.max(sigma)) < 0.03:
            break
        prev_best = best_val

    return {
        "x": best_x,
        "fun": best_val,
        "nit": nit,
        "success": True,
        "message": "CEM finished",
    }


def _solve_optimizer(
    objective_single: Callable[[np.ndarray], Tuple],
    x0: np.ndarray,
    sigma0: np.ndarray,
    method: str,
    options: Optional[Dict] = None,
) -> Dict:
    method_upper = method.upper()
    if method_upper == "CEM":
        cem_options = {
            "population_size": 64,
            "elite_frac": 0.2,
            "max_iter": 90,
            "tol": 1e-3,
            "alpha": 0.7,
            "min_sigma": 1e-3,
            "seed": None,
        }
        if options:
            cem_options.update(options)

        def objective_batch(X):
            return np.array([objective_single(x)[0] for x in X], dtype=float)

        res = _cem_optimize(objective_batch, x0=x0, sigma0=sigma0, **cem_options)
        return {
            "x": res["x"],
            "success": bool(res["success"]),
            "message": str(res["message"]),
            "nit": int(res["nit"]),
            "fun": float(res["fun"]),
        }

    if method_upper in {"CEM-POWELL", "HYBRID"}:
        cem_options = {
            "population_size": 48,
            "elite_frac": 0.2,
            "max_iter": 60,
            "tol": 1e-3,
            "alpha": 0.7,
            "min_sigma": 1e-3,
            "seed": None,
        }
        powell_options = {"maxiter": 80, "xtol": 1e-3, "ftol": 1e-3}
        if options:
            if isinstance(options.get("cem"), dict):
                cem_options.update(options["cem"])
            if isinstance(options.get("powell"), dict):
                powell_options.update(options["powell"])
            for k, v in options.items():
                if k in cem_options:
                    cem_options[k] = v
                if k in powell_options:
                    powell_options[k] = v

        def objective_batch(X):
            return np.array([objective_single(x)[0] for x in X], dtype=float)

        cem_res = _cem_optimize(objective_batch, x0=x0, sigma0=sigma0, **cem_options)
        local_res = minimize(
            lambda x: objective_single(x)[0],
            cem_res["x"],
            method="Powell",
            options=powell_options,
        )
        if float(local_res.fun) <= float(cem_res["fun"]):
            return {
                "x": local_res.x,
                "success": bool(local_res.success),
                "message": f"Hybrid CEM+Powell: {local_res.message}",
                "nit": int(getattr(local_res, "nit", 0)) + int(cem_res["nit"]),
                "fun": float(local_res.fun),
            }
        return {
            "x": cem_res["x"],
            "success": bool(cem_res["success"]),
            "message": "Hybrid CEM+Powell: kept CEM result",
            "nit": int(cem_res["nit"]),
            "fun": float(cem_res["fun"]),
        }

    if method_upper in {"NELDER", "NEAD-MELDER", "NEAD_MELDER"}:
        method = "Nelder-Mead"
        method_upper = "NELDER-MEAD"
    if method_upper == "POWELL":
        scipy_options = {"maxiter": 220, "xtol": 1e-3, "ftol": 1e-3}
    elif method_upper == "NELDER-MEAD":
        scipy_options = {"maxiter": 300, "xatol": 1e-3, "fatol": 1e-3}
    else:
        scipy_options = {"maxiter": 250, "xatol": 1e-3, "fatol": 1e-3}
    if options:
        method_key = method.lower().replace("-", "_")
        if isinstance(options.get(method_key), dict):
            scipy_options.update(options[method_key])
        else:
            scipy_options.update(options)
    scipy_res = minimize(
        lambda x: objective_single(x)[0],
        x0,
        method=method,
        options=scipy_options,
    )
    return {
        "x": scipy_res.x,
        "success": bool(scipy_res.success),
        "message": str(scipy_res.message),
        "nit": int(getattr(scipy_res, "nit", 0)),
        "fun": float(scipy_res.fun),
    }


def optimize_bspline_path(
    scene,
    start: np.ndarray,
    goal: np.ndarray,
    n_vias: int = 3,
    tool_half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    moving_block_size: Optional[Tuple[float, float, float]] = None,
    moving_block_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    collision_ignore_ids: Optional[List[str]] = None,
    safety_margin: float = 0.01,
    n_samples_curve: int = 121,
    collision_check_subsample: int = 1,
    start_yaw_deg: float = 0.0,
    goal_yaw_deg: float = 0.0,
    n_yaw_vias: int = 0,
    combined_4d: bool = True,
    w_len: float = 1.0,
    w_curv: float = 0.1,
    w_yaw_smooth: float = 0.0,
    w_safe: float = 50.0,
    preferred_safety_margin: Optional[float] = None,
    relax_preferred_final_fraction: float = 0.0,
    w_safe_preferred: float = 0.0,
    w_approach_rebound: float = 0.0,
    w_goal_clearance: float = 0.0,
    goal_clearance_target: Optional[float] = None,
    w_goal_clearance_target: float = 0.0,
    approach_only_clearance: Optional[float] = None,
    contact_window_fraction: float = 0.1,
    w_approach_clearance: float = 0.0,
    w_approach_collision: float = 0.0,
    approach_fraction: float = 0.2,
    w_via_dev: float = 0.0,
    w_yaw_dev: float = 0.0,
    w_yaw_monotonic: float = 0.0,
    yaw_goal_reach_u: float = 1.0,
    w_yaw_schedule: float = 0.0,
    init_vias: Optional[np.ndarray] = None,
    init_yaw_vias_deg: Optional[np.ndarray] = None,
    init_offset_scale: float = 1.0,
    method: str = "Powell",
    options: Optional[Dict] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, Dict]:
    """
    Optimize all vias for a cubic B-spline through [start, vias..., goal].
    """
    start = np.asarray(start, float).reshape(3)
    goal = np.asarray(goal, float).reshape(3)
    if n_vias < 2:
        raise ValueError("n_vias must be >= 2 (cubic spline needs >=4 points total).")
    if collision_check_subsample < 1:
        raise ValueError("collision_check_subsample must be >= 1")
    if n_yaw_vias < 0:
        raise ValueError("n_yaw_vias must be >= 0")
    if combined_4d and n_yaw_vias not in (0, n_vias):
        raise ValueError("For combined_4d=True, n_yaw_vias must be 0 or equal to n_vias.")
    if not (0.0 < float(approach_fraction) <= 1.0):
        raise ValueError("approach_fraction must be in (0, 1].")
    if not (0.0 < float(contact_window_fraction) < 1.0):
        raise ValueError("contact_window_fraction must be in (0, 1).")
    if not (0.0 < float(yaw_goal_reach_u) <= 1.0):
        raise ValueError("yaw_goal_reach_u must be in (0, 1].")
    if not (0.0 <= float(relax_preferred_final_fraction) < 1.0):
        raise ValueError("relax_preferred_final_fraction must be in [0, 1).")

    if moving_block_size is None and any(float(v) > 0.0 for v in tool_half_extents):
        hx, hy, hz = map(float, tool_half_extents)
        moving_block_size = (2.0 * hx, 2.0 * hy, 2.0 * hz)

    required_clearance = float(safety_margin)
    preferred_clearance = float(preferred_safety_margin) if preferred_safety_margin is not None else required_clearance
    preferred_clearance = max(preferred_clearance, required_clearance)
    via_init = _default_via_initialization(start, goal, n_vias)
    if init_vias is not None:
        init_v = np.asarray(init_vias, dtype=float).reshape(-1, 3)
        if init_v.shape != (n_vias, 3):
            raise ValueError(f"init_vias must have shape ({n_vias}, 3)")
        via_init = init_v
    x0_pos = via_init.reshape(-1)
    yaw_via_count = n_vias if combined_4d else n_yaw_vias
    has_yaw_opt = bool(yaw_via_count > 0)

    if has_yaw_opt:
        yaw_ctrl_ref = np.linspace(float(start_yaw_deg), float(goal_yaw_deg), yaw_via_count + 2, dtype=float)
        yaw_via_init = np.linspace(
            float(start_yaw_deg),
            float(goal_yaw_deg),
            yaw_via_count + 2,
            dtype=float,
        )[1:-1]
        if init_yaw_vias_deg is not None:
            init_yaw = np.asarray(init_yaw_vias_deg, dtype=float).reshape(-1)
            if init_yaw.shape != (yaw_via_count,):
                raise ValueError(f"init_yaw_vias_deg must have shape ({yaw_via_count},)")
            yaw_via_init = init_yaw
        x0 = np.hstack([x0_pos, yaw_via_init])
    else:
        yaw_ctrl_ref = np.array([float(start_yaw_deg), float(goal_yaw_deg)], dtype=float)
        x0 = x0_pos

    sigma_base = np.linalg.norm(goal - start) * init_offset_scale / max(n_vias, 1)
    sigma0_pos = np.full_like(x0_pos, max(0.05, sigma_base), dtype=float)
    if has_yaw_opt:
        sigma0_yaw = np.full((yaw_via_count,), 20.0, dtype=float)
        sigma0 = np.hstack([sigma0_pos, sigma0_yaw])
    else:
        sigma0 = sigma0_pos

    us = np.linspace(0.0, 1.0, n_samples_curve)

    def _decode_yaw_controls(x: np.ndarray) -> np.ndarray:
        if has_yaw_opt:
            yaw_vias = x[x0_pos.size:].reshape(yaw_via_count)
            return np.hstack([[float(start_yaw_deg)], yaw_vias, [float(goal_yaw_deg)]])
        return np.array([float(start_yaw_deg), float(goal_yaw_deg)], dtype=float)

    def objective_single(x: np.ndarray):
        vias = x[:x0_pos.size].reshape(n_vias, 3)

        yaw_ctrl = _decode_yaw_controls(x)
        if combined_4d:
            W4 = np.hstack(
                [
                    np.vstack([start, vias, goal]),
                    yaw_ctrl.reshape(-1, 1),
                ]
            )
            S4 = build_cubic_bspline(W4)
            Q4 = S4(us)
            P = Q4[:, :3]
            yaw_samples_deg = Q4[:, 3]
        else:
            W = np.vstack([start, vias, goal])
            S = build_cubic_bspline(W)
            P = S(us)
            S_yaw = build_scalar_bspline(yaw_ctrl)
            yaw_samples_deg = S_yaw(us)
        yaw_quats = np.array([yaw_deg_to_quat(y) for y in yaw_samples_deg], dtype=float)

        if collision_check_subsample > 1:
            idx_safe = np.arange(0, P.shape[0], collision_check_subsample, dtype=int)
            if idx_safe[-1] != P.shape[0] - 1:
                idx_safe = np.append(idx_safe, P.shape[0] - 1)
            P_safe = P[idx_safe]
            yaw_quats_safe = yaw_quats[idx_safe]
            us_safe = us[idx_safe]
        else:
            P_safe = P
            yaw_quats_safe = yaw_quats
            us_safe = us

        j_len = path_length(P)
        j_curv = curvature_cost(P)
        j_yaw = yaw_smoothness_cost(yaw_samples_deg)
        d_safe = _path_distances(
            scene,
            P_safe,
            moving_block_size=moving_block_size,
            moving_block_quat=moving_block_quat,
            moving_block_quats=yaw_quats_safe,
            ignore_ids=collision_ignore_ids,
        )
        def_req = np.maximum(0.0, required_clearance - d_safe)
        j_safe = float(np.sum(def_req * def_req))
        j_safe_pref = 0.0
        if preferred_clearance > required_clearance and w_safe_preferred > 0.0:
            if relax_preferred_final_fraction > 0.0:
                keep_n = max(1, int(np.floor((1.0 - relax_preferred_final_fraction) * d_safe.shape[0])))
                d_pref = d_safe[:keep_n]
            else:
                d_pref = d_safe
            def_pref = np.maximum(0.0, preferred_clearance - d_pref)
            j_safe_pref = float(np.sum(def_pref * def_pref))

        # Penalize clearance rebound in the final approach segment.
        n_tail = max(3, int(np.ceil(float(approach_fraction) * d_safe.shape[0])))
        tail = d_safe[-n_tail:]
        tail_inc = np.maximum(0.0, np.diff(tail))
        j_approach_rebound = float(np.sum(tail_inc * tail_inc))

        # Penalize ending farther than preferred clearance (encourages proper approach).
        end_clear = float(d_safe[-1])
        j_goal_clear = float(max(0.0, end_clear - preferred_clearance) ** 2)
        j_goal_target = 0.0
        if goal_clearance_target is not None and w_goal_clearance_target > 0.0:
            j_goal_target = float((end_clear - float(goal_clearance_target)) ** 2)

        # Keep clearance in approach, allow near-contact only in the terminal contact window.
        approach_mask = us_safe < (1.0 - float(contact_window_fraction))
        if np.any(approach_mask):
            d_approach = d_safe[approach_mask]
        else:
            d_approach = d_safe[:-1] if d_safe.shape[0] > 1 else d_safe
        approach_target = preferred_clearance if approach_only_clearance is None else float(approach_only_clearance)
        def_approach = np.maximum(0.0, approach_target - d_approach)
        j_approach_clear = float(np.sum(def_approach * def_approach))
        col_approach = np.maximum(0.0, -d_approach)
        j_approach_col = float(np.sum(col_approach * col_approach))

        # Penalize unnecessary control-point movement from straight-line/yaw interpolation.
        j_via_dev = float(np.sum((vias - via_init) ** 2))
        j_yaw_dev = float(np.sum((yaw_ctrl - yaw_ctrl_ref) ** 2))

        # Enforce one-way yaw motion (no back-and-forth).
        dyaw = np.diff(yaw_samples_deg)
        if float(goal_yaw_deg) >= float(start_yaw_deg):
            backtrack = np.maximum(0.0, -dyaw)
        else:
            backtrack = np.maximum(0.0, dyaw)
        j_yaw_mono = float(np.sum(backtrack * backtrack))

        # Encourage yaw to reach target orientation early (e.g. around u=0.5).
        t_sched = np.clip(us / float(yaw_goal_reach_u), 0.0, 1.0)
        yaw_sched = float(start_yaw_deg) + (float(goal_yaw_deg) - float(start_yaw_deg)) * t_sched
        j_yaw_sched = float(np.sum((yaw_samples_deg - yaw_sched) ** 2))

        j = (
            w_len * j_len
            + w_curv * j_curv
            + w_yaw_smooth * j_yaw
            + w_safe * j_safe
            + w_safe_preferred * j_safe_pref
            + w_approach_rebound * j_approach_rebound
            + w_goal_clearance * j_goal_clear
            + w_goal_clearance_target * j_goal_target
            + w_approach_clearance * j_approach_clear
            + w_approach_collision * j_approach_col
            + w_via_dev * j_via_dev
            + w_yaw_dev * j_yaw_dev
            + w_yaw_monotonic * j_yaw_mono
            + w_yaw_schedule * j_yaw_sched
        )
        return (
            j,
            j_len,
            j_curv,
            j_safe,
            j_yaw,
            j_safe_pref,
            j_approach_rebound,
            j_goal_clear,
            j_goal_target,
            j_approach_clear,
            j_approach_col,
            j_via_dev,
            j_yaw_dev,
            j_yaw_mono,
            j_yaw_sched,
            yaw_samples_deg,
            yaw_quats,
        )

    opt = _solve_optimizer(objective_single, x0=x0, sigma0=sigma0, method=method, options=options)
    x_opt = opt["x"]
    vias_opt = x_opt[:x0_pos.size].reshape(n_vias, 3)
    us_opt = us
    yaw_ctrl_opt = _decode_yaw_controls(x_opt)
    if combined_4d:
        W4_opt = np.hstack(
            [
                np.vstack([start, vias_opt, goal]),
                yaw_ctrl_opt.reshape(-1, 1),
            ]
        )
        S4_opt = build_cubic_bspline(W4_opt)
        Q4_opt = S4_opt(us_opt)
        P_opt = Q4_opt[:, :3]
        yaw_samples_opt = Q4_opt[:, 3]
        def S_opt(uq):
            q = np.asarray(S4_opt(uq), dtype=float)
            if q.ndim == 1:
                return q[:3].reshape(1, 3)
            return q[:, :3]

        def S_yaw_opt(uq):
            q = np.asarray(S4_opt(uq), dtype=float)
            if q.ndim == 1:
                return np.array([q[3]], dtype=float)
            return q[:, 3]
    else:
        W_opt = np.vstack([start, vias_opt, goal])
        S_opt = build_cubic_bspline(W_opt)
        P_opt = S_opt(us_opt)
        S_yaw_opt = build_scalar_bspline(yaw_ctrl_opt)
        yaw_samples_opt = S_yaw_opt(us_opt)
    yaw_quats_opt = np.array([yaw_deg_to_quat(y) for y in yaw_samples_opt], dtype=float)
    (
        _,
        j_len_opt,
        j_curv_opt,
        j_safe_opt,
        j_yaw_opt,
        j_safe_pref_opt,
        j_approach_rebound_opt,
        j_goal_clear_opt,
        j_goal_target_opt,
        j_approach_clear_opt,
        j_approach_col_opt,
        j_via_dev_opt,
        j_yaw_dev_opt,
        j_yaw_mono_opt,
        j_yaw_sched_opt,
        _,
        _,
    ) = objective_single(x_opt)
    d_opt = _path_distances(
        scene,
        P_opt,
        moving_block_size=moving_block_size,
        moving_block_quat=moving_block_quat,
        moving_block_quats=yaw_quats_opt,
        ignore_ids=collision_ignore_ids,
    )

    info = {
        "success": opt["success"],
        "message": opt["message"],
        "fun": opt["fun"],
        "length": j_len_opt,
        "curvature_cost": j_curv_opt,
        "yaw_smoothness_cost": j_yaw_opt,
        "safety_cost": j_safe_opt,
        "preferred_safety_cost": j_safe_pref_opt,
        "approach_rebound_cost": j_approach_rebound_opt,
        "goal_clearance_cost": j_goal_clear_opt,
        "goal_clearance_target_cost": j_goal_target_opt,
        "approach_clearance_cost": j_approach_clear_opt,
        "approach_collision_cost": j_approach_col_opt,
        "via_deviation_cost": j_via_dev_opt,
        "yaw_deviation_cost": j_yaw_dev_opt,
        "yaw_monotonic_cost": j_yaw_mono_opt,
        "yaw_schedule_cost": j_yaw_sched_opt,
        "min_clearance": float(np.min(d_opt)),
        "mean_clearance": float(np.mean(d_opt)),
        "turn_angle_mean_deg": mean_turn_angle_deg(P_opt),
        "yaw_start_deg": float(start_yaw_deg),
        "yaw_goal_deg": float(goal_yaw_deg),
        "yaw_ctrl_deg": yaw_ctrl_opt.copy(),
        "yaw_samples_deg": yaw_samples_opt.copy(),
        "yaw_fn": S_yaw_opt,
        "combined_4d": bool(combined_4d),
        "solver_method": method,
        "required_clearance": required_clearance,
        "preferred_clearance": preferred_clearance,
        "goal_clearance_target": goal_clearance_target,
        "approach_only_clearance": approach_only_clearance,
        "contact_window_fraction": float(contact_window_fraction),
        "yaw_goal_reach_u": float(yaw_goal_reach_u),
        "collision_model": "box" if moving_block_size is not None else "point",
        "nit": opt["nit"],
    }
    return S_opt, vias_opt, info


def optimize_bspline_with_vias(
    scene,
    start: np.ndarray,
    via: np.ndarray,
    goal: np.ndarray,
    n_additional_vias: int = 2,
    tool_half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    safety_margin: float = 0.01,
    n_samples_curve: int = 121,
    collision_check_subsample: int = 1,
    w_len: float = 1.0,
    w_curv: float = 0.1,
    w_safe: float = 50.0,
    init_offset_scale: float = 1.0,
    method: str = "CEM",
    options: Optional[Dict] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, Dict]:
    """
    Optimize N additional via points so the cubic B-spline through
    [start, via, additional_vias..., goal] has low length, low curvature,
    and satisfies a safety margin using the scene SDF.

    Returns:
        S: callable S(u in [0,1])->(m,3) samples
        vias_opt: optimized additional via points (N,3)
        info: dict with objective breakdown
    """
    start = np.asarray(start, float).reshape(3)
    via = np.asarray(via, float).reshape(3)
    goal = np.asarray(goal, float).reshape(3)

    if n_additional_vias < 1:
        raise ValueError("n_additional_vias must be >= 1")
    if collision_check_subsample < 1:
        raise ValueError("collision_check_subsample must be >= 1")

    moving_block_size = None
    if any(float(v) > 0.0 for v in tool_half_extents):
        hx, hy, hz = map(float, tool_half_extents)
        moving_block_size = (2.0 * hx, 2.0 * hy, 2.0 * hz)
    required_clearance = float(safety_margin)

    via_init = _default_via_initialization(via, goal, n_additional_vias)
    x0 = via_init.reshape(-1)

    sigma_base = np.linalg.norm(goal - via) * init_offset_scale / max(n_additional_vias, 1)
    sigma0 = np.full_like(x0, max(0.05, sigma_base), dtype=float)

    def objective_single(x: np.ndarray):
        vias_add = x.reshape(n_additional_vias, 3)
        W = np.vstack([start, via, vias_add, goal])
        S = build_cubic_bspline(W)
        P, _ = sample_curve(S, n=n_samples_curve)
        if collision_check_subsample > 1:
            P_safe = P[::collision_check_subsample]
            if not np.allclose(P_safe[-1], P[-1]):
                P_safe = np.vstack([P_safe, P[-1]])
        else:
            P_safe = P

        j_len = path_length(P)
        j_curv = curvature_cost(P)
        j_safe = safety_cost(
            scene,
            P_safe,
            required_clearance=required_clearance,
            moving_block_size=moving_block_size,
        )

        j = w_len * j_len + w_curv * j_curv + w_safe * j_safe
        return j, j_len, j_curv, j_safe

    opt = _solve_optimizer(objective_single, x0=x0, sigma0=sigma0, method=method, options=options)
    x_opt = opt["x"]

    vias_opt = x_opt.reshape(n_additional_vias, 3)
    W_opt = np.vstack([start, via, vias_opt, goal])
    S_opt = build_cubic_bspline(W_opt)
    P_opt, _ = sample_curve(S_opt, n=n_samples_curve)
    _, j_len_opt, j_curv_opt, j_safe_opt = objective_single(x_opt)
    d_opt = _path_distances(
        scene,
        P_opt,
        moving_block_size=moving_block_size,
    )

    info = {
        "success": opt["success"],
        "message": opt["message"],
        "fun": opt["fun"],
        "length": j_len_opt,
        "curvature_cost": j_curv_opt,
        "safety_cost": j_safe_opt,
        "min_clearance": float(np.min(d_opt)),
        "mean_clearance": float(np.mean(d_opt)),
        "turn_angle_mean_deg": mean_turn_angle_deg(P_opt),
        "required_clearance": required_clearance,
        "collision_model": "box" if moving_block_size is not None else "point",
        "nit": opt["nit"],
    }
    return S_opt, vias_opt, info


def optimize_bspline_two_vias(
    scene,
    start: np.ndarray,
    via: np.ndarray,
    goal: np.ndarray,
    tool_half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    safety_margin: float = 0.01,
    n_samples_curve: int = 121,
    w_len: float = 1.0,
    w_curv: float = 0.1,
    w_safe: float = 50.0,
    init_offset_scale: float = 0.3,
    method: str = "Nelder-Mead",
    options: Dict = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray, Dict]:
    """Backward-compatible wrapper for two additional via points."""
    S, vias, info = optimize_bspline_with_vias(
        scene=scene,
        start=start,
        via=via,
        goal=goal,
        n_additional_vias=2,
        tool_half_extents=tool_half_extents,
        safety_margin=safety_margin,
        n_samples_curve=n_samples_curve,
        w_len=w_len,
        w_curv=w_curv,
        w_safe=w_safe,
        init_offset_scale=init_offset_scale,
        method=method,
        options=options,
    )
    return S, vias[0], vias[1], info
