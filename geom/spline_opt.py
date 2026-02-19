import numpy as np
from typing import Tuple, Callable, Dict

try:
    from scipy.interpolate import make_interp_spline
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError("Please install SciPy: pip install scipy") from e


def build_cubic_bspline(points: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a cubic (C2) B-spline interpolant through the given waypoints.
    points: (N,3) with N >= 3
    Returns S(u) with u in [0,1].
    """
    N = points.shape[0]
    u = np.linspace(0.0, 1.0, N)
    spline = make_interp_spline(u, points, k=3, axis=0)
    return lambda uq: spline(np.asarray(uq, dtype=float))


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
    Simple discrete curvature proxy: sum of squared second differences.
    Low value -> smoother curve.
    """
    # second finite difference
    D2 = P[:-2] - 2.0 * P[1:-1] + P[2:]
    return float(np.sum(np.sum(D2 * D2, axis=1)))


def safety_cost(scene, P: np.ndarray, r_required: float) -> float:
    """
    Safety penalty using distance field: sum of squared deficits where
    d(p) < r_required. Zero if all points have enough clearance.

    cost = sum(max(0, r_required - d(p_i))^2)
    """
    dists = np.array([scene.signed_distance(p) for p in P], dtype=float)
    deficit = np.maximum(0.0, r_required - dists)
    return float(np.sum(deficit * deficit))


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
    """
    Optimize two extra via points (v1, v2) so the cubic B-spline through
    [start, via, v1, v2, goal] has low length, low curvature, and satisfies
    a safety margin using the scene SDF.

    Returns:
        S: callable S(u in [0,1])->(m,3) samples
        v1: optimized via point 1 (3,)
        v2: optimized via point 2 (3,)
        info: dict with objective breakdown
    """
    start = np.asarray(start, float).reshape(3)
    via   = np.asarray(via,   float).reshape(3)
    goal  = np.asarray(goal,  float).reshape(3)

    # Conservative required clearance radius from a box-shaped tool
    hx, hy, hz = map(float, tool_half_extents)
    r_tool = max(np.hypot(hx, hy), np.hypot(hx, hz), np.hypot(hy, hz))
    r_required = r_tool + float(safety_margin)

    # Initialize v1, v2 along the straight line via->goal
    dir_vg = goal - via
    L_vg = np.linalg.norm(dir_vg) + 1e-12
    dir_vg /= L_vg
    v1_init = via + init_offset_scale * (goal - via) * 0.33
    v2_init = via + init_offset_scale * (goal - via) * 0.66

    x0 = np.hstack([v1_init, v2_init])  # 6D variable

    def objective(x):
        v1 = x[0:3]
        v2 = x[3:6]
        W = np.vstack([start, via, v1, v2, goal])
        S = build_cubic_bspline(W)
        P, _ = sample_curve(S, n=n_samples_curve)

        J_len  = path_length(P)
        J_curv = curvature_cost(P)
        J_safe = safety_cost(scene, P, r_required)

        J = w_len * J_len + w_curv * J_curv + w_safe * J_safe
        return J

    res = minimize(
        objective, x0,
        method=method,
        options=({"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3} if options is None else options)
    )

    v1_opt = res.x[0:3]
    v2_opt = res.x[3:6]
    W_opt = np.vstack([start, via, v1_opt, v2_opt, goal])
    S_opt = build_cubic_bspline(W_opt)
    P_opt, _ = sample_curve(S_opt, n=n_samples_curve)

    info = {
        "success": bool(res.success),
        "message": res.message,
        "fun": float(res.fun),
        "length": path_length(P_opt),
        "curvature_cost": curvature_cost(P_opt),
        "safety_cost": safety_cost(scene, P_opt, r_required),
        "r_required": r_required,
        "nit": int(res.nit)
    }
    return S_opt, v1_opt, v2_opt, info