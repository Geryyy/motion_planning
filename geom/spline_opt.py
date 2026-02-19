import numpy as np
from typing import Tuple, Callable, Dict, Optional

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


def _default_via_initialization(via: np.ndarray, goal: np.ndarray, n_additional_vias: int) -> np.ndarray:
    """Place additional vias along the line from via to goal."""
    if n_additional_vias <= 0:
        return np.empty((0, 3), dtype=float)
    t = np.linspace(1.0 / (n_additional_vias + 1), n_additional_vias / (n_additional_vias + 1), n_additional_vias)
    return via[None, :] + t[:, None] * (goal - via)[None, :]


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

    for _ in range(max_iter):
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
        "nit": max_iter,
        "success": True,
        "message": "CEM finished",
    }


def optimize_bspline_with_vias(
    scene,
    start: np.ndarray,
    via: np.ndarray,
    goal: np.ndarray,
    n_additional_vias: int = 2,
    tool_half_extents: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    safety_margin: float = 0.01,
    n_samples_curve: int = 121,
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

    hx, hy, hz = map(float, tool_half_extents)
    r_tool = max(np.hypot(hx, hy), np.hypot(hx, hz), np.hypot(hy, hz))
    r_required = r_tool + float(safety_margin)

    via_init = _default_via_initialization(via, goal, n_additional_vias)
    x0 = via_init.reshape(-1)

    sigma_base = np.linalg.norm(goal - via) * init_offset_scale / max(n_additional_vias, 1)
    sigma0 = np.full_like(x0, max(0.05, sigma_base), dtype=float)

    def objective_single(x: np.ndarray):
        vias_add = x.reshape(n_additional_vias, 3)
        W = np.vstack([start, via, vias_add, goal])
        S = build_cubic_bspline(W)
        P, _ = sample_curve(S, n=n_samples_curve)

        j_len = path_length(P)
        j_curv = curvature_cost(P)
        j_safe = safety_cost(scene, P, r_required)

        j = w_len * j_len + w_curv * j_curv + w_safe * j_safe
        return j, j_len, j_curv, j_safe

    if method.upper() == "CEM":
        cem_options = {
            "population_size": 96,
            "elite_frac": 0.2,
            "max_iter": 120,
            "tol": 1e-3,
            "alpha": 0.7,
            "min_sigma": 1e-3,
            "seed": None,
        }
        if options:
            cem_options.update(options)

        def objective_batch(X):
            return np.array([objective_single(x)[0] for x in X], dtype=float)

        result = _cem_optimize(objective_batch, x0=x0, sigma0=sigma0, **cem_options)
        x_opt = result["x"]
        success = result["success"]
        message = result["message"]
        nit = result["nit"]
        fun = float(result["fun"])
    else:
        scipy_options = {"maxiter": 250, "xatol": 1e-3, "fatol": 1e-3}
        if options:
            scipy_options.update(options)

        scipy_res = minimize(
            lambda x: objective_single(x)[0],
            x0,
            method=method,
            options=scipy_options,
        )
        x_opt = scipy_res.x
        success = bool(scipy_res.success)
        message = str(scipy_res.message)
        nit = int(getattr(scipy_res, "nit", 0))
        fun = float(scipy_res.fun)

    vias_opt = x_opt.reshape(n_additional_vias, 3)
    W_opt = np.vstack([start, via, vias_opt, goal])
    S_opt = build_cubic_bspline(W_opt)
    P_opt, _ = sample_curve(S_opt, n=n_samples_curve)
    _, j_len_opt, j_curv_opt, j_safe_opt = objective_single(x_opt)

    info = {
        "success": success,
        "message": message,
        "fun": fun,
        "length": j_len_opt,
        "curvature_cost": j_curv_opt,
        "safety_cost": j_safe_opt,
        "r_required": r_required,
        "nit": nit,
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
