#!/usr/bin/env python3
"""Benchmark path planning strategies and run simple hyperparameter search.

Strategies:
- Existing local optimizer: Powell
- Cross-Entropy Method: CEM
- Hybrid global-local: CEM-POWELL

The script:
1) Loads scenarios from YAML.
2) Runs random-search hyperparameter optimization per strategy on train scenarios.
3) Benchmarks best configs on all scenarios.
4) Writes results to JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from geom.spline_opt import optimize_bspline_path
from scenarios import WorldModel


BASE_CONFIG: Dict[str, Any] = {
    "n_vias": 2,
    "safety_margin": 0.0,
    "preferred_safety_margin": 0.02,
    "relax_preferred_final_fraction": 0.25,
    "approach_only_clearance": 0.015,
    "contact_window_fraction": 0.08,
    "n_yaw_vias": 2,
    "combined_4d": True,
    "approach_fraction": 0.25,
    "w_via_dev": 0.06,
    "w_yaw_monotonic": 80.0,
    "yaw_goal_reach_u": 0.5,
    "goal_approach_window_fraction": 0.12,
    "init_offset_scale": 0.7,
    "goal_clearance_target": 0.0,
    "w_len": 5.0,
    "n_samples_curve": 101,
    "collision_check_subsample": 1,
    "w_curv": 0.12,
    "w_yaw_smooth": 0.008,
    "w_safe": 380.0,
    "w_safe_preferred": 24.0,
    "w_approach_rebound": 280.0,
    "w_goal_clearance": 35.0,
    "w_goal_clearance_target": 260.0,
    "w_approach_clearance": 420.0,
    "w_approach_collision": 1400.0,
    "w_yaw_dev": 0.05,
    "w_yaw_schedule": 55.0,
    "w_goal_approach_normal": 80.0,
}


def _sample_strategy_config(method: str, rng: np.random.Generator) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = dict(BASE_CONFIG)
    cfg["init_offset_scale"] = float(rng.choice([0.5, 0.7, 1.0]))
    cfg["w_len"] = float(rng.choice([3.5, 5.0, 6.0]))
    cfg["w_curv"] = float(rng.choice([0.08, 0.12, 0.18]))
    cfg["w_safe"] = float(rng.choice([260.0, 380.0, 520.0]))
    cfg["w_goal_approach_normal"] = float(rng.choice([40.0, 80.0, 120.0]))
    cfg["w_approach_collision"] = float(rng.choice([1000.0, 1400.0, 1800.0]))

    if method.upper() == "POWELL":
        options = {
            "maxiter": int(rng.choice([80, 140, 220])),
            "xtol": float(rng.choice([3e-3, 1e-3])),
            "ftol": float(rng.choice([3e-3, 1e-3])),
        }
    elif method.upper() == "CEM":
        options = {
            "population_size": int(rng.choice([48, 64, 96])),
            "elite_frac": float(rng.choice([0.15, 0.2, 0.25])),
            "max_iter": int(rng.choice([60, 90, 140])),
            "alpha": float(rng.choice([0.6, 0.7, 0.8])),
            "min_sigma": float(rng.choice([5e-4, 1e-3])),
            "tol": 1e-3,
            "seed": int(rng.integers(0, 1_000_000)),
        }
    elif method.upper() in {"CEM-POWELL", "HYBRID"}:
        options = {
            "cem": {
                "population_size": int(rng.choice([48, 64, 96])),
                "elite_frac": float(rng.choice([0.15, 0.2, 0.25])),
                "max_iter": int(rng.choice([45, 60, 90])),
                "alpha": float(rng.choice([0.6, 0.7, 0.8])),
                "min_sigma": float(rng.choice([5e-4, 1e-3])),
                "tol": 1e-3,
                "seed": int(rng.integers(0, 1_000_000)),
            },
            "powell": {
                "maxiter": int(rng.choice([60, 100, 140])),
                "xtol": float(rng.choice([3e-3, 1e-3])),
                "ftol": float(rng.choice([3e-3, 1e-3])),
            },
        }
    else:
        raise ValueError(f"Unsupported method: {method}")

    return cfg, options


def _scenario_score(info: Dict[str, Any], runtime_s: float) -> float:
    # Lower is better. Heavily penalize collision and non-success.
    min_clear = float(info.get("min_clearance", -1.0))
    collision_penalty = 50_000.0 * max(0.0, -min_clear) ** 2
    success_penalty = 10_000.0 if not bool(info.get("success", False)) else 0.0
    return float(info["fun"]) + collision_penalty + success_penalty + 0.15 * runtime_s


def _run_single(
    wm: WorldModel,
    scenario_name: str,
    method: str,
    config: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    sc = wm.build_scenario(scenario_name)
    t0 = time.perf_counter()
    _, _, info = optimize_bspline_path(
        scene=sc.scene,
        start=sc.start,
        goal=sc.goal,
        moving_block_size=sc.moving_block_size,
        start_yaw_deg=sc.start_yaw_deg,
        goal_yaw_deg=sc.goal_yaw_deg,
        goal_approach_normals=np.asarray(sc.goal_normals, dtype=float),
        method=method,
        options=options,
        **config,
    )
    dt = float(time.perf_counter() - t0)
    straight_len = float(np.linalg.norm(np.asarray(sc.goal, dtype=float) - np.asarray(sc.start, dtype=float)))
    return {
        "scenario": scenario_name,
        "runtime_s": dt,
        "score": _scenario_score(info, dt),
        "success": bool(info.get("success", False)),
        "fun": float(info.get("fun", 1e9)),
        "length": float(info.get("length", 0.0)),
        "path_efficiency": float(info.get("length", 0.0)) / max(straight_len, 1e-9),
        "curvature_cost": float(info.get("curvature_cost", 0.0)),
        "turn_angle_mean_deg": float(info.get("turn_angle_mean_deg", 0.0)),
        "yaw_smoothness_cost": float(info.get("yaw_smoothness_cost", 0.0)),
        "safety_cost": float(info.get("safety_cost", 0.0)),
        "preferred_safety_cost": float(info.get("preferred_safety_cost", 0.0)),
        "approach_rebound_cost": float(info.get("approach_rebound_cost", 0.0)),
        "goal_clearance_cost": float(info.get("goal_clearance_cost", 0.0)),
        "goal_clearance_target_cost": float(info.get("goal_clearance_target_cost", 0.0)),
        "approach_clearance_cost": float(info.get("approach_clearance_cost", 0.0)),
        "approach_collision_cost": float(info.get("approach_collision_cost", 0.0)),
        "goal_approach_normal_cost": float(info.get("goal_approach_normal_cost", 0.0)),
        "min_clearance": float(info.get("min_clearance", -1.0)),
        "mean_clearance": float(info.get("mean_clearance", 0.0)),
        "nit": int(info.get("nit", 0)),
        "message": str(info.get("message", "")),
    }


def _aggregate_numeric(per_scenario: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys = [
        "score",
        "runtime_s",
        "fun",
        "length",
        "path_efficiency",
        "curvature_cost",
        "turn_angle_mean_deg",
        "yaw_smoothness_cost",
        "safety_cost",
        "preferred_safety_cost",
        "approach_rebound_cost",
        "goal_clearance_cost",
        "goal_clearance_target_cost",
        "approach_clearance_cost",
        "approach_collision_cost",
        "goal_approach_normal_cost",
        "min_clearance",
        "mean_clearance",
        "nit",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([float(r[k]) for r in per_scenario], dtype=float)
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def _evaluate_config(
    wm: WorldModel,
    scenario_names: List[str],
    method: str,
    config: Dict[str, Any],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    per_scenario: List[Dict[str, Any]] = []
    for name in scenario_names:
        try:
            per_scenario.append(_run_single(wm, name, method, config, options))
        except Exception as exc:
            per_scenario.append(
                {
                    "scenario": name,
                    "runtime_s": 0.0,
                    "score": 1e9,
                    "success": False,
                    "fun": 1e9,
                    "length": 0.0,
                    "min_clearance": -1.0,
                    "nit": 0,
                    "message": f"Exception: {exc}",
                }
            )
    success_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in per_scenario]))
    agg = _aggregate_numeric(per_scenario)
    return {
        "mean_score": float(agg["score"]["mean"]),
        "std_score": float(agg["score"]["std"]),
        "success_rate": success_rate,
        "metrics": agg,
        "per_scenario": per_scenario,
    }


def _hyperopt(
    wm: WorldModel,
    train_scenarios: List[str],
    method: str,
    n_trials: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    trials: List[Dict[str, Any]] = []
    best_idx = -1
    best_score = float("inf")

    for i in range(n_trials):
        cfg, opts = _sample_strategy_config(method, rng)
        res = _evaluate_config(wm, train_scenarios, method, cfg, opts)
        trial = {
            "trial": i + 1,
            "method": method,
            "config": cfg,
            "options": opts,
            "mean_score": res["mean_score"],
            "std_score": res["std_score"],
            "success_rate": res["success_rate"],
        }
        trials.append(trial)
        if res["mean_score"] < best_score:
            best_score = res["mean_score"]
            best_idx = i

    best = trials[best_idx]
    return {"trials": trials, "best": best}


def _benchmark_best(
    wm: WorldModel,
    scenario_names: List[str],
    best_entry: Dict[str, Any],
) -> Dict[str, Any]:
    method = str(best_entry["method"])
    cfg = dict(best_entry["config"])
    opts = dict(best_entry["options"])
    eval_res = _evaluate_config(wm, scenario_names, method, cfg, opts)
    return {
        "method": method,
        "config": cfg,
        "options": opts,
        "aggregate": {
            "mean_score": eval_res["mean_score"],
            "std_score": eval_res["std_score"],
            "success_rate": eval_res["success_rate"],
            "metrics": eval_res["metrics"],
        },
        "per_scenario": eval_res["per_scenario"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark path planners and run hyperparameter search.")
    parser.add_argument(
        "--scenarios-file",
        default=str(Path(__file__).with_name("generated_scenarios.yaml")),
        help="Path to scenarios YAML file.",
    )
    parser.add_argument(
        "--methods",
        default="Powell,CEM,CEM-POWELL",
        help="Comma-separated methods to benchmark (supported: Powell,CEM,CEM-POWELL).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
        help="Hyperparameter trials per method.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for hyperparameter search.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("benchmark_results.json")),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Optional comma-separated scenario names to evaluate (subset of YAML).",
    )
    args = parser.parse_args()

    wm = WorldModel(scenarios_file=args.scenarios_file)
    all_scenarios = wm.list_scenarios()
    if args.scenarios.strip():
        wanted = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        missing = [s for s in wanted if s not in all_scenarios]
        if missing:
            raise ValueError(f"Unknown scenario(s) in --scenarios: {', '.join(missing)}")
        all_scenarios = wanted
    if not all_scenarios:
        raise ValueError("No scenarios found for benchmark.")

    # Simple split: keep one scenario as held-out test if possible.
    if len(all_scenarios) > 1:
        train_scenarios = all_scenarios[:-1]
        test_scenarios = all_scenarios
    else:
        train_scenarios = all_scenarios
        test_scenarios = all_scenarios

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m.upper() not in {"POWELL", "CEM", "CEM-POWELL", "HYBRID"}:
            raise ValueError(f"Unsupported method '{m}'. Use Powell, CEM, and/or CEM-POWELL.")

    hyperopt_results: Dict[str, Any] = {}
    benchmark_results: Dict[str, Any] = {}

    for method in methods:
        print(f"[hyperopt] method={method} trials={args.trials} train_scenarios={len(train_scenarios)}")
        hres = _hyperopt(
            wm=wm,
            train_scenarios=train_scenarios,
            method=method,
            n_trials=args.trials,
            seed=args.seed + (
                0 if method.upper() == "POWELL" else 10_000 if method.upper() == "CEM" else 20_000
            ),
        )
        hyperopt_results[method] = hres
        best = hres["best"]
        print(
            f"[best] method={method} mean_score={best['mean_score']:.4f} "
            f"success_rate={best['success_rate']:.2f}"
        )
        bres = _benchmark_best(wm, test_scenarios, best)
        benchmark_results[method] = bres
        agg = bres["aggregate"]
        print(
            f"[benchmark] method={method} mean_score={agg['mean_score']:.4f} "
            f"std={agg['std_score']:.4f} success_rate={agg['success_rate']:.2f}"
        )

    payload = {
        "scenarios_file": str(args.scenarios_file),
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "methods": methods,
        "trials_per_method": int(args.trials),
        "seed": int(args.seed),
        "hyperopt": hyperopt_results,
        "benchmark": benchmark_results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote benchmark results to {out}")


if __name__ == "__main__":
    main()
