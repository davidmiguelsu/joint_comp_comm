"""
Comparison run for the joint communication-computing framework.

Produces a standard results table comparing two split strategies on the
scenario defined in ``config.yaml``:

    1. Fair          : a_k = 0.5 for every UE.
    2. Random Search : best of N random split vectors (N from config, default 100).

For each strategy the table reports the number of feasible UEs, total energy
(split into the communication and computation planes), the deadline-violation
penalty, and wall-clock time.

Usage
-----
    python run_comparison.py
"""

import os
import time

import yaml
import numpy as np

from joint_evaluator import get_or_build_snapshot, evaluate

FEASIBILITY_TOL = 1e-3  # violations below this (seconds) are treated as feasible


def count_feasible(r, deadlines):
    total_time = r['upload_times'] + r['exec_times']
    return int(np.sum(total_time <= deadlines + FEASIBILITY_TOL))


def calculate_metrics(r, deadlines, lam):
    """Return (fitness, comm_J, comp_J, energy_J, penalty_J, violation_s)."""
    comm_energy = float(r['energy_infra'])            # AP + CPU infrastructure
    comp_energy = float(np.sum(r['energy_comp'])) + float(r['edge_idle_energy'])
    energy = comm_energy + comp_energy
    total_time = r['upload_times'] + r['exec_times']
    violations = np.maximum(0.0, total_time - deadlines)
    penalty = lam * float(np.sum(violations))
    return float(energy + penalty), comm_energy, comp_energy, energy, penalty, float(np.sum(violations))


def resource_usage(r):
    """Return (active_aps, total_cores) — the per-plane heuristic decisions."""
    active_aps = int(np.sum(np.asarray(r['D_DCC']).sum(axis=1) > 0))
    placed = [x for x in r['assignments'] if x is not None]
    return active_aps, sum(int(x['num_cores']) for x in placed)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    lam = float(cfg.get('evaluation', {}).get('lambda_penalty', 2000.0))
    rs_cfg = cfg.get('random_search', {})
    rs_n = int(rs_cfg.get('n_samples', 100))
    rs_seed = int(rs_cfg.get('seed', 42))
    a_min = float(rs_cfg.get('a_min', 0.05))
    a_max = float(rs_cfg.get('a_max', 0.95))

    print("Building network snapshot...")
    snapshot, tasks, servers, params = get_or_build_snapshot(cfg)
    K = len(tasks)
    deadlines = np.array([t['deadline'] for t in tasks], dtype=float)

    results = []

    # --- 1. Fair split (a = 0.5) ---
    print("Evaluating Fair split (a = 0.5)...")
    t0 = time.time()
    r = evaluate(tasks, np.full(K, 0.5), snapshot, servers, params, cfg)
    dt = time.time() - t0
    fit, comm, comp, e, p, v = calculate_metrics(r, deadlines, lam)
    results.append(("Fair (0.5)", fit, comm, comp, e, p, v, count_feasible(r, deadlines), dt)
                   + resource_usage(r))

    # --- 2. Random search ---
    print(f"Running Random Search ({rs_n} samples)...")
    t0 = time.time()
    rng = np.random.default_rng(rs_seed)
    best_fit, best_r = np.inf, None
    for _ in range(rs_n):
        a_vec = rng.uniform(a_min, a_max, size=K)
        r = evaluate(tasks, a_vec, snapshot, servers, params, cfg)
        fit, *_ = calculate_metrics(r, deadlines, lam)
        if fit < best_fit:
            best_fit, best_r = fit, r
    dt = time.time() - t0
    fit, comm, comp, e, p, v = calculate_metrics(best_r, deadlines, lam)
    results.append((f"Random Search ({rs_n})", fit, comm, comp, e, p, v, count_feasible(best_r, deadlines), dt)
                   + resource_usage(best_r))

    # --- Results table ---
    results.sort(key=lambda x: x[1])
    print("\n" + "=" * 118)
    print(f"  SPLIT-STRATEGY COMPARISON   (K = {K},  penalty lambda = {lam:g} J/s)")
    print("=" * 118)
    print(f"  {'Strategy':<22} | {'Feasible':>8} | {'Fitness':>12} | {'comm(J)':>9} | "
          f"{'comp(J)':>9} | {'total(J)':>9} | {'Penalty(J)':>10} | {'APs':>4} | {'cores':>5} | {'Time':>8}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-"
          f"{'-'*4}-+-{'-'*5}-+-{'-'*8}")
    for name, fit, comm, comp, e, p, v, feas, dt, aps, cores in results:
        print(f"  {name:<22} | {feas:>4}/{K:<3} | {fit:12.4f} | {comm:9.2f} | "
              f"{comp:9.2f} | {e:9.2f} | {p:10.2f} | {aps:>4} | {cores:>5} | {dt:7.2f}s")
    print("=" * 118)


if __name__ == "__main__":
    main()
