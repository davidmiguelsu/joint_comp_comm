"""
Comparison run for the joint communication-computing framework.

Produces a standard results table for the Fair split (a_k = 0.5 for every UE)
on the scenario defined in ``config.yaml``.

The table reports the number of feasible UEs, total energy (split into the
communication and computation planes), the deadline-violation penalty, and
wall-clock time.

Usage
-----
    python run_comparison.py
"""

import os
import time

import yaml
import numpy as np

from joint_evaluator import get_or_build_snapshot, evaluate, plane_violation

FEASIBILITY_TOL = 1e-3  # violations below this (seconds) are treated as feasible


def count_feasible(r):
    return int(np.sum(plane_violation(r) <= FEASIBILITY_TOL))


def calculate_metrics(r, lam):
    """Return (fitness, comm_J, comp_J, energy_J, penalty_J, violation_s)."""
    comm_energy = float(r['energy_infra'])            # AP + CPU infrastructure
    comp_energy = float(np.sum(r['energy_comp'])) + float(r['edge_idle_energy'])
    energy = comm_energy + comp_energy
    violations = plane_violation(r)
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

    print("Building network snapshot...")
    snapshot, tasks, servers, params = get_or_build_snapshot(cfg)
    K = len(tasks)

    results = []

    # --- Fair split (a = 0.5) ---
    print("Evaluating Fair split (a = 0.5)...")
    t0 = time.time()
    r = evaluate(tasks, np.full(K, 0.5), snapshot, servers, params, cfg)
    dt = time.time() - t0
    fit, comm, comp, e, p, v = calculate_metrics(r, lam)
    results.append(("Fair (0.5)", fit, comm, comp, e, p, v, count_feasible(r), dt)
                   + resource_usage(r))

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
