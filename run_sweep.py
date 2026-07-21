"""
Uniform split sweep for the joint communication-computing framework.

Evaluates a single *uniform* split (the same a_k for every UE) across a range of
values and prints energy + feasibility for each.  Useful as a quick, optimizer-
free baseline / sanity check.

The scenario (network, tasks, edge servers) is taken from ``config.yaml``.

Usage
-----
    python run_sweep.py                       # sweep a = 0.1 .. 0.9 (step 0.1)
    python run_sweep.py --a 0.5               # single uniform split a = 0.5 (fair)
    python run_sweep.py --start 0.1 --stop 0.9 --step 0.05
"""

import os
import argparse

import yaml
import numpy as np

from joint_evaluator import get_or_build_snapshot, evaluate

FEASIBILITY_TOL = 1e-3  # violations below this (seconds) are treated as feasible


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=float, default=None,
                        help="Evaluate a single uniform split instead of a sweep.")
    parser.add_argument("--start", type=float, default=0.1, help="Sweep start (default 0.1).")
    parser.add_argument("--stop", type=float, default=0.9, help="Sweep stop, inclusive (default 0.9).")
    parser.add_argument("--step", type=float, default=0.1, help="Sweep step (default 0.1).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file.")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(root, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    lam = float(cfg.get('evaluation', {}).get('lambda_penalty', 2000.0))

    print("Building network snapshot...")
    snapshot, tasks, servers, params = get_or_build_snapshot(cfg)
    K = len(tasks)
    deadlines = np.array([t['deadline'] for t in tasks], dtype=float)

    if args.a is not None:
        a_values = [args.a]
    else:
        a_values = np.arange(args.start, args.stop + 1e-9, args.step)

    # Per-plane energy: comm plane = AP+CPU infrastructure, comp plane = edge
    # servers (active execution + idle cores).  The resource columns expose the
    # per-plane heuristic decisions: active APs (H_c), total assigned cores and
    # cumulated assigned frequency (H_e).
    print(f"\n  {'a':>5} | {'comm(J)':>10} | {'comp(J)':>10} | {'total(J)':>10} | "
          f"{'penalty(J)':>10} | {'feasible':>9} | {'APs':>4} | {'cores':>5} | {'sumGHz':>7}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-"
          f"{'-'*4}-+-{'-'*5}-+-{'-'*7}")
    for a in a_values:
        r = evaluate(tasks, np.full(K, float(a)), snapshot, servers, params, cfg)
        comm_energy = float(r['energy_infra'])           # AP + CPU infrastructure
        comp_energy = float(np.sum(r['energy_comp'])) + float(r['edge_idle_energy'])
        total_energy = comm_energy + comp_energy
        total_time = r['upload_times'] + r['exec_times']
        penalty = lam * float(np.sum(np.maximum(0.0, total_time - deadlines)))
        n_feas = int(np.sum(total_time <= deadlines + FEASIBILITY_TOL))
        active_aps = int(np.sum(np.asarray(r['D_DCC']).sum(axis=1) > 0))
        placed = [x for x in r['assignments'] if x is not None]
        cores = sum(int(x['num_cores']) for x in placed)
        sum_ghz = sum(float(x['frequency_hz']) for x in placed) / 1e9
        print(f"  {a:5.2f} | {comm_energy:10.2f} | {comp_energy:10.2f} | "
              f"{total_energy:10.2f} | {penalty:10.2f} | {n_feas:>5}/{K} | "
              f"{active_aps:>4} | {cores:>5} | {sum_ghz:>7.1f}")


if __name__ == "__main__":
    main()
