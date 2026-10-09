"""
Evaluate a custom per-UE split vector with the heuristics and print the results.

Unlike ``run_sweep.py`` (which applies the *same* a to every UE), this runner
takes a full vector a = (a_1, ..., a_K) — one split fraction per UE — runs both
heuristics via the joint evaluator, and prints a per-UE breakdown plus the
per-plane energy summary.

The scenario (network, tasks, edge servers) is taken from ``config.yaml``; the
vector length must match the number of UEs K.

Usage
-----
    # Comma-separated, one value per UE:
    python run_vector.py --a 0.5,0.6,0.3,0.9,...

    # Or read the values from a file (whitespace/comma/newline separated):
    python run_vector.py --file my_split.txt
"""

import os
import argparse

import yaml
import numpy as np

from joint_evaluator import get_or_build_snapshot, evaluate, plane_violation

FEASIBILITY_TOL = 1e-3  # violations below this (seconds) are treated as feasible


def parse_vector(text):
    """Parse a list of floats from a string (comma / whitespace / newline separated)."""
    tokens = text.replace(",", " ").split()
    return np.array([float(tok) for tok in tokens], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--a", type=str, help="Comma-separated per-UE split values (length K).")
    src.add_argument("--file", type=str, help="File with per-UE split values (any whitespace/comma separated).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file.")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(root, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    lam = float(cfg.get('evaluation', {}).get('lambda_penalty', 2000.0))

    if args.file is not None:
        with open(args.file) as f:
            a_vec = parse_vector(f.read())
    else:
        a_vec = parse_vector(args.a)

    if np.any(a_vec < 0) or np.any(a_vec > 1):
        parser.error("All split values must be in [0, 1].")

    # Fail fast on length mismatch before building the (slow) snapshot.
    expected_K = sum(int(p['count']) for p in cfg['task_profiles'].values())
    if a_vec.size != expected_K:
        parser.error(f"Split vector has length {a_vec.size}, but the scenario has K = {expected_K} UEs.")

    print("Building network snapshot...")
    snapshot, tasks, servers, params = get_or_build_snapshot(cfg)
    K = len(tasks)

    if a_vec.shape != (K,):
        parser.error(f"Split vector has length {a_vec.size}, but the scenario has K = {K} UEs.")

    r = evaluate(tasks, a_vec, snapshot, servers, params, cfg)

    comm_energy = float(r['energy_infra'])            # AP + CPU infrastructure
    comp_active = float(np.sum(r['energy_comp']))     # edge computation (active)
    comp_idle = float(r['edge_idle_energy'])          # edge computation (idle cores)
    comp_energy = comp_active + comp_idle
    total_energy = comm_energy + comp_energy
    violations = plane_violation(r)
    viol = float(np.sum(violations))
    penalty = lam * viol
    n_feas = int(np.sum(violations <= FEASIBILITY_TOL))

    D_DCC = np.asarray(r['D_DCC'])
    cluster_sizes = D_DCC.sum(axis=0)
    active_aps = int(np.sum(D_DCC.sum(axis=1) > 0))
    assignments = r['assignments']
    infra = r['infra_detail']

    # Per-UE breakdown: comm plane (H_c) on the left, comp plane (H_e) on the right
    print(f"\n  {'UE':>3} | {'profile':<11} | {'a_k':>4} | {'t_upl(s)':>8} | "
          f"{'SEach':>6} | {'SEreq':>6} | {'APs':>3} | {'t_exec(s)':>9} | "
          f"{'srv':>6} | {'cores':>5} | {'f(GHz)':>6} | {'deadline':>8} | {'feasible':>8}")
    print(f"  {'-'*3}-+-{'-'*11}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*3}-+-"
          f"{'-'*9}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for k, t in enumerate(tasks):
        ok = (r['upload_times'][k] <= r['t_comm'][k] + FEASIBILITY_TOL
              and r['exec_times'][k] <= r['t_proc'][k] + FEASIBILITY_TOL)
        asgn = assignments[k]
        srv = asgn['server_id'] if asgn else '-'
        cores = str(int(asgn['num_cores'])) if asgn else '-'
        f_ghz = f"{asgn['frequency_hz'] / 1e9:6.2f}" if asgn else f"{'-':>6}"
        be = '*' if (asgn and asgn.get('best_effort')) else ''
        print(f"  {k:>3} | {t.get('profile', '?'):<11} | {a_vec[k]:>4.2f} | "
              f"{r['upload_times'][k]:8.4f} | {r['per_user_se'][k]:6.2f} | "
              f"{r['required_se'][k]:6.2f} | {int(cluster_sizes[k]):>3} | "
              f"{r['exec_times'][k]:9.4f} | {srv:>6} | {cores:>5}{be:<0} | {f_ghz} | "
              f"{t['deadline']:8.4f} | {'yes' if ok else 'NO':>8}")
    if any(a and a.get('best_effort') for a in assignments):
        print("  (*) best-effort placement: deadline could not be met with available capacity")

    # Per-server utilization (H_e output)
    print(f"\n  {'Server':<8} | {'tasks':>5} | {'cores':>7} | {'power(W)':>9} | {'idle(J)':>8}")
    print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}")
    for s_idx, srv in enumerate(servers):
        placed = [a for a in assignments if a and a['server_id'] == srv['id']]
        if not placed and r['per_server_idle_energy'][s_idx] == 0:
            continue
        cores_used = sum(int(a['num_cores']) for a in placed)
        util = f"{cores_used}/{srv['capacity']}"
        print(f"  {srv['id']:<8} | {len(placed):>5} | {util:>7} | "
              f"{r['server_power'][s_idx]:9.2f} | {r['per_server_idle_energy'][s_idx]:8.2f}")

    # Summary
    print("\n" + "=" * 60)
    print(f"  Custom split vector over K = {K} UEs")
    print("=" * 60)
    print(f"  Feasible UEs        : {n_feas}/{K}")
    print(f"  Comm plane (AP+CPU) : {comm_energy:10.2f} J")
    print(f"    AP processing     : {float(np.sum(infra['per_ap_processing_energy_J'])):10.2f} J "
          f"(active {float(np.sum(infra['per_ap_processing_energy_active_J'])):.2f} / "
          f"sleep {float(np.sum(infra['per_ap_processing_energy_idle_J'])):.2f})")
    print(f"    Antennas          : {float(np.sum(infra['per_ap_antenna_energy_J'])):10.2f} J "
          f"(active {float(np.sum(infra['per_ap_antenna_energy_active_J'])):.2f} / "
          f"sleep {float(np.sum(infra['per_ap_antenna_energy_idle_J'])):.2f})")
    print(f"    CPU (combining)   : {float(infra['cpu_energy_J']):10.2f} J")
    print(f"  Comp plane (edge)   : {comp_energy:10.2f} J "
          f"(active {comp_active:.2f} / idle {comp_idle:.2f})")
    print(f"  Total energy        : {total_energy:10.2f} J")
    print(f"  Active APs          : {active_aps}/{len(D_DCC)}  "
          f"(avg cluster size {float(np.mean(cluster_sizes)):.1f})")
    print(f"  Penalty (lam={lam:g})  : {penalty:10.2f} J")
    print(f"  Total violation     : {viol:10.4f} s")
    print(f"  Fitness (E+penalty) : {total_energy + penalty:10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
