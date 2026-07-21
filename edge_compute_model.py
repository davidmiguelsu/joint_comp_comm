"""
Edge Computing Simulator Module.

Implements the edge server model and two-pass computation heuristic
for the joint communication-computation optimization framework.

Multi-core server model:
    Each server s is characterised by:
        capacity    — number of physical cores (used to derive total_capacity_hz)
        κ_s         — effective switched capacitance  (P_per_core = κ · f³)
        f_min       — minimum operating frequency (Hz)
        f_max       — maximum operating frequency (Hz)
        b_s         — acceleration coefficient
        freq_levels — number of discrete frequency levels (config-tunable, default 11)

    Derived at load time:
        total_capacity_hz = capacity × f_max
        discrete levels   = np.linspace(f_min, f_max, freq_levels)

Key formulas (n_c cores at quantised frequency f, task workload w_k cycles):
    Throughput needed:    F_need = w_k / (t_budget · b_s)    [Hz]
    Cores needed:         n_c = ⌈F_need / f⌉
    Processing time:      t_proc = w_k / (n_c · f · b_s)     [≤ t_budget ✓]
    Power per core:       P = κ · f³
    Total task power:     P_total = n_c · κ · f³
    Energy (n_c cancels): E = κ · f² · w_k / b_s

    All n_c cores of a task must run on a single server at the same frequency.

Heuristic (two-pass greedy):
    Pass 1 — Sort tasks by urgency (w_k / deadline, descending).
             Assign each task to the server yielding minimum energy at the
             highest feasible frequency level (fewest cores, most capacity
             preserved for subsequent tasks).
    Pass 2 — Greedy descent: for each assigned task, step down through
             frequency levels; apply each level where the capacity change
             (Δ = new_cap − cur_cap) fits within the server's remaining
             budget.  Non-monotone ceiling effects are handled by
             continuing past failed levels rather than stopping early.

All parameters configurable via config.yaml.
"""

import math
import os
import numpy as np
import yaml
from typing import Dict, List, Optional

# Cores are always allocated in multiples of this batch size.
CORE_BATCH = 4


def _ceil_to_batch(n: int, batch: int = CORE_BATCH) -> int:
    """Round n up to the nearest multiple of batch (minimum = batch)."""
    return max(batch, math.ceil(n / batch) * batch)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Warning: config.yaml not found at {CONFIG_PATH}.")
    CONFIG = None


# ---------------------------------------------------------------------------
# Data-structure helpers
# ---------------------------------------------------------------------------

def load_servers_from_config(config: Optional[Dict] = None) -> List[Dict]:
    """
    Parse the ``edge_servers`` section of config.yaml into server dicts.

    Each returned dict has:
        id, capacity, remaining_cores, total_capacity_hz, remaining_capacity_hz,
        kappa, f_min (Hz), f_max (Hz), b_s, freq_levels (list[float]), task_count

    ``total_capacity_hz`` is derived as ``capacity × f_max_hz``.
    ``freq_levels`` is a list of ``n_levels`` evenly-spaced frequencies
    from f_min to f_max (inclusive), where ``n_levels`` comes from the
    config ``freq_levels`` field (default 11).
    """
    if config is None:
        if CONFIG is None:
            raise ValueError("config must be provided if config.yaml is not found")
        config = CONFIG

    servers = []
    for spec in config['edge_servers']:
        capacity = int(spec['capacity'])
        f_min = float(spec['f_min_ghz']) * 1e9
        f_max = float(spec['f_max_ghz']) * 1e9
        n_levels = int(spec.get('freq_levels', 11))
        freq_levels = list(np.linspace(f_min, f_max, n_levels))
        total_capacity_hz = capacity * f_max

        servers.append({
            'id':                    spec['id'],
            'capacity':              capacity,
            'remaining_cores':       capacity,
            'total_capacity_hz':     total_capacity_hz,
            'remaining_capacity_hz': total_capacity_hz,
            'kappa':                 float(spec['kappa']),
            'f_min':                 f_min,
            'f_max':                 f_max,
            'b_s':                   float(spec['b_s']),
            'freq_levels':           freq_levels,
            'task_count':            0,
        })
    return servers


def _compute_nominal_deadline(
    data_size_bits: float,
    workload_cycles: float,
    deadline_model: Dict,
) -> float:
    """
    Compute the nominal end-to-end deadline from upload + execution baselines.
    """
    nominal_upload_rate_bps = float(deadline_model['nominal_upload_rate_mbps']) * 1e6
    nominal_compute_cores = float(deadline_model['nominal_compute_cores'])
    nominal_compute_freq_hz = float(deadline_model['nominal_compute_freq_ghz']) * 1e9
    nominal_compute_b_s = float(deadline_model.get('nominal_compute_b_s', 1.0))

    t_ul_nom = data_size_bits / nominal_upload_rate_bps
    t_comp_nom = workload_cycles / (nominal_compute_cores * nominal_compute_freq_hz * nominal_compute_b_s)
    return t_ul_nom + t_comp_nom


def generate_profiled_tasks(
    config: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Generate tasks from ``task_profiles`` with reverse-engineered deadlines.

    For each task:
        nominal_deadline = data_size / nominal_upload_rate
                         + workload / (nominal_cores * nominal_freq * nominal_b_s)
        deadline = nominal_deadline * skew,
        where skew ~ Uniform[deadline_skew_min, deadline_skew_max].
    """
    if config is None:
        if CONFIG is None:
            raise ValueError("config must be provided if config.yaml is not found")
        config = CONFIG

    if 'task_profiles' not in config:
        raise ValueError("config must contain 'task_profiles'")
    if 'deadline_model' not in config:
        raise ValueError("config must contain 'deadline_model'")

    profiles = config['task_profiles']
    deadline_model = config['deadline_model']
    rng = np.random.default_rng(seed)

    skew_min = float(deadline_model['deadline_skew_min'])
    skew_max = float(deadline_model['deadline_skew_max'])
    if skew_min <= 0 or skew_max <= 0 or skew_min > skew_max:
        raise ValueError("deadline skew bounds must be positive and satisfy min <= max")

    expected_k = int(config.get('network', {}).get('K', sum(int(p['count']) for p in profiles.values())))
    total_weight = sum(int(p['count']) for p in profiles.values())

    # Distribute expected_k tasks proportionally based on profile counts
    profile_counts = {}
    accumulated_k = 0
    accumulated_fraction = 0.0
    for pname, prof in profiles.items():
        fraction = int(prof['count']) / total_weight
        accumulated_fraction += fraction
        target_k = int(np.round(accumulated_fraction * expected_k))
        count = target_k - accumulated_k
        profile_counts[pname] = count
        accumulated_k = target_k

    tasks = []
    task_id = 0
    for pname, prof in profiles.items():
        n_prof = profile_counts[pname]
        if n_prof <= 0:
            continue
            
        data_sizes = rng.uniform(
            float(prof['data_size_range'][0]),
            float(prof['data_size_range'][1]),
            size=n_prof,
        )
        intensities = rng.uniform(
            float(prof['processing_intensity_range'][0]),
            float(prof['processing_intensity_range'][1]),
            size=n_prof,
        )
        skews = rng.uniform(skew_min, skew_max, size=n_prof)

        for j in range(n_prof):
            data_size = float(data_sizes[j])
            intensity = float(intensities[j])
            workload = float(data_size * intensity)
            nominal_deadline = _compute_nominal_deadline(data_size, workload, deadline_model)
            skew = float(skews[j])
            deadline = nominal_deadline * skew
            tasks.append({
                'id':                   task_id,
                'profile':              pname,
                'data_size':            data_size,
                'processing_intensity': intensity,
                'workload_cycles':      workload,
                'nominal_deadline':     nominal_deadline,
                'deadline_skew':        skew,
                'deadline':             deadline,
            })
            task_id += 1
    return tasks


# ---------------------------------------------------------------------------
# Core energy helper
# ---------------------------------------------------------------------------

def compute_task_energy(
    kappa: float,
    frequency_hz: float,
    workload_cycles: float,
    b_s: float,
) -> float:
    """
    Compute energy consumed to process a single task (independent of core count).

    E = κ · f² · w_k / b_s

    Derivation:
        P_per_core = κ · f³
        t_proc     = w_k / (n_c · f · b_s)
        E          = n_c · P_per_core · t_proc = κ · f² · w_k / b_s   (n_c cancels)

    Parameters
    ----------
    kappa : float
        Effective switched capacitance of the server.
    frequency_hz : float
        Operating frequency in Hz (must be a discrete level from freq_levels).
    workload_cycles : float
        Total CPU cycles required (w_k = s_k × δ_k).
    b_s : float
        Server acceleration coefficient.

    Returns
    -------
    float
        Energy in Joules.
    """
    return kappa * (frequency_hz ** 2) * workload_cycles / b_s


# ---------------------------------------------------------------------------
# Two-pass computation heuristic
# ---------------------------------------------------------------------------

def run_comp_heuristic(
    tasks: List[Dict],
    server_specs: List[Dict],
    t_proc_budget: np.ndarray,
    T_horizon: Optional[float] = None,
    placement: str = 'greedy',
) -> Dict:
    """
    Two-pass greedy computation heuristic.

    Pass 1 — Urgency-first (w_k / deadline, descending) assignment at the
             highest feasible frequency level: minimises core usage and
             preserves GHz capacity for subsequent tasks.
    Pass 2 — Greedy descent: for each assigned task, step down through
             frequency levels.  A level is applied when its capacity delta
             fits within the server's remaining budget; non-monotone ceiling
             effects are handled by continuing past infeasible levels.

    Parameters
    ----------
    tasks : list[dict]
        Each dict must contain ``id``, ``workload_cycles``, ``deadline``.
    server_specs : list[dict]
        Each dict must contain ``id``, ``total_capacity_hz``,
        ``remaining_capacity_hz``, ``kappa``, ``f_min``, ``f_max``,
        ``b_s``, ``freq_levels`` (list[float]).
    t_proc_budget : ndarray (K,)
        Per-task processing time budget in seconds.
    T_horizon : float, optional
        System horizon (seconds).  When provided, idle cores draw
        κ · (0.5 · f_min)³ watts for the time they are not processing.
    placement : {'greedy', 'first_fit'}, optional
        Assignment policy.  ``'greedy'`` (default): energy-minimizing server +
        DVFS frequency descent.  ``'first_fit'``: energy-unaware ablation
        baseline — place each task on the first server (config order) that fits
        at its own f_max, with no DVFS descent.  Best-effort/idle accounting is
        identical for both.

    Returns
    -------
    dict
        ``energy_comp``  — ndarray (K,) per-task energy (J)
        ``assignments``  — list[dict | None]
                           ``{task_id, server_id, frequency_hz, num_cores}``
                           or None if infeasible
        ``feasible``     — ndarray (K,) bool
        ``server_power`` — ndarray (S,) instantaneous power per server (W)
        ``total_energy`` — float, sum of all task energies + idle energy
    """
    K = len(tasks)
    S = len(server_specs)

    # -- Deep-copy server state so caller's list is not mutated --
    servers = []
    for spec in server_specs:
        capacity = int(spec.get('capacity', round(float(spec['total_capacity_hz']) / float(spec['f_max']))))
        servers.append({
            'id':                    spec['id'],
            'capacity':              capacity,
            'remaining_cores':       int(spec.get('remaining_cores', capacity - int(spec.get('task_count', 0)))),
            'total_capacity_hz':     float(spec['total_capacity_hz']),
            'remaining_capacity_hz': float(spec['remaining_capacity_hz']),
            'kappa':                 float(spec['kappa']),
            'f_min':                 float(spec['f_min']),
            'f_max':                 float(spec['f_max']),
            'b_s':                   float(spec['b_s']),
            'freq_levels':           list(spec['freq_levels']),
            'task_count':            int(spec.get('task_count', 0)),
        })

    energy_comp = np.zeros(K)
    feasible = np.zeros(K, dtype=bool)
    assigned = np.zeros(K, dtype=bool)
    best_effort = np.zeros(K, dtype=bool)
    assignments = [None] * K

    # Urgency = w_k / deadline (descending) — most cycles-per-second-of-deadline first
    urgency = [tasks[i]['workload_cycles'] / tasks[i]['deadline'] for i in range(K)]
    sorted_indices = sorted(range(K), key=lambda i: -urgency[i])

    # ----------------------------------------------------------------
    # Pass 1: assign at highest feasible frequency (min cores used)
    # ----------------------------------------------------------------
    for idx in sorted_indices:
        task = tasks[idx]
        w_k = task['workload_cycles']
        t_budget = float(t_proc_budget[idx])

        if t_budget <= 0:
            continue

        if placement == 'first_fit':
            # First-fit (energy-unaware): scan servers in fixed config order and
            # place the task on the first one that fits at its own f_max.  No
            # DVFS, no heterogeneity-aware server choice.  f_max minimises the
            # core count, so this rarely fails feasibility spuriously.
            for s_idx, srv in enumerate(servers):
                lvl_idx = len(srv['freq_levels']) - 1
                f = srv['freq_levels'][lvl_idx]
                F_need = w_k / (t_budget * srv['b_s'])
                n = _ceil_to_batch(math.ceil(F_need / f))
                if n <= srv['remaining_cores'] and n * f <= srv['remaining_capacity_hz']:
                    srv['remaining_capacity_hz'] -= n * f
                    srv['remaining_cores'] -= n
                    srv['task_count'] += 1
                    energy_comp[idx] = compute_task_energy(srv['kappa'], f, w_k, srv['b_s'])
                    feasible[idx] = True
                    assigned[idx] = True
                    assignments[idx] = {
                        'task_id':      task['id'],
                        'server_id':    srv['id'],
                        'frequency_hz': f,
                        'num_cores':    n,
                        'best_effort':  False,
                        '_s_idx':       s_idx,
                        '_lvl_idx':     lvl_idx,
                    }
                    break
            continue

        best_energy = np.inf
        best_s_idx = -1
        best_freq = 0.0
        best_n_cores = 0
        best_lvl_idx = -1

        for s_idx, srv in enumerate(servers):
            b_s = srv['b_s']
            kappa_s = srv['kappa']
            levels = srv['freq_levels']
            remaining = srv['remaining_capacity_hz']
            remaining_cores = srv['remaining_cores']

            # Total frequency throughput needed on this server
            F_need = w_k / (t_budget * b_s)

            # Scan from highest level downward; take first that satisfies capacity
            for lvl_idx in range(len(levels) - 1, -1, -1):
                f = levels[lvl_idx]
                n = _ceil_to_batch(math.ceil(F_need / f))
                if n <= remaining_cores and n * f <= remaining:
                    energy = compute_task_energy(kappa_s, f, w_k, b_s)
                    if energy < best_energy:
                        best_energy = energy
                        best_s_idx = s_idx
                        best_freq = f
                        best_n_cores = n
                        best_lvl_idx = lvl_idx
                    break  # highest feasible level found for this server

        if best_s_idx >= 0:
            servers[best_s_idx]['remaining_capacity_hz'] -= best_n_cores * best_freq
            servers[best_s_idx]['remaining_cores'] -= best_n_cores
            servers[best_s_idx]['task_count'] += 1
            energy_comp[idx] = best_energy
            feasible[idx] = True
            assigned[idx] = True
            assignments[idx] = {
                'task_id':      task['id'],
                'server_id':    servers[best_s_idx]['id'],
                'frequency_hz': best_freq,
                'num_cores':    best_n_cores,
                'best_effort':  False,
                '_s_idx':       best_s_idx,   # internal; stripped after pass 2
                '_lvl_idx':     best_lvl_idx,  # internal; stripped after pass 2
            }

    # ----------------------------------------------------------------
    # Best-effort pass: tasks that cannot meet their deadline still get
    # the fastest remaining single-server allocation, so callers can see
    # how far the achieved execution time is from the requested budget.
    # ----------------------------------------------------------------
    for idx in sorted_indices:
        if assigned[idx]:
            continue

        task = tasks[idx]
        w_k = task['workload_cycles']

        best_s_idx = -1
        best_freq = 0.0
        best_n_cores = 0
        best_throughput = 0.0
        best_lvl_idx = -1

        for s_idx, srv in enumerate(servers):
            remaining_cores = srv['remaining_cores']
            remaining = srv['remaining_capacity_hz']
            if remaining_cores <= 0 or remaining <= 0:
                continue

            for lvl_idx in range(len(srv['freq_levels']) - 1, -1, -1):
                f = srv['freq_levels'][lvl_idx]
                n_raw = min(remaining_cores, int(math.floor(remaining / f)))
                n = (n_raw // CORE_BATCH) * CORE_BATCH
                if n <= 0:
                    continue
                throughput = n * f * srv['b_s']
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_s_idx = s_idx
                    best_freq = f
                    best_n_cores = n
                    best_lvl_idx = lvl_idx
                break

        if best_s_idx >= 0:
            srv = servers[best_s_idx]
            servers[best_s_idx]['remaining_capacity_hz'] -= best_n_cores * best_freq
            servers[best_s_idx]['remaining_cores'] -= best_n_cores
            servers[best_s_idx]['task_count'] += 1
            energy_comp[idx] = compute_task_energy(srv['kappa'], best_freq, w_k, srv['b_s'])
            assigned[idx] = True
            best_effort[idx] = True
            assignments[idx] = {
                'task_id':      task['id'],
                'server_id':    srv['id'],
                'frequency_hz': best_freq,
                'num_cores':    best_n_cores,
                'best_effort':  True,
                '_s_idx':       best_s_idx,
                '_lvl_idx':     best_lvl_idx,
            }

    # ----------------------------------------------------------------
    # Remaining unassigned tasks: try to allocate CORE_BATCH cores at
    # the highest frequency.  If literally 0 cores remain anywhere,
    # re-schedule on resources freed by the shortest-running tasks.
    # ----------------------------------------------------------------
    no_capacity = np.zeros(K, dtype=bool)
    wait_times = np.zeros(K, dtype=float)   # extra delay waiting for freed cores

    for idx in sorted_indices:
        if assigned[idx]:
            continue

        task = tasks[idx]
        w_k = task['workload_cycles']

        # --- Try direct allocation of CORE_BATCH at highest freq ----------
        best_s_idx = -1
        best_freq = 0.0
        best_lvl_idx = -1

        for s_idx, srv in enumerate(servers):
            if srv['remaining_cores'] < CORE_BATCH:
                continue
            lvl_idx = len(srv['freq_levels']) - 1
            f = srv['freq_levels'][lvl_idx]
            if CORE_BATCH * f > srv['remaining_capacity_hz']:
                continue
            if best_s_idx == -1 or f > best_freq:
                best_s_idx = s_idx
                best_freq = f
                best_lvl_idx = lvl_idx

        if best_s_idx >= 0:
            srv = servers[best_s_idx]
            servers[best_s_idx]['remaining_capacity_hz'] -= CORE_BATCH * best_freq
            servers[best_s_idx]['remaining_cores'] -= CORE_BATCH
            servers[best_s_idx]['task_count'] += 1
            energy_comp[idx] = compute_task_energy(srv['kappa'], best_freq, w_k, srv['b_s'])
            assigned[idx] = True
            best_effort[idx] = True
            assignments[idx] = {
                'task_id':      task['id'],
                'server_id':    srv['id'],
                'frequency_hz': best_freq,
                'num_cores':    CORE_BATCH,
                'best_effort':  True,
                '_s_idx':       best_s_idx,
                '_lvl_idx':     best_lvl_idx,
            }
        else:
            # --- No cores anywhere: re-schedule on freed resources ---------
            # Collect exec times of all currently-assigned tasks
            donor_candidates = []
            for j in range(K):
                if assignments[j] is not None and '_s_idx' in assignments[j]:
                    asgn_j = assignments[j]
                    sj = servers[asgn_j['_s_idx']]
                    t_exec_j = tasks[j]['workload_cycles'] / (
                        asgn_j['num_cores'] * asgn_j['frequency_hz'] * sj['b_s']
                    )
                    donor_candidates.append((t_exec_j, j, asgn_j))

            if not donor_candidates:
                # No assigned tasks to borrow capacity from: leave unscheduled.
                assigned[idx] = True
                best_effort[idx] = True
                no_capacity[idx] = True
                continue

            # Sort by exec time ascending (shortest-running first)
            donor_candidates.sort(key=lambda x: x[0])

            # Iteratively free donors until CORE_BATCH cores are available
            # on at least one server; pick the best server from freed resources.
            freed_donors = []       # (t_exec, task_idx, assignment)
            found_allocation = False
            alloc_s_idx = -1
            alloc_freq = 0.0
            alloc_lvl_idx = -1
            alloc_t_wait = 0.0

            for t_exec_d, d_idx, d_asgn in donor_candidates:
                freed_donors.append((t_exec_d, d_idx, d_asgn))
                # The wait time is the latest finish among freed donors
                alloc_t_wait = max(fd[0] for fd in freed_donors)

                # Temporarily restore freed resources
                for _, _, fd_asgn in freed_donors:
                    fd_s = servers[fd_asgn['_s_idx']]
                    fd_s['remaining_cores'] += fd_asgn['num_cores']
                    fd_s['remaining_capacity_hz'] += fd_asgn['num_cores'] * fd_asgn['frequency_hz']

                # Try to find CORE_BATCH at the best frequency
                cand_s_idx = -1
                cand_freq = 0.0
                cand_lvl_idx = -1
                for s_idx, srv in enumerate(servers):
                    if srv['remaining_cores'] < CORE_BATCH:
                        continue
                    lvl_idx = len(srv['freq_levels']) - 1
                    f = srv['freq_levels'][lvl_idx]
                    if CORE_BATCH * f > srv['remaining_capacity_hz']:
                        continue
                    if cand_s_idx == -1 or f > cand_freq:
                        cand_s_idx = s_idx
                        cand_freq = f
                        cand_lvl_idx = lvl_idx

                # Undo the temporary restoration
                for _, _, fd_asgn in freed_donors:
                    fd_s = servers[fd_asgn['_s_idx']]
                    fd_s['remaining_cores'] -= fd_asgn['num_cores']
                    fd_s['remaining_capacity_hz'] -= fd_asgn['num_cores'] * fd_asgn['frequency_hz']

                if cand_s_idx >= 0:
                    alloc_s_idx = cand_s_idx
                    alloc_freq = cand_freq
                    alloc_lvl_idx = cand_lvl_idx
                    found_allocation = True
                    break

            if found_allocation:
                srv = servers[alloc_s_idx]
                # Note: we do NOT actually consume capacity for the waiting
                # task because the freed donors' resources are restored only
                # after t_wait.  The capacity ledger stays consistent with
                # the concurrent snapshot (donors are still running).
                energy_comp[idx] = compute_task_energy(
                    srv['kappa'], alloc_freq, w_k, srv['b_s']
                )
                assigned[idx] = True
                best_effort[idx] = True
                wait_times[idx] = alloc_t_wait
                freed_task_ids = [fd[1] for fd in freed_donors]
                assignments[idx] = {
                    'task_id':       task['id'],
                    'server_id':     srv['id'],
                    'frequency_hz':  alloc_freq,
                    'num_cores':     CORE_BATCH,
                    'best_effort':   True,
                    'wait_time':     alloc_t_wait,
                    'freed_task_ids': freed_task_ids,
                    '_s_idx':        alloc_s_idx,
                    '_lvl_idx':      alloc_lvl_idx,
                }
            else:
                # Exhausted all donors and still cannot fit anywhere.
                assigned[idx] = True
                best_effort[idx] = True
                no_capacity[idx] = True

    # ----------------------------------------------------------------
    # Pass 2: greedy descent — lower frequency one bin at a time.
    # Skipped under first-fit, which runs flat-out at f_max (no DVFS).
    # ----------------------------------------------------------------
    descent_indices = sorted_indices if placement != 'first_fit' else []
    for idx in descent_indices:
        if not feasible[idx]:
            continue

        asgn = assignments[idx]
        s_idx = asgn['_s_idx']
        srv = servers[s_idx]
        levels = srv['freq_levels']

        w_k = tasks[idx]['workload_cycles']
        t_budget = float(t_proc_budget[idx])
        b_s = srv['b_s']
        kappa_s = srv['kappa']
        F_need = w_k / (t_budget * b_s)

        cur_cap = asgn['num_cores'] * asgn['frequency_hz']
        cur_cores = asgn['num_cores']

        for lvl_idx in range(asgn['_lvl_idx'] - 1, -1, -1):
            f_new = levels[lvl_idx]
            n_new = _ceil_to_batch(math.ceil(F_need / f_new))
            cap_new = n_new * f_new
            delta = cap_new - cur_cap
            delta_cores = n_new - cur_cores

            if delta <= srv['remaining_capacity_hz'] and delta_cores <= srv['remaining_cores']:
                # Apply downgrade
                srv['remaining_capacity_hz'] -= delta
                srv['remaining_cores'] -= delta_cores
                asgn['frequency_hz'] = f_new
                asgn['num_cores'] = n_new
                asgn['_lvl_idx'] = lvl_idx
                energy_comp[idx] = compute_task_energy(kappa_s, f_new, w_k, b_s)
                cur_cap = cap_new
                cur_cores = n_new
            # Continue regardless — ceiling non-monotonicity means a lower
            # level may still fit even if this one did not.

    # ----------------------------------------------------------------
    # Compute per-server power, actual execution times, then strip
    # internal fields
    # ----------------------------------------------------------------
    server_power = np.zeros(S)
    exec_times = np.full(K, np.inf)
    for idx in range(K):
        if assigned[idx] and assignments[idx] is not None:
            asgn = assignments[idx]
            s_idx = asgn['_s_idx']
            srv = servers[s_idx]
            kappa_s = srv['kappa']
            server_power[s_idx] += asgn['num_cores'] * kappa_s * (asgn['frequency_hz'] ** 3)
            raw_exec = tasks[idx]['workload_cycles'] / (
                asgn['num_cores'] * asgn['frequency_hz'] * srv['b_s']
            )
            exec_times[idx] = wait_times[idx] + raw_exec

    # ----------------------------------------------------------------
    # Idle core energy: each unoccupied core draws κ·(0.5·f_min)³ W
    # ----------------------------------------------------------------
    per_server_idle_energy = np.zeros(S)
    edge_idle_energy = 0.0
    if T_horizon is not None and T_horizon > 0:
        # Per-server active core-time (cores × raw execution seconds)
        per_server_active_core_time = np.zeros(S)
        for idx in range(K):
            if assigned[idx] and assignments[idx] is not None:
                asgn = assignments[idx]
                s_idx = asgn['_s_idx']
                srv = servers[s_idx]
                raw_exec = tasks[idx]['workload_cycles'] / (
                    asgn['num_cores'] * asgn['frequency_hz'] * srv['b_s']
                )
                per_server_active_core_time[s_idx] += asgn['num_cores'] * raw_exec

        for s_idx, srv in enumerate(servers):
            capacity = srv['capacity']
            if capacity <= 0:
                continue
            used_cores = capacity - srv['remaining_cores']
            if used_cores <= 0:
                continue
            f_idle = 0.5 * srv['f_min']
            p_idle_per_core = srv['kappa'] * (f_idle ** 3)
            idle_core_time = max(used_cores * T_horizon - per_server_active_core_time[s_idx], 0.0)
            per_server_idle_energy[s_idx] = idle_core_time * p_idle_per_core

        edge_idle_energy = float(np.sum(per_server_idle_energy))

    for idx in range(K):
        if assignments[idx] is not None:
            assignments[idx].pop('_s_idx', None)
            assignments[idx].pop('_lvl_idx', None)

    return {
        'energy_comp':  energy_comp,
        'assignments':  assignments,
        'feasible':     feasible,
        'assigned':     assigned,
        'best_effort':  best_effort,
        'no_capacity':  no_capacity,
        'wait_times':   wait_times,
        'server_power': server_power,
        'exec_times':   exec_times,
        'edge_idle_energy': edge_idle_energy,
        'per_server_idle_energy': per_server_idle_energy,
        'total_energy': float(np.sum(energy_comp)) + edge_idle_energy,
    }

