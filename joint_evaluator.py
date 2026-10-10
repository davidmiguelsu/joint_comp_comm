"""
Joint Evaluator Module.

Given a scenario (UEs, APs, edge servers) and a split vector a ∈ [0, 1]^K,
evaluates the total energy cost of the joint communication + computation system.

Split vector semantics:
    a_k  = fraction of deadline t_k allocated to communication
    1-a_k = fraction of deadline t_k allocated to computation

    t_comm_k = a_k * t_k
    t_proc_k = (1 - a_k) * t_k

Interface:
    evaluate(tasks, split_vector, snapshot, server_specs, network_params, config)
        -> EvalResult dict

EvalResult keys:
    total_energy      — float, E_comm + E_comp (Joules)
    energy_comm       — ndarray (K,), per-UE comm energy (Joules)
    energy_comp       — ndarray (K,), per-UE comp energy (Joules)
    feasible_comm     — ndarray (K,), bool
    feasible_comp     — ndarray (K,), bool
    feasible          — ndarray (K,), bool  (both comm and comp feasible)
    num_feasible      — int
    rates             — ndarray (K,), achieved data rates (bit/s)
    per_user_se       — ndarray (K,), achieved SE (bit/s/Hz)
    required_se       — ndarray (K,), split-derived required SE (bit/s/Hz)
    required_rates    — ndarray (K,), split-derived required rates (bit/s)
    total_infra_power — float, AP + CPU infrastructure power (Watts)
    comm_heuristic_time_s — float, wireless heuristic wall-clock time (seconds)
    comp_heuristic_time_s — float, computation heuristic wall-clock time (seconds)
    total_heuristic_time_s — float, comm + comp heuristic wall-clock time (seconds)
    server_power      — ndarray (S,), per-server computation power (Watts)
    assignments       — list[dict | None], per-task server assignment
    upload_times      — ndarray (K,), achieved upload times (seconds)
    exec_times        — ndarray (K,), actual computation exec times (seconds)
    achieved_comm_fraction — ndarray (K,), bits delivered during t_comm / data_size
    achieved_comp_fraction — ndarray (K,), cycles executable during t_proc / workload
    surplus_comm      — ndarray (K,), t_comm - upload_times (seconds)
    surplus_comp      — ndarray (K,), t_proc - exec_times (seconds)
    t_comm            — ndarray (K,), comm time budgets (seconds)
    t_proc            — ndarray (K,), processing time budgets (seconds)
    split_vector      — ndarray (K,), input split vector
"""

import os
import copy
import json
import time
import pickle
import hashlib
import numpy as np
import yaml
from typing import Dict, List, Optional

import cellfree_mimo
import comm_heuristic
import comm_energy_model
import edge_compute_model

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
# Snapshot cache — build the network/tasks once, reload instantly.
# Hashes the relevant config keys + seeds into a fingerprint; if
# data/snapshot_{fingerprint}.pkl exists it is loaded, otherwise the network is
# built from scratch and saved.
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_FINGERPRINT_KEYS = [
    'L', 'K', 'N', 'area', 'tau_p', 'tau_c', 'realizations',
    'realizations_final', 'carrier_frequency', 'p_k', 'bandwidth', 'delta_h',
    'max_users_per_ap', 'min_channel_gain_db',
]


def _fingerprint(cfg, net_seed, ch_seed, task_seed):
    net = cfg['network']
    d = {k: net.get(k) for k in _FINGERPRINT_KEYS}
    d['net_seed'] = net_seed
    d['ch_seed'] = ch_seed
    d['task_seed'] = task_seed
    d['edge_servers'] = cfg.get('edge_servers')
    d['task_profiles'] = cfg.get('task_profiles')
    d['deadline_model'] = cfg.get('deadline_model')
    blob = json.dumps(d, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def get_or_build_snapshot(cfg, net_seed=42, ch_seed=100, task_seed=7):
    """
    Return (snapshot, tasks, servers, params), from cache if available.

    Parameters
    ----------
    cfg : dict
        Full config (loaded from config.yaml).
    net_seed, ch_seed, task_seed : int
        Seeds for simulation_area, channel_generation, and task generation.

    Returns
    -------
    snapshot : dict   — {H, H_hat, B}
    tasks    : list[dict]
    servers  : list[dict]
    params   : dict   — network_params with realizations = realizations_final
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    fp = _fingerprint(cfg, net_seed, ch_seed, task_seed)
    pkl_path = os.path.join(_DATA_DIR, f"snapshot_{fp}.pkl")

    if os.path.exists(pkl_path):
        print(f"Loading cached snapshot from {os.path.basename(pkl_path)} ...")
        t0 = time.time()
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded in {time.time() - t0:.2f}s")
        return data['snapshot'], data['tasks'], data['servers'], data['params']

    # --- Build from scratch ---
    print("Building network (no cache found) ...")
    t0 = time.time()

    net = cfg['network']
    L, K, N = net['L'], net['K'], net['N']

    params = {
        'area': net['area'],
        'L': L, 'K': K, 'N': N,
        'tau_p': net['tau_p'],
        'tau_c': net['tau_c'],
        'realizations': net['realizations'],
        'bandwidth': net['bandwidth'],
        'carrier_frequency': net['carrier_frequency'],
        'p_k': net['p_k'],
        'delta_h': net.get('delta_h', 11),
        'max_users_per_ap': net.get('max_users_per_ap', 6),
        'min_channel_gain_db': net.get('min_channel_gain_db', 0.0),
    }

    np.random.seed(net_seed)
    R, B, H_mean, kappa = cellfree_mimo.simulation_area(params)

    # Snapshot stores R = realizations_final channel draws; the heuristic slices
    # the first R~ = realizations of them for its intermediate SE evaluations.
    params_100 = dict(params)
    params_100['realizations'] = int(net.get('realizations_final', 100))

    np.random.seed(ch_seed)
    H, Hm, R_cov = cellfree_mimo.channel_generation(
        params_100, R.copy(), H_mean.copy(), kappa.copy(),
    )
    pilot_idx = np.arange(K).reshape(-1, 1) % params['tau_p']
    H_hat = cellfree_mimo.channel_estimation(
        params_100, R_cov, pilot_idx, H, Hm, params['p_k'],
    )

    H = H.transpose(3, 0, 2, 1)
    H_hat = H_hat.transpose(3, 0, 2, 1)

    snapshot = {'H': H, 'H_hat': H_hat, 'B': B}
    tasks = edge_compute_model.generate_profiled_tasks(cfg, seed=task_seed)
    servers = edge_compute_model.load_servers_from_config(cfg)

    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'snapshot': snapshot,
            'tasks': tasks,
            'servers': servers,
            'params': params_100,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Built + saved in {time.time() - t0:.1f}s  ({os.path.basename(pkl_path)})")
    return snapshot, tasks, servers, params_100


# ---------------------------------------------------------------------------
# Communication adapter
# Wraps the CF-mMIMO communication heuristic (comm_heuristic) to expose a
# task-level interface and computes AP+CPU infrastructure energy.
#
# Key mapping:  data_size[k] / (BW * t_comm_budget[k])  ->  se_target[k]  (bit/s/Hz)
# ---------------------------------------------------------------------------

def data_size_to_se_target(
    data_size: np.ndarray,
    bandwidth: float,
    t_comm_budget: np.ndarray,
) -> np.ndarray:
    """
    Convert per-UE data sizes and comm time budgets into SE targets.

        SE_target[k] = data_size[k] / (bandwidth * t_comm_budget[k])

    Returns np.inf for zero-budget entries (inherently infeasible).
    """
    se_targets = np.full_like(data_size, np.inf, dtype=float)
    valid = t_comm_budget > 0
    se_targets[valid] = data_size[valid] / (bandwidth * t_comm_budget[valid])
    return se_targets


def compute_comm_energy(
    p_k_watts,
    t_comm_budget: np.ndarray,
) -> np.ndarray:
    """Per-UE uplink transmit energy:  E_comm[k] = p_k[k] * t_comm_budget[k]."""
    return np.asarray(p_k_watts) * np.maximum(t_comm_budget, 0.0)


def run_comm_heuristic(
    tasks: List[Dict],
    t_comm_budget: np.ndarray,
    snapshot: Dict,
    network_params: Dict,
    config: Optional[Dict] = None,
    energy_model: str = 'upload',
) -> Dict:
    """
    Communication module adapter — standard task-level interface.

    Converts task-level inputs to SE targets, runs the communication
    heuristic, and maps the results back to per-UE infrastructure energy.

    Parameters
    ----------
    tasks : list[dict]
        Each dict must contain ``id`` and ``data_size`` (bits).
    t_comm_budget : ndarray (K,)
        Per-UE communication time budget in seconds.
    snapshot : dict
        CF-mMIMO channel snapshot containing:
            ``H``       — (R, L, N, K) true channel matrix
            ``H_hat``   — (R, L, N, K) estimated channel matrix
            ``B``       — (L, K) large-scale fading coefficients
    network_params : dict
        Network configuration (L, K, N, tau_p, tau_c, bandwidth, p_k, ...).
    config : dict, optional
        Full configuration dict.  Falls back to module-level CONFIG.
    energy_model : {'upload', 'deadline'}, optional
        AP active-time model.  ``'upload'`` (default): AP on until its slowest
        UE finishes uploading.  ``'deadline'``: AP on until max(upload_time,
        deadline) for the slowest UE it serves.

    Returns
    -------
    dict
        ``energy_comm``       — ndarray (K,) per-UE comm energy in Joules.
        ``rates``             — ndarray (K,) achieved data rate per UE (bit/s).
        ``feasible``          — ndarray (K,) boolean feasibility flag.
        ``total_infra_power`` — float, total AP + CPU power (Watts).
        ``per_user_se``       — ndarray (K,) achieved SE (bit/s/Hz).
        ``required_se``       — ndarray (K,) required SE (bit/s/Hz).
        ``required_rates``    — ndarray (K,) required rate per UE (bit/s).
        ``upload_times``      — ndarray (K,) achieved upload time (s).
        ``total_energy``      — float, sum of per-UE comm energy.
    """
    if config is None:
        if CONFIG is None:
            raise ValueError("config must be provided if config.yaml is not found")
        config = CONFIG

    K = len(tasks)
    bandwidth = float(network_params.get('bandwidth', config['network']['bandwidth']))

    # --- Step 1: Convert data sizes + time budgets -> SE targets ---
    data_sizes = np.array([t['data_size'] for t in tasks], dtype=float)
    se_targets = data_size_to_se_target(data_sizes, bandwidth, t_comm_budget)

    # Clamp infinite SE targets to a very high value so the heuristic
    # attempts to serve them but will naturally fail (infeasible).
    se_targets_clamped = np.where(np.isinf(se_targets), 1e6, se_targets)

    # --- Step 2: Prepare inputs ---
    H = snapshot['H']
    H_hat = snapshot['H_hat']
    B = snapshot['B']

    p_max_mw = float(network_params.get('p_k', config['network']['p_k']))
    p_k_watts = p_max_mw / 1000.0  # uniform max power in Watts

    L = network_params['L']
    net_params = copy.deepcopy(network_params)
    net_params['L'] = L
    net_params['K'] = K
    net_params['comm_heuristic'] = (config.get('ablation') or {}).get('comm_heuristic', 'v6')

    # --- Step 3: Run the comm heuristic ---
    # R~ (reduced) realizations for the intermediate SE evaluations inside H_c;
    # R realizations for the final SE evaluation.
    net_cfg = config.get('network', {})
    se_realizations = int(net_cfg.get('realizations', net_params.get('realizations', 20)))
    se_realizations_final = int(net_cfg.get('realizations_final', 100))
    D_DCC, results, p_k_vec = comm_heuristic.run_new_heuristic_v6_se_fpc(
        H_input=H,
        H_hat_input=H_hat,
        B_channel_gains=B,
        network_params=net_params,
        se_targets=se_targets_clamped,
        se_realizations=se_realizations,
        se_realizations_final=se_realizations_final,
    )

    # --- Step 4: Extract results and compute task-level outputs ---
    per_user_se = results['per_user_se']                         # (K,) bit/s/Hz
    rates = per_user_se * bandwidth                             # (K,) bit/s
    upload_times = np.divide(data_sizes, rates, out=np.full_like(data_sizes, np.inf), where=rates > 0)
    required_rates = np.divide(data_sizes, t_comm_budget, out=np.full_like(data_sizes, np.inf), where=t_comm_budget > 0)

    # Feasibility: UE k is feasible if it can transmit its data within budget
    transmitted_bits = np.where(
        t_comm_budget > 0,
        rates * t_comm_budget,
        0.0,
    )
    feasible = transmitted_bits >= data_sizes * (1.0 - 1e-6)     # small tolerance

    # --- Step 5: Compute infrastructure energy (AP + CPU) via Split 7.2 ---
    gops_params = config.get('gops_params', {})
    comm_energy_params = config.get('comm_energy_params', {})

    # Use actual upload time for all UEs — infeasible UEs (upload_time > budget) are charged
    # for their true transmission duration. Zero-rate UEs (rate=0 → inf upload_time) fall
    # back to the budget since they never transmit anything meaningful.
    t_ul_for_energy = np.where(np.isinf(upload_times), t_comm_budget, upload_times)

    # Deadline model: AP stays on until max(upload_time, deadline) per UE.
    # Feasible UEs (upload_time <= deadline) → deadline; infeasible → upload_time.
    # Zero-rate UEs (inf upload_time) fall back to the budget to avoid inf T_frame.
    t_horizon_for_energy = None
    if energy_model == 'deadline':
        t_horizon_for_energy = np.where(
            np.isinf(upload_times),
            t_comm_budget,
            np.maximum(upload_times, t_comm_budget),
        )

    # Fixed frame duration from config (common timeframe for fair comparison)
    T_frame_fixed = float(config.get('network', {}).get('experiment_duration_s', 0)) or None

    energy_infra = comm_energy_model.compute_energy_split72(
        D_DCC=D_DCC,
        network_params=net_params,
        gops_params=gops_params,
        comm_energy_params=comm_energy_params,
        t_ul_k=t_ul_for_energy,
        t_horizon_k=t_horizon_for_energy,
        T_frame_fixed=T_frame_fixed,
    )

    # AP + CPU infrastructure power from split-7.2 model (varies with active APs / D_DCC)
    total_infra_power = energy_infra.get('total_power_W', 0.0)
    total_infra_energy = energy_infra['total_comm_energy_J']

    # UE uplink TX energy is not counted — only AP+CPU infrastructure energy matters.
    energy_comm = np.zeros(K, dtype=float)
    energy_comm_total = total_infra_energy

    return {
        'energy_comm':       energy_comm,
        'energy_infra':      energy_infra,
        'rates':             rates,
        'feasible':          feasible,
        'total_infra_power': total_infra_power,
        'total_infra_energy': total_infra_energy,
        'per_user_se':       per_user_se,
        'required_se':       se_targets,
        'required_rates':    required_rates,
        'upload_times':      upload_times,
        'total_energy':      energy_comm_total,
        'D_DCC':             D_DCC,
        'energy_model':      energy_model,
    }


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate(
    tasks: List[Dict],
    split_vector: np.ndarray,
    snapshot: Dict,
    server_specs: List[Dict],
    network_params: Dict,
    config: Optional[Dict] = None,
    energy_model: str = 'upload',
) -> Dict:
    """
    Evaluate total energy for a given split vector a.

    For each UE k:
        t_comm_k = a_k * t_k        → passed to run_comm_heuristic
        t_proc_k = (1 - a_k) * t_k → passed to run_comp_heuristic

    Parameters
    ----------
    tasks : list[dict]
        One dict per UE/task.  Must contain:
            ``id``         — task identifier
            ``data_size``  — bits to transmit (for comm)
            ``workload_cycles`` — total CPU cycles  (for comp)
            ``deadline``   — end-to-end deadline t_k (seconds)
    split_vector : ndarray (K,)
        Per-UE split fractions a_k ∈ [0, 1].
    snapshot : dict
        CF-mMIMO channel snapshot (H, H_hat, B).
    server_specs : list[dict]
        Edge server specifications (from load_servers_from_config).
    network_params : dict
        Network configuration (L, K, N, bandwidth, p_k, …).
    config : dict, optional
        Full configuration dict.  Falls back to module-level CONFIG.
    energy_model : {'upload', 'deadline'}, optional
        AP active-time model passed to run_comm_heuristic.  Default ``'upload'``.

    Returns
    -------
    dict
        See module docstring for full key list.
    """
    if config is None:
        if CONFIG is None:
            raise ValueError("config must be provided if config.yaml is not found")
        config = CONFIG

    K = len(tasks)
    a = np.asarray(split_vector, dtype=float)
    if a.shape != (K,):
        raise ValueError(f"split_vector must have shape ({K},), got {a.shape}")
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError("All split values must be in [0, 1]")

    # --- Time budget decomposition ---
    deadlines = np.array([t['deadline'] for t in tasks], dtype=float)
    eps = 1e-12
    t_comm = np.where(a > 0, a * deadlines, 0.0)                 # communication budgets
    t_proc = np.where((1.0 - a) > 0, (1.0 - a) * deadlines, 0.0) # processing budgets
    required_rates = np.full(K, np.inf, dtype=float)
    data_sizes = np.array([t['data_size'] for t in tasks], dtype=float)
    positive_comm = t_comm > eps
    required_rates[positive_comm] = data_sizes[positive_comm] / t_comm[positive_comm]
    bandwidth = float(network_params.get('bandwidth', config['network']['bandwidth']))
    required_se = required_rates / bandwidth

    # --- Run communication heuristic ---
    comm_t0 = time.perf_counter()
    comm_result = run_comm_heuristic(
        tasks=tasks,
        t_comm_budget=t_comm,
        snapshot=snapshot,
        network_params=network_params,
        config=config,
        energy_model=energy_model,
    )
    comm_heuristic_time_s = time.perf_counter() - comm_t0

    # --- Run computation heuristic ---
    T_exp = float(config.get('network', {}).get('experiment_duration_s', 0))
    T_horizon = T_exp if T_exp > 0 else float(np.max(deadlines))
    comp_t0 = time.perf_counter()
    comp_placement = (config.get('ablation') or {}).get('comp_heuristic', 'greedy')
    comp_result = edge_compute_model.run_comp_heuristic(
        tasks=tasks,
        server_specs=server_specs,
        t_proc_budget=t_proc,
        T_horizon=T_horizon,
        placement=comp_placement,
    )
    comp_heuristic_time_s = time.perf_counter() - comp_t0

    # --- Aggregate results ---
    energy_comm   = comm_result['energy_comm']        # (K,) zeros — UE TX not counted
    energy_comp   = comp_result['energy_comp']        # (K,) per-task computation energy
    energy_infra  = comm_result['total_infra_energy'] # float, AP+CPU infra energy (J)
    infra_detail  = comm_result['energy_infra']       # dict from compute_energy_split72
    feasible_comm = comm_result['feasible']           # (K,) bool
    feasible_comp = comp_result['feasible']           # (K,) bool
    feasible      = feasible_comm & feasible_comp     # (K,) bool

    # Total network energy: infra (AP+CPU) + edge computation + edge idle.
    # Include best-effort computation assignments even when deadlines are missed.
    edge_idle_energy = comp_result.get('edge_idle_energy', 0.0)
    total_energy = energy_infra + float(np.sum(energy_comp)) + edge_idle_energy

    # --- Time surplus per domain ---
    upload_times = comm_result['upload_times']        # (K,)
    exec_times   = comp_result['exec_times']          # (K,)
    surplus_comm = t_comm - upload_times              # positive = slack in CF domain
    surplus_comp = t_proc - exec_times                # positive = slack in edge domain
    achieved_comm_fraction = np.divide(
        comm_result['rates'] * t_comm,
        data_sizes,
        out=np.zeros(K, dtype=float),
        where=data_sizes > 0,
    )
    workloads = np.array([t['workload_cycles'] for t in tasks], dtype=float)
    achieved_comp_fraction = np.divide(
        workloads * np.divide(
            t_proc,
            exec_times,
            out=np.zeros(K, dtype=float),
            where=np.isfinite(exec_times) & (exec_times > 0),
        ),
        workloads,
        out=np.zeros(K, dtype=float),
        where=workloads > 0,
    )

    return {
        'total_energy':      total_energy,
        'energy_comm':       energy_comm,
        'energy_infra':      energy_infra,
        'infra_detail':      infra_detail,
        'energy_comp':       energy_comp,
        'feasible_comm':     feasible_comm,
        'feasible_comp':     feasible_comp,
        'feasible':          feasible,
        'num_feasible':      int(np.sum(feasible)),
        'rates':             comm_result['rates'],
        'per_user_se':       comm_result['per_user_se'],
        'required_se':       comm_result['required_se'],
        'required_rates':    comm_result['required_rates'],
        'total_infra_power': comm_result['total_infra_power'],
        'comm_heuristic_time_s': comm_heuristic_time_s,
        'comp_heuristic_time_s': comp_heuristic_time_s,
        'total_heuristic_time_s': comm_heuristic_time_s + comp_heuristic_time_s,
        'server_power':      comp_result['server_power'],
        'assignments':       comp_result['assignments'],
        'assigned_comp':     comp_result.get('assigned', feasible_comp),
        'best_effort_comp':  comp_result.get('best_effort', np.zeros(K, dtype=bool)),
        'no_capacity_comp':  comp_result.get('no_capacity', np.zeros(K, dtype=bool)),
        'upload_times':      upload_times,
        'exec_times':        exec_times,
        'achieved_comm_fraction': achieved_comm_fraction,
        'achieved_comp_fraction': achieved_comp_fraction,
        'surplus_comm':      surplus_comm,
        'surplus_comp':      surplus_comp,
        't_comm':            t_comm,
        't_proc':            t_proc,
        'split_vector':      a,
        'D_DCC':             comm_result.get('D_DCC'),
        'edge_idle_energy':  edge_idle_energy,
        'per_server_idle_energy': comp_result.get('per_server_idle_energy', np.zeros(len(server_specs))),
    }


def plane_violation(r):
    """Per-UE deadline violation (s), summed over planes: upload past a_k*D_k plus exec past (1-a_k)*D_k."""
    return np.maximum(0.0, -r['surplus_comm']) + np.maximum(0.0, -r['surplus_comp'])
