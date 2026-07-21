import numpy as np
import yaml
import os
from typing import Dict, Optional

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Warning: config.yaml not found at {CONFIG_PATH}. Functions will require explicit arguments.")
    CONFIG = None


# ============================================================================
# GOPS COMPUTATION FUNCTIONS
# ============================================================================

def compute_gops_filter(N: int, gops_params: Optional[Dict] = None) -> float:
    """
    Waveform filtering GOPS per AP.

    Formula: C_filter = (40 * N * f_s) / 10^9  GOPS

    Parameters
    ----------
    N : int
        Number of antennas at AP.
    gops_params : dict, optional
        Must contain 'f_s' (sampling rate in Hz).
    """
    if gops_params is None:
        if CONFIG is None:
            raise ValueError("gops_params must be provided if config.yaml is not found")
        gops_params = CONFIG['gops_params']

    f_s = gops_params['f_s']
    return (40 * N * f_s) / 1e9


def compute_gops_dft(N: int, gops_params: Optional[Dict] = None) -> float:
    """
    DFT GOPS per AP.

    Formula: C_DFT = (8 * N * N_DFT * log2(N_DFT)) / (T_s * 10^9)  GOPS

    Parameters
    ----------
    N : int
        Number of antennas at AP.
    gops_params : dict, optional
        Must contain 'N_DFT' (DFT size) and 'T_s' (symbol period in seconds).
    """
    if gops_params is None:
        if CONFIG is None:
            raise ValueError("gops_params must be provided if config.yaml is not found")
        gops_params = CONFIG['gops_params']

    N_DFT = gops_params['N_DFT']
    T_s = gops_params['T_s']
    return (8 * N * N_DFT * np.log2(N_DFT)) / (T_s * 1e9)


def compute_gops_ce(N: int, tau_p: int, K_served: int, gops_params: Optional[Dict] = None, tau_c: int = None) -> float:
    """
    Channel estimation GOPS per AP.

    Formula: C_CE = (N_used / (T_s * tau_c * 10^9)) *
                    (8*N*tau_p^2 + 8*N^2*(tau_p + K_served))  GOPS

    Parameters
    ----------
    N : int
        Number of antennas at AP.
    tau_p : int
        Pilot length (symbols).
    K_served : int
        Number of UEs served by this AP.
    gops_params : dict, optional
        Must contain 'N_used', 'T_s'.
    tau_c : int, optional
        Coherence block length override.
    """
    if gops_params is None:
        if CONFIG is None:
            raise ValueError("gops_params must be provided if config.yaml is not found")
        gops_params = CONFIG['gops_params']

    N_used = gops_params['N_used']
    T_s = gops_params['T_s']

    if tau_c is None:
        tau_c = gops_params.get('tau_c', 200)

    term1 = 8 * N * tau_p ** 2
    term2 = 8 * (N ** 2) * (tau_p + K_served)

    return (N_used / (T_s * tau_c * 1e9)) * (term1 + term2)


def compute_gops_comb(M_k_list: np.ndarray, gops_params: Optional[Dict] = None, tau_c: int = None) -> float:
    """
    Centralized UE-centric combining GOPS at CPU.

    Formula: C_comb = (N_used / (T_s * tau_c * 10^9)) *
                      sum_k (8*M_k^2 + 8*(M_k^3 - M_k)/3)  GOPS

    where M_k = total antennas serving UE k = cluster_size * N.

    Parameters
    ----------
    M_k_list : ndarray (K,)
        Total number of antennas serving each UE (cluster_size * N per UE).
    gops_params : dict, optional
        Must contain 'N_used', 'T_s'.
    tau_c : int, optional
        Coherence block length override.
    """
    if gops_params is None:
        if CONFIG is None:
            raise ValueError("gops_params must be provided if config.yaml is not found")
        gops_params = CONFIG['gops_params']

    N_used = gops_params['N_used']
    T_s = gops_params['T_s']

    if tau_c is None:
        tau_c = gops_params.get('tau_c', 200)

    total = 0.0
    for M_k in M_k_list:
        total += 8 * M_k ** 2 + 8 * (M_k ** 3 - M_k) / 3

    return (N_used / (T_s * tau_c * 1e9)) * total


def compute_total_gops_per_ap(
    N: int,
    K_served: int,
    tau_p: int,
    gops_params: Optional[Dict] = None,
    tau_c: int = None
) -> float:
    """
    Compute total GOPS for a single AP given its load.

    Parameters
    ----------
    N : int
        Number of antennas at AP.
    K_served : int
        Number of UEs served by this AP.
    tau_p : int
        Pilot length (symbols).
    gops_params : dict, optional
        GOPS computation parameters.
    tau_c : int, optional
        Coherence block length override.
    """
    if gops_params is None:
        if CONFIG is None:
            raise ValueError("gops_params must be provided if config.yaml is not found")
        gops_params = CONFIG['gops_params']

    if K_served == 0:
        return 0.0

    C_filter = compute_gops_filter(N, gops_params)
    C_DFT = compute_gops_dft(N, gops_params)
    C_CE = compute_gops_ce(N, tau_p, K_served, gops_params, tau_c)

    return C_filter + C_DFT + C_CE


# ============================================================================
# POWER MODEL FUNCTIONS
# ============================================================================

def compute_processor_power(C_load_GOPS: float, spec: Dict) -> float:
    """
    Compute processor power consumption.

    Formula: P_proc = P_proc_0 + Delta_proc * (C_load / C_max)

    Parameters
    ----------
    C_load_GOPS : float
        Current computational load in GOPS.
    spec : dict
        Processor specification with keys 'P_proc_0_W', 'Delta_proc_W', 'C_max_GOPS'.
    """
    P_0 = spec['P_proc_0_W']
    Delta = spec['Delta_proc_W']
    C_max = spec['C_max_GOPS']

    load_normalized = C_load_GOPS / max(C_max, 1e-9)

    return P_0 + Delta * load_normalized


# ============================================================================
# FUNCTIONAL SPLIT 7.2 ENERGY MODEL
# ============================================================================

def compute_energy_split72(
    D_DCC: np.ndarray,
    network_params: Optional[Dict] = None,
    gops_params: Optional[Dict] = None,
    comm_energy_params: Optional[Dict] = None,
    t_ul_k: Optional[np.ndarray] = None,
    t_horizon_k: Optional[np.ndarray] = None,
    T_frame_fixed: Optional[float] = None,
) -> Dict:
    """
    Compute split-7.2 communication power and, when upload times are provided,
    communication energy.

    Split 7.2 mapping:
    - AP side: C_filter, C_DFT, C_CE
    - CPU side: C_comb

    Processing power model (linear):
    - AP processing power:  P_ap,l = P_static_ap + slope_ap * (C_ap,l / C_max_ap)
    - CPU combining power per UE k:  slope_cpu * (C_comb,k / C_max_cpu)
    - CPU total power:  P_static_cpu + slope_cpu * (C_comb_total / C_max_cpu)

    Energy model when t_ul_k is provided:
    - T_frame = T_frame_fixed if given, else max_k t_horizon_k (or t_ul_k)
    - T_act,l = max_{k in K_l} t_ul_k  (or t_horizon_k if provided)
    - E_ap_proc,l = P_ap,proc,l * T_act,l + P_static_ap * sleep_frac * (T_frame - T_act,l)
    - E_ant,l = N*P_ant*T_act,l + N*P_ant*sleep_fraction*(T_frame - T_act,l)
    - E_cpu = P_static_cpu * T_frame + sum_k slope_cpu*(C_comb,k/C_max_cpu) * t_ul_k

    t_horizon_k : ndarray (K,), optional
        Alternative per-UE horizon for AP active-time and T_frame (deadline model).
        CPU combining energy still uses t_ul_k.  When None, falls back to t_ul_k.
    T_frame_fixed : float, optional
        Fixed frame duration (seconds).  When provided, overrides the derived
        T_frame = max(t_horizon_k).  AP active times still come from per-UE
        upload/horizon times.  Use this to compare energy on a common timeframe
        (e.g. experiment_duration_s).
    """
    if network_params is None:
        if CONFIG is None:
            raise ValueError("network_params must be provided if config.yaml is not found")
        network_params = CONFIG['network']

    if gops_params is None:
        if CONFIG is None:
            raise ValueError("gops_params must be provided if config.yaml is not found")
        gops_params = CONFIG['gops_params']

    if comm_energy_params is None:
        if CONFIG is not None and 'comm_energy_params' in CONFIG:
            comm_energy_params = CONFIG['comm_energy_params']
        else:
            raise ValueError("comm_energy_params must be provided if config.yaml is not found")

    antenna_params = {'P_antenna_W': 6.8, 'sleep_fraction': 0.2}
    if CONFIG is not None:
        antenna_params.update(CONFIG.get('antenna_power_params', {}))
    P_ant = antenna_params['P_antenna_W']
    sleep_frac = antenna_params['sleep_fraction']

    L = network_params['L']
    K = network_params['K']
    N = network_params['N']
    tau_p = network_params['tau_p']
    tau_c = network_params.get('tau_c', gops_params.get('tau_c', 200))
    T_coh = tau_c * gops_params['T_s']

    C_max_ap = float(comm_energy_params['C_max_ap_GOPS'])
    P_static_ap = float(comm_energy_params['P_static_ap_W'])
    slope_ap = float(comm_energy_params['slope_power_ap_W'])
    C_max_cpu = float(comm_energy_params['C_max_cpu_GOPS'])
    P_static_cpu = float(comm_energy_params['P_static_cpu_W'])
    slope_cpu = float(comm_energy_params['slope_power_cpu_W'])

    per_ap_gops = np.zeros(L)
    per_ap_processing_power = np.zeros(L)
    per_ap_processing_energy = np.zeros(L)
    per_ap_antenna_energy = np.zeros(L)
    per_ap_total_energy = np.zeros(L)
    per_ap_processing_energy_active = np.zeros(L)
    per_ap_processing_energy_idle = np.zeros(L)
    per_ap_antenna_energy_active = np.zeros(L)
    per_ap_antenna_energy_idle = np.zeros(L)
    per_ap_act_time = np.zeros(L)

    active_count = 0
    idle_count = 0

    T_frame = 0.0
    if t_ul_k is not None:
        t_ul_k = np.asarray(t_ul_k, dtype=float)
        if t_ul_k.shape != (K,):
            raise ValueError(f"t_ul_k must have shape ({K},), got {t_ul_k.shape}")
    if t_horizon_k is not None:
        t_horizon_k = np.asarray(t_horizon_k, dtype=float)
        if t_horizon_k.shape != (K,):
            raise ValueError(f"t_horizon_k must have shape ({K},), got {t_horizon_k.shape}")
    # Horizon array used for AP active time
    _t_ap = t_horizon_k if t_horizon_k is not None else t_ul_k
    if _t_ap is not None:
        # Frame duration: use fixed value if provided, else derive from per-UE horizons
        if T_frame_fixed is not None:
            T_frame = float(T_frame_fixed)
        else:
            T_frame = float(np.max(_t_ap))

    for l in range(L):
        K_served = int(np.sum(D_DCC[l, :]))

        if K_served == 0:
            if t_ul_k is not None:
                per_ap_antenna_energy[l] = 0.0
                per_ap_processing_energy[l] = 0.0
                per_ap_total_energy[l] = 0.0
            idle_count += 1
            continue

        active_count += 1

        C_filter = compute_gops_filter(N, gops_params)
        C_DFT = compute_gops_dft(N, gops_params)
        C_CE = compute_gops_ce(N, tau_p, K_served, gops_params, tau_c=tau_c)
        C_total = C_filter + C_DFT + C_CE
        per_ap_gops[l] = C_total

        per_ap_processing_power[l] = P_static_ap + slope_ap * (C_total / C_max_ap)

        if _t_ap is not None:
            served_idx = D_DCC[l, :] > 0
            # AP active until its slowest served UE's horizon expires
            t_act = float(np.max(_t_ap[served_idx]))
            per_ap_act_time[l] = t_act
            idle_time = max(T_frame - t_act, 0.0)
            proc_act = per_ap_processing_power[l] * t_act
            proc_idle = (P_static_ap * sleep_frac) * idle_time
            per_ap_processing_energy_active[l] = proc_act
            per_ap_processing_energy_idle[l] = proc_idle
            per_ap_processing_energy[l] = proc_act + proc_idle
            ant_act = N * P_ant * t_act
            ant_idle = N * (P_ant * sleep_frac) * idle_time
            per_ap_antenna_energy_active[l] = ant_act
            per_ap_antenna_energy_idle[l] = ant_idle
            per_ap_antenna_energy[l] = ant_act + ant_idle
            per_ap_total_energy[l] = per_ap_processing_energy[l] + per_ap_antenna_energy[l]

    M_k_list = np.array([int(np.sum(D_DCC[:, k])) * N for k in range(K)])
    per_ue_comb_gops = np.zeros(K)
    for k, M_k in enumerate(M_k_list):
        if M_k == 0:
            continue
        per_ue_comb_gops[k] = compute_gops_comb(np.array([M_k]), gops_params, tau_c)

    C_comb = float(np.sum(per_ue_comb_gops))
    per_ue_cpu_dynamic_power = slope_cpu * (per_ue_comb_gops / C_max_cpu)
    cpu_dynamic_power = float(np.sum(per_ue_cpu_dynamic_power))
    cpu_power = P_static_cpu + cpu_dynamic_power

    total_power = float(np.sum(per_ap_processing_power)) + cpu_power

    results = {
        'total_power_W': total_power,
        'per_ap_power_W': per_ap_processing_power,
        'per_ap_processing_power_W': per_ap_processing_power,
        'cpu_power_W': cpu_power,
        'cpu_dynamic_power_W': cpu_dynamic_power,
        'cpu_idle_power_W': P_static_cpu,
        'cpu_gops': C_comb,
        'per_ap_gops': per_ap_gops,
        'per_ue_comb_gops': per_ue_comb_gops,
        'per_ue_cpu_dynamic_power_W': per_ue_cpu_dynamic_power,
        'active_count': active_count,
        'idle_count': idle_count,
        'M_k_antennas': M_k_list,
    }

    if t_ul_k is not None:
        # CPU dynamic energy: each UE's combining runs for its actual upload time
        per_ue_cpu_energy = per_ue_cpu_dynamic_power * t_ul_k   # (K,)
        # CPU static energy: active for the full frame
        cpu_energy = float(P_static_cpu * T_frame + np.sum(per_ue_cpu_energy))
        total_ap_processing_energy = float(np.sum(per_ap_processing_energy))
        total_antenna_energy = float(np.sum(per_ap_antenna_energy))
        total_ap_energy = float(np.sum(per_ap_total_energy))
        total_comm_energy = total_ap_energy + cpu_energy

        results['total_antenna_energy_J'] = total_antenna_energy
        results['total_ap_processing_energy_J'] = total_ap_processing_energy
        results['total_ap_energy_J'] = total_ap_energy
        results['cpu_energy_J'] = cpu_energy
        results['total_comm_energy_J'] = total_comm_energy
        results['per_ap_processing_energy_J'] = per_ap_processing_energy
        results['per_ap_processing_energy_active_J'] = per_ap_processing_energy_active
        results['per_ap_processing_energy_idle_J'] = per_ap_processing_energy_idle
        results['per_ap_antenna_energy_J'] = per_ap_antenna_energy
        results['per_ap_antenna_energy_active_J'] = per_ap_antenna_energy_active
        results['per_ap_antenna_energy_idle_J'] = per_ap_antenna_energy_idle
        results['per_ap_total_energy_J'] = per_ap_total_energy
        results['per_ue_cpu_energy_J'] = per_ue_cpu_energy
        results['T_frame_s'] = T_frame
        results['per_ap_active_time_s'] = per_ap_act_time

    return results
