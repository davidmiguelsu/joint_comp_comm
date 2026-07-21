"""
Communication heuristic: reuse-aware AP clustering with SE-aware fractional
power control (FPC) initialization for the cell-free massive-MIMO uplink.

Before clustering, uplink powers use eligible-AP aggregates S_k and SE targets s_k:

    t_k = S_k^{-nu} * (s_k / s_ref)^gamma,   s_ref = median(s),
    p_k = P_max * t_k / max_j t_j

Reads ``fpc_nu``, ``fpc_se_gamma`` from ``network_params``.  Does not load
``config.yaml`` directly — callers pass merged ``network_params`` / ``se_targets``
(see ``joint_evaluator.py``).  Depends on ``cellfree_mimo`` and
``comm_energy_model``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cellfree_mimo
import comm_energy_model as power_model


# ---------------------------------------------------------------------------
# Cluster-formation constraint helpers (capacity limits, load checks, infra
# power). The final infrastructure energy is computed by joint_evaluator; the
# value returned by `_compute_infrastructure_power` is only used as heuristic
# metadata.
# ---------------------------------------------------------------------------

def get_capacity_limits(network_params: dict) -> tuple[float, float, int, float]:
    """Return AP/CPU capacity and clustering limits from local config."""
    cfg = power_model.CONFIG or {}
    comm_energy_params = cfg.get("comm_energy_params", {})

    c_max_ap = float(
        network_params.get(
            "C_max_ap_GOPS",
            comm_energy_params.get("C_max_ap_GOPS", 180.0),
        )
    )
    c_max_cpu = float(
        network_params.get(
            "C_max_cpu_GOPS",
            comm_energy_params.get("C_max_cpu_GOPS", 180.0),
        )
    )
    max_users_per_ap = int(network_params.get("max_users_per_ap", 6))
    min_channel_gain = float(network_params.get("min_channel_gain_db", -np.inf))
    return c_max_ap, c_max_cpu, max_users_per_ap, min_channel_gain


def check_constraints(
    D_DCC: np.ndarray,
    K: int,
    N: int,
    tau_p: int,
    c_max_ap: float,
    c_max_cpu: float,
    max_users_per_ap: int,
    gops_params: dict,
    tau_c: int = 200,
) -> tuple[bool, list[str]]:
    """Validate local AP/CPU load constraints for a candidate cluster matrix."""
    reasons: list[str] = []
    D = np.asarray(D_DCC)

    ap_user_counts = np.sum(D > 0, axis=1)
    overloaded_aps = np.where(ap_user_counts > max_users_per_ap)[0]
    if overloaded_aps.size:
        reasons.append("max_users_per_ap")

    for ap_idx, users_served in enumerate(ap_user_counts):
        ap_gops = power_model.compute_total_gops_per_ap(
            N=N,
            K_served=int(users_served),
            tau_p=tau_p,
            gops_params=gops_params,
            tau_c=tau_c,
        )
        if ap_gops > c_max_ap + 1e-9:
            reasons.append(f"ap_gops:{ap_idx}")
            break

    cluster_sizes = np.sum(D > 0, axis=0)[:K]
    antennas_per_ue = cluster_sizes * N
    cpu_gops = power_model.compute_gops_comb(
        antennas_per_ue,
        gops_params=gops_params,
        tau_c=tau_c,
    )
    if cpu_gops > c_max_cpu + 1e-9:
        reasons.append("cpu_gops")

    return len(reasons) == 0, reasons


def _compute_infrastructure_power(D_DCC: np.ndarray, network_params: dict) -> float:
    """Heuristic metadata helper; final energy is computed by joint_evaluator."""
    cfg = power_model.CONFIG or {}
    gops_params = cfg.get("gops_params", {})
    comm_energy_params = cfg.get("comm_energy_params", {})
    energy = power_model.compute_energy_split72(
        D_DCC=D_DCC,
        network_params=network_params,
        gops_params=gops_params,
        comm_energy_params=comm_energy_params,
        t_ul_k=np.ones(D_DCC.shape[1], dtype=float),
    )
    return float(energy.get("total_power_W", energy.get("total_comm_energy_J", 0.0)))


def _normalize_se_metric(metric: str) -> str:
    metric = (metric or "uatf").strip().lower()
    if metric in ("uatf", "both"):
        return metric
    if metric in ("cd", "coherent", "coherent_decoding"):
        return "both"
    raise ValueError(f"Unsupported SE metric {metric!r}")


def _compute_se_combiner(
    combiner: str,
    params_dict: dict,
    p_k: np.ndarray,
    D_DCC: np.ndarray,
    H_hat: np.ndarray,
    H: np.ndarray,
    batch: int,
    SE_metric: str,
    target_k: Optional[int] = None,
) -> np.ndarray:
    combiner = (combiner or "mmse").strip().lower().replace("-", "")
    if combiner != "mmse":
        raise ValueError(f"Unsupported combiner {combiner!r}")

    if target_k is not None:
        return _compute_se_mmse_target(
            params_dict, p_k, D_DCC, H_hat, H, batch=batch,
            SE_metric=SE_metric, target_k=target_k,
        )
    se = cellfree_mimo.compute_SE_MMSE(
        params_dict, p_k, D_DCC, H_hat, H, batch=batch, SE_metric=SE_metric
    )

    if SE_metric == "both":
        se = se[1]
    if target_k is not None:
        out = np.zeros(D_DCC.shape[1], dtype=float)
        out[target_k] = se[target_k]
        return out
    return se


def _compute_se_mmse_target(
    params_dict: dict,
    p_k: np.ndarray,
    D_DCC: np.ndarray,
    H_hat: np.ndarray,
    H: np.ndarray,
    batch: int,
    SE_metric: str,
    target_k: int,
) -> np.ndarray:
    """Compute centralized MMSE SE for one UE, matching cellfree_mimo.compute_SE_MMSE."""
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]
    tau_c = params_dict["tau_c"]
    realizations = params_dict["realizations"]
    k = int(target_k)

    se = np.zeros(K, dtype=float)
    serving_ap = np.asarray(np.where(D_DCC[:, k] == 1)).reshape(-1)
    if serving_ap.size == 0:
        return se

    la = len(serving_ap)
    H_k = H[:, serving_ap, :, :]
    H_k = np.reshape(H_k, [H_k.shape[0], -1, H_k.shape[-1]])
    Hhat_k = H_hat[:, serving_ap, :, :]
    Hhat_k = np.reshape(Hhat_k, [Hhat_k.shape[0], -1, Hhat_k.shape[-1]])

    E_k = H_k - Hhat_k
    E_k_H = cellfree_mimo.H_(E_k)
    W = np.tile(np.diag(p_k), (realizations, 1, 1))
    Phi_k = np.mean(E_k @ W @ E_k_H, 0)

    v_k = np.zeros((realizations, la * N, 1), dtype=complex)
    eye_term = Phi_k + np.eye(la * N)
    for r in range(0, realizations, batch):
        end = min(r + batch, realizations)
        span = end - r
        inverse_term = np.linalg.inv(
            Hhat_k[r:end] @ W[r:end] @ cellfree_mimo.H_(Hhat_k[r:end])
            + np.tile(eye_term, (span, 1, 1))
        )
        v_k[r:end] = inverse_term @ Hhat_k[r:end] @ np.sqrt(
            W[r:end, :, k, np.newaxis]
        )

    prelog_factor = (tau_c - tau_p) / tau_c
    se[k] = prelog_factor * cellfree_mimo.Compute_SE_uatf(v_k, H_k, p_k, k)
    return se


def _initial_uplink_powers_se_fpc(
    B_channel_gains: np.ndarray,
    p_max: float,
    nu: float,
    min_channel_gain_db: float,
    se_targets: np.ndarray,
    se_gamma: float,
) -> np.ndarray:
    """FPC weights from S_k (eligible APs) and (s_k/s_ref)^gamma; max-normalize to P_max."""
    K = B_channel_gains.shape[1]
    se_targets = np.asarray(se_targets, dtype=float).reshape(K)
    eligible = B_channel_gains >= min_channel_gain_db
    linear = np.where(eligible, cellfree_mimo.db2pow(B_channel_gains), 0.0)
    S = np.sum(linear, axis=0)

    p_k = np.full(K, p_max, dtype=float)
    nu = float(nu)
    se_gamma = float(max(0.0, se_gamma))

    valid = S > 1e-20
    if not np.any(valid):
        return p_k

    s_safe = np.maximum(se_targets, 1e-12)
    s_ref = float(np.median(s_safe))
    if s_ref <= 0.0:
        s_ref = max(float(np.max(s_safe)), 1e-12)

    S_safe = np.maximum(S, 1e-20)
    if nu > 0.0:
        path = S_safe ** (-nu)
    else:
        path = np.ones(K, dtype=float)

    if se_gamma > 0.0:
        qos = (s_safe / s_ref) ** se_gamma
    else:
        qos = np.ones(K, dtype=float)

    t = np.zeros(K, dtype=float)
    t[valid] = path[valid] * qos[valid]
    denom = float(np.max(t[valid]))
    if denom <= 0.0:
        return p_k

    p_k[valid] = p_max * (t[valid] / denom)
    return np.clip(p_k, 0.0, p_max)


def run_new_heuristic_v6_se_fpc(
    H_input: np.ndarray,
    H_hat_input: np.ndarray,
    B_channel_gains: np.ndarray,
    network_params: dict,
    se_targets: np.ndarray,
    se_realizations: Optional[int] = None,
    combiner: Optional[str] = None,
    se_metric: Optional[str] = None,
    se_metric_final: Optional[str] = None,
    se_realizations_final: Optional[int] = None,
    frozen_combiner: Optional[str] = None,
) -> tuple[np.ndarray, dict, np.ndarray]:
    L, K = B_channel_gains.shape
    se_targets = np.asarray(se_targets, dtype=float)

    ev = network_params.get("evaluation", {})

    if combiner is None:
        combiner = network_params.get("combiner", ev.get("combiner", "mmse"))
    comb_key = (combiner or "mmse").strip().lower().replace("-", "")

    if se_metric is None:
        se_metric = network_params.get("se_metric", ev.get("se_metric", "uatf"))
    se_metric = _normalize_se_metric(se_metric)

    if se_metric_final is None:
        se_metric_final = network_params.get(
            "se_metric_final", ev.get("se_metric_final", "uatf"),
        )
    se_metric_final = _normalize_se_metric(se_metric_final)

    if frozen_combiner is None:
        raw = network_params.get("frozen_combiner", ev.get("frozen_combiner", "false"))
        if isinstance(raw, bool):
            frozen_combiner = "pmax" if raw else "false"
        else:
            frozen_combiner = str(raw).strip().lower()
    if frozen_combiner not in ("false", "pmax", "adaptive"):
        raise ValueError(
            f"frozen_combiner must be 'false', 'pmax', or 'adaptive', got {frozen_combiner!r}"
        )

    p_max = float(network_params.get("p_k", 100.0))
    fpc_nu = float(network_params.get("fpc_nu", ev.get("fpc_nu", 0.5)))
    fpc_nu = min(1.0, max(0.0, fpc_nu))
    fpc_se_gamma = max(
        0.0,
        float(network_params.get("fpc_se_gamma", ev.get("fpc_se_gamma", 0.0))),
    )

    if se_realizations is None:
        se_realizations = int(network_params.get("realizations", 10))
    if se_realizations_final is None:
        se_realizations_final = int(
            network_params.get(
                "se_realizations_final", ev.get("se_realizations_final", 200),
            )
        )

    R_total = H_input.shape[0]
    n_inter = min(int(se_realizations), R_total)
    n_final = min(int(se_realizations_final), R_total)

    inter_params = dict(network_params)
    inter_params["realizations"] = n_inter
    final_params = dict(network_params)
    final_params["realizations"] = n_final

    H_inter = H_input[:n_inter]
    H_hat_inter = H_hat_input[:n_inter]
    H_final = H_input[:n_final]
    H_hat_final = H_hat_input[:n_final]

    N = network_params["N"]
    tau_p = network_params["tau_p"]
    tau_c = network_params.get("tau_c", 200)

    gops_params = power_model.CONFIG.get("gops_params") if power_model.CONFIG else {}
    if not gops_params:
        gops_params = {
            "T_s": 0.00007143, "N_DFT": 2048,
            "N_used": 1200, "f_s": 30720000.0,
        }

    c_max_ap, c_max_cpu, max_users_per_ap, min_channel_gain = get_capacity_limits(
        network_params
    )
    max_aps_per_ue = int(network_params.get('max_aps_per_ue',
                                             ev.get('max_aps_per_ue', 10)))
    cluster_mode = str(network_params.get('comm_heuristic',
                                          ev.get('comm_heuristic', 'v6'))).strip().lower()

    p_k_phase1 = _initial_uplink_powers_se_fpc(
        B_channel_gains, p_max, fpc_nu, min_channel_gain,
        se_targets, fpc_se_gamma,
    )

    t_phase1_start = time.perf_counter()
    D_DCC = np.zeros((L, K), dtype=float)
    ap_delta_se_matrix = np.zeros((L, K), dtype=float)

    sorted_B = np.sort(B_channel_gains, axis=0)[::-1, :]
    best_B = sorted_B[0, :]
    ratio_score = best_B / se_targets
    ue_order = np.argsort(ratio_score)
    active_aps_set: set[int] = set()

    def _get_current_se_k(k_idx: int, D: np.ndarray) -> float:
        if np.sum(D[:, k_idx]) == 0:
            return 0.0
        se_res = _compute_se_combiner(
            comb_key, inter_params, p_k_phase1, D,
            H_hat_inter, H_inter, batch=n_inter,
            SE_metric=se_metric, target_k=k_idx,
        )
        val = se_res[k_idx]
        return 0.0 if np.isnan(val) else float(val)

    for k in ue_order:
        req_SE = se_targets[k]
        curr_se = _get_current_se_k(k, D_DCC)

        valid_aps_for_k = [
            ap for ap in range(L)
            if B_channel_gains[ap, k] >= min_channel_gain
        ]

        if cluster_mode == 'unaware':
            # Energy-unaware clustering: no active/inactive distinction.  Just
            # add the best-channel APs (gain >= threshold) until the SE target
            # or the max-cluster constraint is hit.  The idle/sleep power model
            # still applies downstream via D_DCC; the algorithm simply does not
            # reason about it (so it activates more APs than the reuse-aware v6).
            unaware_aps = sorted(valid_aps_for_k, key=lambda ap: -B_channel_gains[ap, k])
            for ap in unaware_aps:
                if curr_se >= req_SE:
                    break
                if int(np.sum(D_DCC[:, k])) >= max_aps_per_ue:
                    break
                temp_D = D_DCC.copy()
                temp_D[ap, k] = 1.0
                valid, _ = check_constraints(
                    temp_D, K, N, tau_p, c_max_ap, c_max_cpu,
                    max_users_per_ap, gops_params, tau_c,
                )
                if valid:
                    D_DCC[ap, k] = 1.0
                    active_aps_set.add(ap)
                    new_se = _get_current_se_k(k, D_DCC)
                    ap_delta_se_matrix[ap, k] = max(0.0, new_se - curr_se)
                    curr_se = new_se
        else:
            active_valid = [ap for ap in valid_aps_for_k if ap in active_aps_set]
            active_valid.sort(key=lambda ap: -B_channel_gains[ap, k])

            for ap in active_valid:
                if int(np.sum(D_DCC[:, k])) >= max_aps_per_ue:
                    break
                temp_D = D_DCC.copy()
                temp_D[ap, k] = 1.0
                valid, _ = check_constraints(
                    temp_D, K, N, tau_p, c_max_ap, c_max_cpu,
                    max_users_per_ap, gops_params, tau_c,
                )
                if valid:
                    D_DCC[ap, k] = 1.0
                    new_se = _get_current_se_k(k, D_DCC)
                    ap_delta_se_matrix[ap, k] = max(0.0, new_se - curr_se)
                    curr_se = new_se
                    if curr_se >= req_SE:
                        break

            if curr_se < req_SE:
                inactive_valid = [ap for ap in valid_aps_for_k if ap not in active_aps_set]
                inactive_valid.sort(key=lambda ap: -B_channel_gains[ap, k])

                for ap in inactive_valid:
                    if int(np.sum(D_DCC[:, k])) >= max_aps_per_ue:
                        break
                    temp_D = D_DCC.copy()
                    temp_D[ap, k] = 1.0
                    valid, _ = check_constraints(
                        temp_D, K, N, tau_p, c_max_ap, c_max_cpu,
                        max_users_per_ap, gops_params, tau_c,
                    )
                    if valid:
                        D_DCC[ap, k] = 1.0
                        active_aps_set.add(ap)
                        new_se = _get_current_se_k(k, D_DCC)
                        ap_delta_se_matrix[ap, k] = max(0.0, new_se - curr_se)
                        curr_se = new_se
                        if curr_se >= req_SE:
                            break

        if np.sum(D_DCC[:, k]) == 0:
            all_aps_by_gain = np.argsort(-B_channel_gains[:, k])
            for ap in all_aps_by_gain:
                if int(np.sum(D_DCC[:, k])) >= max_aps_per_ue:
                    break
                temp_D = D_DCC.copy()
                temp_D[ap, k] = 1.0
                valid, _ = check_constraints(
                    temp_D, K, N, tau_p, c_max_ap, c_max_cpu,
                    max_users_per_ap, gops_params, tau_c,
                )
                if valid:
                    D_DCC[ap, k] = 1.0
                    active_aps_set.add(ap)
                    new_se = _get_current_se_k(k, D_DCC)
                    ap_delta_se_matrix[ap, k] = max(0.0, new_se)
                    break

    phase1_time_s = float(time.perf_counter() - t_phase1_start)

    phase1_se = np.nan_to_num(
        _compute_se_combiner(
            comb_key, inter_params, p_k_phase1, D_DCC,
            H_hat_inter, H_inter, batch=n_inter, SE_metric=se_metric,
        ),
        nan=0.0,
    )
    D_after_phase1 = D_DCC.copy()
    active_aps_after_phase1 = int(np.sum(np.sum(D_DCC, axis=1) > 0))

    p_k_vec = p_k_phase1.copy()
    final_se = np.nan_to_num(
        _compute_se_combiner(
            comb_key, final_params, p_k_vec, D_DCC,
            H_hat_final, H_final, batch=n_final, SE_metric=se_metric_final,
        ),
        nan=0.0,
    )
    if n_final != n_inter or se_metric_final != se_metric:
        se_intermediate = np.nan_to_num(
            _compute_se_combiner(
                comb_key, inter_params, p_k_vec, D_DCC,
                H_hat_inter, H_inter, batch=n_inter, SE_metric=se_metric,
            ),
            nan=0.0,
        )
        violated_users_intermediate = int(np.sum(se_intermediate < se_targets))
        violations_sumse_intermediate = float(
            np.sum(np.maximum(0.0, se_targets - se_intermediate))
        )
    else:
        violated_users_intermediate = int(np.sum(final_se < se_targets))
        violations_sumse_intermediate = float(
            np.sum(np.maximum(0.0, se_targets - final_se))
        )
    violated_users = int(np.sum(final_se < se_targets))
    aps_per_user = np.sum(D_DCC, axis=0)
    ap_user_counts = np.sum(D_DCC, axis=1)
    infra_after_p1 = _compute_infrastructure_power(D_DCC, network_params)
    results = {
        "total_power": float(np.sum(p_k_vec)),
        "sum_se": float(np.sum(final_se)),
        "min_se": float(np.min(final_se)) if K > 0 else 0.0,
        "violated_users": violated_users,
        "accumulated_se_violation": float(np.sum(np.maximum(0, se_targets - final_se))),
        "avg_se": float(np.mean(final_se)) if K > 0 else 0.0,
        "active_aps": int(np.sum(ap_user_counts > 0)),
        "per_user_se": final_se,
        "avg_cluster_size": float(np.mean(aps_per_user)) if K > 0 else 0.0,
        "min_cluster_size": float(np.min(aps_per_user)) if K > 0 else 0.0,
        "max_cluster_size": float(np.max(aps_per_user)) if K > 0 else 0.0,
        "avg_users_per_ap": float(np.mean(ap_user_counts[ap_user_counts > 0]))
            if np.any(ap_user_counts > 0) else 0.0,
        "max_users_per_ap": int(np.max(ap_user_counts)) if L > 0 else 0,
        "p_k_vec": p_k_vec,
        "required_se": se_targets,
        "combiner": comb_key,
        "se_metric": se_metric,
        "se_metric_final": se_metric_final,
        "frozen_combiner": frozen_combiner,
        "fpc_nu": fpc_nu,
        "fpc_se_gamma": fpc_se_gamma,
        "v6_scheme": "v6_se_fpc",
        "n_real_intermediate": n_inter,
        "n_real_final": n_final,
        "violated_users_intermediate": violated_users_intermediate,
        "violations_sumse_intermediate": violations_sumse_intermediate,
        "active_aps_after_phase1": active_aps_after_phase1,
        "phase1_time_s": phase1_time_s,
        "phase2_feasible": None,
        "active_aps_after_phase2": active_aps_after_phase1,
        "infra_power_after_phase2": infra_after_p1,
        "phase3_attempted": 0,
        "phase3_deactivated": 0,
        "total_fm_iters": 0,
        "total_fm_time_s": 0.0,
        "phase3_time_s": 0.0,
        "active_aps_before_phase3": active_aps_after_phase1,
        "active_aps_after_phase3": int(np.sum(ap_user_counts > 0)),
        "infra_power_before_phase3": infra_after_p1,
        "infra_power_after_phase3": infra_after_p1,
        "phase1_D_DCC": D_after_phase1,
        "phase1_se": phase1_se,
        "phase1_p_k_vec": p_k_phase1,
        "phase2_se": None,
    }
    return D_DCC, results, p_k_vec


__all__ = ["run_new_heuristic_v6_se_fpc", "_initial_uplink_powers_se_fpc"]
