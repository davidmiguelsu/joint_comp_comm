# -*- coding: utf-8 -*-


import numpy as np
import cmath
import os
from scipy import integrate, linalg
from math import exp, pi, sin, cos, sqrt, radians
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def compute_SE_MMSE(params_dict, p_k, D_DCC, H_hat, H, batch=100, SE_metric="uatf"):
    """ Function to calulcate Spectral efficiency acheived by each UE, when centralized MMSE beamforming is used.
    Input:
        p_k: Kx1, uplink transmit power allocated to each UE in mW, based on any arbitrariy power control scheme
        D_DCC: LxK, D matrix determining which AP will serve which UE. D[l,k] =1 means lth AP will serve kth UE
        H_hat: LxKxNxrealizations,  channel estimates
        H: LxKxNxrealizations,  channels
    Output:
        SE_cd: Kx1 spectral efficiencies for all UEs using coherent decoding scheme in centralized uplink operation
        SE_uatf:  Kx1 spectral efficiencies for all UEs using UatF scheme in centralized uplink operation
    """
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]
    tau_c = params_dict["tau_c"]
    L = params_dict["L"]
    realizations = params_dict["realizations"]

    # To store values
    v_k_ = np.zeros((L, K, N, realizations), dtype=complex)  # combining vectors for all Aps and UEs
    SE_uatf = np.zeros(K)  # UatF SE for each UE
    SE_cd = np.zeros(K)  # CD SE for each UE

    # Compute the prelog factor assuming only uplink data transmission
    tau_u = tau_c - tau_p
    prelogFactor = tau_u / tau_c

    # Go through all UEs
    for k in range(0, K):
        # Compute the number of APs that serve UE k
        servingAP = np.asarray(np.where(D_DCC[:, k] == 1)).reshape(-1)
        La = len(servingAP)

        H_k = H[:, servingAP, :, :]  # RxLaxNxK  channels from only serving APs of UE k
        H_k = np.reshape(H_k, [H_k.shape[0], -1, H_k.shape[-1]])  # Rx(La.N)xK

        Hhat_k = H_hat[:, servingAP, :, :]  # RxLaxNxK  channel estimates  for only serving APs of UE k
        Hhat_k = np.reshape(Hhat_k, [Hhat_k.shape[0], -1, Hhat_k.shape[-1]])  # Rx(La.N)xK

        E_k = H_k - Hhat_k  # Rx(La.N)xK
        E_k_H = H_(E_k)  # RxKx(La.N)

        # to reduce computation time, powers of all UEs are arranged in a matrix form
        W = np.tile(np.diag(p_k), (realizations, 1, 1))  # RxKxK
        Phi_k = np.mean(E_k @ W @ E_k_H, 0)  # La.NxLa.N

        v_k = np.zeros((realizations, La * N, 1), dtype=complex)
        for r in range(0, realizations, batch):
            # Calculating MMSE receive combining vector using eq 31. To reduce computational time of code, summations over all UEs in verse term are done in vector form. Thus equation looks different but mathematical correct and equal.
            inverse_term = np.linalg.inv(
                Hhat_k[r:r + batch] @ W[r:r + batch] @ H_(Hhat_k[r:r + batch]) + np.tile(
                    Phi_k + np.eye(La * N), (batch, 1, 1)))  # Rx(la.N)x(la.N)

            v_k[r:r + batch] = inverse_term @ Hhat_k[r:r + batch] @ np.sqrt(
                W[r:r + batch, :, k, np.newaxis])  # Rx(la.N)x1

        v_k_[servingAP, k, :, :] = np.reshape(v_k.T, [len(servingAP), N, realizations])

        # SEs acheived by MMSE beamforming is calculated using coherent decoding bound (Section 3, Eq 4) and UatF bound (Section 3, Eq 3)

        SE_uatf[k] = prelogFactor * Compute_SE_uatf(v_k, H_k, p_k, k)
        if SE_metric == "both":
            SE_cd[k] = prelogFactor * Compute_SE_cd(v_k, Hhat_k, Phi_k, p_k, k)

    if SE_metric == "both":
        SE = np.stack((SE_cd, SE_uatf))
    else:
        SE = SE_uatf
    return SE


def Compute_SE_cd(v_k, Hhat_k, Z_k, p_k, k):
    """
    Function to calculate SE using coherent decoding bound (Section 3 Eq 4) as metric
    Input:
        v_k: Rx(la.N)x1 receive combining vectors for all serving APs of kth UE for all small scale realizations
        Hhat_k: Rx(La.N)xK  channel estimates of N size between serving APs of kth UE and all other UEs for all small scale realizations
        Z_k: Rx(la.N)x(la.N) Estimation error correlation matrice of NxN size between serving APs of kth UE and all other UEs
        p_k: uplink transmit power of pilot for kth UE
        k: index of UE
    Output:
        SE: 1x1 spectral efficiency for kth UE using coherent decoding lower bound
    """
    v_k_H = H_(v_k)  # Rx1x(la.N) # taking hermition and reshaping for multiplications
    numerator = p_k[k] * abs((v_k_H @ Hhat_k[:, :, k, np.newaxis]) ** 2).flatten()
    denominator = np.sum(p_k.T * abs((v_k_H @ Hhat_k) ** 2), axis=2).flatten() - numerator + (
            v_k_H @ Z_k @ v_k).flatten() + np.sum(abs(v_k[:, :, 0] ** 2), axis=1).flatten()
    SINR = numerator / (denominator + 1e-10)  # instantaneous SINR
    SE = np.mean(np.real(np.log2(1 + SINR)))
    return SE


def Compute_SE_uatf(v_k, H_k, p_k, k):
    """
    Function to calculate SE using UatF bound (Section 3 Eq 2) as metric
    Input:
        v_k: Rx(la.N)x1 receive combining vectors for all serving APs of kth UE for all small scale realizations
        H_k: Rx(La.N)xK  channel of N size between serving APs of kth UE and all other UEs for all small scale realizations
        p_k: uplink transmit power of pilot for kth UE
        k: index of UE
    Output:
        SE: 1x1 spectral efficency for kth UE using UatF lower bound
    """
    v_k_H = H_(v_k)  # Rx1x(la.N) # taking hermition and reshaping for multiplications
    numerator = (p_k[k] * abs(np.mean(v_k_H @ H_k[:, :, k, np.newaxis], axis=0) ** 2)).flatten()
    denominator = np.sum(p_k.T * np.mean(abs((v_k_H @ H_k) ** 2), axis=0), axis=1).flatten() + np.mean(
        np.sum(abs(v_k[:, :, 0] ** 2), axis=1)) - numerator
    SINR = numerator / (denominator + 1e-10)
    SE = np.log2(1 + SINR)
    return SE


def calculate_local_scattering(N, varPhi, varTheta, varAntennaSpacing):
    """ Calculating normalized spatial correlation matrix between one AP and one UE
    Input:
        N: 1x1 total number of antennas at AP
        varPhi: 1x1 azimuth angle
        varTheta: 1x1 angle of elevation
        varAntennaSpacing: 1x1 antenna spacing from one antenna to the next in one AP
    Output:
        R: NxN Normalized spatial correlation matrix
    """
    varASD_phi = radians(5)  # angular standard deviation for phi
    varASD_theta = radians(5)  # angular standard deviation for theta
    spatial_corr_UE = np.zeros((N, 1), dtype=complex)
    spatial_corr_UE[0] = 1
    # Go through all the columns of the first row
    for n in range(1, N):
        # Calculating the joint pdf of Phi and Theta
        inside_integral_real = lambda varthetaDelta, varphiDelta: (cmath.exp(
            1j * 2 * pi * varAntennaSpacing * n * sin(varPhi + varphiDelta) * cos(varTheta + varthetaDelta)) * (
                                                                       exp(-varphiDelta ** 2 / (
                                                                               2 * varASD_phi ** 2))) * (
                                                                       exp(-varthetaDelta ** 2 / (
                                                                               2 * varASD_theta ** 2))) / (
                                                                           2 * pi * varASD_theta * varASD_phi)).real
        inside_integral_img = lambda varthetaDelta, varphiDelta: (cmath.exp(
            1j * 2 * pi * varAntennaSpacing * n * sin(varPhi + varphiDelta) * cos(varTheta + varthetaDelta)) * (
                                                                      exp(-varphiDelta ** 2 / (
                                                                              2 * varASD_phi ** 2))) * (
                                                                      exp(-varthetaDelta ** 2 / (
                                                                              2 * varASD_theta ** 2))) / (
                                                                          2 * pi * varASD_theta * varASD_phi)).imag

        res_real = integrate.dblquad(inside_integral_real, -8 * varASD_phi, 8 * varASD_phi, -8 * varASD_theta,
                                     8 * varASD_theta)
        res_imag = integrate.dblquad(inside_integral_img, -8 * varASD_phi, 8 * varASD_phi, -8 * varASD_theta,
                                     8 * varASD_theta)
        spatial_corr_UE[n] = res_real[0] + 1j * res_imag[0]

    # Compute the spatial correlation matrix by utilizing the Toeplitz structure
    R = linalg.toeplitz(spatial_corr_UE.ravel())  # integral gives Nx1, we need to calculate NxN matrix from it
    R = R / np.trace(R)  # normalizing
    return R


def db2pow(ydb):  # converts dB to power
    return np.power(10, ydb / 10)


def pow2db(y):  # converts power to dB
    return 10 * np.log10(y)


def H_(vec):  # Hermitian of more than 2-dimensional  matrix
    dim = vec.ndim - 1
    vec_H = np.conj(np.swapaxes(vec, dim, dim - 1))
    return vec_H


def _compute_R_Hmean(args):
    """Compute R and H_Mean for a single (l, k) pair."""
    l, k, N, phi, theta, antennaSpacing, gain = args
    R_lk = db2pow(gain) * calculate_local_scattering(N, phi, theta, antennaSpacing)
    H_mean_lk = sqrt(db2pow(gain)) * np.exp(
        1j * 2 * pi * antennaSpacing * np.arange(N) * sin(phi) * cos(theta)
    )
    return l, k, R_lk, H_mean_lk


def simulation_area(params_dict):
    """
    Generate one network scenario (APs on a grid, UEs placed at random) and
    return the per-(AP, UE) Rician statistics: spatial correlation R,
    channel gain over noise, LoS mean H_Mean, and k-factor kappa.

    The (l, k) local-scattering integrals are distributed across processes.
    """
    area = params_dict["area"]
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    bandwidth = params_dict["bandwidth"]
    fc = params_dict["carrier_frequency"]
    h = params_dict["delta_h"]
    antennaSpacing = 0.5
    sigma_sf_los = 8
    noiseVariancedB = -174 + pow2db(bandwidth) + 7

    Beta = np.zeros((L, K))
    R = np.zeros((L, K, N, N), dtype=complex)
    gainOverNoise_B = np.zeros((L, K))
    H_Mean = np.zeros((L, K, N), dtype=complex)
    kappa = np.zeros((L, K))
    dist_UEs_to_APs = np.zeros((L, K))

    UEpositions = (np.random.rand(K, 1) + 1j * np.random.rand(K, 1)) * area
    
    grid_side_x = int(np.ceil(np.sqrt(L)))
    grid_side_y = int(np.ceil(L / grid_side_x))
    x_coords = np.linspace(0, area, grid_side_x, endpoint=False) + (area / grid_side_x) / 2
    y_coords = np.linspace(0, area, grid_side_y, endpoint=False) + (area / grid_side_y) / 2
    xv, yv = np.meshgrid(x_coords, y_coords)
    ap_grid = (xv.flatten() + 1j * yv.flatten())[:L]
    APpositions = ap_grid.reshape(L, 1)

    wrapHorizontal = np.tile([-area, 0, area], (3, 1))
    wrapVertical = wrapHorizontal.T
    wrapLocations = wrapHorizontal.T.flatten() + 1j * wrapVertical.T.flatten()
    APpositionsWrapped = np.tile(APpositions, [1, len(wrapLocations)]) + np.tile(wrapLocations, [L, 1])

    for k in range(K):
        dists = abs(APpositionsWrapped - np.tile(UEpositions[k, :], (L, 9)))
        dist_UEs_to_APs[:, k] = np.sqrt(h ** 2 + dists.min(axis=1) ** 2)
        nearest_AP = dists.argmin(axis=1)

    kappa[:] = np.maximum(0, db2pow(13 - 0.05 * dist_UEs_to_APs[:]))

    # Shadow fading drawn one UE column at a time
    sf_noise = np.zeros((L, K))
    for k in range(K):
        sf_noise[:, k] = np.random.randn(L) * sigma_sf_los

    tasks = []
    for k in range(K):
        Beta[:, k] = 35.4 + 20 * np.log10(fc) + 26 * np.log10(dist_UEs_to_APs[:, k])
        gainOverNoise_B[:, k] = -Beta[:, k] - noiseVariancedB + sf_noise[:, k]

        for l in range(L):
            theta = np.arcsin(h / dist_UEs_to_APs[l, k])
            phi = np.angle(UEpositions[k] - APpositionsWrapped[l, nearest_AP[l]], deg=False)[0]
            tasks.append((l, k, N, phi, theta, antennaSpacing, gainOverNoise_B[l, k]))

    n_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for l, k, R_lk, H_mean_lk in pool.map(_compute_R_Hmean, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))):
            R[l, k, :, :] = R_lk
            H_Mean[l, k, :] = H_mean_lk

    return R, gainOverNoise_B, H_Mean, kappa


def _sqrtm_transform(args):
    """Compute sqrtm(R[l,k]) @ H[l,k] + H_Mean[l,k] for one (l,k)."""
    l, k, R_lk, H_lk, H_mean_lk, kappa_lk = args
    kf = kappa_lk / (1 + kappa_lk)
    sf = 1 / (1 + kappa_lk)
    H_mean_scaled = sqrt(kf) * H_mean_lk
    R_scaled = sf * R_lk
    H_out = H_mean_scaled + linalg.sqrtm(R_scaled) @ H_lk
    return l, k, H_out, H_mean_scaled, R_scaled


def channel_generation(params_dict, R, H_Mean, kappa):
    """
    Draw Rician channel realizations ~ N_c(H_Mean, R) for every (AP, UE) pair.
    The per-pair matrix square roots run on a thread pool (LAPACK releases the GIL).
    """
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    realizations = params_dict["realizations"]

    H_Mean = np.tile(H_Mean[:, :, :, np.newaxis], (1, 1, 1, realizations))
    H = sqrt(0.5) * (np.random.randn(L, K, N, realizations) + 1j * np.random.randn(L, K, N, realizations))

    tasks = []
    for l in range(L):
        for k in range(K):
            tasks.append((l, k, R[l, k, :, :], H[l, k, :, :], H_Mean[l, k, :, :], kappa[l, k]))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for l, k, H_out, H_mean_scaled, R_scaled in pool.map(_sqrtm_transform, tasks):
            H[l, k, :, :] = H_out
            H_Mean[l, k, :, :] = H_mean_scaled
            R[l, k, :, :] = R_scaled

    return H, H_Mean, R


def _estimate_ap(args):
    """Channel estimation for a single AP l."""
    l, tau_p, K, N, realizations, R_l, pilot_indicies_, H_l, H_Mean_l, p_k, N_tk_l = args
    H_hat_l = np.zeros((K, N, realizations), dtype=complex)

    for t in range(tau_p):
        mask = (pilot_indicies_ == t).flatten()
        if not np.any(mask):
            continue

        dim_array = np.ones((1, H_l[mask, :, :].ndim), int).ravel()
        dim_array[0] = -1
        p_k_reshaped = p_k.reshape(dim_array)

        y_pilot = np.sum(
            np.sqrt(p_k_reshaped[mask]) * tau_p * H_l[mask, :, :], axis=0
        ) + sqrt(tau_p) * N_tk_l[:, :, t]

        y_mean = np.sum(
            np.sqrt(p_k_reshaped[mask]) * tau_p * H_Mean_l[mask, :, :], axis=0
        )

        Psi = np.sum(
            p_k_reshaped[mask] * tau_p * R_l[mask, :, :], axis=0
        ) + np.eye(N)

        for k in np.where(np.any(pilot_indicies_ == t, axis=1))[0]:
            RPsi = R_l[k, :, :] @ np.linalg.inv(Psi)
            H_hat_l[k, :, :] = H_Mean_l[k, :, :] + np.sqrt(p_k[k]) * RPsi @ (y_pilot - y_mean)

    return l, H_hat_l


def channel_estimation(params_dict, R, pilot_indicies_, H, H_Mean, p_k):
    """
    MMSE channel estimation (see Section 2B), computed per AP on a thread pool.
    """
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = params_dict["tau_p"]
    realizations = params_dict["realizations"]
    p_k = np.full(K, p_k)
    H_hat = np.zeros((L, K, N, realizations), dtype=complex)
    N_tk = sqrt(0.5) * (np.random.randn(N, realizations, L, tau_p) + 1j * np.random.randn(N, realizations, L, tau_p))

    tasks = []
    for l in range(L):
        tasks.append((
            l, tau_p, K, N, realizations,
            R[l, :, :, :], pilot_indicies_,
            H[l, :, :, :], H_Mean[l, :, :, :],
            p_k, N_tk[:, :, l, :]
        ))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for l, H_hat_l in pool.map(_estimate_ap, tasks):
            H_hat[l, :, :, :] = H_hat_l

    return H_hat
