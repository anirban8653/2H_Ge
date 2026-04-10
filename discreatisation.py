import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# =============================================================================
# define parameters form table 2
# =============================================================================

e = 1.602e-19
hbar = 1.05457e-34
m0 = me = 9.109e-31 
a0 = 5.292e-11
Ev = -0.3622
Ecbp1 = 0.632 
Ecb = 0.298

L = 300
N = 50
a = L/(N+1)


"""Optical transition matrix elements"""
p_perp = 0.4829 * hbar/a0/(e*1e-10)
p_parallel = 0.6431 * hbar/a0/(e*1e-10)

"""conduction band effective parameters"""
eff_mass_unit = hbar**2/(2*m0)
A_c1_parallel = 1.0 * eff_mass_unit/(e*1e-20)
A_c2_perp = 4.1565 * eff_mass_unit/(e*1e-20)
A_c1_perp = 9.5120 * eff_mass_unit/(e*1e-20)
A_c2_parallel = 2.4091 * eff_mass_unit/(e*1e-20)

"""valance band effective parameters"""
A_1 = -4.3636 * eff_mass_unit/(e*1e-20)
A_2 = -2.0833 * eff_mass_unit/(e*1e-20)
A_3 = 2.4545 * eff_mass_unit/(e*1e-20)
A_4 = -2.7504 * eff_mass_unit/(e*1e-20)
A_5 = -2.7232 * eff_mass_unit/(e*1e-20)
A_6 = -3.5421 * eff_mass_unit/(e*1e-20)

"""Energy splittings"""
Delta1 = deltacf = 0.2688 
Delta2 = 0.0934
Delta3 = 0.0908

# =============================================================================
# Create the H_kp matrix
# =============================================================================


# ---------- Hamiltonian builder ----------
def H_kp(kx, ky, kz):
    k_plus = kx + 1j*ky
    k_minus = kx - 1j*ky

    C1 = A_c1_perp*(kx**2 + ky**2) + A_c1_parallel*kz**2
    C2 = A_c2_perp*(kx**2 + ky**2) + A_c2_parallel*kz**2
    beta = A_1*kz**2 + A_2*(kx**2 + ky**2)
    alpha = A_3*kz**2 + A_4*(kx**2 + ky**2)

    kin = 0#hbar**2 * (kx**2 + ky**2 + kz**2) / (2*m0)

    H = np.zeros((10,10), dtype=complex)

    H[0,0] = kin + C2 + Ecbp1
    H[0,2] = -(np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[0,3] = (np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[0,4] = (hbar*kz*p_parallel)/m0

    H[1,1] = kin + C1 + Ecb

    H[2,0] = -(np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[2,2] = kin + Delta1 + Delta2 + Ev + alpha + beta
    H[2,3] = -A_5*k_minus**2
    H[2,4] = -A_6*k_minus*kz

    H[3,0] = (np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[3,2] = -A_5*k_plus**2
    H[3,3] = kin + Delta1 - Delta2 + Ev + alpha + beta
    H[3,4] = A_6*k_plus*kz
    H[3,9] = np.sqrt(2)*Delta3

    H[4,0] = (hbar*kz*p_parallel)/m0
    H[4,2] = -A_6*k_plus*kz
    H[4,3] = A_6*k_minus*kz
    H[4,4] = kin + Ev + beta
    H[4,8] = np.sqrt(2)*Delta3

    H[5,5] = kin + C2 + Ecbp1
    H[5,7] = (np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[5,8] = -(np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[5,9] = (hbar*kz*p_parallel)/m0

    H[6,6] = kin + C1 + Ecb

    H[7,5] = (np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[7,7] = kin + Delta1 + Delta2 + Ev + alpha + beta
    H[7,8] = -A_5*k_plus**2
    H[7,9] = A_6*k_plus*kz

    H[8,4] = np.sqrt(2)*Delta3
    H[8,5] = -(np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[8,7] = -A_5*k_minus**2
    H[8,8] = kin + Delta1 - Delta2 + Ev + alpha + beta
    H[8,9] = -A_6*k_minus*kz

    H[9,3] = np.sqrt(2)*Delta3
    H[9,5] = (hbar*kz*p_parallel)/m0
    H[9,7] = A_6*k_minus*kz
    H[9,8] = -A_6*k_plus*kz
    H[9,9] = kin + Ev + beta

    # Hermitianize (safeguard against numerical asymmetry)
    H = (H + H.conj().T)/2

    return H


# =============================================================================
# discreate hamiltonian
# =============================================================================


def H_dis(kx, ky, kz):
    """kane hamiltonian part"""
    # Apply lattice substitutions
    kx_sin = 1/a * np.sin(kx * a)
    ky_sin = 1/a * np.sin(ky * a)
    kz_sin = 1/a * np.sin(kz * a)

    kx2 = 2/(a**2) * (1 - np.cos(kx * a))
    ky2 = 2/(a**2) * (1 - np.cos(ky * a))
    kz2 = 2/(a**2) * (1 - np.cos(kz * a))

    k_plus = kx_sin + 1j*ky_sin
    k_minus = kx_sin - 1j*ky_sin

    C1 = A_c1_perp*(kx2 + ky2) + A_c1_parallel*kz2
    C2 = A_c2_perp*(kx2 + ky2) + A_c2_parallel*kz2
    beta = A_1*kz2 + A_2*(kx2 + ky2)
    alpha = A_3*kz2 + A_4*(kx2 + ky2)

    kin = 0
    
    H = np.zeros((10,10), dtype=complex)

    H[0,0] = kin + C2 + Ecbp1
    H[0,2] = -(np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[0,3] = (np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[0,4] = (hbar*kz_sin*p_parallel)/m0

    H[1,1] = kin + C1 + Ecb

    H[2,0] = -(np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[2,2] = kin + Delta1 + Delta2 + Ev + alpha + beta
    H[2,3] = -A_5*k_minus**2
    H[2,4] = -A_6*k_minus*kz_sin

    H[3,0] = (np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[3,2] = -A_5*k_plus**2
    H[3,3] = kin + Delta1 - Delta2 + Ev + alpha + beta
    H[3,4] = A_6*k_plus*kz_sin
    H[3,9] = np.sqrt(2)*Delta3

    H[4,0] = (hbar*kz_sin*p_parallel)/m0
    H[4,2] = -A_6*k_plus*kz_sin
    H[4,3] = A_6*k_minus*kz_sin
    H[4,4] = kin + Ev + beta
    H[4,8] = np.sqrt(2)*Delta3

    H[5,5] = kin + C2 + Ecbp1
    H[5,7] = (np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[5,8] = -(np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[5,9] = (hbar*kz_sin*p_parallel)/m0

    H[6,6] = kin + C1 + Ecb

    H[7,5] = (np.sqrt(2)*hbar*k_plus*p_perp)/(2*m0)
    H[7,7] = kin + Delta1 + Delta2 + Ev + alpha + beta
    H[7,8] = -A_5*k_plus**2
    H[7,9] = A_6*k_plus*kz_sin

    H[8,4] = np.sqrt(2)*Delta3
    H[8,5] = -(np.sqrt(2)*hbar*k_minus*p_perp)/(2*m0)
    H[8,7] = -A_5*k_minus**2
    H[8,8] = kin + Delta1 - Delta2 + Ev + alpha + beta
    H[8,9] = -A_6*k_minus*kz_sin

    H[9,3] = np.sqrt(2)*Delta3
    H[9,5] = (hbar*kz_sin*p_parallel)/m0
    H[9,7] = A_6*k_minus*kz_sin
    H[9,8] = -A_6*k_plus*kz_sin
    H[9,9] = kin + Ev + beta

    # Hermitianize (safeguard against numerical asymmetry)
    H = (H + H.conj().T)/2

    return H


kpevals,_ = eigh(H_kp(0, 0, 0))
# print(kpevals)


disevals,_ = eigh(H_dis(0, 0, 0))
# print(disevals)


plt.rcParams.update({'figure.autolayout': True})
plt.figure(figsize=(5, 8))
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


colors = ['r', 'b', 'g', 'c']    # 1 kp + 4 discrete

nk = 300
div = 1
Mpoint = np.array([np.pi / 1, 0.0, 0.0])
GM = np.array([[i, 0, 0] for i in np.arange(0, Mpoint[0] / div, Mpoint[0] / (div * nk))])
k_points = GM[:, 0]

# Compute eigenvalues for analytical model
eigvals = [eigh(H_kp(kx, ky, kz))[0] for kx, ky, kz in GM]

# === Plot kp model ===
for i, band in enumerate(np.array(eigvals).T):
    plt.plot(k_points, band, color=colors[0], lw=1.5, label="k·p" if i == 1 else "")

# === Plot discrete models ===

Mpoint = np.array([np.pi / a, 0.0, 0.0])
GM = np.array([[i, 0, 0] for i in np.arange(0, Mpoint[0] / div, Mpoint[0] / (div * nk))])
k_points = GM[:, 0]

# Pass `a` into Hamiltonian functions
eigvals_dis = [eigh(H_dis(kx, ky, kz))[0] for kx, ky, kz in GM]

for i, band in enumerate(np.array(eigvals_dis).T):
    plt.plot(k_points, band, '--', color=colors[1],
              label=f"a={a:.1f} Å" if i == 1 else "")

plt.axhline(y=0, color='black', ls='--', lw=0.5)
plt.xlabel("k$_x$ (1/Å)", fontsize=25)
plt.ylabel("Energy (eV)", fontsize=25)
plt.ylim(-1.2, 1.5)
plt.xlim(0.0, 0.15)
# plt.grid(True, linestyle='--', alpha=0.3)
plt.tick_params(direction='in', length=4, width=1, labelsize=25)
plt.legend(frameon=False, fontsize=15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("10-band_comparison_all_a_values_single_plot.png", dpi=300)
plt.savefig("10-band_comparison_all_a_values_single_plot.pdf")
plt.show()


# a = 20

# G = np.array([0.0, 0.0, 0.0])
# M = np.array([np.pi / a, 0.0, 0.0])
# X = np.array([0.0, np.pi / a, 0.0])
# R = np.array([np.pi / a, np.pi / a, np.pi / a])

# GX = np.linspace(G[1],X[1], 100)
# GX_path = np.array([[0,i,0] for i in GX])

# XM_kx = np.linspace(X[0], M[0], 100)
# XM_ky = np.linspace(X[1], M[1], 100)
# XM_path = np.array([[kx, ky, 0] for kx, ky in zip(XM_kx, XM_ky)])


# MR_ky = np.linspace(M[1], R[1], 100)
# MR_kz = np.linspace(M[2], R[2], 100)

# MR_path = np.array([[M[0], ky, kz] for ky, kz in zip(MR_ky, MR_kz)])

# RG_kx = np.linspace(R[0], G[0], 100)
# RG_ky = np.linspace(R[1], G[1], 100)
# RG_kz = np.linspace(R[2], G[2], 100)

# RG_path = np.array([[kx, ky, kz] for kx, ky, kz in zip(RG_kx, RG_ky, RG_kz)])


# k_path = np.vstack([GX_path, XM_path, MR_path, RG_path])


# k_dist = [0]

# for i in range(1, len(k_path)):
#     dk = np.linalg.norm(k_path[i] - k_path[i-1])
#     k_dist.append(k_dist[-1] + dk)

# k_dist = np.array(k_dist)



# evals_list = []
# for k in k_path:
#     val, vec = eigh(H_dis(k[0], k[1], k[2]))
#     evals_list.append(val)
    
# evals_list = np.array(evals_list)
# print(evals_list.shape)

# k_nodes = [0, 99, 199, 299, 399]
# k_labels = [r'$\Gamma$', 'X', 'M', 'R', r'$\Gamma$']


# plt.figure(figsize=(8,6))

# # Plot all bands
# for i in range(evals_list.shape[1]):
#     plt.plot(k_dist, evals_list[:, i], lw=1)

# # Vertical lines at symmetry points
# for node in k_nodes:
#     plt.axvline(k_dist[node], color='k', linestyle='--', linewidth=0.5)

# # Formatting
# plt.xticks([k_dist[i] for i in k_nodes], k_labels)
# plt.ylabel("Energy")
# plt.xlabel("k-path")
# plt.title("Band Structure")
# # plt.ylim(-1,1)

# plt.tight_layout()
# plt.show()





# a = 20
# def random_unit_vector():
#     v = np.random.randn(3)
#     return v / np.linalg.norm(v)



# N = 500
# gap_threshold = 0.298  # eV

# bad_sign_count = 0
# small_gap_count = 0

# tol = 1e-6
# k0 = 1e5   # very large momentum
# for i in range(N):
#     direction = random_unit_vector()
#     kx, ky, kz = k0 * direction
    
#     evals, _ = eigh(H_dis(kx, ky, kz))
    
#     # Count positive/negative
#     n_pos = np.sum(evals > tol)
#     n_neg = np.sum(evals < -tol)
    
#     # TRUE band gap
#     current_gap = evals[6] - evals[5]
#     # min_gap = min(min_gap, current_gap)
    
#     # Check sign condition
#     if not (n_pos == 4 and n_neg == 6):
#         bad_sign_count += 1
#         print(f"[FAIL SIGN] Iter {i}: n_pos={n_pos}, n_neg={n_neg}")
    
#     # Check gap condition
#     if current_gap < gap_threshold:
#         small_gap_count += 1
#         print(f"[SMALL GAP] Iter {i}: gap={current_gap:.4f} eV")
    
#     # Debug print if something fails
#     if (n_pos != 4 or n_neg != 6) or (current_gap < gap_threshold):
#         print("Eigenvalues:", evals)
#         print("gap:", current_gap)
#         print("-"*50)

# print("\n===== SUMMARY =====")
# print(f"Sign failures     : {bad_sign_count} / {N}")
# print(f"Small gap failures: {small_gap_count} / {N}")





# all_evals = []

# for _ in range(200):
#     direction = random_unit_vector()
#     kx, ky, kz = k0 * direction
#     evals,_ = eigh(H_dis(kx, ky, kz))
#     all_evals.extend(evals)

# plt.hist(all_evals, bins=100)
# plt.title("Eigenvalue distribution at large k")
# plt.xlabel("Energy")
# plt.ylabel("Count")

