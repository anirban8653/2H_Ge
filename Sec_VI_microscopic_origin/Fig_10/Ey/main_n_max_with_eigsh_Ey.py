import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
import time
import matplotlib.pyplot as plt

# ---------------------------------
# TIMER START
# ---------------------------------
t0 = time.time()

# ---------------------------------
# Import Hamiltonian blocks
# ---------------------------------
# from Hamiltonian_real_april import build_H_real, H_dkx
from Hamiltonian_mathematica_v2 import gso, dgso, psi_new_basis

# ---------------------------------
# Parameters
# ---------------------------------
N = 100 #int(input("Enter the value of N: "))                   # FIXED
kx = 0.001 #float(input("Enter the value of kx : "))              # FIXED


Nband = 10
L = 300
Ny = Nz = N

# number of states and around the specific energy (sigma)
num_v_bands_2, sigma_v2 = 250, -0.35
num_v_bands_1, sigma_v1 = 20, 0
num_c_bands, sigma_c = 2, 0.65


print(f"\nBuilding system for N = {N}")


lat = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system
# ---------------------------------
def make_system(kx, Ef):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            V = -Ef * ((y + 1) * a - L / 2)
            syst[lat(y, z)] = V * np.eye(Nband) + gso_cache[(0, 0)]

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue

            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso_cache[(dy, dz)]

    return syst.finalized()




lat_kx = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system
# ---------------------------------
def make_system_kx(kx):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    gso_cache_kx = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache_kx[(dy, dz)] = dgso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            syst[lat_kx(y, z)] =  gso_cache_kx[(0, 0)]

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat_kx, lat_kx)] = gso_cache_kx[(dy, dz)]

    return syst.finalized()




lat_E = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system
# ---------------------------------
def make_system_E():

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            V = - ((y + 1) * a - L / 2)
            syst[lat(y, z)] = V * np.eye(Nband) 

    return syst.finalized()


# ---------------------------------
# Sweep Ef
# ---------------------------------
Ef_values = [0.5e-5] 

#---------------------------------------------------------
#  Schriffer wolf 
#---------------------------------------------------------


# =================================================================
# 1. HAMILTONIAN CONSTRUCTION
# =================================================================

# Building the system components (Sparse matrices)
syst0 = make_system(0.0, 0.0)
H0_sparse = syst0.hamiltonian_submatrix(sparse=True)

syst1 = make_system_kx(0.0)
V_kx = syst1.hamiltonian_submatrix(sparse=True)

syst2 = make_system_E()
V_Ey = syst2.hamiltonian_submatrix(sparse=True)

# =================================================================
# 2. DIAGONALIZATION & SUBSPACE SELECTION
# =================================================================


print(f"\ncalculating for {num_v_bands_1} VB levels")
print(f"calculating for {num_c_bands} CB levels")
print(f"calculating for {num_v_bands_2} VB2 levels")

E_valence_1, vb_vectors_1 = eigsh(H0_sparse, k = num_v_bands_1, sigma = sigma_v1, which="LM")
E_valence_2, vb_vectors_2 = eigsh(H0_sparse, k = num_v_bands_2, sigma = sigma_v2, which="LM")
E_conduction, cb_vectors = eigsh(H0_sparse, k = num_c_bands, sigma = sigma_c, which="LM")

print("\nDiagonalisation is done.")

# # Ensure orthogonality via economic QR decomposition
# vb_vectors_1, _ = qr(vb_vectors_1, mode="economic")
# vb_vectors_2, _ = qr(vb_vectors_2, mode="economic")
# cb_vectors, _ = qr(cb_vectors, mode="economic") 

# -----------------------------
# Sort each eigsh output properly
# -----------------------------
idx_v1 = np.argsort(E_valence_1)
idx_v2 = np.argsort(E_valence_2)
idx_c  = np.argsort(E_conduction)


E_valence_1 = E_valence_1[idx_v1]
vb_vectors_1 = vb_vectors_1[:, idx_v1]


E_valence_2 = E_valence_2[idx_v2]
vb_vectors_2 = vb_vectors_2[:, idx_v2]

E_conduction = E_conduction[idx_c]
cb_vectors = cb_vectors[:, idx_c]


# -----------------------------
# Define A subspace: top two VB states closest to zero
# Since E_valence_1 is ascending, top valence = last two
# -----------------------------
idx_A = [-1, -2]

E_A = E_valence_1[idx_A]
psi1 = vb_vectors_1[:, idx_A[0]]
psi2 = vb_vectors_1[:, idx_A[1]]

psi1, psi2 = psi_new_basis(psi1, psi2, N)

print("\nBasis rotation is completed.")
print("E_A =", E_A)


# -----------------------------
# Keep remaining VB1 states, excluding A
# -----------------------------
mask_B_v1 = np.ones(len(E_valence_1), dtype=bool)
mask_B_v1[idx_A] = False

E_vb1_B = E_valence_1[mask_B_v1]
V_vb1_B = vb_vectors_1[:, mask_B_v1]


# -----------------------------
# Combine B subspace:
# negative-most VB -> positive-most CB
# i.e. globally ascending energy
# -----------------------------
E_B_all = np.concatenate(( E_valence_2, E_vb1_B, E_conduction))
V_B_all = np.column_stack((vb_vectors_2, V_vb1_B, cb_vectors))

idx_B_sort = np.argsort(E_B_all)

E_B_sorted = E_B_all[idx_B_sort]
V_B_sorted = V_B_all[:, idx_B_sort]


print(f"\nvb energy range: {E_valence_2[0]:.4f} -- {E_vb1_B[-1]:.4f} eV")
print(f"cb energy range: {E_conduction[0]:.4f} -- {E_conduction[-1]:.4f} eV")


# ==========================================================
# Individual matrix-element contributions from each state
# ==========================================================


splittings = []
prefactor = kx * Ef_values[0]   # Ey = 0.6e-5


# ==========================================================
# Helper function
# ==========================================================
def add_contribution(Ei, Vi):

    # Energy denominators
    d1 = E_A[0] - Ei
    d2 = E_A[1] - Ei

    # Symmetrized denominator
    d_sym = 0.5 * (1/d1 + 1/d2)

    # ------------------------------------------------------
    # Diagonal matrix elements
    # ------------------------------------------------------
    t11 = (
        np.vdot(psi1, V_Ey @ Vi) * np.vdot(Vi, V_kx @ psi1)
        +
        np.vdot(psi1, V_kx @ Vi) * np.vdot(Vi, V_Ey @ psi1)
    )

    t22 = (
        np.vdot(psi2, V_Ey @ Vi) * np.vdot(Vi, V_kx @ psi2)
        +
        np.vdot(psi2, V_kx @ Vi) * np.vdot(Vi, V_Ey @ psi2)
    )

    # ------------------------------------------------------
    # Off-diagonal matrix elements
    # ------------------------------------------------------
    c12 = (
        np.vdot(psi1, V_Ey @ Vi) * np.vdot(Vi, V_kx @ psi2)
        +
        np.vdot(psi1, V_kx @ Vi) * np.vdot(Vi, V_Ey @ psi2)
    )

    # ------------------------------------------------------
    # Effective Hamiltonian contributions
    # ------------------------------------------------------
    m11 = t11 / d1
    m22 = t22 / d2

    m12 = c12 * d_sym
    m21 = np.conj(c12) * d_sym

    return Ei, m11, m12, m21, m22

# ==========================================================
# Stage 1 : Valence-band states
# ==========================================================
for i in range(len(E_B_sorted)):

    Ei, m11, m12, m21, m22 = add_contribution(
        E_B_sorted[i],
        V_B_sorted[:, i]
    )

    splittings.append([
        Ei,
        m11,
        m12,
        m21,
        m22
    ])



splittings = np.array(splittings, dtype=complex)


np.savetxt(
    f"matrix_elements_H11_r_N{N}_numv2_{num_v_bands_2}_v2_{sigma_v2}_numv1_{num_v_bands_1}_v1_{sigma_v1}_numcb_{num_c_bands}_c_{sigma_c}.dat",
    np.real(splittings)
)

np.savetxt(
    f"matrix_elements_H11_i_N{N}_numv2_{num_v_bands_2}_v2_{sigma_v2}_numv1_{num_v_bands_1}_v1_{sigma_v1}_numcb_{num_c_bands}_c_{sigma_c}.dat",
    np.imag(splittings)
)

print(f"Saved : matrix_elements_H11_r_N{N}_numv2_{num_v_bands_2}_v2_{sigma_v2}_numv1_{num_v_bands_1}_v1_{sigma_v1}_numcb_{num_c_bands}_c_{sigma_c}.dat")
print(f"Saved : matrix_elements_H11_i_N{N}_numv2_{num_v_bands_2}_v2_{sigma_v2}_numv1_{num_v_bands_1}_v1_{sigma_v1}_numcb_{num_c_bands}_c_{sigma_c}.dat")

# ---------------------------------
# Timer
# ---------------------------------
t1 = time.time()
mins, secs = divmod(t1 - t0, 60)

print(f"time taken : {mins} min , {secs:.2f} sec")