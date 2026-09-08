
import os
# import time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import kwant
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
import time
from scipy.sparse.linalg import eigsh  # <-- We will use this
from pymablock import block_diagonalize
from multiprocessing import Pool
from tqdm import tqdm
from Hamiltonian_mathematica_v2 import gso, dgso, psi_new_basis
np.set_printoptions(linewidth=200, suppress=True, precision=5)




# =========================
# TIMER START
# =========================
t0 = time.time()


# ---------------------------------
# Parameters
# ---------------------------------
N = 50 
kx = 0.0005
Nband = 10
Ny = Nz = N
Ef = 0.15e-5
flagy = 1

print(f"\nBuilding system for N = {N}")


lat = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system main Hamiltonian
# ---------------------------------
def make_system(kx, Ef, L):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            if flagy == 1:
                V = -Ef * ((y + 1) * a - L / 2)
            else:
                V = -Ef * ((z + 1) * a - L / 2)
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
# Build system dH/dkx
# ---------------------------------
def make_system_kx(kx,L):

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
# Build system H_E
# ---------------------------------
def make_system_E(L):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            if flagy == 1:
                V = - ((y + 1) * a - L / 2)
            else:
                V = - ((z + 1) * a - L / 2)
            syst[lat(y, z)] = V * np.eye(Nband) 

    return syst.finalized()






# ---------------------------------
# Sweep L
# ---------------------------------
L_values = np.linspace(200,800,21)
results = []
for L in L_values:
    # ---------------------------------
    # Compute GS at zero field (once)
    # ---------------------------------
    syst0 = make_system(kx, 0.0, L)
    H0 = syst0.hamiltonian_submatrix(sparse=True)
    vals0 = eigsh(H0, k=2, sigma=0.0, return_eigenvectors=False)
    vals0 = np.sort(vals0)

    GS_0_meV = vals0[0] * 1000   # meV

    syst = make_system(kx,  Ef,L)
    H = syst.hamiltonian_submatrix(sparse=True)

    vals = eigsh(H, k=2, sigma=0.0, return_eigenvectors=False)
    vals = np.sort(vals)

    GS1_meV = vals[0] * 1000
    GS2_meV = vals[1] * 1000

    # Energy difference (gap) in micro eV
    shift_microeV = (GS2_meV - GS1_meV) * 1000

    results.append([L/10, GS1_meV, GS2_meV, shift_microeV])

    print(f"L: {(L/10)} nm")

results = np.array(results)


print("\nEf (V/A) | GS1 (meV) | GS2 (meV) | Gap (micro-eV)")
print("-" * 65)

for row in results:
    print(f"{row[0]:.2e} | {row[1]:<12.6f} | {row[2]:<12.6f} | {row[3]:<12.6f}")

if flagy == 1:
    np.savetxt(f"numerical_splitting_E_{(Ef*10000):.2f}_N{N}_y.dat", results, header="Ef (V/um) | GS1 (meV) | GS2 (meV) | Gap (micro-eV)", comments='')
    print(f"numerical data saved: numerical_splitting_N{N}_L{L}_y.dat")
else:
    np.savetxt(f"numerical_splitting_N{N}_L{L}_z.dat", results, header="Ef (V/um) | GS1 (meV) | GS2 (meV) | Gap (micro-eV)", comments='')
    print(f"numerical data saved: numerical_splitting_E_{(Ef*10000):.2f}_N{N}_z.dat")

print(" ")
print("-"*30)
print("PYMABLOCK")
print("-"*30)
print(" ")




#--------------------------------------------------
# Starting Pymablock
#--------------------------------------------------



rashba_list = []
for L in L_values:    
    kx_value = kx

    # -----------------------------------------
    # Build Hamiltonians for this specific L
    # -----------------------------------------
    syst0 = make_system(0.0, 0.0, L)
    H0_sparse = syst0.hamiltonian_submatrix(sparse=True)

    syst1 = make_system_kx(0.0, L)
    V_kx = syst1.hamiltonian_submatrix(sparse=True)

    syst2 = make_system_E(L)
    V_Ey = syst2.hamiltonian_submatrix(sparse=True)

    H_list_sparse = [H0_sparse, V_kx, V_Ey]

    # -----------------------------------------
    # Get the two states near zero
    # -----------------------------------------
    num_eigenvectors = 2

    evals, evecs = eigsh(
        H0_sparse,
        k=num_eigenvectors,
        sigma=0.0,
        which="LM"
    )

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Optional orthogonalization
    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    # Rotate Kramers pair / make your preferred basis
    psi_new = psi_new_basis(psi1, psi2, N)
    psi_new = np.array(psi_new).T

    # -----------------------------------------
    # Pymablock for this L
    # -----------------------------------------
    Heff_coeffs, *_ = block_diagonalize(
        H_list_sparse,
        subspace_eigenvectors=[psi_new]
    )

    H00 = Heff_coeffs[(0, 0, 0, 0)]
    H11 = Heff_coeffs[(0, 0, 1, 1)]
    # H13 = Heff_coeffs[(0, 0, 1, 3)]
    # H31 = Heff_coeffs[(0, 0, 3, 1)]

    # -----------------------------------------
    # Extract Rashba and gap
    # -----------------------------------------
    if flagy == 1:
        alpha = np.real(H11[0, 0]) * Ef
    else:
        alpha = np.imag(H11[0, 1]) * Ef

    H_eff_at_kx_2nd = (
        H00
        + H11 * kx_value * Ef
    )

    # H_eff_at_kx_4th = (
    #     H00
    #     + H11 * kx_value * Ef
    #     + H13 * kx_value * Ef**3
    #     + H31 * kx_value**3 * Ef
    # )

    modified_eigenvalues_2nd, _ = eigh(H_eff_at_kx_2nd)
    # modified_eigenvalues_4th, _ = eigh(H_eff_at_kx_4th)

    gap_2nd = np.abs(modified_eigenvalues_2nd[1] - modified_eigenvalues_2nd[0])
    # gap_4th = np.abs(modified_eigenvalues_4th[1] - modified_eigenvalues_4th[0])
    

    rashba_list.append([L / 10, float(alpha), float(gap_2nd * 1e6)])#, float(gap_4th * 1e6)])

rashba_list =  np.array(rashba_list)

if flagy == 1:
    np.savetxt(f"Pymablock_Y_kx{kx_value}_E_{(Ef*10000):.3f}_N{N}.dat",
            rashba_list,
            header="L  alpha(eV.A)  gap_ueV_2nd  gap_ueV_4th")
    print(f"Exported: Pymablock_Y_kx{kx_value}_E_{(Ef*10000):.3f}_N{N}.dat")
else:
    np.savetxt(f"Pymablock_Z_kx{kx_value}_E_{(Ef*10000):.3f}_N{N}.dat",
            rashba_list,
            header="L  alpha(eV.A)  gap_ueV_2nd  gap_ueV_4th")
    print(f"Exported: Pymablock_Z_kx{kx_value}_E_{(Ef*10000):.3f}_N{N}.dat")



print("\n L_nm | alpha(eV.A) | gap_ueV_2nd |  relative error")
for i in range(len(rashba_list)):
    re_err = abs(rashba_list[i,2] - results[i,3])/results[i,3] * 100
    print(np.concatenate((rashba_list[i], [re_err])))

# =========================
# Timer
# =========================
t1 = time.time()
mins, secs = divmod(t1 - t0, 60)
print(f"Execution time: {int(mins)} min {secs:.2f} sec")






