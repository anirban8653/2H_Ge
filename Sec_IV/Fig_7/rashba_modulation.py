
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
from Hamiltonian_mathematica import gso, dgso, psi_new_basis
np.set_printoptions(linewidth=200, suppress=True, precision=5)




# =========================
# TIMER START
# =========================
t0 = time.time()


# ---------------------------------
# Parameters
# ---------------------------------
N = 100 
kx = 0.001 
Nband = 10

Ny = Nz = N
flagy = 1
ncore = 50
sigma_val = 0.12

if flagy == 1:
    imported_file = "valid_regions_data_y.dat"
else:
    imported_file = "valid_regions_data_z.dat"

L_E_data = np.loadtxt(imported_file)
# Columns: L [nm], E [V/um], gap [eV]. Each row is one (L, E) point to compute.
# Convert to the code's units: L in Angstrom, Ef in V/Angstrom (1 V/um = 1e-4 V/A).
L_E_data = np.atleast_2d(L_E_data)

print(f"\nBuilding system for N = {N}")


lat = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system main Hamiltonian
# ---------------------------------
def make_system(L, kx, Ef):

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
def make_system_kx(L, kx):

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


#--------------------------------------------------
# Starting Pymablock
#--------------------------------------------------


def compute_one_L(L, Ef):
    # Build the unperturbed Hamiltonian and perturbations once.
    H0_sparse = make_system(
        L, 0.0, Ef
    ).hamiltonian_submatrix(sparse=True)

    V_kx = make_system_kx(
        L, 0.0
    ).hamiltonian_submatrix(sparse=True)

    # Select the two states near zero energy.
    evals, evecs = eigsh(H0_sparse, k=2, sigma=sigma_val)

    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    psi_new = np.asarray(
        psi_new_basis(psi1, psi2, N)
    ).T

    # Compute the perturbative coefficients once for this L.
    Heff_coeffs, *_ = block_diagonalize(
        [H0_sparse, V_kx],
        subspace_eigenvectors=[psi_new],
    )

    # Pymablock returns a symbolic Zero for blocks that vanish identically
    # (e.g. the kx-linear term at Ef = 0 by inversion symmetry).

    H0 = np.asarray(Heff_coeffs[0, 0, 0], dtype=complex)
    H1 = Heff_coeffs[0, 0, 1]
    if isinstance(H1, np.ndarray):
        H1 = np.asarray(H1, dtype=complex)
    else:
        H1 = np.zeros_like(H0)

    if flagy == 1:
        alpha = np.real(H1[0, 0])
    else:
        alpha = np.imag(H1[0, 1])

    H_eff_2nd = H0 + H1 * kx

    energies_2nd = eigh(H_eff_2nd, eigvals_only=True)

    gap_2nd = abs(energies_2nd[1] - energies_2nd[0])

    return (
        L / 10,                 # nm, assuming L is in Å
        Ef * 1e4,               # V/µm
        float(alpha),           # eV·Å
        float(gap_2nd * 1e6),   # µeV
    )


def compute_one_task(args):
    return compute_one_L(*args)

if __name__ == "__main__":
    tasks = [(L_nm * 10.0, E_Vum * 1e-4) for L_nm, E_Vum, _ in L_E_data]

    with Pool(processes=min(ncore, len(tasks))) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(compute_one_task, tasks),
                total=len(tasks),
                desc="Computing Rashba vs width and field",
            )
        )

    # Sort by width, then electric field.
    rashba_list = np.array(sorted(results, key=lambda row: (row[0], row[1])))

    direction = "Y" if flagy == 1 else "Z"
    filename = (
        f"Pymablock_{direction}_kx{kx}"
        f"_valid_regions_N{N}.dat"
    )

    header = "L_nm E_V_um alpha_eV_A gap_ueV_2nd"
    np.savetxt(filename, rashba_list, header=header)

    print(f"\nExported: {filename}")
    print(header)
    for row in rashba_list:
        print(row)

    mins, secs = divmod(time.time() - t0, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")









