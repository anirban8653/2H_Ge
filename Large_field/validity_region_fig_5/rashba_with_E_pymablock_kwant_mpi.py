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
# from multiprocessing import Pool
from mpi4py import MPI
from tqdm import tqdm
from Hamiltonian_mathematica_v2 import gso, dgso, psi_new_basis
np.set_printoptions(linewidth=200, suppress=True, precision=5)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------
# Helper: split tasks among MPI ranks
# ---------------------------------
def split_tasks(tasks):
    return tasks[rank::size]


# =========================
# TIMER START
# =========================
t0 = time.time()


# ---------------------------------
# Parameters
# ---------------------------------
N = 50  
Nband = 10
L = 300
Ny = Nz = N
flagy = 1
sigma_val = 0.15
kx_list = np.linspace(0.0, 0.004, 100)
Ef_list = np.linspace(0.0, 1.5e-3, 40)

if rank == 0:
    print(f"\nBuilding system for N = {N}")


lat = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system main Hamiltonian
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





# ---------------------------------
# Build system dH/dkx
# ---------------------------------

lat_kx = kwant.lattice.square(norbs=Nband)

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




# ---------------------------------
# Build system H_E
# ---------------------------------

lat_E = kwant.lattice.square(norbs=Nband)

def make_system_E():

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            if flagy == 1:
                V = - ((y + 1) * a - L / 2)
            else:
                V = - ((z + 1) * a - L / 2)
            syst[lat_E(y, z)] = V * np.eye(Nband) 

    return syst.finalized()


# ---------------------------------
# Parallel kx sweep for each E field
# ---------------------------------

def compute_gap_for_kx(args):
    kx_val, Ef_val = args

    syst = make_system(kx_val, Ef_val)
    H = syst.hamiltonian_submatrix(sparse=True)

    vals = eigsh(
        H,
        k=2,
        sigma=sigma_val,
        return_eigenvectors=False
    )

    vals = np.sort(vals)

    gap_microeV = (vals[1] - vals[0]) * 1e6

    return [L, Ef_val, kx_val, vals[0], vals[1], gap_microeV]

if __name__ == "__main__":

    if rank == 0:
        print("\nStarting numerical MPI sweep...")

    tasks_all = [(kx_val, Ef_val) for Ef_val in Ef_list for kx_val in kx_list]
    tasks_local = split_tasks(tasks_all)

    local_results = []

    for task in tqdm(tasks_local, desc=f"Rank {rank}", disable=(rank != 0)):
        local_results.append(compute_gap_for_kx(task))

    gathered_results = comm.gather(local_results, root=0)

    if rank == 0:
        all_results = []
        for r in gathered_results:
            all_results.extend(r)

        all_results = np.array(
            sorted(all_results, key=lambda x: (x[1], x[2]))
        )

        print("\nL | Ef | kx | Gap (micro-eV)")
        print("-" * 60)

        for row in all_results:
            print(f"{row[0]:.0f} | {row[1]:.8e} | {row[2]:.6f} | {row[3]:.8f}")

        if flagy == 1:
            filename = f"gap_vs_Ef_kx_N{N}_L{int(L/10)}_y.dat"
        else:
            filename = f"gap_vs_Ef_kx_N{N}_L{int(L/10)}_z.dat"

        np.savetxt(
            filename,
            all_results,
            header="L Ef kx gap_microeV",
            comments=""
        )

        print(f"\nSaved: {filename}")


#---------------------------------
# Pymablock block diagonalization
# Ef included in H0, only kx perturbative
# MUMPS/Pymablock only on rank 0
# MPI parallelization over (kx, Ef) after coefficients are computed
#---------------------------------

from pymablock.series import zero as pymablock_zero


def to_dense_eff_matrix(A, dim):
    """
    Convert Pymablock coefficient into a normal dense NumPy matrix.
    Handles Pymablock Zero objects and scalar coefficients.
    """

    if A is pymablock_zero:
        return np.zeros((dim, dim), dtype=complex)

    if A.__class__.__name__.lower() == "zero":
        return np.zeros((dim, dim), dtype=complex)

    A = np.asarray(A, dtype=complex)

    if A.ndim == 0:
        return A.item() * np.eye(dim, dtype=complex)

    return A


def compute_pymablock_coefficients(Ef_val):
    """
    For fixed Ef:

        H(kx, Ef) = H0(Ef) + kx * V_kx

    Ef is included fully in the unperturbed Hamiltonian.
    Only kx is treated perturbatively.
    """

    syst0 = make_system(0.0, Ef_val)
    H0_sparse = syst0.hamiltonian_submatrix(sparse=True)

    syst1 = make_system_kx(0.0)
    V_kx = syst1.hamiltonian_submatrix(sparse=True)

    # Only one perturbation: kx
    H_list_sparse = [H0_sparse, V_kx]

    num_eigenvectors = 2

    evals, evecs = eigsh(
        H0_sparse,
        k=num_eigenvectors,
        sigma=sigm_val,
        which="LM"
    )

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    psi_new = psi_new_basis(psi1, psi2, N)
    psi_new = np.asarray(psi_new, dtype=complex).T

    full_dim = H0_sparse.shape[0]

    if psi_new.shape != (full_dim, 2):
        raise ValueError(
            f"Wrong psi_new shape: {psi_new.shape}. "
            f"Expected {(full_dim, 2)}."
        )

    # Re-orthonormalize final 2-state subspace
    psi_new, _ = qr(psi_new, mode="economic")

    Heff_coeffs, *_ = block_diagonalize(
        H_list_sparse,
        subspace_eigenvectors=[psi_new]
    )

    H0_eff = Heff_coeffs[(0, 0, 0)]
    H1_eff = Heff_coeffs[(0, 0, 1)]

    dim_eff = psi_new.shape[1]

    H0_eff = to_dense_eff_matrix(H0_eff, dim_eff)
    H1_eff = to_dense_eff_matrix(H1_eff, dim_eff)

    return H0_eff, H1_eff


def compute_one_kx_from_precomputed(args):
    """
    Cheap MPI-parallel part.

    Uses precomputed H0_eff(Ef), H1_eff(Ef), then evaluates

        H_eff(kx, Ef) = H0_eff(Ef) + kx * H1_eff(Ef)
    """

    kx_val, iEf = args

    Ef_val = Ef_list[iEf]

    H0_eff = coeffs_by_iEf[iEf]["H0"]
    H1_eff = coeffs_by_iEf[iEf]["H1"]

    H_eff = H0_eff + H1_eff * kx_val

    # Remove tiny numerical non-Hermitian error
    H_eff = 0.5 * (H_eff + H_eff.conj().T)

    eig_eff = np.linalg.eigvalsh(H_eff)

    gap_eff = np.abs(eig_eff[1] - eig_eff[0]) * 1e6

    return [L, Ef_val, kx_val, float(gap_eff)]


if __name__ == "__main__":

    if rank == 0:
        print(" ")
        print("-" * 30)
        print("PYMABLOCK")
        print("-" * 30)
        print(" ")
        print("\nStarting Pymablock coefficient calculation...")
        print("Ef is included in H0.")
        print("Only kx is treated as perturbation.")
        print("Pymablock/MUMPS will run only on rank 0.")
        print(f"Using {size} MPI ranks for the final kx-Ef sweep.")

    # --------------------------------------------------
    # Step 1: rank 0 computes all Pymablock coefficients
    # --------------------------------------------------
    if rank == 0:

        coeffs_by_iEf = {}

        for iEf, Ef_val in enumerate(
            tqdm(Ef_list, desc="Pymablock coefficients")
        ):

            H0_eff, H1_eff = compute_pymablock_coefficients(Ef_val)

            coeffs_by_iEf[iEf] = {
                "Ef": Ef_val,
                "H0": H0_eff,
                "H1": H1_eff,
            }

        print("\nPymablock coefficients computed on rank 0.\n")

    else:
        coeffs_by_iEf = None

    # --------------------------------------------------
    # Step 2: broadcast small 2x2 coefficient matrices
    # --------------------------------------------------
    coeffs_by_iEf = comm.bcast(coeffs_by_iEf, root=0)

    # --------------------------------------------------
    # Step 3: parallelize cheap 2x2 diagonalization
    # over the full (kx, Ef) grid
    # --------------------------------------------------
    if rank == 0:
        print("\nStarting MPI sweep over kx and Ef using precomputed coefficients...")

    tasks_all = [
        (kx_val, iEf)
        for iEf in range(len(Ef_list))
        for kx_val in kx_list
    ]

    tasks_local = split_tasks(tasks_all)

    local_pymablock_results = []

    for task in tqdm(
        tasks_local,
        desc=f"Pymablock Rank {rank}",
        disable=(rank != 0)
    ):
        local_pymablock_results.append(compute_one_kx_from_precomputed(task))

    gathered_pymablock_results = comm.gather(local_pymablock_results, root=0)

    if rank == 0:

        pymablock_gap_list = []

        for r in gathered_pymablock_results:
            pymablock_gap_list.extend(r)

        pymablock_gap_list = np.array(
            sorted(pymablock_gap_list, key=lambda x: (x[1], x[2]))
        )

        print("\nL | Ef | kx | gap_eff (micro-eV)")
        print("-" * 80)

        for row in pymablock_gap_list:
            print(
                f"{row[0]:.0f} | {row[1]:.8e} | {row[2]:.6f} | "
                f"{row[3]:.8f}"
            )

        if flagy == 1:
            filename = f"Pymablock_gap_vs_Ef_kx_Y_N{N}_L{int(L/10)}_kx_only.dat"
        else:
            filename = f"Pymablock_gap_vs_Ef_kx_Z_N{N}_L{int(L/10)}_kx_only.dat"

        np.savetxt(
            filename,
            pymablock_gap_list,
            header="L Ef kx gap_eff_microeV",
            comments=""
        )

        print(f"\nExported: {filename}")


comm.Barrier()

if rank == 0:
    t1 = time.time()
    mins, secs = divmod(t1 - t0, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")
