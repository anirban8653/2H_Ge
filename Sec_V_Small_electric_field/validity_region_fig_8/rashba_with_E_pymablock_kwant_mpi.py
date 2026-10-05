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
L = 800
Ny = Nz = N
flagy = 1
kx_list = np.linspace(0.0, 0.004, 100)
Ef_list = np.linspace(0.0, 0.15e-3, 40)


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
        sigma=0.05,
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
#---------------------------------





if rank == 0:

    print(" ")
    print("-" * 30)
    print("PYMABLOCK")
    print("-" * 30)
    print(" ")

    syst0 = make_system(0.0, 0.0)
    H0_sparse = syst0.hamiltonian_submatrix(sparse=True)

    syst1 = make_system_kx(0.0)
    V_kx = syst1.hamiltonian_submatrix(sparse=True)

    syst2 = make_system_E()
    V_Ey = syst2.hamiltonian_submatrix(sparse=True)

    H_list_sparse = [H0_sparse, V_kx, V_Ey]

    num_eigenvectors = 2

    evals, evecs = eigsh(
        H0_sparse,
        k=num_eigenvectors,
        sigma=0.05
    )

    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    psi_new = psi_new_basis(psi1, psi2, N)
    psi_new = np.array(psi_new).T

    Heff_coeffs, *_ = block_diagonalize(
        H_list_sparse,
        subspace_eigenvectors=[psi_new]
    )

    H00 = Heff_coeffs[(0, 0, 0, 0)]
    H11 = Heff_coeffs[(0, 0, 1, 1)]
    H13 = Heff_coeffs[(0, 0, 1, 3)]
    H31 = Heff_coeffs[(0, 0, 3, 1)]

    print("Pymablock done.\n")

else:
    H00 = H11 = H13 = H31 = None

H00 = comm.bcast(H00, root=0)
H11 = comm.bcast(H11, root=0)
H13 = comm.bcast(H13, root=0)
H31 = comm.bcast(H31, root=0)


# ---------------------------------
# E-field and kx sweep for Pymablock gap
# ---------------------------------

def compute_one_kx(args):

    kx_val, Ef_val = args

    H_eff_at_kx_2nd = H00 + H11 * kx_val * Ef_val

    H_eff_at_kx_4th = (
        H00
        + H11 * kx_val * Ef_val
        + H13 * kx_val * Ef_val**3
        + H31 * kx_val**3 * Ef_val
    )

    eig_2nd, _ = eigh(H_eff_at_kx_2nd)
    eig_4th, _ = eigh(H_eff_at_kx_4th)

    gap_2nd = np.abs(eig_2nd[1] - eig_2nd[0]) * 1e6
    gap_4th = np.abs(eig_4th[1] - eig_4th[0]) * 1e6

    return [L, Ef_val, kx_val, float(gap_2nd), float(gap_4th)]


if __name__ == "__main__":

    if rank == 0:
        print("\nStarting Pymablock MPI sweep...")

    tasks_all = [(kx_val, Ef_val) for Ef_val in Ef_list for kx_val in kx_list]
    tasks_local = split_tasks(tasks_all)

    local_pymablock_results = []

    for task in tqdm(tasks_local, desc=f"Pymablock Rank {rank}", disable=(rank != 0)):
        local_pymablock_results.append(compute_one_kx(task))

    gathered_pymablock_results = comm.gather(local_pymablock_results, root=0)

    if rank == 0:
        pymablock_gap_list = []

        for r in gathered_pymablock_results:
            pymablock_gap_list.extend(r)

        pymablock_gap_list = np.array(
            sorted(pymablock_gap_list, key=lambda x: (x[1], x[2]))
        )

        print("\nL | Ef | kx | gap_2nd (micro-eV) | gap_4th (micro-eV)")
        print("-" * 80)

        for row in pymablock_gap_list:
            print(
                f"{row[0]:.0f} | {row[1]:.8e} | {row[2]:.6f} | "
                f"{row[3]:.8f} | {row[4]:.8f}"
            )

        if flagy == 1:
            filename = f"Pymablock_gap_vs_Ef_kx_Y_N{N}_L{int(L/10)}_short_k_range.dat"
        else:
            filename = f"Pymablock_gap_vs_Ef_kx_Z_N{N}_L{int(L/10)}.dat"

        np.savetxt(
            filename,
            pymablock_gap_list,
            header="L Ef kx gap_2nd_microeV gap_4th_microeV",
            comments=""
        )

        print(f"\nExported: {filename}")


comm.Barrier()

if rank == 0:
    t1 = time.time()
    mins, secs = divmod(t1 - t0, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")









