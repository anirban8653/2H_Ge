import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from Hamiltonian_real_april import build_H_real
import time
# np.set_printoptions(precision=3, suppress=True, linewidth=150)
start_time = time.time()


# =========================
# Parameters
# =========================
N = 100
Nband = 10
L = 300
kx_list = np.linspace(-0.02, 0.02, 101)

Ny = Nz = N
lat = kwant.lattice.square(norbs=Nband)

# =========================
# Worker function
# =========================
def solve_for_kx(kx):

    # Build Hamiltonian blocks
    gso = build_H_real(kx, N, L)

    # Build system
    syst = kwant.Builder()

    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso[(0, 0)]

    hopping_dirs = [
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1)]

    for dy, dz in hopping_dirs:
        syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso[(dy, dz)]

    syst = syst.finalized()

    # Sparse Hamiltonian
    H = syst.hamiltonian_submatrix(sparse=True)

    # Diagonalize near zero
    vals = eigsh(H, k=24, sigma=0.0, return_eigenvectors=False)
    vals = np.sort(vals)

    return kx, vals


# =========================
# Parallel execution
# =========================
if __name__ == "__main__":

    nproc = max(1, cpu_count() - 2)  # leave 2 cores free
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        results = list(tqdm(pool.imap(solve_for_kx, kx_list), total=len(kx_list)))

    # Sort results by kx
    results.sort(key=lambda x: x[0])

    # Extract into arrays
    kx_vals = np.array([r[0] for r in results])
    eig_vals = np.array([r[1] for r in results])  

    # Combine into single array: [kx | eigenvalues]
    final_array = np.column_stack((kx_vals, eig_vals))

    print("\nFinal array shape:", final_array.shape)
    # print("First few rows:\n", final_array[:5] * np.array([1,1000,1000,1000,1000]))

    # =========================
    # Save to file
    # =========================
    np.savetxt(
        f"band_kx_N{N}_L{L}.dat",
        final_array,
        fmt="%.8e"
    )

    end_time = time.time()
    # print minute and seconds
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"\nTotal execution time: {minutes} minutes and {seconds} seconds")
    print("Done!")
