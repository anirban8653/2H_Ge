import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
import time
from multiprocessing import Pool, freeze_support
from tqdm import tqdm

# =========================
# Import Hamiltonian blocks
# =========================
from Hamiltonian_mathematica import gso

# =========================
# Parameters
# =========================

N = 100
Nband = 10
Ny = Nz = N
nk = 51
L = 300
ncore = 30
switch = 1  # 1: y-direction, 0: z-direction

lat = kwant.lattice.square(norbs=Nband)

# =========================
# System builder (kx dependent)
# =========================
def make_system(kx, Ey):
    syst = kwant.Builder()
    a = L / (Ny + 1)

    # Precompute gso for this kx
    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            if switch == 1:
                V = -Ey * ((y + 1) * a - L / 2)
            else:
                V = -Ey * ((z + 1) * a - L / 2)
            syst[lat(y, z)] = V * np.eye(Nband) + gso_cache[(0, 0)]

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso_cache[(dy, dz)]

    return syst.finalized()




def process_kx(task):
    """Compute eigvals and spin polarization for one kx."""
    kx, Ef = task
    syst = make_system(kx, Ef)
    H = syst.hamiltonian_submatrix(sparse=True)
    num_eigenvectors = 2
    w, v = eigsh(H, k=num_eigenvectors, sigma=0.0)
    w = np.sort(w)
    eigval1, eigval2 = w[0], w[1]
    # gap = np.abs(eigval1 - eigval2)

    return [eigval1, eigval2]


def main():
    """Run the field sweep and save the two lowest bands."""
    t0 = time.time()

    kpoints = np.linspace(-0.002, 0.002, nk, endpoint=True)
    Ef_values = np.arange(0.00000, 0.000012, 0.000003)

    # Ef_values = np.array([0.00005])

    print(f"Running over {len(kpoints)} k-points using {ncore} cores...")

    for Ef in Ef_values:
        # Pass Ef explicitly so every worker receives the correct field value.
        tasks = [(kx, Ef) for kx in kpoints]

        with Pool(processes=ncore) as pool:
            energylist = list(
                tqdm(
                    pool.imap(process_kx, tasks),
                    total=len(tasks),
                    desc=f"Diagonalizing H with E-field {(Ef * 1e4):.3f} V/µm",
                )
            )

        energylist = np.asarray(energylist)

        if switch == 1:
            filename = f"band_data_y_E{(Ef * 1e4):.3f}_size{N}_nk{nk}.dat"
        else:
            filename = f"band_data_z_E{(Ef * 1e4):.3f}_size{N}_nk{nk}.dat"

        np.savetxt(filename, energylist)

    print("Done. Data saved.")

    t1 = time.time()
    mins, secs = divmod(t1 - t0, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")


if __name__ == "__main__":
    freeze_support()
    main()
