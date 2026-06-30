import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# =========================
# TIMER START
# =========================
t0 = time.time()

# =========================
# Import Hamiltonian blocks
# =========================
from Hamiltonian_mathematica_v2 import gso

# =========================
# Parameters
# =========================

N = 50
Nband = 10
Ny = Nz = N
nk = 51
L = 300

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
            V = -Ey * ((y + 1) * a - L / 2)
            syst[lat(y, z)] = V * np.eye(Nband) + gso_cache[(0, 0)]

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso_cache[(dy, dz)]

    return syst.finalized()




def process_kx(kx):
    """Compute eigvals and spin polarization for one kx."""
    syst = make_system(kx, Ef)
    H = syst.hamiltonian_submatrix(sparse=True)
    num_eigenvectors = 2
    w, v = eigsh(H, k=num_eigenvectors, sigma=0.0)
    w = np.sort(w)
    eigval1, eigval2 = w[0], w[1]
    # gap = np.abs(eigval1 - eigval2)

    return [eigval1, eigval2]


kpoints = np.linspace(-0.002, 0.002, nk, endpoint=True)
Ef_values = np.arange(0.00000, 0.000012, 0.000003)

# Ef_values = np.array([0.00005])

print(f"Running over {len(kpoints)} k-points using {10} cores...")
for Ef in Ef_values:
    energylist = []
    with Pool(10) as pool:
        for result in tqdm(pool.imap(process_kx, kpoints), total=len(kpoints),
                        desc=f"Diagonalizing H with E-field {(Ef * 1e4):.3f} V/µm"):
            energylist.append(result)
    energylist = np.array(energylist)
    np.savetxt(f"band_data_y_E{(Ef * 1e4):.3f}_size{N}_nk{nk}.dat", energylist)
    # if flagy == 1:
    #     np.savetxt(f"band_data_y_E{(Ef * 1e4):.3f}_size{size}_nk{nk}.dat", energylist)
    # if flagz == 1:  
    #     np.savetxt(f"band_data_z_E{(Ef * 1e4):.3f}_size{size}_nk{nk}.dat", energylist)

print("Done. Data saved.")



# =========================
# Timer
# =========================
t1 = time.time()
mins, secs = divmod(t1 - t0, 60)
print(f"Execution time: {int(mins)} min {secs:.2f} sec")
