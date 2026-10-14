import time
from functools import partial
from multiprocessing import Pool

import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from Hamiltonian_mathematica import gso

# =========================
# Parameters
# =========================
N = 30                 # sites along y and z
Ny = Nz = N
Nband = 10             # orbitals per site
L = 300                # cross-section size
nk = 25
ncore = 4
field_axis = "y"       # direction of the applied E-field: "y" or "z"

sigma_v = 0.13         # shift-invert target for valence states (eV)
sigma_c = 0.16         # shift-invert target for conduction states (eV)
num_eigs = 2

kpoints = np.linspace(-0.0028, 0.0028, nk)
Ef_values = np.linspace(0.0, 15.0e-4, 15)

lat = kwant.lattice.square(norbs=Nband)
HEADER_BANDS = "kx  valence1  valence2  conduction1  conduction2  band_gap"
HEADER_GAP = "Ey_internal  Ey_V_per_um  fundamental_band_gap_eV"


# =========================
# System builder
# =========================
def make_system(kx, Ef):
    syst = kwant.Builder()
    a = L / (Ny + 1)

    blocks = {(dy, dz): gso(a, kx, dy, dz)
              for dy in (-1, 0, 1) for dz in (-1, 0, 1)}

    # Onsite: Hamiltonian block plus linear electrostatic potential
    for y in range(Ny):
        for z in range(Nz):
            coord = y if field_axis == "y" else z
            V = -Ef * ((coord + 1) * a - L / 2)
            syst[lat(y, z)] = V * np.eye(Nband) + blocks[(0, 0)]

    # Hoppings to all 8 neighbours
    for (dy, dz), block in blocks.items():
        if (dy, dz) != (0, 0):
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = block

    return syst.finalized()


# =========================
# Diagonalization for one kx
# =========================
def process_kx(kx, Ef):
    """Return [kx, valence1, valence2, conduction1, conduction2, direct_gap]."""
    H = make_system(kx, Ef).hamiltonian_submatrix(sparse=True)

    w_v = np.sort(np.real(eigsh(H, k=num_eigs, sigma=sigma_v,
                                return_eigenvectors=False)))
    w_c = np.sort(np.real(eigsh(H, k=num_eigs, sigma=sigma_c,
                                return_eigenvectors=False)))

    return [kx, w_v[0], w_v[1], w_c[0], w_c[1], w_c.min() - w_v.max()]


# =========================
# Main calculation
# =========================
def main():
    t0 = time.time()
    print(f"Running over {nk} k-points using {ncore} cores...")

    gap_summary = []
    for Ef in Ef_values:
        E_um = Ef * 1e4  # V/µm

        with Pool(ncore) as pool:
            results = pool.imap(partial(process_kx, Ef=Ef), kpoints)
            energylist = np.asarray(
                list(tqdm(results, total=nk,
                          desc=f"Diagonalizing H with E-field {E_um:.3f} V/µm")),
                dtype=float,
            )

        np.savetxt(f"band_data_{field_axis}_E{E_um:.3f}_size{N}_nk{nk}.dat",
                   energylist, fmt="%.12e", header=HEADER_BANDS)

        overall_gap = energylist[:, 3].min() - energylist[:, 2].max()
        gap_summary.append([Ef, E_um, overall_gap])
        print(f"Overall fundamental band gap for E-field {E_um:.3f} V/µm: "
              f"{overall_gap:.12e} eV")

    np.savetxt(f"band_gap_vs_Ey_size{N}_nk{nk}_{field_axis}.dat",
               np.asarray(gap_summary), fmt="%.12e", header=HEADER_GAP)
    print("Done. Data saved.")

    mins, secs = divmod(time.time() - t0, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")


if __name__ == "__main__":
    main()
