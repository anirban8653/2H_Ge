import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
import time
from multiprocessing import Pool
from tqdm import tqdm

# =========================
# TIMER START
# =========================
t0 = time.time()

# =========================
# Import Hamiltonian blocks
# =========================
# from hamiltonian_mathematica_april import gso_50
from Hamiltonian_mathematica import gso

# =========================
# Parameters
# =========================

N = 80
Nband = 10
Ny = Nz = N
nk = 25
L = 300
ncore = 4
flag = 1

sigma_c = 0.16
sigma_v = 0.13


kpoints = np.linspace(
    -0.0028,
    0.0028,
    nk,
    endpoint=True
)

# Electric fields in the internal units used by the Hamiltonian.
# Multiplication by 1e4 converts them to V/µm.
Ef_values = np.linspace(0e-4, 15.0e-4, 15)
# Ef_values = np.linspace(11.1e-4, 15.5e-4, 5)
# Ef_values = Ef_values[11:12]

# Ef_values = [
#    0.0,
#    0.3e-5,
#    0.6e-5,
#     0.9e-5,
# ]




lat = kwant.lattice.square(norbs=Nband)


# =========================
# System builder
# =========================
def make_system(kx, Ey):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # Precompute gso blocks for this kx
    gso_cache = {}

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:

            gso_cache[(dy, dz)] = gso(
                a,
                kx,
                dy,
                dz
            )

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            if flag == 1:

                V = -Ey * (
                    (y + 1) * a - L / 2
                )
            elif flag == 0:
                V = -Ey * (
                    (z + 1) * a - L / 2
                )                

            syst[lat(y, z)] = (
                V * np.eye(Nband)
                + gso_cache[(0, 0)]
            )

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:

            if dy == 0 and dz == 0:
                continue

            syst[
                kwant.builder.HoppingKind(
                    (dy, dz),
                    lat,
                    lat
                )
            ] = gso_cache[(dy, dz)]

    return syst.finalized()


# =========================
# Diagonalization for one kx
# =========================
def process_kx(kx):
    """
    Compute two valence-band eigenvalues near 0 eV,
    two conduction-band eigenvalues near 0.3 eV,
    and the direct band gap for one value of kx.

    Returns
    -------
    [
        kx,
        valence1,
        valence2,
        conduction1,
        conduction2,
        band_gap
    ]
    """

    syst = make_system(kx, Ef)

    H = syst.hamiltonian_submatrix(
        sparse=True
    )

    num_eigenvectors = 2

    # Valence-band states near 0 eV
    w_valence = eigsh(
        H,
        k=num_eigenvectors,
        sigma=sigma_v,
        return_eigenvectors=False
    )

    w_valence = np.sort(
        np.real(w_valence)
    )

    valence1 = w_valence[0]
    valence2 = w_valence[1]
    valence_band_maximum = np.max(w_valence)

    # Conduction-band states near 0.3 eV
    w_conduction = eigsh(
        H,
        k=num_eigenvectors,
        sigma=sigma_c,
        return_eigenvectors=False
    )

    w_conduction = np.sort(
        np.real(w_conduction)
    )

    conduction1 = w_conduction[0]
    conduction2 = w_conduction[1]
    conduction_band_minimum = np.min(w_conduction)

    # Direct gap at this kx
    band_gap = (
        conduction_band_minimum
        - valence_band_maximum
    )

    return [
        kx,
        valence1,
        valence2,
        conduction1,
        conduction2,
        band_gap
    ]


# =========================
# Main calculation
# =========================
print(
    f"Running over {len(kpoints)} k-points "
    f"using {ncore} cores..."
)


gap_summary = []

for Ef in Ef_values:

    energylist = []

    with Pool(ncore) as pool:

        results = pool.imap(
            process_kx,
            kpoints
        )

        for result in tqdm(
            results,
            total=len(kpoints),
            desc=(
                f"Diagonalizing H with E-field "
                f"{(Ef * 1e4):.3f} V/µm"
            )
        ):
            energylist.append(result)

    energylist = np.asarray(
        energylist,
        dtype=float
    )

    if flag == 1:
        output_filename = (
            f"band_data_y_E{(Ef * 1e4):.3f}"
            f"_size{N}_nk{nk}.dat"
        )

        np.savetxt(
            output_filename,
            energylist,
            fmt="%.12e",
            header=(
                "kx  "
                "valence1  valence2  "
                "conduction1  conduction2  "
                "band_gap"
            )
        )
    else:

        output_filename = (
                    f"band_data_z_E{(Ef * 1e4):.3f}"
                    f"_size{N}_nk{nk}.dat"
                )
        
        np.savetxt(
            output_filename,
            energylist,
            fmt="%.12e",
            header=(
                "kx  "
                "valence1  valence2  "
                "conduction1  conduction2  "
                "band_gap"
            )
        )


    overall_band_gap = (
        np.min(energylist[:, 3])
        - np.max(energylist[:, 2])
    )

    gap_summary.append([
        Ef,
        Ef * 1e4,
        overall_band_gap
    ])

    print(
        f"Overall fundamental band gap for "
        f"E-field {(Ef * 1e4):.3f} V/µm: "
        f"{overall_band_gap:.12e} eV"
    )

gap_summary = np.asarray(
    gap_summary,
    dtype=float
)

if flag == 1:
    gap_output_filename = (
        f"band_gap_vs_Ey_size{N}_nk{nk}_y.dat"
    )

elif flag == 0:
    gap_output_filename = (
        f"band_gap_vs_Ey_size{N}_nk{nk}_z.dat"
    )

np.savetxt(
    gap_output_filename,
    gap_summary,
    fmt="%.12e",
    header=(
        "Ey_internal  "
        "Ey_V_per_um  "
        "fundamental_band_gap_eV"
    )
)

print("Done. Data saved.")


# =========================
# Timer
# =========================
t1 = time.time()

mins, secs = divmod(
    t1 - t0,
    60
)

print(
    f"Execution time: "
    f"{int(mins)} min {secs:.2f} sec"
)







