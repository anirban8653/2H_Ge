import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
from Hamiltonian_mathematica_v2 import gso

# =============================================================================
#   Parameters
# =============================================================================
N     = 100
Nband = 10
NORB  = 5       # spatial orbitals per spin channel
L     = 300
Ny = Nz = N

NUM_VB_2,sigma_v2 = 250, -0.55
NUM_VB_1, sigma_v1 = 20, 0
NUM_CB, sigma_c = 2, 0.65

BULK_H = np.array([
    [ 0.632,      0.,          0.,          0.,          0.,          0.,          0.,          0.,          0.,          0.        ],
    [ 0.,         0.298,       0.,          0.,          0.,          0.,          0.,          0.,          0.,          0.        ],
    [ 0.,         0.,         -0.,          0.,          0.,          0.,          0.,          0.,          0.,          0.        ],
    [ 0.,         0.,          0.,         -0.1868,      0.,          0.,          0.,          0.,          0.,          0.12841059],
    [ 0.,         0.,          0.,          0.,         -0.3622,      0.,          0.,          0.,          0.12841059,  0.        ],
    [ 0.,         0.,          0.,          0.,          0.,          0.632,       0.,          0.,          0.,          0.        ],
    [ 0.,         0.,          0.,          0.,          0.,          0.,          0.298,       0.,          0.,          0.        ],
    [ 0.,         0.,          0.,          0.,          0.,          0.,          0.,         -0.,          0.,          0.        ],
    [ 0.,         0.,          0.,          0.,          0.12841059,  0.,          0.,          0.,         -0.1868,      0.        ],
    [ 0.,         0.,          0.,          0.12841059,  0.,          0.,          0.,          0.,          0.,         -0.3622    ],
], dtype=complex)

# =============================================================================
#   System builder
# =============================================================================
def make_system(kx):
    lat  = kwant.lattice.square(norbs=Nband)
    syst = kwant.Builder()
    a    = L / (Ny + 1)

    gso_cache = {(dy, dz): gso(a, kx, dy, dz) for dy in [-1, 0, 1] for dz in [-1, 0, 1]}

    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso_cache[(0, 0)]

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso_cache[(dy, dz)]

    return syst.finalized()

# =============================================================================
#   Probability density functions
# =============================================================================
def prob_den(psi, Ny, Nz, Nband):
    """Total (orbital-summed) probability density -> shape (Ny, Nz)."""
    return (np.abs(psi)**2).reshape((Ny * Nz, Nband)).sum(axis=1).reshape((Ny, Nz))


def prob_den_orbital(psi, Ny, Nz, Nband):
    """Orbital-resolved probability density -> shape (Ny*Nz, Nband).

    Interleaved spin basis:
        col 2k   -> orbital k, spin-up
        col 2k+1 -> orbital k, spin-down
    """
    assert np.isclose(np.sum(np.abs(psi)**2), 1.0), "Eigenvector not normalised"
    return (np.abs(psi)**2).reshape((Ny * Nz, Nband))

# =============================================================================
#   Overlap with bulk states
# =============================================================================
def overlap_function(nanowire_evecs, nano_idx, bulk_evecs, bulk_idx, N, Nband):
    """Overlap between one nanowire eigenstate and one bulk eigenstate."""
    psi_nano      = np.conj(nanowire_evecs[:, nano_idx]).reshape((N**2, Nband))
    psi_bulk      = bulk_evecs[:, bulk_idx]
    site_overlaps = psi_nano @ psi_bulk
    return np.sum(np.abs(site_overlaps)**2)

# =============================================================================
#   Main
# =============================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Build Hamiltonian
    # ------------------------------------------------------------------
    print(f"Building system for N = {N} ...")
    H0 = make_system(kx=0).hamiltonian_submatrix(sparse=True)

    # ------------------------------------------------------------------
    # Diagonalise  -  valence bands (two energy windows) + conduction
    # ------------------------------------------------------------------
    print("Diagonalising ...")
    E_vb1, V_vb1 = eigsh(H0, k=NUM_VB_1, sigma=sigma_v1,   which="LM")
    E_vb2, V_vb2 = eigsh(H0, k=NUM_VB_2, sigma=sigma_v2, which="LM")
    E_cb,  V_cb  = eigsh(H0, k=NUM_CB,   sigma=sigma_c,  which="LM")

    # -----------------------------
    # Sort each eigsh output
    # -----------------------------
    idx_v1 = np.argsort(E_vb1)
    idx_v2 = np.argsort(E_vb2)
    idx_c  = np.argsort(E_cb)

    E_vb1 = E_vb1[idx_v1]
    V_vb1 = V_vb1[:, idx_v1]

    E_vb2 = E_vb2[idx_v2]
    V_vb2 = V_vb2[:, idx_v2]

    E_cb = E_cb[idx_c]
    V_cb = V_cb[:, idx_c]

    # -----------------------------
    # Combine and sort valence bands
    # -----------------------------
    E_B_val = np.concatenate((E_vb2, E_vb1))
    V_B_val = np.column_stack((V_vb2, V_vb1))

    idx_val_sort = np.argsort(E_B_val)

    E_val_sorted = E_B_val[idx_val_sort]
    V_val_sorted = V_B_val[:, idx_val_sort]

    # -----------------------------
    # Conduction already sorted
    # -----------------------------
    E_con_sorted = E_cb
    V_con_sorted = V_cb
  
    print(f"  VB range : {E_vb2[0]:.6f}  ->  {E_vb1[-1]:.6f}")
    print(f"  CB range : {E_cb[0]:.6f}   ->  {E_cb[-1]:.6f}")

    np.savetxt(f"energies_nw_vb_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat", E_B_val)
    np.savetxt(f"energies_nw_cb_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat", E_cb)
    # print(f"  Energies saved : energies_nw_vb_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat, \n                   energies_nw_cb_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat")

    # ------------------------------------------------------------------
    # Bulk Hamiltonian  (independent of VB/CB)
    # ------------------------------------------------------------------
    wb, bulk_evecs = eigh(BULK_H)
    np.savetxt(f"bulk_energy_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat", wb)
    print(f"  Bulk energies saved : bulk_energy_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat")

    # ------------------------------------------------------------------
    # Save probability densities + overlaps  (loop over VB / CB)
    # ------------------------------------------------------------------
    bands = {
        "vb": {"evecs": V_val_sorted, "n_states": NUM_VB_1 + NUM_VB_2},
        "cb": {"evecs": V_con_sorted, "n_states": NUM_CB},
    }

    for tag, cfg in bands.items():
        evecs    = cfg["evecs"]
        n_states = cfg["n_states"]

        print(f"\n  [{tag.upper()}]")

        # Total (orbital-summed) probability density
        psilist = np.array([
            prob_den(evecs[:, n], Ny, Nz, Nband).flatten()
            for n in range(n_states)
        ])
        np.savetxt(f"psi_data_{tag}_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat", psilist)
        print(f"    Prob. density saved     : psi_data_{tag}_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat")

        # Orbital-resolved probability density
        psilist_orb = np.array([
            prob_den_orbital(evecs[:, n], Ny, Nz, Nband).flatten()
            for n in range(n_states)
        ])
        np.savetxt(f"psi_data_{tag}_orbital_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat", psilist_orb)
        print(f"    Orb. density saved      : psi_data_{tag}_orbital_{N}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat")

        # Overlaps with all Nband bulk states
        overlap = np.array([
            [overlap_function(evecs, j, bulk_evecs, i, N, Nband) for i in range(Nband)]
            for j in range(n_states)
        ])
        fname = f"overlap_{tag}_N{N}_states{n_states}_o_numv2_{NUM_VB_2}_v2_{sigma_v2}_numv1_{NUM_VB_1}_v1_{sigma_v1}_numcb_{NUM_CB}_c_{sigma_c}.dat"
        np.savetxt(fname, overlap)
        print(f"    Overlap saved           : {fname}")

    print("\nDone.")
