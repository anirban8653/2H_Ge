"""
Signed g_xx of the top valence Kramers pair (vb0) of a 2H-Ge nanowire vs
cross-section size L, at E = 0, B || x (along the wire).

Sign convention (spin-referenced, electron picture)
---------------------------------------------------
H_Z = (muB/2) g_xx B sigma_x, with the pseudospin frame fixed so that the
projected TRUE spin is  P^dag S_x P = +s_x sigma_x,  s_x > 0.
Equivalently: g_xx > 0  <=>  the upper Zeeman state has the larger <S_x>
(free-electron-like). This is basis independent and needs only B || x.

Method
------
1. Kramers pair P at B = 0 (kwant, gso blocks, as in gfactor_Bx_1.py).
2. Exact first-order orbital Zeeman operator from the Peierls phases,
   electron charge (-e), gauge A = (0, 0, B (y - y_c)):
       H_ij(B) = H_ij(0) exp(+i e B a^2 y_mid (z_j - z_i) / hbar)
       dH_orb/dB = H_ij(0) * i e a^2 y_mid (z_j - z_i) / hbar
   (i = row site, j = column site; built directly from H0, no rebuild.)
3. Project:  M_orb = P^dag dH_orb/dB P,  M_spin = P^dag (g0 muB S_x) P
   g_vec = Tr(M sigma)/muB for each; axis n = g_spin / |g_spin|
   g_orb  = g_orb_vec . n,  g_spin = |g_spin_vec|,  g_xx = g_orb + g_spin
   'perp' = component of g_orb perpendicular to n (should be ~0 by symmetry).
4. Finite-B splittings of vb0, vb1, vb2 (eV); cross-check for vb0: diagonalize H(Bf) and get
   g = sign(<S_x>_upper - <S_x>_lower) * splitting / (muB Bf).

Output: gxx_signed_vs_L_N{N}.dat
"""
import os
import time
import kwant
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, block_diag, bmat
from scipy.sparse.linalg import eigsh

from Hamiltonian_mathematica_v2 import gso

# ---------------------------------------------
#               PARAMETERS
# ---------------------------------------------
N = int(os.environ.get("GF_N", 100))
Ny = Nz = N
Nband = 10
kx = 0.0

hbar = 1.05457e-34
q = 1.602e-19
muB = 5.78e-5
gspin = 2.0
Bf = 0.01                       # T, for the finite-B splittings

SIGMA_MID = 0.12                # eV, inside the gap; take the highest pair below it

size_list = np.array([100, 120, 140, 160, 180] + list(np.linspace(200, 1000, 20)))  # Angstrom

lat = kwant.lattice.square(norbs=Nband)
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex),
         np.array([[1, 0], [0, -1]], complex)]


# ---------------------------------------------
#               SPIN MATRIX S_x (10x10)
# ---------------------------------------------
def build_spin_matrix():
    sud = np.zeros((5, 5), dtype=complex)
    sud[0, 0] = sud[1, 1] = sud[4, 4] = 1
    sud[2, 3] = sud[3, 2] = 1
    sud = csr_matrix(sud)
    zero = csr_matrix((5, 5))
    return 0.5 * bmat([[zero, sud], [sud.conjugate(), zero]], format="csr")


# ---------------------------------------------
#               KWANT SYSTEM AT B = 0
# ---------------------------------------------
def make_system(L):
    a = L / (N + 1)
    blocks = {(dy, dz): gso(a, kx, dy, dz) for dy in (-1, 0, 1) for dz in (-1, 0, 1)}
    syst = kwant.Builder()
    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = blocks[(0, 0)]

    def hopping(site1, site2):
        # same block convention as gfactor_Bx_1.py: gso(y2 - y1, z2 - z1)
        (yi, zi), (yj, zj) = site1.tag, site2.tag
        return blocks[(yj - yi, zj - zi)]

    for d in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping
    return syst.finalized()


def orbital_operators(H0, tags, L):
    """dH_orb/dB_x (eV/T) and a function giving H_orb(B), both from H0."""
    a_m = L / (N + 1) * 1e-10
    H0 = coo_matrix(H0)
    si, sj = H0.row // Nband, H0.col // Nband
    y_c = (Ny - 1) / 2.0
    y_mid = 0.5 * (tags[si, 0] + tags[sj, 0]) - y_c
    dz = tags[sj, 1] - tags[si, 1]                 # z_j - z_i  (i = row)
    phi_per_T = q * a_m**2 / hbar * y_mid * dz     # phase per tesla
    dH = coo_matrix((H0.data * 1j * phi_per_T, (H0.row, H0.col)), shape=H0.shape).tocsr()

    def H_of_B(B):
        return coo_matrix((H0.data * np.exp(1j * phi_per_T * B), (H0.row, H0.col)),
                          shape=H0.shape).tocsr()
    return dH, H_of_B


def top_pair(H, k=6):
    w, v = eigsh(H, k=k, sigma=SIGMA_MID, which="LM")
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    below = np.where(w < SIGMA_MID)[0]
    return w, v, below


def pauli_vector(M):
    """    
    M_x = P^dag dH/dB P  (2x2)
    g = Tr(M_x siga_x)/mu_B
    """
    return np.array([np.trace(M @ s).real for s in PAULI]) / muB


# ---------------------------------------------
#               MAIN
# ---------------------------------------------
def main():
    Sx = None
    rows = []
    for L in size_list:
        t0 = time.time()
        fsyst = make_system(L)
        tags = np.array([s.tag for s in fsyst.sites], dtype=float)
        H0 = fsyst.hamiltonian_submatrix(sparse=True).tocsr()

        if Sx is None or Sx.shape != H0.shape:
            Sx = block_diag([build_spin_matrix()] * (H0.shape[0] // Nband), format="csr")
        dH_orb, H_of_B = orbital_operators(H0, tags, L)

        # --- projection at B = 0 ---
        w, v, below = top_pair(H0)
        P, _ = np.linalg.qr(v[:, below[-2:]])

        g_orb_vec = pauli_vector(P.conj().T @ (dH_orb @ P))
        g_spin_vec = pauli_vector(P.conj().T @ (gspin * muB * (Sx @ P)))
        
        
        # Then: n = g_spin_vec/|g_spin_vec|,  g_spin = |g_spin_vec|,
        #       g_orb = g_orb_vec . n,  g_xx = g_orb + g_spin  (signed g-factor along B_x)
        n = g_spin_vec / np.linalg.norm(g_spin_vec)
        g_orb = g_orb_vec @ n
        g_spin = np.linalg.norm(g_spin_vec)
        g_xx = g_orb + g_spin
        perp = np.linalg.norm(g_orb_vec - g_orb * n)
        gap_next = w[below[-2]] - w[below[-3]]

        # --- finite-B splittings: vb0 (signed g cross-check), vb1, vb2 ---
        HB = H_of_B(Bf) + gspin * muB * Bf * Sx
        wB, vB, bB = top_pair(HB, k=8)
        
        lo, hi = bB[-2], bB[-1]
        s_lo = np.vdot(vB[:, lo], Sx @ vB[:, lo]).real
        s_hi = np.vdot(vB[:, hi], Sx @ vB[:, hi]).real
        g_split = np.sign(s_hi - s_lo) * (wB[hi] - wB[lo]) / (muB * Bf)
        split_vb0 = wB[bB[-1]] - wB[bB[-2]]
        split_vb1 = wB[bB[-3]] - wB[bB[-4]]
        split_vb2 = wB[bB[-5]] - wB[bB[-6]]

        rows.append([L / 10, g_xx, g_orb, g_spin, perp, g_split, w[below[-1]], gap_next,
                     split_vb0, split_vb1, split_vb2])
        print(f"L={L/10:6.2f} nm  g_xx={g_xx:+.5f}  (orb {g_orb:+.5f}, spin {g_spin:+.5f}, "
              f"perp {perp:.1e})  splitting check {g_split:+.5f}  "
              f"E_vb0={w[below[-1]]*1e3:.3f} meV  splittings vb0/vb1/vb2 = "
              f"{split_vb0*1e6:.4f}/{split_vb1*1e6:.4f}/{split_vb2*1e6:.4f} ueV  ({time.time()-t0:.0f} s)", flush=True)

    data = np.array(rows)
    out = f"gxx_signed_vs_L_N{N}.dat"
    np.savetxt(out, data, header="L(nm)  g_xx  g_orb  g_spin  perp  g_xx_splitting  E_vb0(eV)  gap_next(eV)  "
                     "split_vb0(eV)  split_vb1(eV)  split_vb2(eV)")
    print(f"Saved {out}")

    for i in np.where(np.diff(np.sign(data[:, 1])) != 0)[0]:
        print(f"g_xx changes sign between L = {data[i,0]:.1f} and {data[i+1,0]:.1f} nm")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(data[:, 0], data[:, 1], "o-", label=r"$g_{xx}$")
    ax.plot(data[:, 0], data[:, 2], "--", lw=1, label="orbital")
    ax.plot(data[:, 0], data[:, 3], ":", lw=1, label="spin")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$L$ (nm)")
    ax.set_ylabel(r"$g_{xx}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"gxx_signed_vs_L_N{N}.png", dpi=200)


if __name__ == "__main__":
    main()
