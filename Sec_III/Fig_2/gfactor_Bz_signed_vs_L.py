"""
Signed g_zz of the top valence Kramers pair (vb0) of a 2H-Ge nanowire vs
cross-section size L, at E = 0, B || z.

Sign convention (spin-referenced, electron picture; same as the g_xx script)
---------------------------------------------------------------------------
g_zz > 0  <=>  the upper Zeeman state has the larger <S_z>
          <=>  pseudospin frame with P^dag S_z P = +s_z sigma, s_z > 0.

Method
------
1. Kramers pair P at B = 0 (kwant, gs_shifted_bz with B = 0, i.e. gso).
2. Orbital Zeeman operator, electron charge (-e), gauge A = (-B (y - y_c), 0, 0):
   kx -> kx - e B (y - y_c) / hbar  (gs_shifted_bz), central difference
       dH_orb/dB = (H_orb(+Bf) - H_orb(-Bf)) / (2 Bf)
   gs_shifted_bz gets the site-2 y index for hoppings (it adds +-0.5 itself).
3. Project: g_orb_vec = Tr(P^dag dH_orb P sigma)/muB,
            g_spin_vec = Tr(P^dag (g0 muB S_z) P sigma)/muB
   n = g_spin_vec/|g_spin_vec|, g_orb = g_orb_vec.n, g_spin = |g_spin_vec|,
   g_zz = g_orb + g_spin, perp = |g_orb_vec - g_orb n| (should be ~0).
4. Finite-B splittings of vb0, vb1, vb2 (eV); cross-check for vb0: diagonalize H(Bf) and use
   g = sign(<S_z>_upper - <S_z>_lower) * splitting / (muB Bf).

Output: gzz_signed_vs_L_N{N}.dat
"""
import os
import time
import kwant
import numpy as np
from scipy.sparse import csr_matrix, block_diag, bmat, identity
from scipy.sparse.linalg import eigsh

from Hamiltonian_mathematica_v2 import gs_shifted_bz

# ---------------------------------------------
#               PARAMETERS
# ---------------------------------------------
N = int(os.environ.get("GF_N", 100))
Ny = Nz = N
Nband = 10
kx = 0.0

muB = 5.78e-5
gspin = 2.0
Bf = 0.01                       # T, finite-difference step and splitting check

SIGMA_MID = 0.12                # eV, inside the gap; take the highest pair below it

size_list = np.array([100, 120, 140, 160, 180] + list(np.linspace(200, 1000, 20)))  # Angstrom

lat = kwant.lattice.square(norbs=Nband)
PAULI = [np.array([[0, 1], [1, 0]], complex),
         np.array([[0, -1j], [1j, 0]], complex),
         np.array([[1, 0], [0, -1]], complex)]


# ---------------------------------------------
#               SPIN MATRIX S_z (10x10)
# ---------------------------------------------
def build_spin_matrix():
    one = identity(5, format="csr", dtype=complex)
    zero = csr_matrix((5, 5), dtype=complex)
    return 0.5 * bmat([[one, zero], [zero, -one]], format="csr")


# ---------------------------------------------
#               KWANT SYSTEM (orbital part only)
# ---------------------------------------------
def make_system(L, B):
    a = L / (N + 1)
    y_c = (Ny - 1) / 2.0                      # gauge origin at the center

    def onsite(site):
        y, z = site.tag
        return np.asarray(gs_shifted_bz(a, kx, 0, 0, B, y - y_c), dtype=complex)

    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag
        return np.asarray(gs_shifted_bz(a, kx, int(y1 - y2), int(z1 - z2), B, y2 - y_c),
                          dtype=complex)

    syst = kwant.Builder()
    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite
    for d in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping
    return syst.finalized()


def hamiltonian(L, B):
    return make_system(L, B).hamiltonian_submatrix(sparse=True).tocsr()


def top_pair(H, k=6):
    w, v = eigsh(H, k=k, sigma=SIGMA_MID, which="LM")
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    below = np.where(w < SIGMA_MID)[0]
    return w, v, below


def pauli_vector(M):
    return np.array([np.trace(M @ s).real for s in PAULI]) / muB


# ---------------------------------------------
#               MAIN
# ---------------------------------------------
def main():
    S = None
    rows = []
    for L in size_list:
        t0 = time.time()
        H0 = hamiltonian(L, 0.0)
        Hp = hamiltonian(L, +Bf)
        Hm = hamiltonian(L, -Bf)
        if S is None or S.shape != H0.shape:
            S = block_diag([build_spin_matrix()] * (H0.shape[0] // Nband), format="csr")
        dH_orb = ((Hp - Hm) / (2 * Bf)).tocsr()

        # --- projection at B = 0 ---
        w, v, below = top_pair(H0)
        P, _ = np.linalg.qr(v[:, below[-2:]])
        g_orb_vec = pauli_vector(P.conj().T @ (dH_orb @ P))
        g_spin_vec = pauli_vector(P.conj().T @ (gspin * muB * (S @ P)))
        n = g_spin_vec / np.linalg.norm(g_spin_vec)
        g_orb = g_orb_vec @ n
        g_spin = np.linalg.norm(g_spin_vec)
        g_zz = g_orb + g_spin
        perp = np.linalg.norm(g_orb_vec - g_orb * n)
        gap_next = w[below[-2]] - w[below[-3]]

        # --- finite-B splittings: vb0 (signed g cross-check), vb1, vb2 ---
        HB = Hp + gspin * muB * Bf * S
        wB, vB, bB = top_pair(HB, k=8)
        if len(bB) < 6:
            raise RuntimeError(f"fewer than 3 pairs below SIGMA_MID at L = {L/10} nm")
        lo, hi = bB[-2], bB[-1]
        s_lo = np.vdot(vB[:, lo], S @ vB[:, lo]).real
        s_hi = np.vdot(vB[:, hi], S @ vB[:, hi]).real
        g_split = np.sign(s_hi - s_lo) * (wB[hi] - wB[lo]) / (muB * Bf)
        split_vb0 = wB[bB[-1]] - wB[bB[-2]]
        split_vb1 = wB[bB[-3]] - wB[bB[-4]]
        split_vb2 = wB[bB[-5]] - wB[bB[-6]]

        rows.append([L / 10, g_zz, g_orb, g_spin, perp, g_split, w[below[-1]], gap_next,
                     split_vb0, split_vb1, split_vb2])
        print(f"L={L/10:6.2f} nm  g_zz={g_zz:+.5f}  (orb {g_orb:+.5f}, spin {g_spin:+.5f}, "
              f"perp {perp:.1e})  splitting check {g_split:+.5f}  "
              f"E_vb0={w[below[-1]]*1e3:.3f} meV  splittings vb0/vb1/vb2 = "
              f"{split_vb0*1e6:.4f}/{split_vb1*1e6:.4f}/{split_vb2*1e6:.4f} ueV  ({time.time()-t0:.0f} s)", flush=True)

    data = np.array(rows)
    out = f"gzz_signed_vs_L_N{N}.dat"
    np.savetxt(out, data, header="L(nm)  g_zz  g_orb  g_spin  perp  g_zz_splitting  E_vb0(eV)  gap_next(eV)  "
                     "split_vb0(eV)  split_vb1(eV)  split_vb2(eV)")
    print(f"Saved {out}")

    for i in np.where(np.diff(np.sign(data[:, 1])) != 0)[0]:
        print(f"g_zz changes sign between L = {data[i,0]:.1f} and {data[i+1,0]:.1f} nm")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(data[:, 0], data[:, 1], "o-", label=r"$g_{zz}$")
    ax.plot(data[:, 0], data[:, 2], "--", lw=1, label="orbital")
    ax.plot(data[:, 0], data[:, 3], ":", lw=1, label="spin")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$L$ (nm)")
    ax.set_ylabel(r"$g_{zz}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"gzz_signed_vs_L_N{N}.png", dpi=200)


if __name__ == "__main__":
    main()
