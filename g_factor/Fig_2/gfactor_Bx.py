import kwant
import numpy as np
from scipy.sparse import csr_matrix, block_diag, bmat
from scipy.sparse.linalg import eigsh
import time

from Hamiltonian_mathematica_v2 import gso
np.set_printoptions(linewidth=150)


# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N= 50
Nband = 10
kx = 0.0

hbar = 1.05457e-34
q = 1.602e-19
muB = 5.78e-5
gspin = 2.0
Bf = 0.01 #applied magnetic field in T

Ny = Nz = N

lat = kwant.lattice.square(norbs=Nband)


# ---------------------------------------------
#               SPIN MATRIX
# ---------------------------------------------
def build_spin_matrix():
    sminus = 1

    sud = np.zeros((5, 5), dtype=complex)
    sud[0, 0] = sminus
    sud[1, 1] = sminus
    sud[4, 4] = sminus
    sud[2, 3] = sminus
    sud[3, 2] = sminus

    sud_sparse = csr_matrix(sud)
    sdu_sparse = sud_sparse.conjugate()
    zero = csr_matrix((5, 5))

    smatrix = 0.5 * bmat([
        [zero, sud_sparse],
        [sdu_sparse, zero]
    ], format='csr')

    return smatrix


# ---------------------------------------------
#               KWANT SYSTEM
# ---------------------------------------------
def make_system(Bf, L):
    syst = kwant.Builder()

    a = L / (N + 1)

    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso_cache[(0, 0)]


    def hopping(site1, site2):
    
        yi, zi = site1.tag
        yj, zj = site2.tag

        dy = yj - yi
        dz = zj - zi

        val = gso_cache.get((dy, dz), None)
        if val is None:
            return None

        # lattice spacing in meters
        a_m = a * 1e-10

        # Peierls prefactor: q B a^2 / hbar
        phase_prefactor = q * Bf * a_m**2 / hbar

        # midpoint y coordinate in lattice-index units
        y_mid = 0.5 * (yi + yj)

        # Peierls phase for A_z = Bx y
        phase = np.exp(-1j * phase_prefactor * y_mid * dz)

        return val * phase


    # ---------- Apply hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = hopping

    return syst.finalized()


# ---------------------------------------------
#               G-FACTOR CALCULATION
# ---------------------------------------------
def compute_gfactors(H, smatrixfull):
    H_total = H + gspin * muB * Bf * smatrixfull
    # print("Dimension of H_total:", H_total.shape)
    # print("Number of non-zero elements in H_total:", H_total.nnz)

    # --- Valence bands ---
    vbs = eigsh(H_total, k=6, sigma=0 ,which='LM', return_eigenvectors=False)
    vbs = np.sort(vbs)

    # --- Conduction bands ---
    cbs = eigsh(H_total, k=2, sigma=0.3, which='LM', return_eigenvectors=False)
    cbs = np.sort(cbs)

    vb1, vb2, vb3, vb4, vb5, vb6 = vbs
    cb1, cb2 = cbs

    gap_vb2 = abs(vb1 - vb2)
    gap_vb1 = abs(vb3 - vb4)
    gap_vb0 = abs(vb5 - vb6)
    

    gap_cb = abs(cb1 - cb2)

    gvb = gap_vb0 / (muB * Bf)
    gcb = gap_cb / (muB * Bf)

    return (gvb,  gap_vb0, gap_vb1, gap_vb2, 
            vb5, vb6, vb3, vb4, vb1, vb2, cb1, cb2, gap_cb, gcb)


# ---------------------------------------------
#               MAIN LOOP
# ---------------------------------------------
def main():
    gfactors = []

    smatrix_sparse = build_spin_matrix()
    smatrixfull = block_diag([smatrix_sparse] * (N**2), format='csr')

    size_list = np.linspace(200, 1000, 20, dtype=int)
    # size_list = [300]

    for L in size_list:
        time_start = time.time()
        print(f"\nBuilding system for L = {L/10} nm")

        # --- Build Hamiltonian ---
        syst = make_system(Bf, L)
        H0 = syst.hamiltonian_submatrix(sparse=True)

        # --- Compute g-factors ---
        (gvb, gap_vb0, gap_vb1, gap_vb2, 
         vb5, vb6, vb3, vb4, vb1, vb2, cb1, cb2, gap_cb, gcb) = compute_gfactors(H0, smatrixfull)

        # --- Print ---
        print(f"VB:   {vb5: .8f}, {vb6: .8f}   gap={gap_vb0*1e6: .8f}   g={gvb: .3f}")
        print(f"VB+1: {vb3: .8f}, {vb4: .8f}   gap={gap_vb1*1e6: .8f}")
        print(f"VB+2: {vb1: .8f}, {vb2: .8f}   gap={gap_vb2*1e6: .8f}")
        print(f"CB:   {cb1: .8f}, {cb2: .8f}   gap={gap_cb*1e6: .8f}   g={gcb: .3f}\n")

        gfactors.append([L/10, gvb, gap_vb0, gap_vb1, gap_vb2, gap_cb, gcb ])
        time_end = time.time()
        # time in minutes and seconds
        time_elapsed = time_end - time_start
        minutes = int(time_elapsed // 60)
        seconds = int(time_elapsed % 60)
        print(f"Time taken: {minutes} min {seconds} sec")


    # --- Save ---
    gfactors = np.array(gfactors)
    np.savetxt(
        f"full_gfactor_x_N{N}.dat",
        gfactors,
        header="size(nm)  g_vb   gap_vb(eV)   gap_vb1(eV)   gap_vb2(eV) "
    )


# ---------------------------------------------
#               RUN
# ---------------------------------------------
if __name__ == "__main__":
    main()