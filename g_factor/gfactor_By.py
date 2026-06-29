import kwant
import numpy as np
from scipy.sparse import lil_matrix,block_diag, bmat, csr_matrix
from scipy.sparse.linalg import eigsh
from Hamiltonian_mathematica_v2 import gs_shifted_by
import time


# =========================
# Precompute hopping blocks
# =========================


        
# ---------------------------------------------
#               GLOBAL PARAMETERS
# ---------------------------------------------
N = 100
Nband = 10
kx = 0.0
Bf = 0.01
hbar = 1.05457e-34
q = 1.602e-19
muB = 5.78e-5
gspin = 2.0

def make_system(Ny, Nz, L, kx, Bf):

    a = L / (Ny + 1)

    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=Nband)

    # --- Onsite Function ---
    def onsite(site):
        y, z = site.tag

        H_on = gs_shifted_by(a, kx, 0, 0, Bf, z)

        return np.asarray(H_on, dtype=complex)

    # --- Hopping Function ---
    def hopping(site1, site2):
        y1, z1 = site1.tag
        y2, z2 = site2.tag

        hop_dy = int(y1 - y2)
        hop_dz = int(z1 - z2)

        zmid = 0.5 * (z1 + z2)

        H_hop = gs_shifted_by(a, kx, hop_dy, hop_dz, Bf, zmid)

        return np.asarray(H_hop, dtype=complex)

    # --- Build lattice ---
    syst[(lat(y, z) for y in range(Ny) for z in range(Nz))] = onsite

    hop_directions = [
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
    ]

    for d in hop_directions:
        syst[kwant.builder.HoppingKind(d, lat, lat)] = hopping

    return syst.finalized()




#---------------------------------------------
#           Create the S matrix
#---------------------------------------------
# --- base 5x5 components ---
sminus = -1j

sud = np.zeros((5, 5), dtype=complex)
sud[0, 0] = sminus
sud[1, 1] = sminus
sud[4, 4] = sminus
sud[2, 3] = sminus
sud[3, 2] = sminus

sud_sparse = csr_matrix(sud)
sdu_sparse = sud_sparse.conjugate()

zero = csr_matrix((5, 5))

# --- 10x10 spin matrix (ORDER PRESERVED) ---
smatrix_sparse = 0.5 * bmat([
    [zero, sud_sparse],
    [sdu_sparse, zero]
], format='csr')


#---------------------------------------------
#               Main parameters
#---------------------------------------------

gfactors = []
size_list = np.linspace(200, 1000, 20, dtype=int)
# size_list = [300]
for size in size_list:
    time_start = time.time()
    L = size
    print(f"\nBuilding system for L = {L/10} nm")
    Ny = Nz = N

    # #smatrix
    # smatrixfull = block_diag([smatrix_sparse] * N**2, format='csr')

    # #---------------------------------------------
    # #        Eigenvalue & g-factor calc
    # #---------------------------------------------
    # syst = make_system(Ny, Nz, L, kx, Bf)
    # H0 = syst.hamiltonian_submatrix(sparse=True)
    # H = H0 + gspin * muB * Bf * smatrixfull

    syst = make_system(Ny, Nz, L, kx, Bf)
    H0 = syst.hamiltonian_submatrix(sparse=True)

    n_sites = H0.shape[0] // Nband
    smatrixfull = block_diag([smatrix_sparse] * n_sites, format='csr')

    # print("H0 shape          =", H0.shape)
    # print("smatrixfull shape =", smatrixfull.shape)

    H = H0 + gspin * muB * Bf * smatrixfull
    
    # Compute eigenvalues near zero
    vbs = eigsh(H, k=6, sigma=0, which='LM', return_eigenvectors=False)
    vbs = np.sort(vbs)
    # print(vbs)
    
    cbs = eigsh(H, k=2, sigma=0.3, which='LM', return_eigenvectors=False)
    cbs = np.sort(cbs)
    # print(cbs)
    
    vb1, vb2, vb3, vb4, vb5, vb6 = vbs       # valence band top (closer to zero is vb2)
    cb1, cb2 = cbs    # conduction band bottom (cb1 is closest to zero)
    
    # only gap of vb2, vb1, vb0
    gap_vb2 = abs(vb1 - vb2)
    gap_vb1 = abs(vb3 - vb4)
    gap_vb0 = abs(vb5 - vb6)

    # g-factor of top vb
    gvb = gap_vb0 / (muB * Bf)
    
    # g-factor of bottom cb
    gap_cb = abs(cb1 - cb2)
    gcb = gap_cb / (muB * Bf)
    
    # ---------------------------------------------
    #                 Output
    # ---------------------------------------------
    
    print(f"VB: {vb5: .6f}, {vb6: .6f}   gap={gap_vb0: .8f}   g={gvb: .3f}")
    print(f"VB+1: {vb3: .6f}, {vb4: .6f}   gap={gap_vb1: .8f}   ")
    print(f"VB+2: {vb1: .6f}, {vb2: .6f}   gap={gap_vb2: .8f}   ")
    print(f"CB: {cb1: .6f}, {cb2: .6f}   gap={gap_cb: .8f}   g={gcb: .3f}")
    print("")
    
    # print("---------------------------------")

    gfactors.append([(L/10), gvb,  gap_vb0, gap_vb1, gap_vb2, gap_cb, gcb])

    time_end = time.time()

    #print time in minute and seconds
    elapsed_time = time_end - time_start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Time taken: {minutes} min {seconds} sec")

gfactors = np.array(gfactors)
np.savetxt(f"full_gfactor_y_N{N}.dat", gfactors, 
           header="size(nm)  g_vb   gap_vb(eV)   gap_vb1(eV)   gap_vb2(eV)")
