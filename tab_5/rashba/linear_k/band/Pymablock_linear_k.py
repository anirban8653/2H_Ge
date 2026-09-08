
import os
# import time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import kwant
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
import time
from scipy.sparse.linalg import eigsh  # <-- We will use this
from pymablock import block_diagonalize
from multiprocessing import Pool
from tqdm import tqdm
from Hamiltonian_mathematica_v2 import gso, dgso, ddgso, psi_new_basis
np.set_printoptions(linewidth=200, suppress=True, precision=5)




# =========================
# TIMER START
# =========================
t0 = time.time()

flags = [1,0]
for flagy in flags:
    # ---------------------------------
    # Parameters
    # ---------------------------------
    N           = 100
    # kx        = 0.001 
    nk          = 51
    kpoints     = np.linspace(-0.0028, 0.0028, nk, endpoint=True)
    Nband       = 10
    L           = 300
    Ny = Nz     = N
    emax_V_um   = 3.0
    Ef        = emax_V_um * 1e-4
    # nfield      = 11
    sigma_val   = 0.1
    # Ef_values = np.linspace(0,emax,nfield)
    # print(Ef_values)
    nproc       = 40


    print(f"\nBuilding system for N = {N}")


    lat = kwant.lattice.square(norbs=Nband)

    # ---------------------------------
    # Build system main Hamiltonian
    # ---------------------------------
    def make_system(kx, Ef):

        syst = kwant.Builder()
        a = L / (Ny + 1)

        gso_cache = {}
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

        # ---------- Onsite ----------
        for y in range(Ny):
            for z in range(Nz):
                if flagy == 1:
                    V = -Ef * ((y + 1) * a - L / 2)
                else:
                    V = -Ef * ((z + 1) * a - L / 2)
                syst[lat(y, z)] = V * np.eye(Nband) + gso_cache[(0, 0)]

        # ---------- Hoppings ----------
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dy == 0 and dz == 0:
                    continue

                syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso_cache[(dy, dz)]

        return syst.finalized()




    lat_kx = kwant.lattice.square(norbs=Nband)

    # ---------------------------------
    # Build system dH/dkx
    # ---------------------------------
    def make_system_kx(kx):

        syst = kwant.Builder()
        a = L / (Ny + 1)

        gso_cache_kx = {}
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                gso_cache_kx[(dy, dz)] = dgso(a, kx, dy, dz)

        # ---------- Onsite ----------
        for y in range(Ny):
            for z in range(Nz):
                syst[lat_kx(y, z)] =  gso_cache_kx[(0, 0)]

        # ---------- Hoppings ----------
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dy == 0 and dz == 0:
                    continue
                syst[kwant.builder.HoppingKind((dy, dz), lat_kx, lat_kx)] = gso_cache_kx[(dy, dz)]

        return syst.finalized()





    lat_kx2 = kwant.lattice.square(norbs=Nband)

    # ---------------------------------
    # Build system dH/dkx
    # ---------------------------------
    def make_system_kx2(kx):

        syst = kwant.Builder()
        a = L / (Ny + 1)

        gso_cache_kx2 = {}
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                gso_cache_kx2[(dy, dz)] = ddgso(a, kx, dy, dz)

        # ---------- Onsite ----------
        for y in range(Ny):
            for z in range(Nz):
                syst[lat_kx(y, z)] =  gso_cache_kx2[(0, 0)]

        # ---------- Hoppings ----------
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dy == 0 and dz == 0:
                    continue
                syst[kwant.builder.HoppingKind((dy, dz), lat_kx, lat_kx)] = gso_cache_kx2[(dy, dz)]

        return syst.finalized()


    syst0 = make_system(0.0, Ef)
    H0_sparse = syst0.hamiltonian_submatrix(sparse=True)

    syst1 = make_system_kx(0.0)
    V_kx = syst1.hamiltonian_submatrix(sparse=True)

    syst2 = make_system_kx2(0.0)
    V_kx2 = syst2.hamiltonian_submatrix(sparse=True)

    

    #--------------------------------------------------
    # Starting Pymablock
    #--------------------------------------------------

    

    H_list_sparse = [H0_sparse, V_kx, 0.5 * V_kx2]

    num_eigenvectors = 2
    evals, evecs = eigsh(H0_sparse, 
                            k=num_eigenvectors, 
                            sigma=sigma_val)

    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]

    psi_new = psi_new_basis(psi1, psi2, N)
    psi_new = np.array(psi_new).T

    
    Heff_coeffs, *_ = block_diagonalize(
        H_list_sparse, subspace_eigenvectors=[psi_new])

    H00 = Heff_coeffs[(0,0,0,0)]
    H10 = Heff_coeffs[(0,0,1,0)]
    H20 = Heff_coeffs[(0,0,2,0)]
    H01 = Heff_coeffs[(0,0,0,1)]

    def compute_one_kx(kx):
        
        H_eff_at_kx = H00 + H10 * kx + (H20 + H01) * kx**2
        modified_eigenvalues, _ = eigh(H_eff_at_kx)
        modified_eigenvalues = np.sort(modified_eigenvalues)
        mw0 = modified_eigenvalues[0] * 1e3
        mw1 = modified_eigenvalues[1] * 1e3

        gap = np.abs(mw1 - mw0) * 1e3

        return kx, mw0, mw1, gap

        


    if __name__ == "__main__":
        

        rashba_list = []

        with Pool(nproc) as pool:
            # imap_unordered yields results one by one
            for result in tqdm(pool.imap_unordered(compute_one_kx, kpoints),
                            total=len(kpoints),
                            desc="Diagonalising..."):
                rashba_list.append(result)

        # Convert to nicely-sorted NumPy array
        rashba_list = np.array(sorted(rashba_list, key=lambda x: x[0]))
        

        if flagy == 1:
            np.savetxt(f"Pymablock_y_EF{Ef*1e4:.2f}_N{N}_L{L}.dat",
                    rashba_list,
                    header="kx  w0_meV w1_meV  gap_microeV")
            print(f"Exported: Pymablock_z_EF{Ef*1e4:.2f}_N{N}_L{L}.dat")
        else:
            np.savetxt(f"Pymablock_z_EF{Ef*1e4:.2f}_N{N}_L{L}.dat",
                    rashba_list,
                    header="kx  w0_meV w1_meV  gap_microeV")
            print(f"Exported: Pymablock_z_EF{Ef*1e4:.2f}_N{N}_L{L}.dat")




    print("\n K  | E1  |E2  |gap_meV_2nd ")
    for i in range(len(rashba_list)):
        print(rashba_list[i])

# =========================
# Timer
# =========================
t1 = time.time()
mins, secs = divmod(t1 - t0, 60)
print(f"Execution time: {int(mins)} min {secs:.2f} sec")






