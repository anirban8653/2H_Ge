
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
from Hamiltonian_mathematica_v2 import gso, dgso, psi_new_basis
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
    N = 30
    kx = 0.001 
    Nband = 10
    L = 300
    Ny = Nz = N
    emax_V_um = 7.0
    emax = emax_V_um * 1e-4
    nfield = 11
    sigma_val = 0.1
    Ef_values = np.linspace(0,emax,nfield)
    # print(Ef_values)
    nproc = 10


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




    # ---------------------------------
    # Compute GS at zero field (once)
    # ---------------------------------
    syst0 = make_system(kx, 0.0)
    H0 = syst0.hamiltonian_submatrix(sparse=True)
    vals0 = eigsh(H0, k=2, sigma=0.0, return_eigenvectors=False)
    vals0 = np.sort(vals0)

    GS_0_meV = vals0[0] * 1000   # meV

    # ---------------------------------
    # Sweep Ef
    # ---------------------------------
    
    results = []
    for Ef in Ef_values:

        syst = make_system(kx,  Ef)
        H = syst.hamiltonian_submatrix(sparse=True)

        vals = eigsh(H, k=2, sigma=sigma_val, return_eigenvectors=False)
        vals = np.sort(vals)

        GS1_meV = vals[0] * 1000
        GS2_meV = vals[1] * 1000

        # Energy difference (gap) in micro eV
        shift_microeV = (GS2_meV - GS1_meV) * 1000

        results.append([Ef * 1e4, GS1_meV, GS2_meV, shift_microeV])

        print(f"Ef: {Ef*1e4:.2e}")

    results = np.array(results)


    print("\nEf (V/A) | GS1 (meV) | GS2 (meV) | Gap (micro-eV)")
    print("-" * 65)

    for row in results:
        print(f"{row[0]:.2e} | {row[1]:<12.6f} | {row[2]:<12.6f} | {row[3]:<12.6f}")

    if flagy == 1:
        np.savetxt(f"numerical_splitting_N{N}_L{L}_emax{(emax*10000):.2f}_y.dat", results, header="Ef (V/um) | GS1 (meV) | GS2 (meV) | Gap (micro-eV)", comments='')
        print(f"numerical data saved: numerical_splitting_N{N}_L{L}_emax{(emax*10000):.2f}_y.dat")
    else:
        np.savetxt(f"numerical_splitting_N{N}_L{L}_emax{(emax*10000):.2f}_z.dat", results, header="Ef (V/um) | GS1 (meV) | GS2 (meV) | Gap (micro-eV)", comments='')
        print(f"numerical data saved: numerical_splitting_N{N}_L{L}_emax{(emax*10000):.2f}_z.dat")

    print(" ")
    print("-"*30)
    print("PYMABLOCK")
    print("-"*30)
    print(" ")



    syst1 = make_system_kx(0.0)
    V_kx = syst1.hamiltonian_submatrix(sparse=True)

    kx_value = kx


    def compute_one_Ef(Ef):

        #--------------------------------------------------
        # Starting Pymablock
        #--------------------------------------------------

        syst0 = make_system(0.0, Ef)
        H0_sparse = syst0.hamiltonian_submatrix(sparse=True)

        H_list_sparse = [H0_sparse, V_kx]
        num_eigenvectors = 2
        evals, evecs = eigsh(H0_sparse, 
                                k=num_eigenvectors, 
                                sigma=sigma_val)

        evecs_ortho, _ = qr(evecs, mode="economic")

        psi1 = evecs_ortho[:, 0]
        psi2 = evecs_ortho[:, 1]

        psi_new = psi_new_basis(psi1, psi2, N)
        psi_new = np.array(psi_new).T

        if Ef > 0.0:
            Heff_coeffs, *_ = block_diagonalize(
                H_list_sparse, subspace_eigenvectors=[psi_new])

            H00 = Heff_coeffs[(0,0,0)]
            H11 = Heff_coeffs[(0,0,1)]


            if flagy==1:
                alpha = np.real(H11[0,0])
            else:
                alpha = np.imag(H11[0,1])
            
            H_eff_at_kx = H00 + H11 * kx_value  
            modified_eigenvalues, _ = eigh(H_eff_at_kx)
            gap = np.abs(modified_eigenvalues[1] - modified_eigenvalues[0])

            return Ef*1e4, float(alpha), float(gap * 1e6)

        else:
            return Ef*1e4, 0, 0


    if __name__ == "__main__":
        

        rashba_list = []

        with Pool(nproc) as pool:
            # imap_unordered yields results one by one
            for result in tqdm(pool.imap_unordered(compute_one_Ef, Ef_values),
                            total=len(Ef_values),
                            desc="Computing Rashba vs Electric Field"):
                rashba_list.append(result)

        # Convert to nicely-sorted NumPy array
        rashba_list = np.array(sorted(rashba_list, key=lambda x: x[0]))
        

        if flagy == 1:
            np.savetxt(f"Pymablock_Y_kx{kx_value}_emax_{(emax*10000):.2f}_N{N}_L{L}.dat",
                    rashba_list,
                    header="E  alpha(eV.A)  gap_meV")
            print(f"Exported: Pymablock_Y_kx{kx_value}_emax_{(emax*10000):.2f}_N{N}_L{L}.dat")
        else:
            np.savetxt(f"Pymablock_Z_kx{kx_value}_emax_{(emax*10000):.2f}_N{N}_L{L}.dat",
                    rashba_list,
                    header="E  alpha(eV.A)  gap_meV_2nd")
            print(f"Exported: Pymablock_Z_kx{kx_value}_emax_{(emax*10000):.2f}_N{N}_L{L}.dat")




    print("\n E  |alpha(eV.A)  |gap_meV_2nd ")
    for i in range(len(rashba_list)):
        print(rashba_list[i])

# =========================
# Timer
# =========================
t1 = time.time()
mins, secs = divmod(t1 - t0, 60)
print(f"Execution time: {int(mins)} min {secs:.2f} sec")






