import kwant
import numpy as np
from scipy.sparse import diags, eye, kron, csr_matrix, identity, bmat, block_diag
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, qr
import time
import matplotlib.pyplot as plt
from pymablock import block_diagonalize
from Hamiltonian_mathematica_v2 import gso, dgso, ddgso, psi_new_basis



# ---------------------------------
# Parameters
# ---------------------------------
N           = 100 
Nband       = 10
hbar        = 1.05457e-34  # J*s
e_mass      = 9.11e-31  # kg
e0          = 1.602e-19
dim         = 1.602e-39    # dimensional scaling factor (from energy unit)
Ny = Nz = N
print(f"\nBuilding system for N = {N}")


# ==================================================
# Calculate effective mass
# ==================================================

lat = kwant.lattice.square(norbs=Nband)

def make_system_from_block(block_function, kx, L):
    """
    Build transverse 2D nanowire system for a given Hamiltonian block.

    block_function can be:
        gso   : H(kx)
        dgso  : dH/dkx
        ddgso : d2H/dkx2
    """

    syst = kwant.Builder()
    a = L / (Ny + 1)

    block_cache = {}

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            block_cache[(dy, dz)] = block_function(a, kx, dy, dz)

    # Onsite
    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = block_cache[(0, 0)]

    # Hoppings
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:

            if dy == 0 and dz == 0:
                continue

            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = block_cache[(dy, dz)]

    return syst.finalized()

# ==================================================
# Build H0, dH/dkx, d2H/dkx2 at kx = 0
# ==================================================

print("\nBuilding H0, Hk, Hkk...")

L_values = np.linspace(200, 1000, 20)

eff_data = []
for L in L_values:
    syst_0  = make_system_from_block(gso,   0, L)
    syst_d  = make_system_from_block(dgso,  0, L)
    syst_dd = make_system_from_block(ddgso, 0, L)

    H_0  = syst_0.hamiltonian_submatrix(sparse=True)
    H_d  = syst_d.hamiltonian_submatrix(sparse=True)
    H_dd = syst_dd.hamiltonian_submatrix(sparse=True)

    # print("Hamiltonian matrix size:", H_0.shape)

    H_list_sparse = [H_0, H_d, 0.5*H_dd]

    # 3. Find the low-energy subspace using eigsh
    print("Finding k=2 low-energy eigenvectors using eigsh...")
    num_eigenvectors = 2
    evals, evecs = eigsh(H_0, 
                            k=num_eigenvectors, 
                            sigma=0) 
    Ev = evals[-1]
    print("Low-energy eigenvalues found:", evals)
    print("Eigenvectors shape:", evecs.shape)
    evecs_ortho, _ = qr(evecs, mode="economic")  # orthogonalize the vectors

    # block diagonalization using the symbolic mode (to avoid the crash)
    Heff_coeffs, *_ = block_diagonalize(
        H_list_sparse, 
        subspace_eigenvectors=[evecs_ortho]
    ) 
    print("Pymablock finished.")

    # ---------------------------------------------
    #           calculate effective mass
    # ---------------------------------------------

    m_eff = hbar**2 / (2 * (Heff_coeffs[(0,0,2,0)][1,1] +  Heff_coeffs[(0,0,0,1)][1,1]) * dim)
    m_eff_ratio = m_eff / e_mass
    print("\n")
    print(f"Effective mass : {np.real(m_eff_ratio):.6f} m0")
    eff_data.append([L/10, np.real(m_eff_ratio)])

eff_data = np.array(eff_data)
np.save("eff_data.npy", eff_data)
print("done")


