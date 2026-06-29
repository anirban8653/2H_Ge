import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
import time

# ---------------------------------
# TIMER START
# ---------------------------------
t0 = time.time()

# ---------------------------------
# Import Hamiltonian blocks
# ---------------------------------
from Hamiltonian_real_april import build_H_real

# ---------------------------------
# Parameters
# ---------------------------------
N           = 100                      
Nband       = 10
L           = 300
kx          = 0.001           
switch_y    = 0  # put 1 for y directional field, 0 for z directional field

Ny = Nz = N
if switch_y == 1:
    print("-"*30)
    print(f"E || y")
    print("-"*30)
else:
    print("-"*30)
    print(f"E || z")
    print("-"*30)
print(f"\nBuilding system for N = {N}")

# Build Hamiltonian blocks once
gso = build_H_real(kx, N, L)

# ---------------------------------
# Lattice
# ---------------------------------
lat = kwant.lattice.square(norbs=Nband)

# ---------------------------------
# Build system
# ---------------------------------
def make_system(Ef):

    syst = kwant.Builder()
    a = L / (Ny + 1)

    # ---------- Onsite ----------
    for y in range(Ny):
        for z in range(Nz):


            if switch_y == 1:
                V = -Ef * ((y + 1) * a - L / 2)
            else:
                V = -Ef * ((z + 1) * a - L / 2)

            syst[lat(y, z)] = V * np.eye(Nband) + gso[(0, 0)]

    # ---------- Hoppings ----------
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue

            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso[(dy, dz)]

    return syst.finalized()


# ---------------------------------
# Field sweep
# ---------------------------------
Ef_values = np.linspace(0, 0.6e-5, 10)

results = []

# ---------------------------------
# Compute GS at zero field (once)
# ---------------------------------
syst0 = make_system(0.0)
H0 = syst0.hamiltonian_submatrix(sparse=True)

vals0 = eigsh(H0, k=2, sigma=0.0, return_eigenvectors=False)
vals0 = np.sort(vals0)

GS_0_meV = vals0[0] * 1000   # meV

# ---------------------------------
# Sweep Ef
# ---------------------------------
for Ef in Ef_values:

    syst = make_system(Ef)
    H = syst.hamiltonian_submatrix(sparse=True)

    vals = eigsh(H, k=2, sigma=0.0, return_eigenvectors=False)
    vals = np.sort(vals)

    GS1_meV = vals[0] * 1000
    GS2_meV = vals[1] * 1000

    # Energy difference (gap) in micro eV
    shift_microeV = (GS2_meV - GS1_meV) * 1000

    results.append([Ef * 1e4, GS1_meV, GS2_meV, shift_microeV])

    print(f"Ef: {Ef*1e4:.3f}")

results = np.array(results)

# ---------------------------------
# Timer
# ---------------------------------
t1 = time.time()
mins, secs = divmod(t1 - t0, 60)

print("\nEf (V/A) | GS1 (meV) | GS2 (meV) | Gap (micro-eV)")
print("-" * 65)

for row in results:
    print(f"{row[0]:.2e} | {row[1]:<12.6f} | {row[2]:<12.6f} | {row[3]:<12.6f}")

# ---------------------------------
# Save results
# ---------------------------------

if switch_y == 1:
    np.savetxt(
        f"Ef_sweep_gap_y_{N}_delta3_pperp_A5_A6.dat",
        results,
        fmt="%.6e %.6f %.6f %.6f",
        header="Ef(V/A) GS1(meV) GS2(meV) Gap(micro-eV)",
        comments=""
    )
else:
    np.savetxt(
        f"Ef_sweep_gap_z_{N}_delta3_pperp_A5_A6.dat",
        results,
        fmt="%.6e %.6f %.6f %.6f",
        header="Ef(V/A) GS1(meV) GS2(meV) Gap(micro-eV)",
        comments=""
    )

print("\nDone.")
