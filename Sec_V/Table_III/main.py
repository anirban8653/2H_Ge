import kwant
import numpy as np
from scipy.sparse.linalg import eigsh
import time

t0 = time.time()

from Hamiltonian_real_april import build_H_real

# ---------------------------------
# Parameters
# ---------------------------------
N     = 100
Nband = 10
L     = 300
kx    = 0.001
Ny = Nz = N

# electric field in V/Angstrom (0.05 V/um = 0.05e-4 V/A)
E_Vum = 0.05
Ef    = E_Vum * 1e-4

lat = kwant.lattice.square(norbs=Nband)


def splitting_microeV(direction, p, A5, A6, d3):
    """Splitting of the two states closest to E=0 for E || direction ('y' or 'z')."""
    gso = build_H_real(kx, N, L, p, A5, A6, d3)
    a = L / (Ny + 1)

    syst = kwant.Builder()
    for y in range(Ny):
        for z in range(Nz):
            pos = (y + 1) * a if direction == 'y' else (z + 1) * a
            V = -Ef * (pos - L / 2)
            syst[lat(y, z)] = V * np.eye(Nband) + gso[(0, 0)]

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = gso[(dy, dz)]

    H = syst.finalized().hamiltonian_submatrix(sparse=True)
    vals = np.sort(eigsh(H, k=2, sigma=0.0, return_eigenvectors=False))
    return (vals[1] - vals[0]) * 1e6   # eV -> micro-eV


# ---------------------------------
# Cases: all on, then one switch off at a time
# ---------------------------------
all_on = dict(p=1, A5=1, A6=1, d3=1)
cases = [("all on", all_on)]
for name in all_on:
    cases.append((f"{name} = 0", {**all_on, name: 0}))

split = {d: {} for d in "yz"}
for label, sw in cases:
    for d in "yz":
        split[d][label] = splitting_microeV(d, **sw)
        print(f"{label:8s} E||{d}: {split[d][label]:.6f} micro-eV", flush=True)

# ---------------------------------
# Table
# ---------------------------------
lines = [f"E = {E_Vum} V/um, kx = {kx}, N = {N}, L = {L}",
         f"Splitting (all on): y = {split['y']['all on']:.6f}, z = {split['z']['all on']:.6f} micro-eV",
         "",
         f"{'Parameter':<10}| {'y (%)':>12} | {'z (%)':>12}",
         "-" * 40]
for label, _ in cases[1:]:
    ry = abs(split['y'][label] - split['y']['all on']) / split['y']['all on'] * 100
    rz = abs(split['z'][label] - split['z']['all on']) / split['z']['all on'] * 100
    lines.append(f"{label:<10}| {ry:>12.4f} | {rz:>12.4f}")

table = "\n".join(lines)
print("\n" + table)
with open(f"relative_change_table_N{N}.dat", "w") as f:
    f.write(table + "\n")

print(f"\nDone. Elapsed {time.time() - t0:.1f} s")
