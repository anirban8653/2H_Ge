import numpy as np
import matplotlib.pyplot as plt
from hamiltonian_bulk import params_coupling_m, params_coupling_p, params_no_coupling , Hfull, eff_mass_cb, eff_mass_vb
from scipy.linalg import eigh

kpoints = np.linspace(-0.05, 0.05, 201)


evaldatax = []
evaldataz = []

for kx in kpoints:
    H = Hfull(kx, 0, 0, p = params_coupling_m)
    eval, evec = eigh(H)
    evaldatax.append(eval)
    
for kz in kpoints:
    H = Hfull(0, 0, kz, p = params_coupling_m)
    eval, evec = eigh(H)
    evaldataz.append(eval)
    
evaldatax = np.array(evaldatax)
evaldataz = np.array(evaldataz)

plt.figure(figsize=(4,5))
plt.plot(kpoints, evaldatax)
plt.xlabel(r"k$_x$ [$\AA^{-1}$]")
plt.ylabel(r"Energy [eV]")
plt.show()

vb1_data_x = evaldatax[:,5]
cb1_data_x = evaldatax[:,6]

vb1_data_z = evaldataz[:,5]
cb1_data_z = evaldataz[:,6]


# ============================================================
# EFFECTIVE MASS FUNCTION
# ============================================================


mcbx = eff_mass_cb(kpoints, cb1_data_x)
mvbx = eff_mass_vb(kpoints, vb1_data_x)
mcbz = eff_mass_cb(kpoints, cb1_data_z)
mvbz = eff_mass_vb(kpoints, vb1_data_z)

print("-----------------------")
print("Parallel Effective mass (x/y)")
print("-----------------------")
print("cb : ",mcbx[1])
print("vb : ",mvbx[1]) 


print("-----------------------")
print("Perpendicular Effective mass (z)")
print("-----------------------")
print("cb : ",mcbz[1])
print("vb : ",mvbz[1]) 



