import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
#                      GLOBAL SETTINGS
# ==============================================================

plt.rcParams.update({
    "figure.figsize": [7, 6],
    "font.family": "Serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8
})

# ==============================================================
#                    BAND STRUCTURE PART
# ==============================================================

size = 50
Ny = Nz = size
NN = Ny * Nz
Nband = 10
nk = 51

kpoints  = np.linspace(-0.002, 0.002, nk, endpoint=True)*1000
# kpoints0 = np.linspace(-0.01, 0.01, 201, endpoint=True)*1000

eval_array_E0 = np.loadtxt("band_data_y_E0.000_size50_nk51.dat")

efields_um = np.arange(0.03,0.12,0.03)
cmap = plt.cm.Blues
colors = cmap(np.linspace(0.5, 1, len(efields_um)))

def plot_bands(ax, direction, band_indices, ylim):

    # Zero field
    ax.plot(kpoints, eval_array_E0[:,0]*1e3,
            color='red', lw=0.7,
            label=r"$\mathcal{E}$ = 0.00 V/µm", ls = 'dotted')
    ax.plot(kpoints, eval_array_E0[:,1]*1e3,
            color='red', lw=0.7, ls = '--')
    

    # Finite field
    for idx, efield_um in enumerate(efields_um):

        datafile = f"band_data_{direction}_E{efield_um:.3f}_size{size}_nk{nk}.dat"
        eval_array_E = np.loadtxt(datafile)

        ax.plot(kpoints,
                eval_array_E[:, band_indices[0]]*1e3,
                color=colors[idx], lw=0.7)

        ax.plot(kpoints,
                eval_array_E[:, band_indices[1]]*1e3,
                color=colors[idx], lw=0.7,
                label=rf"$\mathcal{{E}}$ = {efield_um:.2f} V/µm")

    ax.axvline(0, color='black', lw=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(1, color='green', lw=0.8, ls = 'dashed')

    ax.set_xlim(-2,2)
    ax.set_ylim(*ylim)
    ax.set_xticks([-2,0,2])
    ax.tick_params(direction='in', length=6, width=0.8,
                   top=True, right=True, labelsize=16)

    ax.set_ylabel('Energy (meV)', fontsize=16)
    ax.legend(fontsize=9, loc='lower left')


# ==============================================================
#                    SPLITTING PART
# ==============================================================

kx = 0.001

datap2_y = np.loadtxt(f"gap_vs_Ey_per2_{kx}_emax_0.10.dat")
datap4_y = np.loadtxt(f"gap_vs_Ey_per4_{kx}_emax_0.10.dat")
datay_N  = np.loadtxt(f"evals_vs_Ey_{kx}.dat")

datap2_z = np.loadtxt(f"gap_vs_Ez_per2_{kx}_emax_0.10.dat")
datap4_z = np.loadtxt(f"gap_vs_Ez_per4_{kx}_emax_0.10.dat")
dataz_N  = np.loadtxt(f"evals_vs_Ez_{kx}.dat")

E_y   = datap4_y[:,0] * 1e4
E_z   = datap4_z[:,0] * 1e4
E_num = datay_N[:,0]

gap_pert2_y = datap2_y[:,2]*1000
gap_pert4_y = datap4_y[:,2]*1000
gap_num_y   = datay_N[:,3]*1000

gap_pert2_z = datap2_z[:,2]*1000
gap_pert4_z = datap4_z[:,2]*1000
gap_num_z   = dataz_N[:,3]*1000


def plot_splitting_y(ax):

    ax.plot(E_y, gap_pert2_y, 'o--',
            color='cyan', markersize=6,
            markerfacecolor='white',
            label="Pert (2nd)")

    ax.plot(E_y, gap_pert4_y, 'o--',
            color='royalblue', markersize=6,
            markerfacecolor='white',
            label="Pert (4th)")

    ax.plot(E_num, gap_num_y, 'o-',
            color='darkblue', markersize=4,
            label="Numerical")

    ax.set_ylabel(r'$\Delta E_y$ ($\mu$eV)', fontsize=16)
    ax.tick_params(direction='in', top=True, right=True, labelsize=16)
    ax.legend(fontsize=8, loc='upper left')
    


def plot_splitting_z(ax):

    ax.plot(E_z, gap_pert2_z, 'o--',
            color='cyan', markersize=6,
            markerfacecolor='white',
            label="Pert (2nd)")

    ax.plot(E_z, gap_pert4_z, 'o--',
            color='royalblue', markersize=6,
            markerfacecolor='white',
            label="Pert (4th)")

    ax.plot(E_num, gap_num_z, 'o-',
            color='darkblue', markersize=4,
            label="Numerical")

    ax.set_xlabel(r'$\mathcal{E}$ (V/$\mu$m)', fontsize=16)
    ax.set_ylabel(r'$\Delta E_z$ ($\mu$eV)', fontsize=16)
    ax.tick_params(direction='in', top=True, right=True, labelsize=16)
    ax.legend(fontsize=8, loc='upper left')
    ax.text(0.03,0,"$k_x$ = 0.001"r" $\AA^{-1}$", fontsize = 14)


# ==============================================================
#                    CREATE 2x2 FIGURE
# ==============================================================

fig, axs = plt.subplots(2,2)

# ----- Row 1 -----
plot_bands(axs[0,0], 'y', (0,1), (-5.8,-5.5))
plot_splitting_y(axs[0,1])

# ----- Row 2 -----
plot_bands(axs[1,0], 'z', (0,1), (-5.8,-5.4))
plot_splitting_z(axs[1,1])

axs[1,0].set_xlabel(r'$k_x$ ($\times$10$^3$ $\AA^{-1}$)', fontsize=16)
axs[0,0].text(-1.8, -5.54,'$\mathcal{E}$ || $y$', fontsize = 16)
axs[1,0].text(-1.8, -5.45,'$\mathcal{E}$ || $z$', fontsize = 16)
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[0,1].text(0.03,0,"$k_x$ = 0.001"r" $\AA^{-1}$", fontsize = 14)

# ==============================================================
#                Subfigure Labels (a,b,c,d)
# ==============================================================

labels = ['','(c)','','(d)']
for ax, label in zip(axs.flat, labels):
    ax.text(-0.35, 1, label,
            transform=ax.transAxes,
            fontsize=16,
            va='top')
    
labels = ['(a)','','(b)','']
for ax, label in zip(axs.flat, labels):
    ax.text(-0.47, 1, label,
            transform=ax.transAxes,
            fontsize=16,
            va='top')

plt.tight_layout()
plt.savefig("combined_2x2_bands_splitting.png", dpi=300)
plt.savefig("efield.pdf")
plt.show()
