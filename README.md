# 2H-Ge nanowires: code, data, and figures

This repository contains numerical codes, datasets, plotting notebooks, and figure exports supporting the manuscript **“Spin-orbit interaction and g-tensor anisotropy of holes in nanowires of hexagonal germanium”** by Anirban Das, Baksa Kolok, Dániel Varjas, and András Pályi.

The calculations use a ten-band $\mathbf{k}\cdot\mathbf{p}$ Hamiltonian to investigate confined hole states in hexagonal germanium (2H-Ge) nanowires, including their band structure, anisotropic magnetic response, electric-field-induced Stark shifts and Rashba splitting, and the microscopic origin of the Rashba coupling.

## Organization and manuscript correspondence

Top-level folders follow the sections and appendices of the paper. Within them, each figure folder collects the corresponding calculation scripts, saved data, and/or plotting notebooks and exports. Figure numbers refer to the manuscript, rather than numbering within each section. Folder names and capitalization in the links below match the repository exactly.

| Manuscript section | Figure | Folder | Topic |
| --- | --- | --- | --- |
| II — Model of 2H-Ge nanowires | 1 | [Sec_II_Band_structure/Fig_1](Sec_II_Band_structure/Fig_1/) | Crystal and nanowire geometry, bulk bands, and confined valence subbands |
| III — Magnetic-field response | 2 | [Sec_III_Zeeman/Fig_2](Sec_III_Zeeman/Fig_2/) | Zeeman splitting and effective $g$ factors versus nanowire size |
| IV — Electric-field response | 3 | [QSE_FIg_3](Sec_IV_Large_elctric_field/QSE_FIg_3/) | Electric-field-induced band shifts and quadratic Stark effect |
| IV — Electric-field response | 4 | [Rashba_Fig_4](Sec_IV_Large_elctric_field/Rashba_Fig_4/) | Finite-field dispersion and Rashba splitting: numerical and effective-Hamiltonian results |
| IV — Electric-field response | 5 | [validity_region_fig_5](Sec_IV_Large_elctric_field/validity_region_fig_5/) | Validity of the linear-in-$k_x$ Rashba approximation |
| IV — Electric-field response | 6 | [Rashba_modulation_fig_6](Sec_IV_Large_elctric_field/Rashba_modulation_fig_6/) | Nonlinear electric-field dependence and angular anisotropy of the Rashba coefficient |
| V — Weak-field Rashba coupling | 7 | [Comapring_splitting_Fig_7](Sec_V_weak_field/Comapring_splitting_Fig_7/) | Numerical splitting compared with second- and fourth-order perturbative results |
| V — Weak-field Rashba coupling | 8 | [validity_region_fig_8](Sec_V_weak_field/validity_region_fig_8/) | Validity of the second-order weak-field expansion for different nanowire sizes |
| V — Weak-field Rashba coupling | 9 | [Rashba_modulation_Fig_9](Sec_V_weak_field/Rashba_modulation_Fig_9/) | Electric-field and cross-sectional-size dependence of the weak-field Rashba coefficient |
| VI — Microscopic origin of the Rashba coupling | 10 | [Fig_10](Sec_VI_microscopic_origin/Fig_10/) | Intermediate-subband contributions and bulk-band character weights |
| VI — Microscopic origin of the Rashba coupling | 11 | [Fig_11](Sec_VI_microscopic_origin/Fig_11/) | Symmetry-allowed virtual coupling processes |
| Appendix B — Effective mass calculation | 12 | [Fig_12](App_B_mx_modulation/Fig_12/) | Confinement dependence of the longitudinal effective mass |

Additional bulk-band comparisons with density functional theory (DFT), together with effective-mass calculations, are in [comapre_with_DFT](comapre_with_DFT/).

**The two Rashba treatments:** Section IV includes the transverse electric field in the reference Hamiltonian and expands in longitudinal momentum $k_x$. Section V treats both the electric field and momentum perturbatively in a weak-field Schrieffer–Wolff expansion. Their validity plots therefore test different approximations.

## Main scripts and notebooks

The filenames below are relative to the figure folders linked above.

| Figure | Calculation or analysis scripts | Plotting notebooks |
| --- | --- | --- |
| 1 | `main_subband_parallelised.py`; `Hamiltonian_python.py`; `Discretised_Hamiltonian.wls` | `hamiltonian_k_dot_p.ipynb`, `plot.ipynb` |
| 2 | `gfactor_Bx.py`, `gfactor_By.py`, `gfactor_Bz.py` | `g_factor_plot.ipynb` |
| 3 | `main_band_shift.py`, `main_gap.py` | `bands_plot.ipynb`, `plot_gap_variation.ipynb` |
| 4 | `main.py`, `Pymablock_linear_k.py` | `bands_plot.ipynb` |
| 5 | `rashba_with_E_pymablock_kwant_mpi.py` | `plot.ipynb` |
| 6 | `rashba_with_E_pymablock_kwant.py`, `main_E_rotation_theta.py` | `plot_only_y_and_z_direction.ipynb`, `peanut_plot.ipynb` |
| 7 | Saved numerical and perturbative splitting data | `plot.ipynb` |
| 8 | `rashba_with_E_pymablock_kwant_mpi.py` | `plot.ipynb` |
| 9 | `rashba_with_E_pymablock_kwant.py`, `rashba_with_L_pymablock_kwant.py` | `plot_E_sweep.ipynb`, `plot_L_sweep.ipynb` |
| 10 | Saved subband-contribution and overlap data | `plot.ipynb` |
| 11 | Diagram constructed in the notebook | `level_2x1_combined.ipynb` |
| 12 | `effcetive_mass.py` | `plot.ipynb` |

Figure folders also contain local Hamiltonian modules where needed, such as `Hamiltonian_mathematica.py` and `Hamiltonian_mathematica_v2.py`. These are Python modules despite their names. Keep each script with its accompanying module; similarly named modules in different folders should not be assumed interchangeable.

The DFT comparison folder contains `hamiltonian_bulk.py`, `main_numerical_eff_mass.py`, `band_fit.ipynb`, and the supplied bulk-band datasets.

## Software and execution

The inspected Python workflows use NumPy, SciPy, Matplotlib, Kwant, Pymablock, and tqdm, with Jupyter for the notebooks. MPI calculation scripts additionally use `mpi4py` and an MPI runtime. The `.wls` script requires a Wolfram Language installation. Requirements depend on the selected calculation; package versions are not pinned in this repository.

To inspect a result, open the relevant figure folder and view its existing plot exports. To regenerate a plot from saved data, open its plotting notebook with that folder as the working directory and review its input filenames and selected parameters.

To rerun a calculation:

1. Review the script's parameters, including grid size, nanowire size, momentum range, field direction and strength, and process count.
2. Run from the figure folder so that local imports and relative data paths resolve correctly.
3. Check output filenames before execution: scripts may overwrite existing datasets.
4. Use the corresponding notebook to plot the generated data.

For example, after configuring the parameters and process count for Figure 1:

```bash
cd Sec_II_Band_structure/Fig_1
python main_subband_parallelised.py
```

Several scripts use multiprocessing or MPI. The included `.mpi` files are cluster-specific Slurm submission scripts; adapt their resource requests, environment paths, and launch commands to your computing environment.

In the inspected nanowire scripts, lengths are expressed in angstroms and momenta in inverse angstroms; for example, `L = 300` corresponds to 30 nm. Electric-field inputs are converted between V/µm and V/Å, while plotted energies may be converted from eV to meV or µeV. Consult each script and dataset header for the precise convention. Current script defaults need not reproduce every stored parameter sweep without modification.

## Data and figure exports

- `.dat` files contain saved numerical datasets; column layouts and headers vary by calculation.
- `.npy` files contain NumPy arrays, including the effective-mass data for Figure 12.
- `.ipynb` files contain plotting and analysis code, sometimes with multiple figure variants.
- `.pdf`, `.svg`, and `.png` files contain figure exports, component panels, or diagnostic plots.

The Figure 7 and Figure 10 folders contain saved data and plotting notebooks, but no standalone scripts generating those datasets. Figure 11 is a schematic generated directly in its notebook. The repository therefore provides different levels of calculation and plotting coverage for different figures.

Some manuscript figure filenames differ from the available repository exports. The following exact manuscript filenames are absent from the inspected repository; the listed files are related exports in the corresponding folders, not verified identical copies of the manuscript figures.

| Figure | Filename referenced by the manuscript | Related repository exports |
| --- | --- | --- |
| 4 | `rasha_splitting_v2.pdf` | `bands_electrci_field_horizontal.pdf` |
| 6 | `polar_plot_linear_k.pdf` | `alpha_linear_k.pdf`, `polar_alpha_all_E_N50_L300_kx0.001.pdf` |
| 8 | `relative_error_new_2.pdf` | `phase_diagram_Ey.pdf`, `phase_diagram_Ez.pdf` |
| 9 | `rashba_coeff_variation_v2.pdf` | `Rashba_modulation_new.pdf`, `alpha_vs_L_YZ_Ny100_Nz100.pdf`, and other coefficient plots |

Use the section and figure folder mapping to locate the relevant materials, rather than relying solely on an exported filename.
