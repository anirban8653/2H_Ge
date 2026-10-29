# 2H-Ge nanowires: code, data, and figures

This repository contains numerical codes, datasets, plotting notebooks, and figure exports supporting the manuscript **"Spin-orbit interaction and g-tensor anisotropy of holes in nanowires of hexagonal germanium"** by Anirban Das, Baksa Kolok, Dániel Varjas, and András Pályi.

The calculations use a ten-band $\mathbf{k}\cdot\mathbf{p}$ Hamiltonian to study confined hole states in hexagonal germanium (2H-Ge) nanowires: band structure, anisotropic magnetic response, electric-field-induced Stark shifts and Rashba splitting, and the microscopic origin of the Rashba coupling.

## Repository layout and manuscript correspondence

Top-level folders are named after the manuscript sections (`Sec_*`), appendices (`App_B`, `Appendix_H`), and supplementary material (`Extra`). Each `Fig_N` folder holds the calculation scripts, saved data, plotting notebooks, and figure exports for manuscript Figure N; `Sec_V/Table_III` holds the code and data for manuscript Table III. In `Sec_IV/FIg_3` and `Sec_IV/Fig_7`, saved datasets are in a `data/` subfolder and figure exports in a `plots/` subfolder. Folder names and capitalization below match the repository exactly (note `FIg_3` and `Appendix_H`).

| Manuscript part | Figure | Folder | Topic |
| --- | --- | --- | --- |
| Sec. II: Model | 1 | [`Sec_II/Fig_1`](Sec_II/Fig_1) | Bulk bands, confined valence subbands, and Hamiltonian checks |
| Sec. III: Magnetic-field response | 2 | [`Sec_III/Fig_2`](Sec_III/Fig_2) | Zeeman splitting and effective g factors for fields along x, y, z |
| Sec. IV: Electric-field response | 3 | [`Sec_IV/FIg_3`](Sec_IV/FIg_3) | Electric-field-induced band shifts, gap variation, quadratic Stark effect |
| Sec. IV | 4 | [`Sec_IV/Fig_4`](Sec_IV/Fig_4) | Finite-field dispersion and Rashba splitting: numerics vs. linear-$k_x$ effective Hamiltonian (Pymablock) |
| Sec. IV | 5 | [`Sec_IV/Fig_5`](Sec_IV/Fig_5) | Field and angular dependence of the Rashba coefficient and spin-orbit length |
| Sec. IV | 6 | [`Sec_IV/Fig_6`](Sec_IV/Fig_6) | Validity region of the effective-Hamiltonian (Pymablock) description, from gap comparisons |
| Sec. IV | 7 | [`Sec_IV/Fig_7`](Sec_IV/Fig_7) | Rashba modulation evaluated over the valid-region data |
| Sec. V: Microscopic origin of Rashba coupling | 8 | [`Sec_V/Fig_8`](Sec_V/Fig_8) | Intermediate-subband contributions (`Ey/`, `Ez/`) and bulk-band weights (`overlap_function/`) |
| Sec. V | 9 | [`Sec_V/Fig_9`](Sec_V/Fig_9) | Schematic virtual-coupling (selection-rule) diagrams |
| Sec. V | Table III | [`Sec_V/Table_III`](Sec_V/Table_III) | Relative change of the Rashba splitting when individual k·p parameters ($P_\perp$, $A_5$, $A_6$, $\Delta_3$) are switched off |
| App. B: Effective mass | 10 | [`App_B/Fig_10`](App_B/Fig_10) | Confinement dependence of the longitudinal effective mass |
| App. H: Dominant states | 11 | [`Appendix_H/Fig_11`](Appendix_H/Fig_11) | Probability densities of the dominant states (figure exports only) |
| Supplementary | n/a | [`Extra/comparing_k.p_bands_with_DFT`](Extra/comparing_k.p_bands_with_DFT) | $\mathbf{k}\cdot\mathbf{p}$ vs. DFT bulk bands and effective-mass fits |

Figure numbers refer to the manuscript, not to numbering within a section.

Notes on specific folders:

- **Sec_IV/FIg_3**: `main_band_shift.py` and `main_gap.py` compute the field-dependent bands and gaps; `bands_plot.ipynb` and `plot_gap_variation.ipynb` plot them. Band data (`band_data_{y,z}_E*_size100_nk*.dat`) and gap data (`gap_variation_{y,z}_field.dat`) are in `data/`; exports (`Fig-3`, `gap_variation`, `bands_electrci_field_*`, `band_evolution_vs_Ey_*`) are in `plots/`.
- **Sec_IV/Fig_5**: `plot_only_y_and_z_direction.ipynb` plots the field dependence for the y and z directions; `peanut_plot.ipynb` plots the angular dependence in the yz plane (both include spin-orbit-length plots). Convention: $\theta=0^\circ$ is a field along +z, $\theta=90^\circ$ along +y.
- **Sec_IV/Fig_7**: `rashba_modulation.py` evaluates the Rashba coefficient over the valid-region data; `plot.ipynb` produces `Fig-7` and `Fig-7-new`. Inputs (`valid_regions_data_{y,z}.dat`, `Pymablock_{Y,Z}_kx0.001_valid_regions_N100.dat`, `eff_data.npy`) are in `data/`; exports are in `plots/`.
- **Sec_V/Fig_8**: `Ey/` and `Ez/` each contain `main_n_max_with_eigsh_E*.py` and `plot_matrix_elements_data.ipynb`; the combined plot is `plot.ipynb`. `overlap_function/` contains `overlap.py` and `plot_weight.ipynb`. The combined plot reads `paired_splitting_data_joined_N100_{y,z}.dat`, `overlap_data_N100.dat`, and `bulk_energy_o.dat`.
- **Sec_V/Fig_9**: `level_2x1_combined.ipynb` builds the two schematic panels (y-directed and z-directed couplings). Level positions are schematic, not to scale.
- **Sec_V/Table_III**: `main_E_sweep.py` computes the splitting of the two states closest to $E=0$ at $k_x = 0.001$ Å⁻¹, $L = 300$ Å, and $E = 0.05$ V/µm along y and z, first with all parameters on and then with $P_\perp$, $A_5$, $A_6$, or $\Delta_3$ set to zero one at a time (via the switches `p_c`, `A5_c`, `A6_c`, `d3_c` of `build_H_real` in `Hamiltonian_real_april.py`). Results are written to `relative_change_table_N{N}.dat`; tables for `N = 50` and `N = 100` are included.
- **App_B/Fig_10**: `effcetive_mass.py` (filename as in the repository) computes the effective-mass data `eff_data.npy`; `plot.ipynb` plots it and exports `mx_variation.pdf`.
- **Extra**: contains `hamiltonian_bulk.py`, `main_numerical_eff_mass.py`, `band_fit.ipynb`, and the supplied bulk-band datasets.
- Folders contain local Hamiltonian modules (`Hamiltonian_mathematica.py`, `Hamiltonian_mathematica_v2.py`, `Hamiltonian_python.py`, `Hamiltonian_real_april.py`). Keep each script with its module; same-named modules in different folders are not interchangeable.

## Software and execution

The Python workflows use NumPy, SciPy, Matplotlib, Kwant, Pymablock, and tqdm, with Jupyter for the notebooks. The MPI script `Sec_IV/Fig_6/rashba_with_E_pymablock_kwant_mpi.py` additionally needs mpi4py and an MPI runtime. The `.wls` script in `Sec_II/Fig_1` requires Wolfram Language. Package versions are not pinned.

To regenerate a plot from saved data, open its notebook with that figure folder as the working directory.

> **Note on `data/` and `plots/`:** the scripts and notebooks in `Sec_IV/FIg_3` and `Sec_IV/Fig_7` still read and write files by bare filename (e.g. `np.loadtxt("gap_variation_y_field.dat")`, `fig.savefig("Fig-7.pdf")`). Until the paths are updated, either prefix them with `data/` or `plots/`, or copy the needed files back to the figure folder before running. To rerun a calculation:

1. Review the script's parameters (grid size `N`, nanowire size `L`, momentum range, field direction and strength, process count).
2. Run from inside the figure folder so local imports and relative data paths resolve.
3. Check output filenames first: scripts may overwrite existing datasets.
4. Use the matching notebook to plot the new data.

```bash
cd Sec_II/Fig_1
python main_subband_parallelised.py
```

Several scripts use multiprocessing or MPI. The `.mpi` files are cluster-specific Slurm submission scripts; adapt resources, environment paths, and launch commands to your system.

Conventions: lengths are in angstroms and momenta in inverse angstroms (e.g. `L = 300` is 30 nm). Electric fields are converted between V/µm and V/Å; plotted energies may be in eV, meV, or µeV. Check each script and dataset header for the exact convention. Current script defaults need not reproduce every stored parameter sweep without modification.

## Data and figure files

- `.dat`: saved numerical datasets; column layouts vary by calculation.
- `.npy`: NumPy arrays (e.g. effective-mass data in `App_B/Fig_10` and `Sec_IV/Fig_7/data`, bulk levels in `Sec_V/Fig_8/Ey`).
- `.ipynb`: plotting and analysis notebooks, sometimes with several figure variants.
- `.pdf`, `.svg`, `.png`: figure exports, panels, and diagnostic plots.

## Citation

See [`CITATION.cff`](CITATION.cff).
