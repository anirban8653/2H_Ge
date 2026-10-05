# Figure 10 — Microscopic origin of the Rashba coupling

This directory contains the calculation scripts, saved data, and plotting notebooks for the intermediate-subband contributions to Rashba splitting and their bulk-band character.

| Folder | Contents | Calculation and plotting files |
| --- | --- | --- |
| [Ey/](Ey/) | Matrix elements and dominant intermediate-subband contributions for an electric field along y. | `main_n_max_with_eigsh_Ey.py`, `plot_matrix_elements_data.ipynb` |
| [Ez/](Ez/) | Matrix elements and dominant intermediate-subband contributions for an electric field along z. | `main_n_max_with_eigsh_Ez.py`, `plot_matrix_elements_data.ipynb` |
| [overlap_function/](overlap_function/) | Bulk-band overlap weights used to identify the character of the confined states. | `overlap.py`, `plot_weight.ipynb` |

## Combined plots

[plot.ipynb](plot.ipynb) reads the joined datasets in this directory:

- `paired_splitting_data_joined_N100_y.dat` and `paired_splitting_data_joined_N100_z.dat`: paired contributions for the two field directions.
- `overlap_data_N100.dat`: bulk-band character weights.
- `bulk_energy_o.dat`: bulk-energy reference lines.

The notebook contains a three-panel plot of the y and z contributions and the overlap weights, as well as a two-panel variant containing only the splitting contributions. The stored [splitting_energy_plot.pdf](splitting_energy_plot.pdf) is available for inspection.

The annotations identify the dominant states as VB₂(2,1), VB₁(2,1), and CB₂(1,1) for the y direction, and VB₂(2,1), VB₁(2,1), and VB₁(1,2) for the z direction. The plotted contributions are labeled for an electric field of 0.05 V/µm and $k_x=0.001$ Å⁻¹.

Run each calculation from its own subfolder with its accompanying Hamiltonian module. Open the combined plotting notebook from this Figure 10 directory so that its relative data paths resolve correctly.

The associated schematic of the coupling processes is in [Figure 11](../Fig_11/).
