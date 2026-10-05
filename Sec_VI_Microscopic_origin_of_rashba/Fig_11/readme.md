# Figure 11 — Schematic energy-level and coupling diagrams

This directory contains schematic diagrams of the virtual coupling processes associated with the dominant intermediate-subband contributions presented in [Figure 10](../Fig_10/).

The levels illustrate the energy-level structure schematically; their positions and spacings are not drawn to scale and should not be interpreted as quantitative energy values. The notebook specifies the level positions explicitly rather than loading numerical energies from Figure 10.

| File | Contents |
| --- | --- |
| [level_2x1_combined.ipynb](level_2x1_combined.ipynb) | Matplotlib notebook arranging the two selection-rule diagrams vertically as panels (a) and (b). |
| [level_diagram.pdf](level_diagram.pdf) | Exported schematic energy-level and coupling diagrams. |

Panel (a) shows the y-directed electric-field couplings; panel (b) shows the z-directed couplings. Solid blue arrows denote electric-dipole couplings, and dashed black arrows denote the $k_x$-dependent coupling. The diagrams label the bulk-band character, transverse quantum numbers, and symmetry eigenvalues of the states.

To regenerate the schematic, run the notebook in this directory with Matplotlib installed.
