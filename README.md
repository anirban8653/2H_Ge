# 2H-Ge Nanowires: Code, Data, and Figures

This repository contains the numerical codes, data files, and plots supporting the accompanying research paper on hexagonal germanium (2H-Ge) nanowires.

The repository follows the structure of the paper. Top-level folders correspond to individual sections or appendices, and figure subfolders collect the code, data, and plots associated with the corresponding figures in the paper.

## Repository organization

| Folder | Corresponding part of the paper |
| --- | --- |
| `Sec_II_Band_structure/` | Section II: Band structure |
| `Sec_III_Zeeman/` | Section III: Zeeman response |
| `Sec_IV_Large_elctric_field/` | Section IV: Large electric fields |
| `Sec_V_weak_field/` | Section V: Weak-field regime |
| `Sec_VI_microscopic_origin/` | Section VI: Microscopic origin |
| `App_B_mx_modulation/` | Appendix B: Effective-mass modulation |
| `comapre_with_DFT/` | Comparisons with density functional theory (DFT) calculations |

## Finding the files for a figure

Each section folder may contain multiple subfolders named `Fig_<number>`, where `<number>` is the figure number in the paper. For example:

| Path | Contents |
| --- | --- |
| `Sec_II_Band_structure/Fig_1/` | Code, data, and plots associated with Figure 1 |
| `Sec_III_Zeeman/Fig_2/` | Code, data, and plots associated with Figure 2 |
| `App_B_mx_modulation/Fig_12/` | Code, data, and plots associated with Figure 12 |

Within a figure folder, the files provide the numerical calculations, the associated datasets, and the plots used in the paper. The exact file layout may vary between figures.

## Using the repository

1. Locate the section containing the result of interest in the paper.
2. Open the corresponding section folder and figure subfolder.
3. Consult the supplied plots and data to inspect the reported results.
4. To repeat or modify a calculation, review the relevant code for its dependencies, input files, parameter choices, and output paths before running it.

Keep the folder structure intact when downloading or cloning the repository, as scripts may use relative paths to access data files.
