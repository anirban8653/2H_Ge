import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

kpoints = np.linspace(-0.05, 0.05, 201)

results = {}  # keep both parameter sets' data alive

fig, ax = plt.subplots(figsize=(4,5))
colors = {0: 'C0', 1: 'C1'}

for switch, tab in enumerate([2, 5]):
    if switch == 0:
        from hamiltonian_bulk_tab2 import Hfull, params_coupling, eff_mass_cb, eff_mass_vb
    if switch == 1:
        from hamiltonian_bulk_tab5 import Hfull, params_coupling, eff_mass_cb, eff_mass_vb

    evaldatax = []
    evaldataz = []

    for kx in kpoints:
        H = Hfull(kx, 0, 0, p=params_coupling)
        eval, evec = eigh(H)
        evaldatax.append(eval)

    for kz in kpoints:
        H = Hfull(0, 0, kz, p=params_coupling)
        eval, evec = eigh(H)
        evaldataz.append(eval)

    evaldatax = np.array(evaldatax)
    evaldataz = np.array(evaldataz)

    results[switch] = {
        'evaldatax': evaldatax,
        'evaldataz': evaldataz,
        'eff_mass_cb': eff_mass_cb,   # carry the correct function forward
        'eff_mass_vb': eff_mass_vb,
    }

    ax.plot(kpoints, evaldatax, color=colors[switch])
    ax.plot([], [], color=colors[switch], label=f'tab {tab}')  # legend entry

ax.set_xlabel(r"k$_x$ [$\AA^{-1}$]")
ax.set_ylabel(r"Energy [eV]")
ax.legend()
plt.savefig("bulk_eff_mass_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# EFFECTIVE MASS FUNCTION
# ============================================================

for switch, tab in enumerate([2, 5]):
    vb1_data_x = results[switch]['evaldatax'][:, 5]
    cb1_data_x = results[switch]['evaldatax'][:, 6]
    vb1_data_z = results[switch]['evaldataz'][:, 5]
    cb1_data_z = results[switch]['evaldataz'][:, 6]

    eff_mass_cb = results[switch]['eff_mass_cb']
    eff_mass_vb = results[switch]['eff_mass_vb']

    mcbx = eff_mass_cb(kpoints, cb1_data_x)
    mvbx = eff_mass_vb(kpoints, vb1_data_x)
    mcbz = eff_mass_cb(kpoints, cb1_data_z)
    mvbz = eff_mass_vb(kpoints, vb1_data_z)

    print(f"========================")
    print(f"tab {tab}")
    print(f"========================")
    print("-----------------------")
    print("Parallel Effective mass (x/y)")
    print("-----------------------")
    print("cb : ", mcbx[1])
    print("vb : ", mvbx[1])

    print("-----------------------")
    print("Perpendicular Effective mass (z)")
    print("-----------------------")
    print("cb : ", mcbz[1])
    print("vb : ", mvbz[1])




# ============================================================
# EFFECTIVE MASS TABLE : TABLE 2 vs TABLE 5
# ============================================================

def get_effective_mass_data(switch):

    eff_mass_cb = results[switch]['eff_mass_cb']
    eff_mass_vb = results[switch]['eff_mass_vb']

    # Valence and conduction bands
    vb_data_x = results[switch]['evaldatax'][:, 5]
    cb_data_x = results[switch]['evaldatax'][:, 6]

    vb_data_z = results[switch]['evaldataz'][:, 5]
    cb_data_z = results[switch]['evaldataz'][:, 6]

    # Effective masses
    mvbx = eff_mass_vb(kpoints, vb_data_x)
    mcbx = eff_mass_cb(kpoints, cb_data_x)

    mvbz = eff_mass_vb(kpoints, vb_data_z)
    mcbz = eff_mass_cb(kpoints, cb_data_z)

    return {
        'vb_perp': abs(mvbx[1]),   # x/y : perpendicular to c-axis
        'vb_par':  abs(mvbz[1]),   # z   : parallel to c-axis
        'cb_perp': abs(mcbx[1]),
        'cb_par':  abs(mcbz[1]),
    }


# Table 2 and Table 5 data
table2_data = get_effective_mass_data(0)
table5_data = get_effective_mass_data(1)


# ============================================================
# DRAW TABLE
# ============================================================

def plot_effective_mass_tables(table2_data, table5_data,
                               fname="effective_mass_table2_table5.png"):

    fig, ax = plt.subplots(figsize=(7, 6.5))

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Column widths
    col_edges = [0, 1.1, 2.45, 3.0]

    row_h = 1.0

    # --------------------------------------------------------
    # Draw one normal row
    # --------------------------------------------------------
    def draw_row(y, cells, bold=False, fontsize=12):

        for (x0, x1), text in zip(
                zip(col_edges[:-1], col_edges[1:]), cells):

            ax.add_patch(
                plt.Rectangle(
                    (x0, y),
                    x1 - x0,
                    row_h,
                    fill=False,
                    edgecolor='black',
                    lw=1
                )
            )

            ax.text(
                (x0 + x1) / 2,
                y + row_h / 2,
                text,
                ha='center',
                va='center',
                fontsize=fontsize,
                fontweight='bold' if bold else 'normal'
            )


    # --------------------------------------------------------
    # Draw one complete section
    # --------------------------------------------------------
    def draw_section(y_top, title, data):

        y = y_top

        # -------------------------
        # Section title
        # -------------------------
        ax.add_patch(
            plt.Rectangle(
                (0, y),
                3,
                row_h,
                fill=False,
                edgecolor='black',
                lw=1
            )
        )

        ax.text(
            1.5,
            y + row_h / 2,
            title,
            ha='center',
            va='center',
            fontsize=14,
            fontweight='bold'
        )

        y -= row_h

        # -------------------------
        # Header
        # -------------------------
        draw_row(
            y,
            [
                "Band",
                r"Direction w.r.t. $c$ axis",
                r"$m_{\mathrm{eff}}/m_0$"
            ],
            fontsize=12
        )

        y -= row_h

        # ====================================================
        # Valence band
        # ====================================================

        y_val_top = y

        # Direction + mass rows
        draw_row(
            y,
            [
                "",
                r"Perpendicular $(x/y)$",
                f"{data['vb_perp']:.4f}"
            ]
        )

        y -= row_h

        draw_row(
            y,
            [
                "",
                r"Parallel $(z)$",
                f"{data['vb_par']:.4f}"
            ]
        )

        # Remove internal line in first column
        ax.plot(
            [0, col_edges[1]],
            [y + row_h, y + row_h],
            color='white',
            lw=2,
            zorder=3
        )

        # Valence-band merged label
        ax.text(
            col_edges[1] / 2,
            y_val_top,
            "Valence band",
            ha='center',
            va='center',
            fontsize=12
        )

        y -= row_h

        # ====================================================
        # Conduction band
        # ====================================================

        y_cond_top = y

        draw_row(
            y,
            [
                "",
                r"Perpendicular $(x/y)$",
                f"{data['cb_perp']:.4f}"
            ]
        )

        y -= row_h

        draw_row(
            y,
            [
                "",
                r"Parallel $(z)$",
                f"{data['cb_par']:.4f}"
            ]
        )

        # Remove internal line in first column
        ax.plot(
            [0, col_edges[1]],
            [y + row_h, y + row_h],
            color='white',
            lw=2,
            zorder=3
        )

        # Conduction-band merged label
        ax.text(
            col_edges[1] / 2,
            y_cond_top,
            "Conduction band",
            ha='center',
            va='center',
            fontsize=12
        )

        y -= row_h

        return y


    # ========================================================
    # TABLE 2 : UPPER
    # ========================================================

    y = 11

    y = draw_section(
        y,
        "Table 2",
        table2_data
    )


    # ========================================================
    # TABLE 5 : LOWER
    # ========================================================

    y = draw_section(
        y,
        "Table 5",
        table5_data
    )


    plt.tight_layout()

    plt.savefig(
        fname,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# ============================================================
# MAKE FIGURE
# ============================================================

plot_effective_mass_tables(
    table2_data,
    table5_data
)