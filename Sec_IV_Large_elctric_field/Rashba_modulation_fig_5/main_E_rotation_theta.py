import os

# Prevent nested BLAS/OpenMP parallelism inside multiprocessing workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from multiprocessing import get_context

import kwant
import numpy as np
from pymablock import block_diagonalize
from pymablock.series import zero as pymablock_zero
from scipy.linalg import eigh, qr
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from Hamiltonian_mathematica_v2 import dgso, gso, psi_new_basis


np.set_printoptions(linewidth=200, suppress=True, precision=5)


# ==============================================================
# Parameters
# ==============================================================
N = 50
Nband = 10
L = 300                       # Angstrom
Ny = Nz = N
kx_value = 0.001              # 1/Angstrom
sigma_val = 0.1
E_list = [1, 2, 3, 4, 5, 6, 7]  # Electric-field magnitudes in V/micrometer

# These two values are updated for each entry of E_list in the main loop.
E_magnitude_V_per_um = None
E_magnitude = None

theta_values = np.linspace(0.0, 360.0, 151)
num_processes = min(
int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
len(theta_values),
) 
# num_processes = 10


lat = kwant.lattice.square(norbs=Nband)


# These objects are constructed once in the main process and inherited by
# the forked worker processes. For every theta, the electric field is added
# to H(kx=0) before Pymablock is called. Only kx is perturbative.
H_kx0_base = None
H_numerical_base = None
V_Ey = None
V_Ez = None
V_kx = None


def get_heff_coefficient(series, index):
    """Return a Pymablock coefficient as a numerical 2 x 2 matrix."""

    coefficient = series[index]
    if coefficient is pymablock_zero:
        return np.zeros((2, 2), dtype=complex)
    return coefficient


# ==============================================================
# Kwant systems
# ==============================================================
def make_system(kx):
    """Construct the field-free confined Hamiltonian at the given kx."""

    syst = kwant.Builder()
    a = L / (Ny + 1)

    gso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            gso_cache[(dy, dz)] = gso(a, kx, dy, dz)

    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = gso_cache[(0, 0)]

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = (
                gso_cache[(dy, dz)]
            )

    return syst.finalized()


def make_system_kx(kx):
    """Construct dH/dkx at the given kx."""

    syst = kwant.Builder()
    a = L / (Ny + 1)

    dgso_cache = {}
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            dgso_cache[(dy, dz)] = dgso(a, kx, dy, dz)

    for y in range(Ny):
        for z in range(Nz):
            syst[lat(y, z)] = dgso_cache[(0, 0)]

    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dy == 0 and dz == 0:
                continue
            syst[kwant.builder.HoppingKind((dy, dz), lat, lat)] = (
                dgso_cache[(dy, dz)]
            )

    return syst.finalized()


def make_system_E(direction):
    """
    Construct the electric-potential matrix for a unit field.

    This matrix is used to build the electric-field-dressed unperturbed
    Hamiltonian. It is not passed to Pymablock as a perturbation.

    Parameters
    ----------
    direction : {"y", "z"}
        Direction of the unit electric field.
    """

    if direction not in {"y", "z"}:
        raise ValueError("direction must be either 'y' or 'z'")

    syst = kwant.Builder()
    a = L / (Ny + 1)

    for y in range(Ny):
        y_position = (y + 1) * a - L / 2
        for z in range(Nz):
            z_position = (z + 1) * a - L / 2
            coordinate = y_position if direction == "y" else z_position
            potential = -coordinate
            syst[lat(y, z)] = potential * np.eye(Nband)

    return syst.finalized()


# ==============================================================
# One angular point
# ==============================================================
def compute_one_theta(theta_deg):
    """
    Calculate numerical and perturbative splittings for one field angle.

    theta = 0 degrees   -> E || z
    theta = 90 degrees  -> E || y
    """

    theta_rad = np.deg2rad(theta_deg)

    Ey = E_magnitude * np.sin(theta_rad)
    Ez = E_magnitude * np.cos(theta_rad)

    # Remove tiny floating-point remnants at the symmetry directions.
    if abs(Ey) < 1.0e-15:
        Ey = 0.0
    if abs(Ez) < 1.0e-15:
        Ez = 0.0

    # ----------------------------------------------------------
    # Full numerical calculation
    # ----------------------------------------------------------
    H0_theta = H_kx0_base + Ey * V_Ey + Ez * V_Ez
    H_numerical = H_numerical_base + Ey * V_Ey + Ez * V_Ez

    numerical_eigenvalues = eigsh(
        H_numerical,
        k=2,
        sigma=sigma_val,
        return_eigenvectors=False,
    )
    numerical_eigenvalues = np.sort(numerical_eigenvalues)

    GS1_meV = numerical_eigenvalues[0] * 1.0e3
    GS2_meV = numerical_eigenvalues[1] * 1.0e3
    numerical_gap_microeV = (
        numerical_eigenvalues[1] - numerical_eigenvalues[0]
    ) * 1.0e6

    # ----------------------------------------------------------
    # Perturbative calculation: the electric field is included exactly
    # in H0_theta, and only kx is treated as the perturbation.
    # ----------------------------------------------------------
    _, evecs = eigsh(
        H0_theta,
        k=2,
        sigma=sigma_val,
    )
    evecs_ortho, _ = qr(evecs, mode="economic")

    psi1 = evecs_ortho[:, 0]
    psi2 = evecs_ortho[:, 1]
    psi_new = np.asarray(psi_new_basis(psi1, psi2, N)).T

    # Variable ordering in this Pymablock series: (kx,).
    Heff_coeffs, *_ = block_diagonalize(
        [H0_theta, V_kx],
        subspace_eigenvectors=[psi_new],
    )

    H00 = get_heff_coefficient(
        Heff_coeffs,
        (0, 0, 0),
    )
    H_k1 = get_heff_coefficient(
        Heff_coeffs,
        (0, 0, 1),
    )

    # Since the field is already contained in H00, H_k1 includes its
    # dependence on both Ey and Ez to all orders within the model.
    # For the Kramers pair, Delta E = 2 alpha |kx| at leading order.
    alpha_eVA = 0.5 * np.ptp(
        eigh(H_k1, eigvals_only=True)
    )

    H_eff_k1 = H00 + H_k1 * kx_value
    
    perturbative_eigenvalues_k1 = eigh(
        H_eff_k1,
        eigvals_only=True,
    )

    perturbative_gap_k1_microeV = np.ptp(
        perturbative_eigenvalues_k1
    ) * 1.0e6

    return (
        float(theta_deg),
        float(Ey * 1.0e4),
        float(Ez * 1.0e4),
        float(GS1_meV),
        float(GS2_meV),
        float(numerical_gap_microeV),
        float(alpha_eVA),
        float(perturbative_gap_k1_microeV)
    )


# ==============================================================
# Main calculation
# ==============================================================
if __name__ == "__main__":
    t0 = time.time()

    print(f"\nBuilding systems for N = {N}, L = {L} Angstrom")
    print(f"kx = {kx_value} 1/Angstrom")
    print(f"Electric-field list = {E_list} V/micrometer")
    print(
        f"Angular sweep: {theta_values[0]:.1f} to "
        f"{theta_values[-1]:.1f} degrees in {len(theta_values)} points"
    )

    # ----------------------------------------------------------
    # Base matrices shared by all angular workers
    # ----------------------------------------------------------
    H_kx0_base = make_system(
        0.0
    ).hamiltonian_submatrix(sparse=True)
    H_numerical_base = make_system(
        kx_value
    ).hamiltonian_submatrix(sparse=True)
    V_Ey = make_system_E("y").hamiltonian_submatrix(sparse=True)
    V_Ez = make_system_E("z").hamiltonian_submatrix(sparse=True)
    V_kx = make_system_kx(0.0).hamiltonian_submatrix(sparse=True)

    print(
        "\nFor each theta, Pymablock uses the electric-field-dressed "
        "H(kx=0) as H0 and treats only kx perturbatively."
    )

    header = (
        "theta_deg  Ey_V_per_um  Ez_V_per_um  "
        "GS1_meV  GS2_meV  numerical_gap_microeV  "
        "alpha_eV_A  perturbative_gap_k1_microeV  "
    )

    # ----------------------------------------------------------
    # Electric-field and angular sweeps
    # ----------------------------------------------------------
    # A new forked pool is created for each field magnitude so that every
    # worker inherits the current values of E_magnitude_V_per_um and
    # E_magnitude together with the already constructed sparse matrices.
    multiprocessing_context = get_context("fork")

    for E_value in E_list:
        field_t0 = time.time()

        E_magnitude_V_per_um = float(E_value)
        E_magnitude = E_magnitude_V_per_um / 1.0e4  # V/Angstrom

        print("\n" + "=" * 60)
        print(f"|E| = {E_magnitude_V_per_um:g} V/micrometer")
        print("=" * 60)

        angular_results = []
        with multiprocessing_context.Pool(num_processes) as pool:
            iterator = pool.imap_unordered(
                compute_one_theta,
                theta_values,
            )
            for result in tqdm(
                iterator,
                total=len(theta_values),
                desc=f"Field {E_magnitude_V_per_um:g} V/um",
            ):
                angular_results.append(result)

        angular_results = np.asarray(
            sorted(angular_results, key=lambda row: row[0])
        )

        output_file = (
            f"E_rotation_N{N}_L{L}_"
            f"E{E_magnitude_V_per_um:.2f}Vum_"
            f"kx{kx_value}.dat"
        )

        np.savetxt(
            output_file,
            angular_results,
            header=header,
        )

        print(f"\nData saved: {output_file}")
        print("\nSelected angular points:")
        print(header)
        for target_theta in [0.0, 90.0, 180.0, 270.0, 360.0]:
            index = np.argmin(
                np.abs(angular_results[:, 0] - target_theta)
            )
            print(angular_results[index])

        field_minutes, field_seconds = divmod(time.time() - field_t0, 60)
        print(
            f"\nField execution time: {int(field_minutes)} min "
            f"{field_seconds:.2f} sec"
        )

    t1 = time.time()
    minutes, seconds = divmod(t1 - t0, 60)
    print(f"\nTotal execution time: {int(minutes)} min {seconds:.2f} sec")
