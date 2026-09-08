# ---------- Parameters ----------
# hbar = 1   # Reduced Planck constant in J·s
# m0 = 1  # Electron rest mass in kg
# a0 = 1  # Bohr radius in meters
# energy_scale = 27.2113862 * hbar **2 / m0 / a0**2  # Conversion factor from hbar^2/m_e/a_0^2 to eV

import numpy as np

e = 1.602e-19 # Elementary charge in Coulombs
hbar = 1.05457e-34 # Reduced Planck constant in J·s
m0 = me = 9.109e-31 # Electron rest mass in kg
a0 = 5.292e-11 # Bohr radius in meters
# switch = int(input("Enter kinetic energy: "))
# energy_scale = 27.2113862 * hbar **2 / m0 / a0**2  # Conversion factor from hbar^2/m_e/a_0^2 to eV

# params_no_coupling = {
#     "p_perp": 0, #0.4829 * hbar / a0/(e*1e-10),
#     "p_parallel": 0,#0.6431 * hbar / a0/(e*1e-10),
#     "A_c2_perp": 4.1565 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c1_perp": 9.5120 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c2_parallel": 2.4091 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c1_parallel": 1 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_1": -4.3636 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_2": -2.0833 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_3": 2.4545 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_4": -2.7504 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_5": 0,#-2.7232 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_6": 0,#-3.5421 * hbar**2 / (2 * m0)/(e*1e-20),
#     "Delta_1": 0.2688 ,
#     "Delta_2": 0.0934 ,
#     "Delta_3": 0,#0.0908 ,
#     "Ecbp1": 0.632 ,
#     "Ecb": 0.298 ,
#     "Ev": -0.3622 ,
#     "hbar": hbar,
#     "m0": m0,
# }

# params_coupling_m = {
#     "p_perp": 0.4829 * hbar / a0/(e*1e-10),
#     "p_parallel": 0.6431 * hbar / a0/(e*1e-10),
#     "A_c2_perp": 4.1565 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c1_perp": 9.5120 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c2_parallel": 2.4091 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c1_parallel": 1 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_1": -4.3636 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_2": -2.0833 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_3": 2.4545 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_4": -2.7504 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_5": -2.7232 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_6": -3.5421 * hbar**2 / (2 * m0)/(e*1e-20),
#     "Delta_1": 0.2688 ,
#     "Delta_2": 0.0934 ,
#     "Delta_3": 0.0908 ,
#     "Ecbp1": 0.632 ,
#     "Ecb": 0.298 ,
#     "Ev": -0.3622 ,
#     "hbar": hbar,
#     "m0": m0,
# }

# params_coupling_p = {
#     "p_perp": 0.4829 * hbar / a0/(e*1e-10),
#     "p_parallel": 0.6431 * hbar / a0/(e*1e-10),
#     "A_c2_perp": 4.1565 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c1_perp": 9.5120 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c2_parallel": 2.4091 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_c1_parallel": 1 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_1": -4.3636 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_2": -2.0833 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_3": 2.4545 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_4": -2.7504 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_5": 2.7232 * hbar**2 / (2 * m0)/(e*1e-20),
#     "A_6": -3.5421 * hbar**2 / (2 * m0)/(e*1e-20),
#     "Delta_1": 0.2688 ,
#     "Delta_2": 0.0934 ,
#     "Delta_3": 0.0908 ,
#     "Ecbp1": 0.632 ,
#     "Ecb": 0.298 ,
#     "Ev": -0.3622 ,
#     "hbar": hbar,
#     "m0": m0,
# }

# ---------- Hamiltonian builder ----------
def Hfull(kx, ky, kz, p):
    k_plus = kx + 1j*ky
    k_minus = kx - 1j*ky

    C1 = p["A_c1_perp"]*(kx**2 + ky**2) + p["A_c1_parallel"]*kz**2
    C2 = p["A_c2_perp"]*(kx**2 + ky**2) + p["A_c2_parallel"]*kz**2
    beta = p["A_1"]*kz**2 + p["A_2"]*(kx**2 + ky**2)
    alpha = p["A_3"]*kz**2 + p["A_4"]*(kx**2 + ky**2)
    
    # if switch == 0:
    #     kin = 0
    # if switch == 1:
    #     kin = p["hbar"]**2 * (kx**2 + ky**2 + kz**2) / (2*p["m0"]*e*1e-20)
    kin = 0

    H = np.zeros((10,10), dtype=complex)

    H[0,0] = kin + C2 + p["Ecbp1"]
    H[0,2] = -(np.sqrt(2)*p["hbar"]*k_plus*p["p_perp"])/(2*p["m0"])
    H[0,3] = (np.sqrt(2)*p["hbar"]*k_minus*p["p_perp"])/(2*p["m0"])
    H[0,4] = (p["hbar"]*kz*p["p_parallel"])/p["m0"]

    H[1,1] = kin + C1 + p["Ecb"]

    H[2,0] = -(np.sqrt(2)*p["hbar"]*k_minus*p["p_perp"])/(2*p["m0"])
    H[2,2] = kin + p["Delta_1"] + p["Delta_2"] + p["Ev"] + alpha + beta
    H[2,3] = -p["A_5"]*k_minus**2
    H[2,4] = -p["A_6"]*k_minus*kz

    H[3,0] = (np.sqrt(2)*p["hbar"]*k_plus*p["p_perp"])/(2*p["m0"])
    H[3,2] = -p["A_5"]*k_plus**2
    H[3,3] = kin + p["Delta_1"] - p["Delta_2"] + p["Ev"] + alpha + beta
    H[3,4] = p["A_6"]*k_plus*kz
    H[3,9] = np.sqrt(2)*p["Delta_3"]

    H[4,0] = (p["hbar"]*kz*p["p_parallel"])/p["m0"]
    H[4,2] = -p["A_6"]*k_plus*kz
    H[4,3] = p["A_6"]*k_minus*kz
    H[4,4] = kin + p["Ev"] + beta
    H[4,8] = np.sqrt(2)*p["Delta_3"]

    H[5,5] = kin + C2 + p["Ecbp1"]
    H[5,7] = (np.sqrt(2)*p["hbar"]*k_minus*p["p_perp"])/(2*p["m0"])
    H[5,8] = -(np.sqrt(2)*p["hbar"]*k_plus*p["p_perp"])/(2*p["m0"])
    H[5,9] = (p["hbar"]*kz*p["p_parallel"])/p["m0"]

    H[6,6] = kin + C1 + p["Ecb"]

    H[7,5] = (np.sqrt(2)*p["hbar"]*k_plus*p["p_perp"])/(2*p["m0"])
    H[7,7] = kin + p["Delta_1"] + p["Delta_2"] + p["Ev"] + alpha + beta
    H[7,8] = -p["A_5"]*k_plus**2
    H[7,9] = p["A_6"]*k_plus*kz

    H[8,4] = np.sqrt(2)*p["Delta_3"]
    H[8,5] = -(np.sqrt(2)*p["hbar"]*k_minus*p["p_perp"])/(2*p["m0"])
    H[8,7] = -p["A_5"]*k_minus**2
    H[8,8] = kin + p["Delta_1"] - p["Delta_2"] + p["Ev"] + alpha + beta
    H[8,9] = -p["A_6"]*k_minus*kz

    H[9,3] = np.sqrt(2)*p["Delta_3"]
    H[9,5] = (p["hbar"]*kz*p["p_parallel"])/p["m0"]
    H[9,7] = p["A_6"]*k_minus*kz
    H[9,8] = -p["A_6"]*k_plus*kz
    H[9,9] = kin + p["Ev"] + beta

    # Hermitianize (safeguard against numerical asymmetry)
    H = (H + H.conj().T)/2

    return H


def eff_mass_cb(kdata, bdata):
    """
    Compute the effective mass (m*) near the band extremum.
    Fits a quadratic to three points around the band minimum.
    Returns m*/m_e (effective mass in units of electron mass).
    """
    idx_max = np.argmin(bdata)  # locate extremum

    # handle edge cases
    if idx_max == 0:
        idx = 1
    elif idx_max == len(bdata) - 1:
        idx = len(bdata) - 2
    else:
        idx = idx_max

    # select three nearby points
    k_subset = kdata[idx-1:idx+2]
    e_subset = bdata[idx-1:idx+2]

    # quadratic fit: E = p[0]*k^2 + p[1]*k + p[2]
    p = np.polyfit(k_subset, e_subset, 2)

    # 5. Physical Constants and Calculation
    hbar = 1.05457e-34
    me = 9.109e-31

    # 1 eV * Angstrom^2 = 1.602e-39 J * m^2
    dim = 1.602e-39 

    # m* = ħ² / (2a), where a = p[0]
    m_eff = hbar**2 / (2 * p[0] * dim)
    m_eff_r = m_eff / me

    return kdata[idx_max], m_eff_r




def eff_mass_vb(kdata, bdata):
    """
    Compute the effective mass (m*) near the band extremum.
    Fits a quadratic to three points around the band maximum.
    Returns m*/m_e (effective mass in units of electron mass).
    """
    idx_max = np.argmax(bdata)  # locate extremum

    # handle edge cases
    if idx_max == 0:
        idx = 1
    elif idx_max == len(bdata) - 1:
        idx = len(bdata) - 2
    else:
        idx = idx_max

    # select three nearby points
    k_subset = kdata[idx-1:idx+2]
    e_subset = bdata[idx-1:idx+2]

    # quadratic fit: E = p[0]*k^2 + p[1]*k + p[2]
    p = np.polyfit(k_subset, e_subset, 2)

    # 5. Physical Constants and Calculation
    hbar = 1.05457e-34
    me = 9.109e-31

    # 1 eV * Angstrom^2 = 1.602e-39 J * m^2
    dim = 1.602e-39 

    # m* = ħ² / (2a), where a = p[0]
    m_eff = hbar**2 / (2 * p[0] * dim)
    m_eff_r = m_eff / me

    return kdata[idx_max], m_eff_r