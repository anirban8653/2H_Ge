import numpy as np

# constants
e = 1.602e-19
hbar = 1.05457e-34
m0 = 9.109e-31
a0 = 5.292e-11

Ev = -0.3622
Ecbp1 = 0.632
Ecb = 0.298

p_perp = 0.4829 * hbar/a0/(e*1e-10)
p_parallel = 0.6431 * hbar/a0/(e*1e-10)

eff_mass_unit = hbar**2/(2*m0)

A_c1_parallel = 1.0 * eff_mass_unit/(e*1e-20)
A_c2_perp = 4.1565 * eff_mass_unit/(e*1e-20)
A_c1_perp = 9.5120 * eff_mass_unit/(e*1e-20)
A_c2_parallel = 2.4091 * eff_mass_unit/(e*1e-20)

A_1 = -4.3636 * eff_mass_unit/(e*1e-20)
A_2 = -2.0833 * eff_mass_unit/(e*1e-20)
A_3 = 2.4545 * eff_mass_unit/(e*1e-20)
A_4 = -2.7504 * eff_mass_unit/(e*1e-20)
A_5 = -2.7232 * eff_mass_unit/(e*1e-20)
A_6 = -3.5421 * eff_mass_unit/(e*1e-20)

Delta1 = 0.2688
Delta2 = 0.0934
Delta3 = 0.0908

def build_H_real(kx,N,L):
    a = L / (N + 1)


    H = {}

    def add(dy, dz, mat):
        key = (dy, dz)
        if key not in H:
            H[key] = np.zeros((10,10), dtype=complex)
        H[key] += mat

    def B(i,j,val):
        M = np.zeros((10,10), dtype=complex)
        M[i,j] = val
        return M

    # -------- kx ----------
    kx_sin = np.sin(kx*a)/a
    kx2 = 2*(1 - np.cos(kx*a))/(a**2)

    # -------- ky ----------
    ky_lin = 1/(2*a)
    ky2_onsite = 2/(a**2)
    ky2_hop = -1/(a**2)

    # -------- kz ----------
    kz_lin = 1/(2*a)
    kz2_onsite = 2/(a**2)
    kz2_hop = -1/(a**2)

    # =========================================================
    # ----------- DIAGONAL TERMS ------------------------------
    # =========================================================
    
    
    # C2 (bands 0 and 5)
    C2 = A_c2_perp*(kx2 + ky2_onsite) + A_c2_parallel*kz2_onsite

    for i in [0,5]:
        add(0,0, B(i,i, C2 + Ecbp1))
        add(+1,0, B(i,i, A_c2_perp*ky2_hop))
        add(-1,0, B(i,i, A_c2_perp*ky2_hop))
        add(0,+1, B(i,i, A_c2_parallel*kz2_hop))
        add(0,-1, B(i,i, A_c2_parallel*kz2_hop))

    # C1 (bands 1 and 6)
    C1 = A_c1_perp*(kx2 + ky2_onsite) + A_c1_parallel*kz2_onsite

    for i in [1,6]:
        add(0,0, B(i,i, C1 + Ecb))
        add(+1,0, B(i,i, A_c1_perp*ky2_hop))
        add(-1,0, B(i,i, A_c1_perp*ky2_hop))
        add(0,+1, B(i,i, A_c1_parallel*kz2_hop))
        add(0,-1, B(i,i, A_c1_parallel*kz2_hop))

    
    # V1 (bands 2 and 7)
    beta = A_1*kz2_onsite + A_2*(kx2 + ky2_onsite)
    alpha = A_3*kz2_onsite + A_4*(kx2 + ky2_onsite)
    for i in [2,7]:
        add(0,0, B(i,i, Delta1 + Delta2 + Ev + alpha + beta))
        add(+1,0, B(i,i, A_2*ky2_hop + A_4*ky2_hop))
        add(-1,0, B(i,i, A_2*ky2_hop + A_4*ky2_hop))
        add(0,+1, B(i,i, A_1*kz2_hop + A_3*kz2_hop))
        add(0,-1, B(i,i, A_1*kz2_hop + A_3*kz2_hop))

    # V2 (bands 3 and 8)
    for i in [3,8]:
        add(0,0, B(i,i, Delta1 - Delta2 + Ev + alpha + beta))
        add(+1,0, B(i,i, A_2*ky2_hop + A_4*ky2_hop))
        add(-1,0, B(i,i, A_2*ky2_hop + A_4*ky2_hop))
        add(0,+1, B(i,i, A_1*kz2_hop + A_3*kz2_hop))
        add(0,-1, B(i,i, A_1*kz2_hop + A_3*kz2_hop))


    # V3 (bands 4 and 9)
    for i in [4,9]:
        add(0,0, B(i,i, Ev + beta))
        add(+1,0, B(i,i, A_2*ky2_hop ))
        add(-1,0, B(i,i, A_2*ky2_hop ))
        add(0,+1, B(i,i, A_1*kz2_hop ))
        add(0,-1, B(i,i, A_1*kz2_hop ))


    # =========================================================
    # ----------- A5 TERMS (k±²) ------------------------------
    # =========================================================

    # k_plus^2 = (kx + i ky)^2
    # = kx^2 - ky^2 + 2i kx ky

    # ky² → NN hopping
    # ky → NN hopping

    def kpm2_terms(i,j,coefA5):
        # onsite part (kx² + ky²)
        onsite = coefA5*(kx2 + ky2_onsite)
        add(0,0, B(i,j, onsite))

        # ky² hopping
        add(+1,0, B(i,j, coefA5*ky2_hop))
        add(-1,0, B(i,j, coefA5*ky2_hop))

    # apply
    kpm2_terms(3,2, -A_5)
    kpm2_terms(7,8, -A_5)
    kpm2_terms(2,3, -A_5)
    kpm2_terms(8,7, -A_5)
    
    # =========================================================
    # ----------- SPIN-ORBIT -------------------------------
    # =========================================================

    add(0,0, B(3,9, np.sqrt(2)*Delta3))
    add(0,0, B(4,8, np.sqrt(2)*Delta3))
    add(0,0, B(9,3, np.sqrt(2)*Delta3))
    add(0,0, B(8,4, np.sqrt(2)*Delta3))

    
    # =========================================================
    # ----------- k·p COUPLING -------------------------------
    # =========================================================

    coef = (np.sqrt(2)*hbar*p_perp)/(2*m0)

    # k+
    add(0,0, B(0,2, -coef*kx_sin))
    add(0,0, B(2,0, -coef*kx_sin))
    add(0,0, B(5,7, coef*kx_sin))
    add(0,0, B(7,5, coef*kx_sin))
    add(0,0, B(0,3, coef*kx_sin))
    add(0,0, B(3,0, coef*kx_sin))
    add(0,0, B(5,8, -coef*kx_sin))
    add(0,0, B(8,5, -coef*kx_sin))
    
    add(+1,0, B(0,2, coef*(ky_lin)))
    add(+1,0, B(5,7, -coef*(-ky_lin)))
    add(+1,0, B(2,0, +coef*(-ky_lin)))
    add(+1,0, B(7,5, -coef*(ky_lin)))
    
    add(+1,0, B(0,3, coef*(ky_lin)))
    add(+1,0, B(5,8, -coef*(-ky_lin)))
    add(+1,0, B(3,0, coef*(-ky_lin)))
    add(+1,0, B(8,5, -coef*(ky_lin)))
    
    add(-1,0, B(0,2, -coef*(ky_lin)))
    add(-1,0, B(5,7, -coef*(ky_lin)))
    add(-1,0, B(2,0, coef*(ky_lin)))
    add(-1,0, B(7,5, coef*(ky_lin)))
    
    add(-1,0, B(0,3, -coef*(ky_lin)))
    add(-1,0, B(5,8, -coef*(ky_lin)))
    add(-1,0, B(3,0, coef*(ky_lin)))
    add(-1,0, B(8,5, coef*(ky_lin)))
    
    

    # # kz coupling
    coef_z = (hbar*p_parallel)/m0
    add(0,+1, B(0,4, 1j*coef_z*kz_lin))
    add(0,+1, B(4,0, 1j*coef_z*kz_lin))
    add(0,+1, B(5,9, 1j*coef_z*kz_lin))
    add(0,+1, B(9,5, 1j*coef_z*kz_lin))
    
    add(0,-1, B(0,4, -1j*coef_z*kz_lin))
    add(0,-1, B(4,0, -1j*coef_z*kz_lin))
    add(0,-1, B(5,9, -1j*coef_z*kz_lin))
    add(0,-1, B(9,5, -1j*coef_z*kz_lin))
    
    
    # =========================================================
    # ----------- A6 TERMS (ky kz) ----------------------------
    # =========================================================

    # ky kz → hopping in BOTH directions


    coeff_a6 = 1j/(2*a)
    coeff_a62 = 1j/(4*a**2)
    
    #----------------------k_-kz
    add(0, -1, B(2,4, -A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(2,4, -A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(2,4, -A_6* coeff_a62 ))
    add(1, -1, B(2,4, -A_6* -coeff_a62 ))
    add(-1, 1, B(2,4, -A_6* -coeff_a62 ))
    add(1, 1, B(2,4, -A_6* coeff_a62 ))
    
    add(0, -1, B(8,9, -A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(8,9, -A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(8,9, -A_6* coeff_a62 ))
    add(1, -1, B(8,9, -A_6* -coeff_a62 ))
    add(-1, 1, B(8,9, -A_6* -coeff_a62 ))
    add(1, 1, B(8,9, -A_6* coeff_a62 ))
    
    add(0, -1, B(4,3, A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(4,3, A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(4,3, A_6* coeff_a62 ))
    add(1, -1, B(4,3, A_6* -coeff_a62 ))
    add(-1, 1, B(4,3, A_6* -coeff_a62 ))
    add(1, 1, B(4,3, A_6* coeff_a62 ))
    
    add(0, -1, B(9,7, A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(9,7, A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(9,7, A_6* coeff_a62 ))
    add(1, -1, B(9,7, A_6* -coeff_a62 ))
    add(-1, 1, B(9,7, A_6* -coeff_a62 ))
    add(1, 1, B(9,7, A_6* coeff_a62 ))
    
    #---------------------k_+kz-----------------------
    add(0, -1, B(3,4, A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(3,4, A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(3,4, A_6* -coeff_a62 ))
    add(1, -1, B(3,4, A_6* coeff_a62 ))
    add(-1, 1, B(3,4, A_6* coeff_a62 ))
    add(1, 1, B(3,4, A_6* -coeff_a62 ))
    
    add(0, -1, B(7,9, A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(7,9, A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(7,9, A_6* -coeff_a62 ))
    add(1, -1, B(7,9, A_6* coeff_a62 ))
    add(-1, 1, B(7,9, A_6* coeff_a62 ))
    add(1, 1, B(7,9, A_6* -coeff_a62 ))
    
    add(0, -1, B(4,2, -A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(4,2, -A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(4,2, -A_6* -coeff_a62 ))
    add(1, -1, B(4,2, -A_6* coeff_a62 ))
    add(-1, 1, B(4,2, -A_6* coeff_a62 ))
    add(1, 1, B(4,2, -A_6* -coeff_a62 ))
    
    add(0, -1, B(9,8, -A_6* -coeff_a6 * kx_sin))
    add(0, 1, B(9,8, -A_6* coeff_a6 * kx_sin))
    add(-1, -1, B(9,8, -A_6* -coeff_a62 ))
    add(1, -1, B(9,8, -A_6* coeff_a62 ))
    add(-1, 1, B(9,8, -A_6* coeff_a62 ))
    add(1, 1, B(9,8, -A_6* -coeff_a62 ))
    

    return H

# np.set_printoptions(precision=3, suppress=True, linewidth=150)
# print(build_H_real(0.01, N, L)[(0,-1)])

