import sys
import math
import numpy as np

'''
Function: lattice_parameter
Sets up the lattice parameter array. Converts from Angstrom to au.
Parameters:
    - crystal_index - the index corresponding to the desired crystal.
'''
def lattice_parameter(crystal_index):
    # Set up the lattice parameter array in Angstrom
    a0 = np.array([5.43, 5.66, 6.49, 5.64, 5.41, 6.41])

    # Return the lattice parameter converted to au.
    return a0[crystal_index]/0.529


'''
Function: atomic_form_factor
Sets up the atomic form factor matrix.
Returns the corresponding atomic form factors.
Parameters:
    - crystal_index - the index corresponding to the desired crystal.
    - K_minus_Kp_squared - this is (K - K')^2
'''
def atomic_form_factor(crystal_index, K_minus_Kp_squared):
    # Set up the atomic form factor matrix.
    aff = np.array([[-0.21, 0.04, 0.08, 0, 0, 0],
                    [-0.23, 0.01, 0.06, 0, 0, 0],
                    [-0.20, 0, 0.04, 0, 0, 0],
                    [-0.23, 0.01, 0.06, 0.07, 0.05, 0.01],
                    [-0.22, 0.03, 0.07, 0.24, 0.14, 0.04],
                    [-0.20, 0, 0.04, 0.15, 0.09, 0.04]])

    # Assign the atomic form factors for (K - K')^2.
    if abs(K_minus_Kp_squared - 3) < 1e-6:
        V_S = aff[crystal_index, 0]
        V_A = aff[crystal_index, 3]

    elif abs(K_minus_Kp_squared - 4) < 1e-6:
        V_S = 0
        V_A = aff[crystal_index, 4]

    elif abs(K_minus_Kp_squared - 8) < 1e-6:
        V_S = aff[crystal_index, 1]
        V_A = 0

    elif abs(K_minus_Kp_squared - 11) < 1e-6:
        V_S = aff[crystal_index, 2]
        V_A = aff[crystal_index, 5]

    else:
        V_S = 0
        V_A = 0

    # Return the corresponding atomic form factors.
    return V_S, V_A


'''
Function: bz_path
Sets up the vectors 'k' that are along the desired path on the first Brillouin zone.
Vectors are modulo 2pi/a_0.
Parameters:
    none
'''
def bz_path():
    k = np.zeros((38,3))

    # Set the value for points 'L', 'X', 'U' and 'K'. Since 'Gamma' is the origin, we can skip it.
    k[0, :] = np.array([1/2, 1/2, 1/2])        # L
    k[20, :] = np.array([1, 0, 0])             # X
    k[24, :] = np.array([1, 1/4, 1/4])         # U
    k[25, :] = np.array([3/4, 3/4, 0])         # K

    # L -> Gamma.
    for i in range(1, 10):
        k[i, :] = np.array([(10-i)/20, (10-i)/20, (10-i)/20])

    # Gamma -> X
    for i in range(11, 20):
        k[i, :] = np.array([(i-10)/10, 0, 0])

    # X -> U
    for i in range(21, 24):
        k[i, :] = np.array([1, (i-20)/16, (i-20)/16])

    # K -> Gamma
    for i in range(26, 37):
        k[i, :] = np.array([(37-i)/16, (37-i)/16, 0])

    return k


