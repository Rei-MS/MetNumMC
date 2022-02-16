import sys
import math
import numpy as np
import scipy.special as sc
from scipy import integrate as scint


'''
Function: harmonic
Sets up the matrix elements corresponding to the Harmonic oscillator.
Parameters:
    array - array (square) to store matrix elements
    omega - frequency
'''
def harmonic(array, omega):
    if np.shape(array)[0] == np.shape(array)[1]:
        dimension = np.shape(array)[0]
    else:
        sys.exit('Script Stopped. Expected square matrix; harmonic function.')

    # Fill array.
    for i in range(0, dimension):
        array[i, i] = i + (1/2)

    # Multiply by omega.
    array = omega*array

    return array


'''
Function: x_1
Sets up the matrix elements corresponding to X.
Parameters:
    array - array (square) to store matrix elements
    omega - frequency
'''
def x_1(array, omega):
    if np.shape(array)[0] == np.shape(array)[1]:
        dimension = np.shape(array)[0]
    else:
        sys.exit('Script Stopped. Expected square matrix; x_1 function.')

    # Fill array.
    for i in range(0, dimension-1):
        array[i, i+1] = math.sqrt(i+1)
        array[i+1, i] = array[i, i+1]

    # Multiply by the square root of 1/omega.
    array = np.sqrt(1/omega)*array

    return array

'''
Function: x_2
Sets up the matrix elements corresponding to X^2.
Parameters:
    array - array (square) to store matrix elements
    omega - frequency
'''
'''
def x_2(array, omega):
    if np.shape(array)[0] == np.shape(array)[1]:
        dimension = np.shape(array)[0]
    else:
        sys.exit('Script Stopped. Expected square matrix; x_2 function.')

    # Fill array.
    for i in range(0, dimension-2):
        array[i, i+2] = math.sqrt((i+1)(i+2))
        array[i+2, i] = array[i, i+2]

    for i in range(0, dimension):
        array[i, i] = (2*i + 1)

    # Multiply by 1/omega.
    array = (1/omega)*array

    return array
'''

'''
Function: x_4
Sets up the matrix elements corresponding to X^4.
Parameters:
    array - array (square) to store matrix elements
    omega - frequency
'''
def x_4(array, omega):
    if np.shape(array)[0] == np.shape(array)[1]:
        dimension = np.shape(array)[0]
    else:
        sys.exit('Script Stopped. Expected square matrix; x_4 function.')

    # Fill array.
    for i in range(0, dimension-4):
        array[i, i+4] = math.sqrt((i+1)*(i+2)*(i+3)*(i+4))
        array[i+4, i] = array[i, i+4]

    for i in range(0, dimension-2):
        array[i, i+2] = 2*(2*i + 3)*math.sqrt((i+1)*(i+2))
        array[i+2, i] = array[i, i+2]

    for i in range(0, dimension):
        array[i, i] = (6*i*i + 6*i + 3)

    # Multiply by (1/omega)^2.
    array = (1/omega)*(1/omega)*array

    return array


'''
Function: p_2
Sets up the matrix elements corresponding to P^2/2m.
Parameters:
    array - array (square) to store matrix elements
    omega - frequency
'''
def p_2(array, omega):
    if np.shape(array)[0] == np.shape(array)[1]:
        dimension = np.shape(array)[0]
    else:
        sys.exit('Script Stopped. Expected square matrix; x_2 function.')

    # Fill array.
    for i in range(0, dimension-2):
        array[i, i+2] = math.sqrt((i+1)*(i+2))
        array[i+2, i] = array[i, i+2]

    for i in range(0, dimension):
        array[i, i] = -(2*i + 1)

    # Multiply by -omega/4.
    array = (-omega/4)*array

    return array

'''
Function: pwell
Sets up the matrix elements corresponding to the potential of the finite square well.
Parameters:
    array - array (square) to store matrix elements
    omega - frequency
    V_0 - depth of the potential well (V_0 is positive)
    b - half-width of the well (b is positive)
'''
def pwell(array, omega, V_0, b):
    if np.shape(array)[0] == np.shape(array)[1]:
        dimension = np.shape(array)[0]
    else:
        sys.exit('Script Stopped. Expected square matrix; x_2 function.')

    # Set up necessary constants.
    x_0 = math.sqrt(2/omega)
    constant = -V_0/(math.sqrt(np.pi * x_0 * x_0))
    integral_bound = b/x_0

    # Set up arrays for integration.
    y = np.linspace(-integral_bound, integral_bound, 1000, endpoint=True)
    expy2 = np.exp(-(y*y))

    # Fill array.
    # The 'i' counter is for rows. For each row, the for loop runs through each possible column,
    # excluding entries below the main diagonal. Then it assigns both a(ij) and a(ji) the same value.
    try:
        i = 0
        while i < dimension:
            c_i = 1/( math.sqrt(pow(2, i) * math.factorial(i)) )
            H_i = sc.eval_hermite(i, y)
            for j in range (i, dimension):
                H_j = sc.eval_hermite(j, y)
                c_j = 1/( math.sqrt(pow(2, j) * math.factorial(j)) )
                array[i, j] = constant * c_i * c_j * scint.trapz(expy2*H_i*H_j, y)
                array[j, i] = array[i, j]
            i += 1
    except KeyboardInterrupt:
        sys.exit('Script stopped.')

    return array

'''
Function: hydro_h
Sets up the matrix elements corresponding to the Hamiltonian of the hydrogen atom and the overlap matrix.
Parameters:
    array - array (square) to store matrix elements of the Hamiltonian
    overlap - array (square) to store matrix elements of the overlap matrix
    coefficients - array with the coefficients of the Gaussian functions
'''
def hydro_h(array, overlap, coefficients):
    if np.shape(array)[0] == np.shape(array)[1] and np.shape(overlap)[0] == np.shape(overlap)[1]:
        if np.shape(array)[0] == np.shape(coefficients)[0] and np.shape(overlap)[0] == np.shape(coefficients)[0]:
            dimension = np.shape(array)[0]
        else:
            sys.exit('Script Stopped. Matrix dimensions are different; hydro_h function.')
    else:
        sys.exit('Script Stopped. Expected square matrix; hydro_h function.')

    # Fill array.
    # The 'i' counter is for rows. For each row, the for loop runs through each possible column,
    # excluding entries below the main diagonal. Then it assigns both a(ij) and a(ji) the same value.
    try:
        i = 0
        while i < dimension:
            for j in range (i, dimension):
                overlap[i, j] = (np.pi / (coefficients[i] + coefficients[j]))**(1.5)
                overlap[j, i] = overlap[i, j]

                array[i, j] = ((6 * coefficients[i] * coefficients[j] * overlap[i, j])/(coefficients[i] + coefficients[j])) - ((4 * np.pi)/(coefficients[i] + coefficients[j]))
                array[j, i] = array[i, j]
            i += 1
    except KeyboardInterrupt:
        sys.exit('Script stopped.')

    return array, overlap
