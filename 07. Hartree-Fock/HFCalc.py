import sys
import math
import numpy as np
import itertools

'''
Function: overlap
Sets up the overlap array S.
Parameters:
    coefficients - array with coefficients of the gaussians used in the expansion.
'''
def overlap(coefficients):
    num_coeff = int(len(coefficients))
    overlap_array = np.zeros([num_coeff, num_coeff])

    for index1, index2 in itertools.product(range(0, num_coeff), range(0, num_coeff)):
        overlap_array[index1, index2] = (np.pi / (coefficients[index1] + coefficients[index2]) )**(3/2)

    return overlap_array

'''
Function: se_hamiltonian
Sets up the single electron hamiltonian array T.
Parameters:
    coefficients - array with coefficients of the gaussians used in the expansion.
    Z - atomic number.
'''
def se_hamiltonian(coefficients, Z):
    num_coeff = int(len(coefficients))
    overlap_array = overlap(coefficients)
    T = np.zeros([num_coeff, num_coeff])

    for index1, index2 in itertools.product(range(0, num_coeff), range(0, num_coeff)):
        T[index1, index2] = overlap_array[index1, index2] * 6 * coefficients[index1] * coefficients[index2] / (coefficients[index1] + coefficients[index2])
        T[index1, index2] -= 2 * Z * (2*np.pi) / (coefficients[index1] + coefficients[index2])

    return T

'''
Function: G_array
Sets up the array G.
Parameters:
    coefficients - array with coefficients of the gaussians used in the expansion.
'''
def G_array(coefficients):
    num_coeff = int(len(coefficients))
    G = np.zeros([num_coeff, num_coeff, num_coeff, num_coeff])

    for index1, index2, index3, index4 in itertools.product(range(0, num_coeff), range(0, num_coeff), range(0, num_coeff), range(0, num_coeff)):
        alpha13 = coefficients[index1] + coefficients[index3]
        alpha24 = coefficients[index2] + coefficients[index4]
        denominator = alpha13 * alpha24 * np.sqrt(alpha13 + alpha24)
        G[index1, index2, index3, index4] = 2 * 2 * (np.pi**(5/2)) / denominator

    return G

'''
Function: fock
Sets up the Fock operator array.
Parameters:
    coefficients_exp - array with coefficients of the gaussians used in the expansion.
    T - single electron hamiltonian array T.
    G - array G.
'''
def fock(coefficients_exp, T, G):
    num_coeff = int(len(coefficients_exp))
    fock = np.zeros([num_coeff, num_coeff])

    for index1, index2 in itertools.product(range(0, num_coeff), range(0, num_coeff)):
        fock[index1, index2] = T[index1, index2]
        for index3, index4 in itertools.product(range(0, num_coeff), range(0, num_coeff)):
            fock[index1, index2] += (2*G[index1, index3, index2, index4] - G[index1, index2, index3, index4])*coefficients_exp[index3]*coefficients_exp[index4]

    return fock
