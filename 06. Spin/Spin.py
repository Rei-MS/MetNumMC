import sys
import math
import itertools
import numpy as np
import SpinCalc as spc
from matplotlib import pyplot as plt
from scipy.special import comb
from scipy.linalg import eigh_tridiagonal
from scipy.linalg import eigh

'''
Diagonalization of the one dimensional 1/2-spin isotropic Heisenberg spin chain.
Units used: Rydberg atomic units. hbar = 1.
Enegies in units of |J| and spins dimensionless.
'''


# Ask whether the chain is open or closed.
print(' ')
open_chain = input('Is it an open spin chain? (y/n): ')
print(' ')

if open_chain == str('y'):
    open_chain = True

elif open_chain == str('n'):
    open_chain = False

else:
    sys.exit('Script stopped. Must input "y" or "n".')



# Ask whether J is positive or negative.
J_sign = input('Is the interaction energy J positive (ferromagnetic)? (y/n): ')
print(' ')

if J_sign == str('y'):
    J_sign = 1

elif J_sign == str('n'):
    J_sign = -1

else:
    sys.exit('Script stopped. Must input "y" or "n".')


# Choose number of spins in the chain.
num_spins = int(input('Input the number of spins in the chain: '))
print(' ')

# Choose number of up spins.
num_upspins = int(input('Input the number of up spins in the chain: '))
print(' ')

# Check that 'num_upspins' is not bigger than 'num_spins'.
if num_upspins > num_spins:
    sys.exit('Script stopped. Number of up spins cannot be bigger than total number of spins.')

# Calculate the dimension of the Hilbert subspace.
num_hil = comb(num_spins, num_upspins, exact=True)

# Get the array containing the index of the spin neighbors to the right.
neighbor = spc.neighbors(num_spins, open_chain)

# Get the array containing the possible states with 'num_upspins' up spins.
states = spc.up_states(num_spins, num_upspins, num_hil)

# Get the Hamiltonian matrix.
hamiltonian = spc.hamiltonian(num_spins, num_hil, neighbor, states, J_sign, open_chain)

# Choose number of steps to be performed by the Lanczos algorithm.
num_lanczos = int(input('Input the number of steps to be performed by the Lanczos algorithm: '))
print(' ')
# If it's bigger than the dimension of the Hilbert subspace, bring it down.
if num_lanczos > num_hil:
    num_lanczos = num_hil


# Diagonalize with conventional diagonalization.
try:
    spectrum_c, vectors_c = eigh(hamiltonian, subset_by_index=[0, 2])
except ValueError:
    spectrum_c, vectors_c = eigh(hamiltonian, subset_by_index=[0, 0])

# Apply the Lanczos algorithm to the Hamiltonian matrix.
alpha, beta = spc.lanczos(hamiltonian, num_hil, num_lanczos)

# Diagonalize the tridiagonal matrix. Get 3 lowest eigenvalues.
try:
    spectrum, vectors = eigh_tridiagonal(alpha, beta, select='i', select_range=(0,2))
except ValueError:
    spectrum, vectors = eigh_tridiagonal(alpha, beta)

print('3(1) lowest eigenvalues:')
print('Lanczos =')
print(spectrum)
print(vectors)
print('Conventional = ')
print(spectrum_c)
print(vectors_c)

print(' ')
end = input('Press enter to end the script.')
