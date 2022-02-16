import sys
import math
import itertools
import numpy as np
import HFCalc as hfc
from matplotlib import pyplot as plt
from scipy.linalg import eigh


'''
Hartree-Fock method for the helium atom ground state.
Expansion of orbital into S Gaussians.
Units used: Rydberg atomic units.
'''

# Set atomic number.
Z = 2

# Input number of Gaussians used in the expansion of the orbital.
print(' ')
num_gaussians = int(input('Input the number of Gaussians to be used in the expansion: '))
print(' ')

# Input coefficients for said Gaussians (alpha coefficients).
a_coefficients = np.zeros(num_gaussians)

for gaussian in range(0, num_gaussians):
    a_coefficients[gaussian] = input('Input coefficient for Gaussian number ' + str(gaussian + 1) + ': ')

# Initialize and/or fill necessary arrays.
overlap = hfc.overlap(a_coefficients)
T = hfc.se_hamiltonian(a_coefficients, Z)
G = hfc.G_array(a_coefficients)
coefficient = np.zeros(num_gaussians)       # Coefficients of the expansion of the orbital into Gaussians.
fock = np.zeros([num_gaussians, num_gaussians])

# Iteration.
coefficient[0] = 1          # Initial guess.
energy_current = 0
energy_old = 0

try:
    while True:
        # Set new energies.
        energy_old = energy_current
        energy_current = 0

        # Get Fock array and diagonalize it.
        fock = hfc.fock(coefficient, T, G)
        energy_hfock, coefficient = eigh(fock, overlap, subset_by_index=[0, 0])

        # Get energy for current iteration.
        for index1, index2 in itertools.product(range(0, num_gaussians), range(0, num_gaussians)):
            energy_current += 2 * coefficient[index1] * coefficient[index2] * T[index1, index2]
            for index3, index4 in itertools.product(range(0, num_gaussians), range(0, num_gaussians)):
                energy_current += G[index1, index3, index2, index4] * coefficient[index1] * coefficient[index2] * coefficient[index3] * coefficient[index4]

        if abs(energy_current - energy_old) < 1e-6:
            break
        else:
            pass
except KeyboardInterrupt:
    sys.exit('Script stopped.')

print('Energy obtained:')
print(energy_current)
print(' ')
print('Coefficients of the expansion:')
print(coefficient)


print(' ')
end = input('Press enter to end the script.')
