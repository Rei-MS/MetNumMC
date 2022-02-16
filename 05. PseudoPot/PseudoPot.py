import sys
import math
import itertools
import numpy as np
import PseudoPotCalc as ppc
from matplotlib import pyplot as plt
from scipy.linalg import eigh

'''
Plot the band structures of Si, Ge, Sn, GeAs, ZnS and CdTe by implementing
the pseudopotential method as presented by Cohen & Bergstresser, PRB 141, 789 (1966).

Units used: Rydberg atomic units. hbar^2 = 2m = 1.
'''

# Turn interactive mode on for pyplot.
plt.ion()

# Set up primitive vectors of the reciprocal lattice. Vectors are modulo 2pi/a_0.
b_1 = np.array([1, 1, -1])
b_2 = np.array([1, -1, 1])
b_3 = np.array([-1, 1, 1])

# Choose desired crystal.
print(' ')
print('Choose crystal:')
print('1 - Si.')
print('2 - Ge.')
print('3 - Sn.')
print('4 - GeAs.')
print('5 - ZnS.')
print('6 - CdTe.')
selection = int(input('Input the number corresponding to the desired crystal: ')) - 1
print(' ')
if selection not in {0, 1, 2, 3, 4, 5}:
    sys.exit('Script Stopped. Input invalid.')

# Get corresponding lattice parameter.
a_0 = ppc.lattice_parameter(selection)

# Set upper bound for kinetic energy.
e_up = float(input('Input upper bound for kinetic energy [Ry]: '))
print(' ')
if e_up < 0:
    sys.exit('Script Stopped. Input invalid.')

# Get the vectors 'k' that are along the desired path on the first Brillouin zone.
# Vectors are modulo 2pi/a_0.
wave_vector_k = ppc.bz_path()

# Set up an array to store the energy values obtained by diagonalization.
energy = np.zeros((int(len(wave_vector_k)), 8))

# Estimate the number of plane waves needed, for iteration purposes.
num_pw_estimate = int(round( ( math.sqrt(e_up) / (((2*np.pi)/a_0) * math.sqrt(3)) ) + (1/2)    ) + 1)

# Solve using the variational method.
for k_index in range(0, int(len(wave_vector_k))):
    # Set up a dictionary to store k + K values.
    k_K = {}

    # Set up a counter for the number of plane waves that will be used in diagonalization.
    num_pw = 0

    # Set up k + K possible values and compare with upper bound for kinetic energy.
    pw_iter = range(-num_pw_estimate, num_pw_estimate + 1)

    for m_1, m_2, m_3 in itertools.product(pw_iter, pw_iter, pw_iter):
        # Calculate k + K for current iteration. Again, modulo 2pi/a_0.
        k_plus_K = wave_vector_k[k_index, :] + m_1*b_1 + m_2*b_2 + m_3*b_3

        # Calculate (k + K)^2 for current iteration.
        k_plus_K_squared = ((2*np.pi) / a_0)**2 * (k_plus_K[0]**2 + k_plus_K[1]**2 + k_plus_K[2]**2)

        # Check if the upper bound for the kinetic energy condition is satisfied.
        # If so, store k + K in dictionary with 'num_pw' as it's key.
        if k_plus_K_squared <= e_up:
            k_K['{}'.format(num_pw)] = k_plus_K  # Note that first key is '0'.
            num_pw += 1

    # Set up the matrix elements of the Hamiltonian.
    hamiltonian = np.zeros((num_pw, num_pw), dtype=np.complex_)

    for i, l in itertools.product(range(0, num_pw), range(0, num_pw)):
        # Calculate K - K'. Note that (k + K) - (k + K') = K - K'.
        K_minus_Kp = k_K['{}'.format(i)] - k_K['{}'.format(l)]

        # Determine which (if any) atomic form factors are needed for the current iteration.
        K_minus_Kp_squared = K_minus_Kp[0]**2 + K_minus_Kp[1]**2 + K_minus_Kp[2]**2

        V_S, V_A = ppc.atomic_form_factor(selection, K_minus_Kp_squared)

        # Assign the corresponding value to the matrix element.
        if i == l:
            k_plus_K = k_K['{}'.format(i)]
            hamiltonian[i, l] = ((2*np.pi) / a_0)**2 * (k_plus_K[0]**2 + k_plus_K[1]**2 + k_plus_K[2]**2)

        else:
            # Calculate (K - K') dot x_1.
            K_minus_Kp_dot_x_1 = (a_0/8) * (K_minus_Kp[0] + K_minus_Kp[1] + K_minus_Kp[2])
            hamiltonian[i, l] = (V_S * math.cos(K_minus_Kp_dot_x_1)) + (V_A * math.sin(K_minus_Kp_dot_x_1))*1j


    # Diagonalize hamiltonian and store the 8 lowest eigenvalues.
    energy[k_index, :] = eigh(hamiltonian, eigvals_only=True)[0: 8]


# Plot the band structure obtained.
crystal = {"0": "Si", "1": "Ge", "2": "Sn", "3": "GeAs", "4": "ZnS", "5": "CdTe"}
plt.figure(frameon=False)

colors = ['ko-', 'co-', 'go-', 'yo-', 'ro-', 'mo-', 'ko-', 'bo-']

for index_k in range(0, int(len(wave_vector_k))):
    for eigen_index in range(0, 8):
        plt.plot(index_k, energy[index_k, eigen_index], colors[eigen_index], markersize=2)

plt.ylabel('E [Ry]')
plt.xticks(np.array([0, 10, 20, 24, 25, 37]), ('L', r'$\Gamma$', 'X', 'U', 'K', r'$\Gamma$'))
plt.title('Band structure of ' + crystal['{}'.format(selection)] + '.')

plt.grid(True)
plt.show()

print(' ')
end = input('Press enter to end the script.')
