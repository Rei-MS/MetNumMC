import sys
import math
import numpy as np
import VariationalCalc as vac
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from cycler import cycler

'''
Solve the variational problem for four different potentials.
Expansion into the quantum harmonic oscillator eigenfunctions:
   - The modified quantum oscillator.
   - The quartic anharmonic oscillator.
   - The symmetric potential well.

Expansion into Gaussian functions:
    - The hydrogen atom (ground state)
Units used: Rydberg Atomic Units (hbar = 2m = 1)
'''

# Turn interactive mode on for pyplot.
plt.ion()

# Choose Hamiltonian.
print(' ')
print('Choose Hamiltonian:')
print('1 - Modified Harmonic Oscillator.')
print('2 - Quartic Anharmonic Oscillator.')
print('3 - Symmetric Potential Well.')
print('4 - The Hydrogen Atom.')
selection = int(input('Input the number corresponding to the desired Hamiltonian: '))
print(' ')
if selection not in {1, 2, 3, 4}:
    sys.exit('Script Stopped. Input invalid.')

'''
This part of the code solves the problem for the first three Hamiltonian for one set of parameters.
Be careful since many variables in this piece of code share names with variables in the rest of the code.

# Input number of functions to consider for the expansion:
n = int(input('Input the number of functions to consider for the expansion (Must be natural number): '))

# Initialize required arrays.
hamiltonian = np.zeros((n, n))
energy = np.zeros(n)

# Build the Hamiltonian Matrix.
# Modified Harmonic Oscillator
if selection == 1:
    #alpha = float(input('Input the parameter for the linear term of the Hamiltonian: '))
    alpha = math.sqrt(omega)
    hamiltonian = vac.harmonic(np.zeros((n, n)), omega) - alpha*vac.x_1(np.zeros((n, n)), omega)
# Quartic Anharmonic Oscillator.
elif selection == 2:
    #beta = float(input('Input the parameter for the quartic term of the Hamiltonian: '))
    beta = omega*omega
    hamiltonian = vac.harmonic(np.zeros((n, n)), omega) + beta*vac.x_4(np.zeros((n, n)), omega)
# Symmetric Potential Well.
else:
    #V_0 = float(input('Input the depth of the potential well (must be positive): '))
    #b = float(input('Input the half-width of the potential well (must be positive): '))
    V_0 = 10
    b = math.sqrt(2/omega)*5
    hamiltonian = vac.p_2(np.zeros((n, n)), omega) + vac.pwell(np.zeros((n, n)), omega, V_0, b)

# Diagonalize. Eigenvalues only.
energy = eigh(hamiltonian, lower=False, eigvals_only=True)
print(energy)
'''

# Since we do different things with each Hamiltonian, we will separate between Hamiltonians.
# The modified harmonic oscillator and the quartic anharmonic oscillator will be handled together.

if selection in {1, 2}:
    # Set value for the frequency: (default is 1)
    omega = 1

    # Initialize an array with the number of functions desired for the expansion.
    number = np.array([5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 75, 100])

    # Initialize array to store the five lowest eigenvalues, for each diagonalization.
    energies = np.zeros((5, np.size(number)))

    # Diagonalize.
    # Modified harmonic oscillator.
    if selection == 1:
        alpha = math.sqrt(omega)
        for index in range(0, np.size(number)):
            # Define the matrix dimension 'n' and the hamiltonian for the current iteration.
            n = number[index]
            hamiltonian = vac.harmonic(np.zeros((n, n)), omega) - alpha*vac.x_1(np.zeros((n, n)), omega)

            # Store energy values, taking only the lowest five.
            energies[:, index] = eigh(hamiltonian, lower=False, eigvals_only=True)[0: 5]

    # Quartic anharmonic oscillator.
    else:
        beta = omega*omega
        for index in range(0, np.size(number)):
            # Define the matrix dimension 'n' and the hamiltonian for the current iteration.
            n = number[index]
            hamiltonian = vac.harmonic(np.zeros((n, n)), omega) + beta*vac.x_4(np.zeros((n, n)), omega)

            # Store energy values, taking only the lowest five.
            energies[:, index] = eigh(hamiltonian, lower=False, eigvals_only=True)[0: 5]


    # Plot energies as a function of the number of functions in the expansion.
    plt.figure(frameon=False)

    colors = ['bo', 'co', 'go', 'yo', 'ro']
    for index in range(0, np.size(number)):
        for eigen_index in range(0, 5):
            plt.plot(number[index], energies[eigen_index, index], colors[eigen_index], markersize=2)

    plt.ylabel('E [Ry]')
    plt.xlabel('Number of functions in the expansion.')
    if selection == 1:
        plt.title('Eigenvalue distribution. Five lowest eigenvalues. Modified Harmonic Oscillator.')

    else:
        plt.title('Eigenvalue distribution. Five lowest eigenvalues. Quartic Anharmonic Oscillator.')

    plt.grid(True)
    plt.show()

    # Print energies obtained with the minimum and maximum number of funtions in the expansion.
    print('Eigenvalues obtained for ' + str(number[0]) + ' functions in the expansion.')
    print(energies[:, 0])
    print(' ')
    print('Eigenvalues obtained for ' + str(number[np.size(number)-1]) + ' functions in the expansion.')
    print(energies[:, np.size(number)-1])

elif selection == 3:
    # Set value for the frequency: (default is 1)
    omega = 1

    # Define posible values for V_0 (V_0 is positive).
    V_0 = np.linspace(0, 3, num=60)

    # Initialize array to store the ten lowest eigenvalues, for each diagonalization.
    energies = np.zeros((10, np.size(V_0)))

    # Diagonalize.
    b = math.sqrt(2/omega)*5
    for index in range(0, np.size(V_0)):
        # Define the matrix dimension 'n' and the hamiltonian for the current iteration.
        n = 50
        hamiltonian = vac.p_2(np.zeros((n, n)), omega) + vac.pwell(np.zeros((n, n)), omega, V_0[index], b)

        # Store energy values, taking only the lowest ten.
        energies[:, index] = eigh(hamiltonian, lower=False, eigvals_only=True)[0: 10]


    # Plot energies as a function of the depth of the well.
    plt.figure(frameon=False)

    colors = ['bo', 'co', 'go', 'yo', 'ro', 'ko', 'ko', 'ko', 'ko', 'ko']
    for index in range(0, np.size(V_0)):
        for eigen_index in range(0, 10):
            plt.plot(V_0[index], energies[eigen_index, index], colors[eigen_index], markersize=2)

    plt.ylabel('E [Ry]')
    plt.xlabel('V_0 [Ry]')
    plt.title('Eigenvalue distribution. Ten lowest eigenvalues. Finite Square Well.')

    plt.grid(True)
    plt.show()

else:
    # Input number of Gaussians to be used. Either 3 or 4.
    gaussians = int(input('Input the number of Gaussians to be used in the expansion (either 3 or 4): '))
    print(' ')

    if gaussians not in {3, 4}:
        sys.exit('Script Stopped. Input invalid.')

    # Input number of sampling points for each coefficient.
    sample = int(input('Input the number of sampling points for each coefficient: '))
    print(' ')

    if sample % 2 == 1:
        sample = sample + 1
    sample = int(sample/2)

    # Set up array with optimized values.
    if gaussians == 3:
        # Values given by Cramer and Gianozzi.
        #optimized_coefficients = np.array([0.109818, 0.405771, 2.227660])
        # Values found by me.
        optimized_coefficients = np.array([0.151218, 0.679771, 4.48666])
    else:
        optimized_coefficients = np.array([0.121949, 0.444529, 1.962079, 13.00773])

    # Set up array with coefficient sample.
    coefficient_sample = np.zeros((np.size(optimized_coefficients), 2*sample + 1))
    # Store the samples.
    step = (0.03 / sample)
    for coefficient in range(0, int(np.shape(coefficient_sample)[0])):
        for sampling_point in range(0, int(np.shape(coefficient_sample)[1])):
            coefficient_sample[coefficient, sampling_point] = optimized_coefficients[coefficient] -0.03 + sampling_point * step
        # Make sure optimized value is in the 'middle'.
        coefficient_sample[coefficient, sample] = optimized_coefficients[coefficient]

    # For every combination of coefficients, get matrix elements and diagonalize.
    # This keeps track of the energy value, keeping the lowest possible.
    # Also keeps track of the index of the respective coefficients.
    hamiltonian = np.zeros((gaussians, gaussians))
    overlap = np.zeros((gaussians, gaussians))
    index_matrix = np.zeros(gaussians)
    energy = 10

    if gaussians == 3:
        for index_0, coeff_0 in enumerate(coefficient_sample[0, :]):
            for index_1, coeff_1 in enumerate(coefficient_sample[1, :]):
                for index_2, coeff_2 in enumerate(coefficient_sample[2, :]):
                    # Store coefficients of current iteration in array.
                    coefficient_array = np.array([coeff_0, coeff_1, coeff_2])

                    # Get matrix elements for current iteration.
                    hamiltonian, overlap = vac.hydro_h(hamiltonian, overlap, coefficient_array)

                    # Diagonalize and get lowest energy.
                    iteration_energy = eigh(hamiltonian, b=overlap, lower=False, eigvals_only=True, type=1)[0]

                    # Compare to energy from previous iteration. Keep lowest and store coefficient index if necessary.
                    if iteration_energy < energy:
                        energy = iteration_energy
                        index_matrix = np.array([int(index_0), int(index_1), int(index_2)])

                    else:
                        pass

        print('Energy obtained: ' + str(energy))
        print(' ')
        print('Coefficients for this energy value:')
        print(coefficient_sample[0, index_matrix[0]])
        print(coefficient_sample[1, index_matrix[1]])
        print(coefficient_sample[2, index_matrix[2]])


    else:
        for index_0, coeff_0 in enumerate(coefficient_sample[0, :]):
            for index_1, coeff_1 in enumerate(coefficient_sample[1, :]):
                for index_2, coeff_2 in enumerate(coefficient_sample[2, :]):
                    for index_3, coeff_3 in enumerate(coefficient_sample[3, :]):
                        # Store coefficients of current iteration in array.
                        coefficient_array = np.array([coeff_0, coeff_1, coeff_2, coeff_3])

                        # Get matrix elements for current iteration.
                        hamiltonian, overlap = vac.hydro_h(hamiltonian, overlap, coefficient_array)

                        # Diagonalize and get lowest energy.
                        iteration_energy = eigh(hamiltonian, b=overlap, lower=False, eigvals_only=True, type=1)[0]

                        # Compare to energy from previous iteration. Keep lowest and store coefficient index, if necessary.
                        if iteration_energy < energy:
                            energy = iteration_energy
                            index_matrix = np.array([int(index_0), int(index_1), int(index_2), int(index_3)])

                        else:
                            pass

        print('Energy obtained: ' + str(energy))
        print(' ')
        print('Coefficients for this energy value:')
        print(coefficient_sample[0, index_matrix[0]])
        print(coefficient_sample[1, index_matrix[1]])
        print(coefficient_sample[2, index_matrix[2]])
        print(coefficient_sample[3, index_matrix[3]])

print(' ')
end = input('Press enter to end the script.')
