import sys
import math
import numpy as np
import TDPotCalc as tpc
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from scipy.linalg import eigh_tridiagonal
import scipy.sparse as scs

'''
Solve the two-dimensional Schrödinger equation using the direct matrix method and Lanczos algorithm.
Lowest eigenvalues are found for the following potentials:
   - Infinite rectangular well.
   - Harmonic oscillator.
   - Gaussian well.

Units used: Scaled axes, energy in units of hbar^2/2m except for harmonic oscillator.
            Harmonic oscillator is dimensionless.
'''

# Turn interactive mode on for pyplot.
plt.ion()

# Choose potential.
print(' ')
print('Choose potential:')
print('1 - Infinite rectangular well.')
print('2 - Harmonic oscillator.')
print('3 - Gaussian well.')
selection = int(input('Input the number corresponding to the desired potential: '))
print(' ')
if selection not in {1, 2, 3}:
    sys.exit('Script Stopped. Input invalid.')

'''
Each potential is treated separately first to set up some parameters.
These are parameters relation to the potential, side length of the grid, and eta & mu parameters.
'''

if selection == 1:
    '''
    Set up parameters relating to the potential.
    For the rectangular potential well we need it's dimensions.
    '''
    a = float(input('Input the width of the well in the x-direction: '))
    b = float(input('Input the width of the well in the y-direction: '))

    if a <= 0 or b <= 0:
        sys.exit('Script Stopped. Width must be strictly positive.')

    # Store parameters in array.
    parameters = np.zeros(2)
    parameters[0] = a
    parameters[1] = b

    '''
    The grid is a 1 x 1 square centered at the origin.
    We need the side length of the grid.
    '''
    l = 1


elif selection == 2:
    '''
    Set up parameters relating to the potential.
    For the harmonic oscillator potential we don't need parameters.
    Still, set up a null array to avoid errors.
    '''
    parameters = np.zeros(2)


    '''
    The grid is an l x l square centered at the origin.
    We need the side length of the grid.
    '''
    l = float(input('Input the side length of the square grid (Recommended >= 12) : '))

    if l <= 0:
        sys.exit('Script Stopped. Side length must be strictly positive.')


else:
    '''
    Set up parameters relating to the potential.
    For the Gaussian well we need it's depth and screening parameter.
    '''
    V_0 = float(input('Input the depth (2mV_0 / hbar^2) of the Gaussian well: '))
    alpha = float(input('Input the parameter alpha of the Gaussian well: '))

    if V_0 <= 0 or alpha <= 0:
        sys.exit('Script Stopped. Both parameters must be strictly positive.')

    # Store parameters in array.
    parameters = np.zeros(2)
    parameters[0] = V_0
    parameters[1] = alpha

    '''
    The grid is a l x l square centered at the origin.
    We need the side length of the grid.
    '''
    l = float(input('Input the side length of the square grid (Recommended >= 8) : '))

    if l <= 0:
        sys.exit('Script Stopped. Side length must be strictly positive.')


'''
Set up grid; a l x l square centered at the origin.
'''
print(' ')
print('A grid will be set over the square [{:.2f}, {:.2f}]x[{:.2f}, {:.2f}].'.format(-l/2, l/2, -l/2, l/2))

# Step size in x-direction.
delta_x = float(input('Input the step size of the grid in the x-direction: '))

if delta_x <= 0 or delta_x >= l:
    sys.exit('Script Stopped. Step size in the x-direction must be strictly between 0 and {}.'.format(l))

# Step size in y-direction.
if selection == 1:
    delta_y = float(input('Input the step size of the grid in the y-direction (Recommended: {:.4f}): '.format(b*delta_x/a)))
else:
    delta_y = float(input('Input the step size of the grid in the y-direction (Recommended: {}): '.format(delta_x)))

if delta_y <= 0 or delta_y >= l:
    sys.exit('Script Stopped. Step size in the y-direction must be strictly between 0 and {}.'.format(l))

# Calculate number of points in each direction and set grid arrays.
N_x = int(round((l / delta_x) + 1))
N_y = int(round((l / delta_y) + 1))
grid_x = np.linspace(-l/2, l/2, N_x)
grid_y = np.linspace(-l/2, l/2, N_y)

print(' ')
print('With these values, a {}x{} grid is set over the square [{:.2f}, {:.2f}]x[{:.2f}, {:.2f}].'.format(N_x, N_y, -l/2, l/2, -l/2, l/2))


'''
Set the eta and mu parameters.
'''
if selection == 1:
    eta = 1 / (a * a * delta_x * delta_x)
    mu = 1 / (b * b * delta_y * delta_y)
elif selection == 2:
    eta = 1 / (2 * delta_x * delta_x)
    mu = 1 / (2 * delta_y * delta_y)
else:
    eta = alpha / (delta_x * delta_x)
    mu = alpha / (delta_y * delta_y)


'''
Lanczos iteration step.
'''
# Choose number of steps to be performed by the Lanczos algorithm.
print(' ')
num_lanczos = int(input('Input the number of steps to be performed by the Lanczos algorithm (<= {}): '.format((N_x - 2) * (N_y - 2))))


if num_lanczos <= 1 or num_lanczos > ((N_x - 2) * (N_y - 2)):
    sys.exit('Script Stopped. Number of steps must be strictly between 1 and {}.'.format((N_x - 2) * (N_y - 2)))

###########
# The following is various attempts to implement the algorithm.
# First, by using SciPy's 'sparse' package to set the Hamiltonian matrix.
# Second, by calculating the products H|q> in the Lanczos algorithm on the fly (convergence failure; don't know why.)
# Third, by setting the whole(!) Hamiltonian matrix. (not practical, fails if too big.)
###########

#####################################
#####################################
'''
    Defining the hamiltonian matrix.
'''

hamdim = (N_x - 2) * (N_y - 2)

# Set arrays for diagonals.
nu_diag = np.zeros(hamdim)
eta_diag = np.zeros(hamdim - 1)
mu_diag = np.zeros(hamdim - (N_x - 2))

for index in range(0, hamdim):
    nu_diag[index] = 2*(eta + mu) + tpc.potential(selection, parameters, index, grid_x, grid_y, N_x)

    try:
        if (index + 1) % (N_x - 2) == 0:
            eta_diag[index] = 0
        else:
            eta_diag[index] = -1 * eta
    except IndexError:
        pass

    try:
        mu_diag[index] = -1 * mu
    except IndexError:
        pass

# Set the Hamiltonian matrix.
offsets = [0, 1, -1, (N_x - 2), -(N_x - 2)]
diagonals = [nu_diag, eta_diag, eta_diag, mu_diag, mu_diag]
hamiltonian = scs.diags(diagonals, offsets)

# Call the lanczos_direct function.
sel_ort = input('Use Lanczos algorithm with selective orthogonalization? (y/n): ')
print(' ')

if sel_ort == str('y'):
    ritz_values, ritz_vectors = tpc.lanczos_direct_s(hamiltonian, hamdim, num_lanczos)
elif sel_ort == str('n'):
    alpha, beta = tpc.lanczos_direct(hamiltonian, hamdim, num_lanczos)
    ritz_values = eigh_tridiagonal(alpha, beta, eigvals_only=True)
else:
    sys.exit('Script stopped. Must input "y" or "n".')

# Print eigenvalues.
np.set_printoptions(edgeitems=3, infstr='inf',
linewidth=75, nanstr='nan', precision=None,
suppress=False, threshold=1000, formatter=None)
print('Eigenvalues by Lanczos algorithm.')
#ritz_values = 36*ritz_values/(np.pi*np.pi)
print(ritz_values[0:40])
print(' ')


# Get eigenvalues by conventional diagonalization.
conv_diag = input('Get eigenvalues by conventional diagonalization? (y/n): ')
print(' ')

if conv_diag == str('y'):
    eigvals = scs.linalg.eigsh(hamiltonian, k=15, which='SA', tol=1e-4, return_eigenvectors=False)
    print(eigvals)
    indices_repetidos = np.zeros(0)

    for i in range(0, 15):
        for j in range(i, 15):
            if '%.3f'%(eigvals[i]) == '%.3f'%(eigvals[j]):
                if i == j:
                    pass
                else:
                    if j not in indices_repetidos:
                        indices_repetidos = np.append(indices_repetidos, int(j))
            else:
                pass

    print(indices_repetidos)

    eigvals = np.delete(eigvals, indices_repetidos.astype(int))

    print(' ')
    print('Eigenvalues by conventional diagonalization.')
    #eigvals = 36*eigvals/(np.pi*np.pi)
    print(eigvals)
else:
    pass

#####################################
#####################################
'''
# Calculating H|q> on the fly.
# Call the lanczos function.
sel_ort = input('Use Lanczos algorithm with selective orthogonalization? (y/n): ')

if sel_ort == str('y'):
    ritz_values, ritz_vectors = tpc.lanczos_s(num_lanczos, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y)
elif sel_ort == str('n'):
    alpha, beta = tpc.lanczos(num_lanczos, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y)
    ritz_values = eigh_tridiagonal(alpha, beta[1:], eigvals_only=True)
else:
    sys.exit('Script stopped. Must input "y" or "n".')

# Print eigenvalues.
print('Eigenvalues by Lanczos algorithm.')
print(ritz_values[0:20])
'''
#####################################
#####################################
'''
# Try by conventional diagonalization by defining the whole(!) Hamiltonian matrix.
# Fails if N_x and/or N_y are large enough.

hamdim = (N_x - 2) * (N_y - 2)
hamiltonian = np.zeros([hamdim, hamdim])

for index in range(0, hamdim):
    hamiltonian[index, index] = 2*(eta + mu) + tpc.potential(selection, parameters, index, grid_x, grid_y, N_x)

    try:
        hamiltonian[index, index + 1] = -1 * eta
    except IndexError:
        pass

    try:
        hamiltonian[index + 1, index] = -1 * eta
    except IndexError:
        pass

    if (index + 1) % (N_x - 2) == 0:
        try:
            hamiltonian[index, index + 1] = 0
            hamiltonian[index + 1, index] = 0
        except IndexError:
            pass

    try:
        hamiltonian[index, index + (N_x - 2)] = -1 * mu
    except IndexError:
        pass

    try:
        hamiltonian[index + (N_x - 2), index] = -1 * mu
    except IndexError:
        pass


eigvals = eigh(hamiltonian, eigvals_only=True, subset_by_index=[0, 20])
#eigvals = eigh(hamiltonian, eigvals_only=True)
print('eigenvalues.')
print(eigvals)
'''
#####################################
#####################################
print(' ')
end = input('Press enter to end the script.')
