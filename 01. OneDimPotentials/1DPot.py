import sys
import math
import numpy as np
from matplotlib import pyplot as plt
import scipy.special as sc

'''
Script to find bound states in 1D potentials, by Numerov's algorithm.

Potentials considered will be the harmonic potential, double-well potential
and the Morse potential.
'''

# Turn interactive mode on for pyplot.
plt.ion()

# Choose potential.
print('Choose potential:')
print('1 - Harmonic Potential.')
print('2 - Double-well Potential.')
print('3 - Morse Potential.')
selection = int(input('Input the number corresponding to the desired potential: '))
print(' ')
if selection not in {1, 2, 3}:
    sys.exit('Script Stopped. Input invalid.')

xmax = float(input('Input max value for x (recommended: 10): '))
print(' ')
n = int(input('Input number of grid points (increase with number of nodes): '))
print(' ')

# Create arrays for the required quantities. Vector of length 'mesh'
# whose entries are our x values, evenly spaced. Vectors of length 'mesh'
# for the 'y' values, for the potential, and for the auxiliary array 'f'
# are initialized as zeroes.
x = np.linspace(0.0, xmax, n)
y = np.zeros([n])
pot = np.zeros([n])
f = np.zeros([n])

# Set the step 'dx' and for convenience the quantity (dx^2)/12
dx = xmax/n
dxd12 = (dx*dx)/12

# Set the entries for the potential array to what they need to be.
# First, the harmonic potential. Here, we just x^2 and the algorithm will return twice the value of the desired result.
if selection == 1:
    pot = np.square(x)
# Second, the double-well. Here we take epsilon to be a half, and delta/b to be the square root of 3.
elif selection == 2:
    pot = (np.float_power(x, 4)/9 - 2*np.square(x)/3 + 1)/2
# The Morse potential. We take lambda to be 5 and x_e to be 3.
else:
    pot = 25*(np.exp( -2*(x - 3) ) - 2*np.exp( -(x - 3)) )

# Input number of nodes. For the Morse potential, this number has to
# be between 0 and 4.
print('Input the number of nodes.')
if selection == 3:
    node = int(input('This has to be a whole number between 0 and 4: '))
else:
    node = int(input('This has to be a whole number: '))
print('')

if node not in {0, 1, 2, 3, 4} and selection == 3:
    sys.exit('Script Stopped. Forbidden node value inputted.')
else:
    pass

# Set the initial lower and upper bounds for the eigenvalue.
# The algorithm will search eigenvalues with bisection.
eup = np.amax(pot)
elw = np.amin(pot)
en = (eup + elw) / 2

# Set the values for y_0 and y_1. Also, if not Morse potential, set the number of nodes over half of the grid.
if selection == 3:
    # Morse potential
    # Since the vector 'y' was initialized as zeroes, we only need to set the y_1 value.
    if (node % 2) == 1:
        y[1] = -dxd12
    else:
        y[1] = dxd12

else:
    if (node % 2) == 1:
        # For odd number of nodes, we only need to set the y_1 value, since y_0 should be zero.
        y[1] = 1/sc.eval_hermite(node, x[1])
        node = int((node - 1)/2)
        even = False

    else:
        y[0] = 1/sc.eval_hermite(node, 0)
        y[1] = ( (12 - 10 * (1 + dxd12 * (en - pot[0]))) * y[0] ) / 2*(1 + dxd12 * (en - pot[1]))
        node = int(node/2)
        even = True


# Set up a function to handle the calculation.
def calculation(en, eup, elw, n, dxd12, node, pot, f, y, selection):
    # Set the 'f' array defined by the algorithm.
    f = 1 + dxd12 * (en - pot)

    # Forward integration and crossing couting.
    cross = 0

    for i in range (1, n-1):
        y[i+1] = ( (12 - 10 * f[i]) * y[i] - y[i-1] * f[i-1] ) / f[i+1]
        if y[i] != math.copysign(y[i], y[i+1]):
            cross += 1
        else:
            pass

    # Check number of crossings and update 'en', 'eup' and 'elw' values.
    if cross > node:
        eup = en
    else:
        elw = en

    en = (eup + elw) / 2

    # If not Morse potential, and even number of nodes, update y_1.
    if selection != 3 and even:
        y[1] = ( (12 - 10 * f[0]) * y[0]) / 2*f[1]

    return en, eup, elw, f, y


# Now, the script will search the eigenvalue using bisection. It will call
# the 'calculation' function, until the bounds for the eigenvalue are
# close enough of one another.
try:
    while True:
        en, eup, elw, f, y = calculation(en, eup, elw, n, dxd12, node, pot, f, y, selection)
        if eup - elw < 1e-10:
            break
        else:
            pass
except KeyboardInterrupt:
    sys.exit('Script stopped.')

# Construct eigenfunction for x<0 for harmonic and double-well potentials.
# Also, set the variable 'node' back to being the number of nodes over the whole grid.
if selection != 3:
    x = np.linspace(-xmax, xmax, 2*n+1)
    func = np.zeros(2*n + 1)
    if even:
        node = 2*node
        for i in range(0,n):
            func[n+i] = y[i]
            func[n-i] = y[i]

    else:
        node = 2*node + 1
        for i in range(0,n):
            func[n+i] = y[i]
            func[n-i] = -y[i]
        pass

else:
    func = y

# Define exact values for eigenvalues and eigenfunctions.
# Harmonic potential.
if selection == 1:
    # Define the exact value for the eigenvalue.
    e_exact = node + 1/2
    # Define the exact eigenfunction over the grid.
    N_n = math.sqrt(1 / ((2**node) * math.factorial(node) * math.sqrt(math.pi)))

    psi = np.zeros(2*n + 1)
    for i in range (0, 2*n + 1):
        psi[i] = N_n * math.exp(-x[i]*x[i] / 2) * sc.eval_hermite(node, x[i])

    # Since the algorithm converges to twice the desired value for epsilon, we halve it.
    en = en/2

# Morse Potential.
if selection == 3:
    # Define the exact value for the eigenvalue.
    e_exact = -(9/2 - node)*(9/2 - node)

    # Define the exact eigenfunction over the grid.
    N_n = math.sqrt( ( math.factorial(node) * (9 - 2*node) ) / sc.gamma(10 - node) )

    psi = np.zeros([n])
    z = np.zeros([n])

    for i in range (0, n):
        z[i] = 10 * math.exp(-(x[i] - 3))
        psi[i] = N_n * ( (z[i])**(9/2 - node) ) * math.exp(-z[i] / 2) * sc.eval_genlaguerre(node, 9 - 2*node, z[i])



# Plot comparison between calculated results and exact results, unless it's the double-well potential.
if selection == 2:
    plt.figure(frameon=False)

    plt.plot(x[6000:24000], func[6000:24000], 'bo', markersize=2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$\Psi_{{{n}}}(\chi)$'.format(n=str(node)))
    plt.title('Double-well. Nodes: ' + str(node) + '. Calculated eigenvalue: ' + r'$E_{{{n}}}$'.format(n=str(node)) + ' = ' + str(round(en, 8)), y=1.03)

    plt.grid(True)
    plt.show()

elif selection == 1:
    plt.figure(frameon=False)

    plt.subplot(121)
    plt.plot(x[6000:24000], func[6000:24000], 'bo', markersize=2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$\Psi_{{{n}}}(\chi)$'.format(n=str(node)))
    plt.title('Harmonic oscillator. Nodes: ' + str(node) + '. Calculated eigenvalue: ' + r'$\varepsilon_{{{n}}}$'.format(n=str(node)) + ' = ' + str(round(en, 8)), y=1.03)
    plt.grid(True)

    plt.subplot(122)
    plt.plot(x[6000:24000], psi[6000:24000], 'r', linewidth=2)
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$\Psi_{{{n}}}(\chi)$'.format(n=str(node)))
    plt.title('Harmonic oscillator. Nodes: ' + str(node) + '. Exact eigenvalue: ' + r'$\varepsilon_{{{n}}}$'.format(n=str(node)) + ' = ' + str(e_exact), y=1.03)
    plt.grid(True)

    plt.show()

else:
    plt.figure(frameon=False)

    plt.subplot(121)
    plt.plot(x, func, 'bo', markersize=2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\Psi_{{{n}}}(x)$'.format(n=str(node)))
    plt.title('The Morse potential. Nodes: ' + str(node) + '. Calculated eigenvalue: ' + r'$\xi_{{{n}}}$'.format(n=str(node)) + ' = ' + str(round(en, 8)), y=1.03)
    plt.grid(True)

    plt.subplot(122)
    plt.plot(x, psi, 'r', linewidth=2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\Psi_{{{n}}}(x)$'.format(n=str(node)))
    plt.title('The Morse potential. Nodes: ' + str(node) + '. Exact eigenvalue: ' + r'$\xi_{{{n}}}$'.format(n=str(node)) + ' = ' + str(e_exact), y=1.03)
    plt.grid(True)

    plt.show()




# Plot potential.
if selection == 2:
    pot = (np.float_power(x, 4)/9 - 2*np.square(x)/3 + 1)/2

    plt.figure(frameon=False)

    plt.subplot(121)
    plt.plot(x, pot, 'g', linewidth=2)
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$V(\chi)$')
    plt.title('The double-well potential for ' + r'$2\varepsilon = 1$' + ' and ' + r'$\delta = \sqrt{3}b$' + '.')
    plt.grid(True)

    plt.subplot(122)
    plt.plot(x, pot, 'g', linewidth=2)
    plt.axis([-4, 4, -0.5, 2.5])
    plt.xlabel(r'$\chi$')
    plt.ylabel(r'$V(\chi)$')
    plt.title('The double-well potential near the origin.')
    plt.grid(True)

    plt.show()

elif selection == 3:
    plt.figure(frameon=False)

    plt.subplot(121)
    plt.plot(x, pot, 'g', linewidth=2)
    plt.axis([0, 10, -30, 1000])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$V(x)$')
    plt.title('The Morse potential for ' + r'$\lambda$' + ' = 5 and ' r'$x_e$' + ' = 3.')
    plt.grid(True)

    plt.subplot(122)
    plt.plot(x, pot, 'g', linewidth=2)
    plt.axis([2, 5, -30, 10])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$V(x)$')
    plt.title('The Morse potential near minimum for ' + r'$\lambda$' + ' = 5 and ' r'$x_e$' + ' = 3.')
    plt.grid(True)

    plt.show()


end = input('Press enter to end the script.')
