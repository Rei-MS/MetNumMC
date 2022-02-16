import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as scint
import scipy.special as sc
import HydroRadCalc as hrc

# Find a solution with given 'n', 'l' for the hydrogen atom solving the
# radial schrodinger equation by Numerov's method.
# Can be modified slightly to solve this problem with a small perturbation
# to the potential.
# Units used: Atomic (Rydberg)


# Define the parameters necessary to set up the logarithmic grid.
zeta = 1
'''
To change the atomic number simply replace the statement 'zeta = 1' with:
zeta = int(input('Input the atomic number: '))
print(' ')
'''
zmesh = zeta
rmax = 100
xmin = -8
dx = 0.01

mesh = int((math.log(zmesh*rmax) - xmin)/dx)

# Initialize required arrays as zeroes.
r = np.zeros([mesh])
sqr = np.zeros([mesh])
r2 = np.zeros([mesh])
pot = np.zeros([mesh])
y = np.zeros([mesh])
chi = np.zeros([mesh])

# Set up the arrays corresponding to the grid.
r, sqr, r2 = hrc.radial_grid(zmesh, xmin, dx, mesh, r, sqr, r2)

# Set up the potential array.
pot, epsilon = hrc.potential(zeta, r, pot, mesh)

# Input principal and azimuthal quantum numbers 'n' and 'l'.
n = int(input('Input principal quantum number n: '))
print(' ')

l = int(input('Input azimuthal quantum number l: '))
print(' ')


if n < 1:
    sys.exit('Script Stopped. Principal quantum number must be a positive integer.')
elif l not in np.arange(n):
    sys.exit('Script Stopped. Azimuthal quantum number must be a positive integer between 0 and n-1.')
else:
    pass

# Solve the schrodinger equation in radial corrdinates by Numerov's method.
e = 0
e, y = hrc.solve(n, l, e, mesh, dx, r, sqr, r2, pot, zeta, y)


# Set up the 'chi' function
chi = y * sqr


# Define the exact eigenvalue
e_exact = - zeta*zeta/(n*n)

# Define the exact eigenfunction
cte = math.sqrt( (2 / n) * (2 / n) * (2 / n) * ( math.factorial(n-l-1) / ( 2*n * math.factorial(n+l) ) ) )

rho = np.zeros([mesh])
chi_exact = np.zeros([mesh])

for i in range (0, mesh):
    rho[i] = 2*zeta*r[i]/n
    chi_exact[i] = cte * ( (rho[i])**(l+1) ) * math.exp(-rho[i]/2) * sc.eval_genlaguerre(n-l-1, 2*l + 1, rho[i])

# To print eigenvalue obtained
print('Calculated eigenvalue: ' + str(e))
print(' ')

# To print exact eigenvalue
print('Exact eigenvalue: ' + str(e_exact))
print(' ')

'''
# Define the first order correction to the eigenvalue.
sum_nu = 0
for i in range(0, n-l):
    sum_nu = sum_nu + math.factorial(i + 2*l)/math.factorial(i)

e_corr = e_exact + (epsilon) * math.factorial(n-l-1) * sum_nu / (2 * n * math.factorial(n+l))

# To print eigenvalue correction with perturbation theory
print('First order correction to the eigenvalue: ' + str(e_corr))
print(' ')
'''

# Plot the calculated values for the eigenfunction and the eigenvalue
plt.plot(r, chi, 'bo', markersize=1.3)
plt.xlabel('r')
plt.ylabel(r'$\chi_{nl}$' + '(r)')
plt.title('n = ' + str(n) + ', l = ' + str(l) + ', Z = ' + str(zeta) + '. Calculated eigenvalue = ' + str(round(e, 4)))
#plt.xlim(-1, 50)


#To plot the exact values for the eigenfunction and the eigenvalue:
#plt.plot(r, chi_exact, 'r', linewidth=2)
#plt.xlabel('r')
#plt.ylabel(r'$\chi_{nl}$' + '(r)')
#plt.title('n = ' + str(n) + ', l = ' + str(l) + '. Exact eigenvalue = ' + str(round(e_exact, 4)))
#plt.xlim(-1, 50)


'''
# To plot the potential
plt.plot(r, pot, 'g', linewidth=2)
plt.xlabel('r')
plt.ylabel('V(r)')
plt.title('Coulomb potential for Z = 1.')
'''

'''
# To restrict the axis values for the area near the origin
plt.axis([0, 6, -15, 1])
'''

# Calculate expectation values of r and r^-1.
# Expectation value of r.
evr = n*n*scint.trapz(r*chi*chi, r)/4
print('Expectation value of r: ' + str(evr))
print(' ')

# Expectation value of r^-1.
evr_1 = n*n*scint.trapz((chi*chi)/r, r)/4
print('Expectation value of r^-1: ' + str(evr_1))
print(' ')


plt.grid(True)
plt.show()

