import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as scint
import ScatteringCalc as sca

'''
Find total cross section for scattering of a particle by a potential
implementing Numerov's method.
Units used: energies in meV and distances in Angstrom.
'''

# Define the ratio between reduced Planck's constant squared and 2m, where
# m is the mass of the particle.
hbar2_div_2mu = 2.08

# Define parameters of the Lennard-Jones potential. Here, epsilon is in meV
# and sigma is in A (Angstrom). Default values are 5.9 and 3.57, respectively.
epsilon = 5.9
sigma = 3.57

# Define start (r_min) and end point (r_max) for integration. r_max is also
# the point from which we can ignore the potential.
r_min = 1.8
r_max = 18

# Define the number of grid points.
mesh = 2000

# Define a range for energies (e_min, e_max), and a step for the energy
# (delta_e). Also, calculate the number of 'steps' in the interval.
e_min = 0.1
e_max = 3.5
delta_e = 0.01
e_mesh = int(round((e_max - e_min)/delta_e))

# Define the upper bound for 'l' (l_max).
l_max = 9

# Define arrays for the energy and the total cross section
energy = np.zeros([e_mesh])

for i in range(0, e_mesh):
    energy[i] = e_min + i * delta_e

total_cross = np.zeros([e_mesh])


# Call the solve function
try:
    total_cross, r, potential, delta_l =  sca.solve(e_mesh, e_min, delta_e, hbar2_div_2mu, r_max, r_min, mesh, epsilon, sigma, l_max, total_cross)
except KeyboardInterrupt:
    sys.exit('Script stopped.')


'''
# Set up partial cross sections. Uses the same index choices as delta_l.
partial_cross = np.zeros([(l_max+1)*e_mesh])
for i in range(0, e_mesh):
    k2 = energy[i] / hbar2_div_2mu
    for l in range(0, l_max+1):
        partial_cross[l*e_mesh + i] = ( 4*math.pi / (k2) ) * (2*l + 1) * ((math.sin(delta_l[l*e_mesh + i]))**2)
'''

# To plot total cross section as a function of energy
plt.plot(energy, total_cross, 'bo', markersize=1.3)

'''
# To plot partial cross sections.
for l in range(0, l_max+1):
    plt.plot(energy, partial_cross[l*e_mesh:(l+1)*e_mesh], 'o', markersize=1, label='l = {}'.format(l))
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
'''

plt.xlabel('E [meV]')
plt.ylabel(r'$\sigma_{t}$' + '(E)')
plt.title('Total cross section as a function of energy.')


# To plot the phase shifts, for each value of 'l', as a function of energy
#for l in range(0, l_max+1):
    #plt.plot(energy, delta_l[l*e_mesh:(l+1)*e_mesh], 'o', markersize=1, label='l = {}'.format(l))

#plt.xlabel('E [meV]')
#plt.ylabel(r'$\delta_{l}$' + '(E)')
#plt.title('Phase shift, for each l, as a function of energy.')
#plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)


# To plot the potential
#plt.plot(r, potential, 'g', linewidth=2)
#plt.xlabel('r [Angstrom]')
#plt.ylabel('V(r) [meV]')
#plt.title('Lennard-Jones potential.')

'''
# To plot effective potential as a function of l.
eff_potential = np.zeros([(l_max+1)*mesh])
for i in range(0, mesh):
    for l in range(0, l_max+1):
        eff_potential[l*mesh + i] = potential[i] + hbar2_div_2mu * l * (l+1) / (r[i]*r[i])

for l in range(0, l_max+1):
    plt.plot(r, eff_potential[l*mesh:(l+1)*mesh], 'o', markersize=1, label='l = {}'.format(l))
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.xlabel('r [Angstrom]')
plt.ylabel(r'$V_{eff}$' '[meV]')
plt.title('Effective potential as a function of l.')
plt.axis([2, 7, -10, 70])
'''

'''
# To calculate phase shifts by their integral approximation and compare with the ones obtained from the algorithm:
delta_app = np.zeros([(l_max+1)*e_mesh])
jbessel = np.zeros([mesh])

# Compute the integral for all energies.
for i in range(0, e_mesh):
    k = math.sqrt(energy[i] / hbar2_div_2mu)
    for l in range(0, l_max+1):
        # Set up the bessel j function over the grid.
        for m in range(0, mesh):
            jbessel[m] = sca.sbessel_j(k*r[m], l)

        delta_app[l*e_mesh + i] = - (k / hbar2_div_2mu) * scint.trapz(r*r*potential*jbessel*jbessel, r)
'''

# Plot comparisons.
#for l in {0, 1, 2, 3, 4, 5, 6}:
#    plt.figure()
#    plt.plot(energy, delta_l[l*e_mesh:(l+1)*e_mesh], 'bo', markersize=1, label='By algorithm')
#    plt.plot(energy, delta_app[l*e_mesh:(l+1)*e_mesh], 'go', markersize=1, label='By integration')
#    plt.legend(loc="upper left")
#    plt.xlabel('E [meV]')
#    plt.ylabel(r'$\delta_{{{val}}}(E)$'.format(val=str(l)))
#    plt.title('Phase shift comparison.  l = {}.'.format(l))
#    plt.grid(True)
#    plt.show()


plt.grid(True)
plt.show()
