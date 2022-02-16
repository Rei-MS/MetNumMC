import math
import numpy as np


# Function: setup
# Sets up necessary parameters to solve the scattering problem.
# Parameters:
#   e_min -- lower bound for the energy interval
#   delta_e -- step for the energy interval
#   hbar2_div_2mu -- h bar squared over two m
#   r_max -- end point for integration
#   r_min -- start point for integration
#   mesh -- number of grid points
#   i -- current iteration
def setup(e_min, delta_e, hbar2_div_2mu, r_max, r_min, mesh, i):
    # Set current energy, and the corresponding value for k.
    e = e_min + i*delta_e
    k = math.sqrt(e / hbar2_div_2mu)

    # Set 1/4 of the wavelength.
    lambda_4 = math.pi / (2 * k)

    # Set the step in the radial grid.
    delta_r = (r_max + lambda_4 - r_min) / mesh

    # Set the (step)^2/12 parameter used in Numerov's algorithm.
    deldelr12 = delta_r*delta_r / 12

    # Set necessary parameters for the calculation of the phase shift.
    r2 = r_max + lambda_4
    r1 = r2 - delta_r * int(round( lambda_4 / delta_r ))

    i2 = mesh - 1
    i1 = mesh - 1 - int(round( lambda_4 / delta_r ))

    return e, k, lambda_4, delta_r, deldelr12, r2, r1, i2, i1

# Function: radial_grid
# Sets up the array for the radial grid.
# Parameters:
#   r -- array to store the values for the radial grid
#   r_min -- lower bound for the radial coordinate
#   delta_r -- size of the step in the radial grid
#   mesh -- number of grid points
def radial_grid(r, r_min, delta_r, mesh):
    for i in range(0, mesh):
        r[i] = r_min + i * delta_r

    return r


# Function: ljpotential
# Sets up the potential array for the Lennard-Jones potential.
# Parameters:
#   mesh -- number of grid points.
#   potential -- array to store the potential values
#   epsilon -- parameter of the Lennard-Jones potential
#   sigma -- parameter of the Lennard-Jones potential
#   r -- array with the values of the radial coordinate
def ljpotential(mesh, potential, epsilon, sigma, r):
    for i in range(0, mesh):
        potential[i] = epsilon * ( (sigma/r[i])**12 - 2*((sigma/r[i])**6) )

    return potential


# Function: f_function
# Sets up the f-function used by Numerov's algorithm.
# Parameters:
#   f -- array to store the values for the f-function
#   deldelr12 -- the (step)^2/12 parameter used in Numerov's algorithm
#   hbar2_div_2mu == h bar squared over two m
#   r -- array with the values of the radial coordinate
#   potential -- array with the values of the potential
#   e -- current value for the energy
#   l -- current iteration
#   mesh -- number of grid points
def f_function(f, deldelr12, hbar2_div_2mu, r, potential, e, mesh, l):
    for i in range(0, mesh):
        f[i] = (deldelr12 / hbar2_div_2mu) * ((hbar2_div_2mu*l*(l+1)/(r[i]*r[i])) + potential[i] - e)
        f[i] = 1 - f[i]

    return f


# Function: initial_conditions
# Sets up the initial conditions for the integration
# Parameters:
#   chi0 -- first entry of the array for the wave function
#   chi1 -- second entry of the array for the wave function
#   r0 -- first entry of the array for the radial coordinate
#   r1 -- second entry of the array for the radial coordinate
#   epsilon -- parameter of the Lennard-Jones potential
#   sigma -- parameter of the Lennard-Jones potential
#   hbar2_div_2mu -- h bar squared over two m
def initial_conditions(chi0, chi1, r0, r1, epsilon, sigma, hbar2_div_2mu):
    chi0 = math.exp( - math.sqrt( epsilon * sigma**12 / (hbar2_div_2mu * 25)) / r0**5 )
    chi1 = math.exp( - math.sqrt( epsilon * sigma**12 / (hbar2_div_2mu * 25)) / r1**5 )

    return chi0, chi1


# Function: integrate
# Integrates (forward only) using Numerov's method.
# Parameters:
#   chi -- array to store the values for the wave function
#   f -- array with the values of the f-function
#   mesh -- number of grid points
def integrate(chi, f, mesh):
    for i in range(1, mesh - 1):
        chi[i+1] = ( ( 12 - 10*f[i] ) * chi[i] - chi[i-1]*f[i-1] ) / f[i+1]

    return chi

# Function: normalize
# Normalizes the wave function
# Parameters:
#   chi -- array with the values of the wave function
#   delta_r -- size of the step in the radial grid
def normalize(chi, delta_r):
    norm = np.dot(chi, chi) * delta_r
    chi = np.divide(chi, math.sqrt(norm))

    return chi


# Function: phase_shift
# Calculates the phase shift
# Parameters:
#   chi -- array with the values of the wave function
#   r2 -- value for the coordinate of the further point
#   r1 -- value for the coordinate of the closer point
#   i1 -- index corresponding to r1
#   i2 -- index corresponding to r2
#   k -- value of k for the current energy
#   l -- current iteration
def phase_shift(chi, r2, r1, i1, i2, k, l):
    kappa = r1*chi[i2] / (r2*chi[i1])

    tan_delta = ( kappa * sbessel_j(k*r1, l) - sbessel_j(k*r2, l) ) / ( kappa * sbessel_n(k*r1, l) - sbessel_n(k*r2, l) )

    delta = math.atan(tan_delta)

    return delta


# Function: solve
# Solves the scattering problem. Calculates total cross section, and the phase shifts.
# Parameters:
#   e_mesh -- the number of steps in the energy interval
#   e_min -- lower bound for the energy interval
#   delta_e -- step for the energy interval
#   hbar2_div_2mu -- h bar squared over two m
#   r_max -- end point for integration
#   r_min -- start point for integration
#   mesh -- number of grid points
#   epsilon -- parameter of the Lennard-Jones potential
#   sigma -- parameter of the Lennard-Jones potential
#   l_max -- the upper bound for 'l'
#   total_cross -- array to store the total cross section
def solve(e_mesh, e_min, delta_e, hbar2_div_2mu, r_max, r_min, mesh, epsilon, sigma, l_max, total_cross):

    # Set up necessary arrays.
    r = np.zeros([mesh])                        # The radial coordinate
    chi = np.zeros([mesh])                      # The wave function.
    potential = np.zeros([mesh])                # The potential
    f = np.zeros([mesh])                        # The f-function for Numerov's method.
    cross = np.zeros([l_max+1])                 # To store the partial cross sections.
    delta_l = np.zeros([(l_max+1)*e_mesh])      # To store the phase shifts.


    for i in range(0, e_mesh):

        # Set up necessary parameters for the algorithm.
        e, k, lambda_4, delta_r, deldelr12, r2, r1, i2, i1 = setup(e_min, delta_e, hbar2_div_2mu, r_max, r_min, mesh, i)

        # Fill the array for the radial grid.
        r = radial_grid(r, r_min, delta_r, mesh)

        # Fill the array for the potential.
        potential = ljpotential(mesh, potential, epsilon, sigma, r)
        #potential = pwell(r, mesh, potential)

        # Reset the cross array.
        cross = np.zeros([l_max+1])

        # Start integration.
        for l in range(0, l_max+1):

            # Set up the f-function used by Numerov's algorithm.
            f = f_function(f, deldelr12, hbar2_div_2mu, r, potential, e, mesh, l)

            # Reset the wave-function array:
            chi = np.zeros([mesh])

            # Set up the initial conditions.
            chi[0], chi[1] = initial_conditions(chi[0], chi[1], r[0], r[1], epsilon, sigma, hbar2_div_2mu)
            #chi[0], chi[1] = initial_conditions_pwell(chi[0], chi[1], r[0], r[1], l)

            # Integrate forward
            chi = integrate(chi, f, mesh)

            # Normalize the wave function
            chi = normalize(chi, delta_r)

            # Calculate the phase shift.
            delta = phase_shift(chi, r2, r1, i1, i2, k, l)

            # Store the calculated phase shift. See final lines for explanation.
            delta_l[l*e_mesh + i] = delta

            # Calculate the partial cross section
            cross[l] = cross[l] + ( 4*math.pi / (k**2) ) * (2*l + 1) * ((math.sin(delta))**2)

        # Fill the corresponding entry of the array for the total cross section
        total_cross[i] = np.sum(cross)


    return total_cross, r, potential, delta_l


# Function: sbessel_j
# Calculates the spherical Bessel function of the first kind at a given point.
# Parameters:
#   z -- value for the coordinate where the function is calculated
#   n -- order of the function needed
def sbessel_j(z, n):
    f_n_1 = math.cos(z) / z
    f_n = math.sin(z) / z
    if n==0:
        return f_n
    else:
        for l in range(0, n):
            f_np1 = (2*l + 1) * f_n / z - f_n_1
            f_n_1 = f_n
            f_n = f_np1

        return f_n


# Function: sbessel_n
# Calculates the spherical Bessel function of the second kind at a given point.
# Parameters:
#   z -- value for the coordinate where the function is calculated
#   n -- order of the function needed
def sbessel_n(z, n):
    f_n_1 = math.sin(z) / z
    f_n = -1 * math.cos(z) / z
    if n==0:
        return f_n
    else:
        for l in range(0, n):
            f_np1 = (2*l + 1) * f_n / z - f_n_1
            f_n_1 = f_n
            f_n = f_np1

        return f_n

# Function: pwell
# Sets up the potential array for a well.
# Parameters:
#   r -- array to store the values for the radial grid
#   mesh -- number of grid points.
#   potential -- array to store the potential values
def pwell(r, mesh, potential):
    for i in range(0, mesh):
        if r[i] < 5:
            potential[i] = -5
        else:
            potential[i] = 0

    return potential

# Function: initial_conditions_pwell
# Sets up the initial conditions for the integration
# Parameters:
#   chi0 -- first entry of the array for the wave function
#   chi1 -- second entry of the array for the wave function
#   r0 -- first entry of the array for the radial coordinate
#   r1 -- second entry of the array for the radial coordinate
#   l -- current iteration
def initial_conditions_pwell(chi0, chi1, r0, r1, l):
    chi0 = r0**(l+1)
    chi1 = r1**(l+1)

    return chi0, chi1

###########################
# Explanation for storing the phase shifts.

# This method stores the phase shifts, for equal 'l', one after the other.

# This way, the first 'e_mesh' elements (i.e. indices 0, 1, ..., e_mesh-1)
# are the phase shifts, with increasing energy, for l=0.

# The second 'e_mesh' elements (indices e_mesh, e_mesh+1, ..., 2*e_mesh-1)
# are the phase shifts, with increasing energy, for l=1.

# For a given value of l, the indices l*e_mesh, l*e_mesh+1, l*e_mesh+2, ...,
# l*e_mesh + e_mesh -1 = (l+1)*e_mesh - 1, correspond to the phase shifts
# for that value of l, with increasing energy.

# The order is according to the 'energy grid'. That is, the first index for
# a given l is the phase shift for energy = e_min. The second index is for
# energy = e_min + delta_e, and so on, up to energy = e_min + e_mesh*delta_e, which
# equals e_max.

# This method for storing elements was chosen so as to make the plot a lot
# easier without having to define auxiliary arrays for the plot of the phase shifts.
###########################
