import math
import numpy as np

# Function: radial_grid
# Sets up the arrays that correspond to the grid.
# Parameters:
#   zmesh -- atomic number
#   xmin -- minimum value for the logarithmic grid
#   dx -- the step for the logarithmic grid
#   mesh -- maximum value for the logarithmic grid
#   r -- array to store the values for the radial coordinate over the grid
#   sqr -- array to store the values for the square root of the radial coordinate over the grid
#   r2 -- array to store the values for the square of the radial coordinate over the grid
def radial_grid(zmesh, xmin, dx, mesh, r, sqr, r2):
    x = np.zeros([mesh])
    for i in range (0, mesh):
        x[i] = xmin + dx * i

    r = np.divide(np.exp(x), zmesh)
    sqr = np.sqrt(r)
    r2 = np.multiply(r, r)

    return r, sqr, r2



# Function: potential
# Sets up the potential array.
# Parameters:
#   zeta -- atomic number
#   r -- values for the radial coordinate over the grid
#   pot -- array to store the values for the potential over the grid
#   mesh -- maximum value for the logarithmic grid
def potential(zeta, r, pot, mesh):
    for i in range (0, mesh):
        pot[i] = -2* (zeta / r[i])
    return pot, 0

    # To build perturbed potential with an epsilon
    #epsilon = float(input('Input epsilon for perturbed potential: '))
    #print(' ')
    #for i in range (0, mesh):
        #pot[i] = -2/r[i] + (epsilon)/(r[i]*r[i])
    #return pot, epsilon



# Function: fcross
# Set up the f-function and determine the position of it's last change of sign
# Parameters:
#   f -- array to store the values for the f-function defined by Numerov's Method - 1
#   ddx12 -- square of the step for the logarithmic grid divided by 12
#   sqlhf -- azimuthal quantum number plus a half, all squared
#   r2 -- values of the square of the radial coordinate over the grid
#   pot -- values of the potential over the grid
#   e -- estimate for the energy (eigenvalue)
#   mesh -- maximum value for the logarithmic grid
def fcross(f, ddx12, sqlhf, r2, pot, e, mesh):
    icl = -1
    f[0] = ddx12 * (sqlhf + r2[0] * (pot[0] - e))
    for i in range (1, mesh):
        f[i] = ddx12 * (sqlhf + r2[i] * (pot[i] - e))
        if f[i] == 0:
            f[i] = 1e-20
        elif f[i] != math.copysign(f[i], f[i-1]):
            icl = i
        else:
            pass

    return icl, f


# Function: energyl
# Set up a new smaller estimate for the energy (eigenvalue)
# Parameters:
#   e -- current estimate for the energy (eigenvalue)
#   elw -- lower bound to the energy (eigenvalue)
#   eup -- upper bound to the energy (eigenvalue)
def energyl(e, elw, eup):
    eup = e
    e = (eup + elw) / 2
    return e, elw, eup


# Function: energyu
# Set up a new higher estimate for the energy (eigenvalue)
# Parameters:
#   e -- current estimate for the energy (eigenvalue)
#   elw -- lower bound to the energy (eigenvalue)
#   eup -- upper bound to the energy (eigenvalue)
def energyu(e, elw, eup):
    elw = e
    e = (eup + elw) / 2
    return e, elw, eup


# Function: forwardint
# Integrate forward with Numerov's method, count the number of crossings.
# Parameters:
#   icl -- classical inversion point
#   y --  array for the y function defined as the 'chi' function over the square root of the radial coordinate
#   f -- array for the 'f' function as defined by Numerov's method
#   ncross - variable to count the number of crossings
def forwardint(icl, y, f, ncross):
    ncross = 0
    for i in range (1, icl):
        y[i+1] = ( (12 - 10 * f[i]) * y[i] - f[i-1] * y[i-1]) / f[i+1]
        if y[i] != math.copysign(y[i], y[i+1]):
            ncross += 1
        else:
            pass

    return y, ncross


# Function: backwardint
# Integrate backward with Numerov's method.
# Parameters:
#   mesh -- maximum value for the logarithmic grid
#   icl -- classical inversion point
#   y -- array for the y function defined as the 'chi' function over the square root of the radial coordinate
#   f -- array for the 'f' function as defined by Numerov's method
def backwardint(mesh, icl, y, f):
    for i in range (mesh - 2, icl, -1):
        y[i-1] = ( (12 - 10 * f[i]) * y[i] - f[i+1] * y[i+1]) / f[i-1]

        # If the 'y' values get too big, re-scale them.
        if y[i-1] > 1e10:
            for j in range (mesh - 1, i - 2, -1):
                y[j] = y[j] / y[i-1]
        else:
            pass

    return y


# Function: normalize
# Normalization of the 'y' function.
# Parameters:
#   norm -- variable to store the normalization constant
#   mesh -- maximum value for the logarithmic grid
#   y -- array for the y function defined as the 'chi' function over the square root of the radial coordinate
#   r2 -- values of the square of the radial coordinate over the grid
#   dx -- the step for the logarithmic grid
def normalize(norm, mesh, y, r2, dx):
    norm = 0
    for i in range (0, mesh):
        norm = norm + y[i]*y[i] * r2[i] * dx
    norm = math.sqrt(norm)
    y = y / norm
    return y


# Function: eigenupdate
# Improve convergence with perturbation theory. Give an estimate of the difference between the current estimate and it's final value.
#   y -- array for the y function defined as the 'chi' function over the square root of the radial coordinate
#   f -- array for the 'f' function as defined by Numerov's method
#   icl -- classical inversion point
#   ddx12 -- square of the step for the logarithmic grid divided by 12
#   dx -- the step for the logarithmic grid
#   e -- current estimate for the energy (eigenvalue)
#   elw -- lower bound to the energy (eigenvalue)
#   eup -- upper bound to the energy (eigenvalue)
#   de -- estimate of the difference between the current estimate of the eigenvalue and it's final value
def eigenupdate(y, f, icl, ddx12, dx, e, elw, eup, de):
    # Find the value of the cusp at the matching point.
    ycusp = ( y[icl-1]*f[icl-1] + f[icl+1]*y[icl+1] + 10*f[icl]*y[icl] ) / 12
    fcusp = f[icl] * (y[icl]/ycusp - 1)

    # Update eigenvalue
    de = (fcusp / ddx12) * ycusp * ycusp * dx
    if de > 0:
        elw = e
    elif de < 0:
        eup = e
    else:
        pass

    # Get new estimate while avoiding the estimate from falling outside the bounds:
    e = max(min(e+de , eup) , elw)
    return e, elw, eup, de

# Function: solve
# Solves the schrodinger equation in radial corrdinates on a logarithmic grid by Numerov's method.
# Parameters:
#   n -- principal quantum number
#   l -- azimuthal quantum number
#   e -- estimate for the energy (eigenvalue)
#   mesh -- maximum value for the logarithmic grid
#   dx -- the step for the logarithmic grid
#   r -- values of the radial coordinate over the grid
#   sqr -- values of the square root of the radial coordinate over the grid
#   r2 -- values of the square of the radial coordinate over the grid
#   pot -- values of the potential over the grid
#   zeta -- atomic number
#   y -- array for the y function defined as the 'chi' function over the square root of the radial coordinate
def solve(n, l, e, mesh, dx, r, sqr, r2, pot, zeta, y):
    # Initialize some parameters for later use.
    eps = 1e-10
    f = np.zeros([mesh])
    ddx12 = (dx * dx) / 12
    sqlhf = (l + 0.5)**2
    x2l2 = (2 * l) + 2
    icl = 0
    ncross = 0
    fac = 0
    norm = 0
    de = 0

    # Set some initial lower and upper bounds to the eigenvalue.
    eup = pot[mesh-1]
    elw = np.amin((sqlhf / r2) + pot)
    e = (eup + elw) / 2

    # Solve the schrodinger equation in radial corrdinates on a logarithmic grid by Numerov's method.

    try:
        while True:
            # Check convergence.
            if eup - elw < eps:
                break
            else:
                pass

            # Find the classical inversion point.
            icl, f = fcross(f, ddx12, sqlhf, r2, pot, e, mesh)
            if icl < 0 or icl >= mesh - 2:
                e, elw, eup = energyl(e, elw, eup)
                continue
            else:
                pass

            # Set up the 'f' function as defined by Numerov's method.
            f = 1 - f
            nodes = n - l - 1

            # Wave-fcuntion in the first two points.
            y[0] = (r[0]**(l + 1)) * (1 - 2 * zeta * r[0] / x2l2) / sqr[0]
            y[1] = (r[1]**(l + 1)) * (1 - 2 * zeta * r[1] / x2l2) / sqr[1]

            # Forwards integration, while counting number of crossings.
            y, ncross = forwardint(icl, y, f, ncross)

            # Store the value of 'y' at the classical inversion point for later
            fac = y[icl]

            # Check number of crossings and update energy accordingly
            if ncross != nodes:
                if ncross > nodes:
                    e, elw, eup = energyl(e, elw, eup)
                else:
                    e, elw, eup = energyu(e, elw, eup)
                continue
            else:
                pass

            # Wave-function in the last two points. Assumes y[mesh] = 0 and y[mesh - 1] = dx
            y[mesh - 1] = dx
            y[mesh - 2] = ( 12 - 10 * f[mesh-1] ) * y[mesh-1] / f[mesh-2]

            # Backwards integration.
            y = backwardint(mesh, icl, y, f)

            # Re-scale the function at the right to match both integrations at the classical inversion point
            fac = fac / y[icl]
            for i in range (icl, mesh):
                y[i] = y[i]*fac

            # Normalization
            y = normalize(norm, mesh, y, r2, dx)

            # Eigenvalue update using perturbation theory (See Giannozzi [Reference 2], page 22 for an explanation)
            e, elw, eup, de = eigenupdate(y, f, icl, ddx12, dx, e, elw, eup, de)

            # Check convergence.
            if abs(de) < eps:
                break
            else:
                pass
    except KeyboardInterrupt:
        sys.exit('Script stopped.')

    # I use different normalization for the 'chi' function than Giannozzi; this line is to adjust to that.
    y = y * (2/n)

    return e, y

