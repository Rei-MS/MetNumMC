import sys
import math
import numpy as np
from scipy.linalg import eigh_tridiagonal
import scipy.sparse.linalg as ssl

'''
Function: potential
Returns the potential evaluated at a specific point of the grid.
Parameters:
    selection - selection of potential.
    parameters - array containing parameters related to the potential.
    k - current iteration in the hamiltonian_product function.
    grid_x - array for the grid in the x-direction.
    grid_y - array for the grid in the y-direction.
    N_x - number of grind points in the x-direction.
'''
def potential(selection, parameters, k, grid_x, grid_y, N_x):
    # Calculate the i,j indices for the current iteration.
    i = (k % (N_x - 2)) + 1
    j = math.floor(k / (N_x - 2)) + 1

    if selection == 1:
        return 0
    elif selection == 2:
        return (grid_x[i]*grid_x[i] + grid_y[j]*grid_y[j])/2
    elif selection == 3:
        return -1 * parameters[0] * np.exp(-(grid_x[i]*grid_x[i] + grid_y[j]*grid_y[j]))
    else:
        sys.exit('Script Stopped. Function: potential.')

'''
Function: hamiltonian_product
Returns the product of the Hamiltonian matrix H and a vector.
Parameters:
    vector - array of the vector.
    selection - selection of potential.
    parameters - array containing parameters related to the potential.
    eta - eta parameter for the corresponding potential.
    mu - mu parameter for the corresponding potential.
    N_x - number of grind points in the x-direction.
    N_y - numbeer of grid points in the y-direction.
    grid_x - array for the grid in the x-direction.
    grid_y - array for the grid in the y-direction.
'''
def hamiltonian_product(vector, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y):
    dim = (N_x - 2) * (N_y - 2)
    vector_result = np.zeros(dim)

    for k in range(0, dim):
        # Store appropriate vector entries.
        # Check for boundary conditions ruled out by the k-indexing scheme.
        # This is for values that fall outside the allowed k values.
        try:
            v1 = vector[k - (N_x - 2)]
        except IndexError:
            v1 = 0

        try:
            v2 = vector[k - 1]
        except IndexError:
            v2 = 0

        v3 = vector[k]

        try:
            v4 = vector[k + 1]
        except IndexError:
            v4 = 0

        try:
            v5 = vector[k + (N_x - 2)]
        except IndexError:
            v5 = 0

        # Check for 'discontinuities'.
        # These arise when k or k+1 is a multiple of (N_x - 2).
        if (k + 1) % (N_x - 2) == 0:
            v4 = 0

        if k % (N_x - 2) == 0:
            v2 = 0

        # Get the value of nu_ij.
        nu_ij = 2 * (eta + mu) + potential(selection, parameters, k, grid_x, grid_y, N_x)

        # Store value
        vector_result[k] = - mu*v1 - eta*v2 + nu_ij*v3 - eta*v4 - mu*v5

    return vector_result


'''
Function: lanczos
Performs the Lanczos algorithm.
Parameters:
    num_lanczos - the number of steps to be performed by the algorithm.
    selection - selection of potential.
    parameters - array containing parameters related to the potential.
    eta - eta parameter for the corresponding potential.
    mu - mu parameter for the corresponding potential.
    N_x - number of grind points in the x-direction.
    N_y - numbeer of grid points in the y-direction.
    grid_x - array for the grid in the x-direction.
    grid_y - array for the grid in the y-direction.
'''
def lanczos(num_lanczos, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y):
    # Set arrays to store the alpha and beta coefficients.
    alpha = np.zeros(0)
    beta = np.zeros(0)

    # Random vector |q_1>. Normalized.
    dim = (N_x - 2)*(N_y - 2)
    q_current = normalize(10*np.random.rand(dim))

    # Initialize |q_0> and beta_1 as zero.
    q_prev = np.zeros(dim)
    beta = np.append(beta, 0)

    # Loop. Index j calculates q_(j+1), alpha_(j) and beta_(j+1).
    for j in range(1, num_lanczos + 1):
        # Let u be the product of the Hamiltonian matrix and |q_j>.
        # (u = H|q_j>)
        u = hamiltonian_product(q_current, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y)

        # alpha_(j) is q_(j) dot u.
        # (alpha_(j) = <q_(j)|H|q_(j)>)
        current_alpha = np.dot(q_current, u)
        alpha = np.append(alpha, current_alpha)

        # w_(j+1) is u - alpha_(j)*q_(j) - beta_(j)*q_(j-1).
        # (|w_(j+1)> = H|q_(j)> - alpha_(j)|q_(j)> - beta_(j)|q_(j-1)>)
        w = u - alpha[j-1]*q_current - beta[j-1]*q_prev

        # Compute beta_(j+1) as the norm of w_(j+1).
        current_beta = np.linalg.norm(w)
        if current_beta == 0:
            print('Lanczos iteration came to an end due to null beta coefficient. Iterations: {}.'.format(j))
            break
        else:
            beta = np.append(beta, current_beta)

        # Assign q_(j) to q_(j-1) for the next iteration.
        q_prev = q_current

        # Set the next Lanczos vector.
        # (|q_(j+1)> = |w_(j+1)> / beta_(j+1))
        q_current = w / current_beta

    # Calculate last alpha.
    u = hamiltonian_product(q_current, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y)
    current_alpha = np.dot(q_current, u)
    alpha = np.append(alpha, current_alpha)

    # Once Lanczos algorithm has stopped, return alpha and beta as arrays.
    return alpha, beta


'''
Function: lanczos_s
Performs the Lanczos algorithm with selective re-orthogonalization.
Parameters:
    num_lanczos - the number of steps to be performed by the algorithm.
    selection - selection of potential.
    parameters - array containing parameters related to the potential.
    eta - eta parameter for the corresponding potential.
    mu - mu parameter for the corresponding potential.
    N_x - number of grind points in the x-direction.
    N_y - numbeer of grid points in the y-direction.
    grid_x - array for the grid in the x-direction.
    grid_y - array for the grid in the y-direction.
'''
def lanczos_s(num_lanczos, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y):
    # Set arrays to store the alpha and beta coefficients.
    alpha = np.zeros(0)
    beta = np.zeros(0)

    # Random vector |q_1>. Normalized.
    dim = (N_x - 2)*(N_y - 2)
    q_current = normalize(10*np.random.rand(dim))

    # Initialize |q_0> and beta_1 as zero.
    q_prev = np.zeros(dim)
    beta = np.append(beta, 0)

    # Store the Lanczos vector in the Q matrix.
    # The function atleast_2d let's us treat the vector as a row matrix.
    # The T transposes said matrix.
    Q = np.atleast_2d(q_current).T

    # Loop. Index j calculates q_(j+1), alpha_(j) and beta_(j+1).
    for j in range(1, num_lanczos + 1):
        # Let u be the product of the Hamiltonian matrix and |q_j>.
        # (u = H|q_j>)
        u = hamiltonian_product(q_current, selection, parameters, eta, mu, N_x, N_y, grid_x, grid_y)

        # alpha_(j) is q_(j) dot u.
        # (alpha_(j) = <q_(j)|H|q_(j)>)
        current_alpha = np.dot(q_current, u)
        alpha = np.append(alpha, current_alpha)

        # w_(j+1) is u - alpha_(j)*q_(j) - beta_(j)*q_(j-1).
        # (|w_(j+1)> = H|q_(j)> - alpha_(j)|q_(j)> - beta_(j)|q_(j-1)>)
        w = u - alpha[j-1]*q_current - beta[j-1]*q_prev

        # Selective orthogonalization step.
        w, ritz_values, ritz_vectors = selective_orthogonalization(alpha, beta, w, Q, j)

        # Compute beta_(j+1) as the norm of w_(j+1).
        current_beta = np.linalg.norm(w)
        if current_beta == 0:
            print('Lanczos iteration came to an end due to null beta coefficient. Iterations: {}.'.format(j))
            break
        else:
            beta = np.append(beta, current_beta)

        # Assign q_(j) to q_(j-1) for the next iteration.
        q_prev = q_current

        # Set the next Lanczos vector.
        # (|q_(j+1)> = |w_(j+1)> / beta_(j+1))
        q_current = w / current_beta

        # Add the Lanczos vector in the Q matrix.
        Q = np.append(Q, np.atleast_2d(q_current).T, axis = 1)


    # Once Lanczos algorithm has stopped, return Ritz values as array and ritz vectors as matrix.
    return ritz_values, ritz_vectors



'''
Function: selective_orthogonalization
Check for bound errors and orthogonalize accordingly.
Parameters:
    alpha - alpha coefficients of the tridiagional matrix T_k for current iteration.
    beta - beta coefficients of the tridiagonal matrix T_k for current iteration.
    vector - Lanczos vector to check for large components in the direction of the Ritz vectors.
    Q - Q matrix, containing Lanczos vectors as columns.
    k - index of the current iteration in the lanczos function.
'''
def selective_orthogonalization(alpha, beta, vector, Q, k):
    # Diagonalize the T_k matrix. Eigenvalues in ascending order.
    if np.size(alpha) == np.size(beta):
        ritz_values, v_eigenvectors = eigh_tridiagonal(alpha, beta[1:])
    else:
        k = k + 1
        ritz_values, v_eigenvectors = eigh_tridiagonal(alpha, beta)

    # Calculate Ritz vectors.
    ritz_vectors = np.matmul(Q, v_eigenvectors)

    # Calculate corresponding beta.
    next_beta = np.linalg.norm(vector)
    if next_beta == 0:
        return vector, ritz_values, ritz_vectors

    # Calculate square root machine epsilon.
    sqrt_machine_epsilon = np.sqrt(np.finfo(float).eps)

    # Calculate the absolutely largest eigenvalue of T_k.
    if np.abs(ritz_values[0]) > np.abs(ritz_values[k-1]):
        maxeig = np.abs(ritz_values[0])
    else:
        maxeig = np.abs(ritz_values[k-1])

    # Orthogonalization loop.
    for i in range(0, k):
        if next_beta*np.abs(v_eigenvectors[k-1, i]) <= sqrt_machine_epsilon*maxeig:
            vector = vector - (np.dot(ritz_vectors[:, i], vector))*vector

    return vector, ritz_values, ritz_vectors



'''
Function: normalize
Normalize an array
Parameters:
    - vector - the vector to normalize
'''
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
       return vector
    return vector / norm


'''
Function: lanczos_direct.
Apply the lanczos algorithm to a Hermitian matrix.
Parameters:
    - matrix - a Hermitian matrix.
    - dim_matrix - dimension of the matrix.
    - num_lanczos - the number of steps to be performed by the algorithm
'''
def lanczos_direct(matrix, dim_matrix, num_lanczos):
    # Arrays to store the alpha and beta coefficients.
    alpha = np.zeros(num_lanczos)
    beta = np.zeros(num_lanczos - 1)

    # Random vector. Normalized.
    q = normalize(10*np.random.rand(dim_matrix))

    # Let u be the product of 'matrix' and 'q'.
    u = matrix.dot(q)

    # Let alpha[0] (alpha_1) be 'q' dot 'u'
    alpha[0] = np.dot(q, u)

    # Compute w_1:
    w = u - alpha[0]*q

    # Loop for the rest.
    for j in range(1, num_lanczos):
        # Compute beta_j as the norm of w_(j-1)
        beta[j-1] = np.linalg.norm(w)

        # q_j is w_(j-1) / beta_j
        qj = w / beta[j-1]

        # u_j is the product of 'matrix' and q_j
        u = matrix.dot(qj)

        # alpha_j is q_j dot u_j
        alpha[j] = np.dot(qj, u)

        # w_j is u_j - alpha_j*q_j - beta_j*q_(j-1)
        w = u - alpha[j]*qj - beta[j-1]*q

        # Assign q_j to q_(j-1) for the next loop.
        q = qj

    # Now, return alpha and beta as arrays.
    return alpha, beta


'''
Function: lanczos_direct_s.
Apply the lanczos algorithm to a Hermitian matrix with selective orthogonalization.
Parameters:
    - matrix - a Hermitian matrix.
    - dim_matrix - dimension of the matrix.
    - num_lanczos - the number of steps to be performed by the algorithm
'''
def lanczos_direct_s(matrix, dim_matrix, num_lanczos):
    # Arrays to store the alpha and beta coefficients.
    #alpha = np.zeros(num_lanczos)
    #beta = np.zeros(num_lanczos - 1)
    alpha = np.zeros(0)
    beta = np.zeros(0)

    # Random vector. Normalized.
    q = normalize(10*np.random.rand(dim_matrix))

    # Store the Lanczos vector in the Q matrix.
    # The function atleast_2d let's us treat the vector as a row matrix.
    # The T transposes said matrix.
    Q = np.atleast_2d(q).T

    # Let u be the product of 'matrix' and 'q'.
    u = matrix.dot(q)

    # Let alpha[0] (alpha_1) be 'q' dot 'u'
    #alpha[0] = np.dot(q, u)
    current_alpha = np.dot(q, u)
    alpha = np.append(alpha, current_alpha)

    # Compute w_1:
    w = u - alpha[0]*q

    # Loop for the rest.
    for j in range(1, num_lanczos):
        # Compute beta_j as the norm of w_(j-1)
        #beta[j-1] = np.linalg.norm(w)
        current_beta = np.linalg.norm(w)
        if current_beta == 0:
            print('Lanczos iteration came to an end due to null beta coefficient. Iterations: {}.'.format(j))
            return ritz_values, ritz_vectors
        else:
            beta = np.append(beta, current_beta)

        # q_j is w_(j-1) / beta_j
        qj = w / beta[j-1]

        # Add the Lanczos vector in the Q matrix.
        Q = np.append(Q, np.atleast_2d(qj).T, axis = 1)

        # u_j is the product of 'matrix' and q_j
        u = matrix.dot(qj)

        # alpha_j is q_j dot u_j
        #alpha[j] = np.dot(qj, u)
        current_alpha = np.dot(qj, u)
        alpha = np.append(alpha, current_alpha)

        # w_j is u_j - alpha_j*q_j - beta_j*q_(j-1)
        w = u - alpha[j]*qj - beta[j-1]*q

        # Selective orthogonalization step.
        w, ritz_values, ritz_vectors = selective_orthogonalization(alpha, beta, w, Q, j)

        # Assign q_j to q_(j-1) for the next loop.
        q = qj

    # Now, return ritz values and ritz vectors as arrays.
    return ritz_values, ritz_vectors
