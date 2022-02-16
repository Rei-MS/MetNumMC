import sys
import math
import numpy as np

'''
Function: neighbors
Sets up an array that contains the index of the neighbor to the right of the spin.
This means that the index of the neighbor of the i-th spin will be the i-th
element of the array.
Parameters:
    - num_spins - number of spins in the chain.
    - open_chain - if True, the chain will be open.
'''
def neighbors(num_spins, open_chain):
    if open_chain:
        neighbor = np.arange(num_spins - 1) + 1
        pass
    else:
        neighbor = np.arange(num_spins) + 1
        neighbor[num_spins - 1] = 0
    return neighbor


'''
Function: up_states
Sets up an array that contains the possible states with a particular number of up spins.
States are treated as binary numbers. 1 means up spin, while 0 means down spin.
Parameters:
    - num_spins - number of spins in the chain.
    - num_upspins - number of up spins in the chain.
    - num_hil - dimension of the Hilbert subspace.
'''
def up_states(num_spins, num_upspins, num_hil):
    # Set the array and a counter.
    states = np.zeros(num_hil)
    counter = 0

    # Fill the array.
    for state in range(0, 2**num_spins):
        # This if checks whether the number of 1's in the binary representation of 'state' is
        # exactly 'num_upsins'.
        if bin(state).count("1") == num_upspins:
            try:
                states[counter] = state
                counter += 1
            except IndexError:
                sys.exit('Script stopped. IndexError in up_states function.')

        else:
            pass


    # Check whether 'counter' == 'num_hil'
    if counter != num_hil:
        sys.exit('Script stopped. Error in up_states function. Number of states different from dimension of Hilbert subspace.')

    return states

'''
Function: hamiltonian
Sets up an array that contains the matrix elements of the Hamiltonian.
Parameters:
    - num_spins - number of spins in the chain.
    - num_hil - dimension of the Hilbert subspace.
    - neighbor - array containing the neighbors to the right of the spins.
    - states - array containing the possible states in the Hilbert subspace.
    - J_sign - +1 for ferromagnetic, -1 for antiferromagnetic.
    - open_chain - if True, the chain will be open.
'''
def hamiltonian(num_spins, num_hil, neighbor, states, J_sign, open_chain):
    hamiltonian = np.zeros((num_hil, num_hil))

    # If the chain is open, the last spin has no neighbor to the right.
    # To avoid rewriting the loop this statement is added.
    if open_chain:
        # Reduce the number of spins by 1.
        num_spins -= 1


    # 'state' are the indexes of the states of the Hilbert subspace.
    for state in range(0, num_hil):
        # For clarification purposes, get the state of the current iteration
        current_state = int(states[state])
        # 'spin' are the spins of the chain. These are the spin that have a neighbor to the right.
        for spin in range(0, num_spins):
            try:
                '''
                The expression (x & (1<<n)) is:
                    - True if the n-th bit of x is set (1)
                    - False if the n-th bit of x is not set (0)
                '''

                # Fill the parts corresponding to (S+)_i(S-)_j and (Sz)_i(Sz)_j.
                # This checks whether the 'spin'-th digit of 'state' is a 0 (i-th spin is down)
                # and if the 'neighbor(spin)'-th digit of 'state' is a 1. (j-th spin is up)
                # If so, evaluate and add the corresponding matrix element to the hamiltonian matrix.
                if ((not(current_state & (1<<spin))) and (current_state & (1<<neighbor[spin]))):
                    # Now, find the corresponding 'other_state' so that <other_state|(S+)_i(S-)_j|state> is non zero.
                    # First, calculate 'other_state'.
                    # For this, remember the 'other_state' has i-th spin up and j-th spin down.
                    # So, we get the state |state> first, and then flip those two spins.
                    other_state = current_state + 2**(spin) - 2**(neighbor[spin])

                    # Then, we get the index of 'other_state' in the 'states' array.
                    other_state_index = np.where(states == other_state)[0][0]

                    # Finally, we evaluate and add the matrix element corresponding to (S+)_i(S-)_j.
                    hamiltonian[state, other_state_index] += -J_sign/2

                    # Fill the part corresponding to (Sz)_i(Sz)_j.
                    # These matrix elements are non zero only for |other_state> = |state>
                    # So, we only care about 'state'.
                    # But if only one spin is down, <state|(Sz)_i(Sz)_j|state> will be negative.
                    # Here it's positive because there is a minus sign in the Hamiltonian.
                    hamiltonian[state, state] += J_sign/4

                # Fill the part corresponding to (S-)_i(S+)_j and (Sz)_i(Sz)_j.
                # This checks whether the 'spin'-th digit of 'state' is a 1 (i-th spin is up)
                # and if the 'neighbor(spin)'-th digit of 'state' is a 0. (j-th spin is down)
                # If so, evaluate and add the corresponding matrix element to the hamiltonian matrix.
                elif ((current_state & (1<<spin)) and (not(current_state & (1<<neighbor[spin])))):
                    # Now, find the corresponding 'other_state' so that <other_state|(S-)_i(S+)_j|state> is non zero.
                    # First, calculate 'other_state'.
                    # For this, remember the 'other_state' has i-th spin down and j-th spin up.
                    # So, we get the state |state> first, and then flip those two spins.
                    other_state = current_state - 2**(spin) + 2**(neighbor[spin])

                    # Then, we get the index of 'other_state' in the 'states' array.
                    other_state_index = np.where(states == other_state)[0][0]

                    # Finally, we evaluate and add the corresponding matrix element.
                    hamiltonian[state, other_state_index] += -J_sign/2

                    # Fill the part corresponding to (Sz)_i(Sz)_j.
                    # These matrix elements are non zero only for |other_state> = |state>
                    # So, we only care about 'state'.
                    # But if only one spin is down, <state|(Sz)_i(Sz)_j|state> will be negative.
                    # Here it's positive because there is a minus sign in the Hamiltonian.
                    hamiltonian[state, state] += J_sign/4

                # The only case remaining is where both neighboring spins are up, or both are down.
                # Fill the part corresponding to (Sz)_i(Sz)_j.
                # Again, <state|(Sz)_i(Sz)_j|state> will be positive.
                # Here it's negative because there is a minus sign in the Hamiltonian.
                else:
                    hamiltonian[state, state] += -J_sign/4
            except IndexError:
                print(state)
                print(spin)
                print(current_state)
                print(other_state)
                print(states)
                sys.exit('Script stopped. IndexError in hamiltonian function.')
    return hamiltonian


'''
Function: lanczos
Apply the lanczos algorithm to a Hermitian matrix.
Parameters:
    - matrix - a Hermitian matrix.
    - dim_matrix - dimension of the matrix.
    - num_lanczos - the number of steps to be performed by the algorithm
'''
def lanczos(matrix, dim_matrix, num_lanczos):
    # Arrays to store the alpha and beta coefficients.
    alpha = np.zeros(num_lanczos)
    beta = np.zeros(num_lanczos - 1)

    # Random vector. Normalized.
    v = normalize(10*np.random.rand(dim_matrix))

    # Let u be the product of 'matrix' and 'v'.
    u = np.matmul(matrix, v)

    # Let alpha[0] (alpha_1) be 'v' dot 'u'
    alpha[0] = np.dot(v, u)

    # Compute w_1:
    w = u - alpha[0]*v

    # Loop for the rest.
    for j in range(1, num_lanczos):
        # Compute beta_j as the norm of w_(j-1)
        beta[j-1] = np.linalg.norm(w)

        # v_j is w_(j-1) / beta_j
        vj = w / beta[j-1]

        # u_j is the product of 'matrix' and v_j
        u = np.matmul(matrix, vj)

        # alpha_j is v_j dot u_j
        alpha[j] = np.dot(vj, u)

        # w_j is u_j - alpha_j*v_j - beta_j*v_(j-1)
        w = u - alpha[j]*vj - beta[j-1]*v

        # Assign v_j to v_(j-1) for the next loop.
        v = vj

    # Now, return alpha and beta as arrays.
    return alpha, beta



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
