import math
import numpy as np

# Use the following script to diagonalize a matrix and get it's eigenvalues and normalized eigenvectors.

print('This script is used to get eigenvalues and eigenvectors of a real, symmetric matrix.')

# Get the dimension of the square matrix.

n = int(input('Input the dimension of the square matrix: '))
print(' ')

# Create a square matrix of size 'n' with zeroes as it's entries.

a = np.zeros((n, n))

# Now to input the elements of the matrix. Since it's symmetric, a(ij) = a(ji).
# The 'i' counter is for rows. For each row, the for loop runs through each possible column,
# excluding entries below the main diagonal. Then it assigns both a(ij) and a(ji) the same value.

i = 0

while i < n:
    for j in range (i, n):
            a[i, j] = input('Input entry in row number ' + str(i + 1) + ' and column number ' + str(j + 1) + ': ')
            a[j, i] = a[i, j]
            print(' ')
    i += 1

# To print matrix:
#print(a)

# Now to get the eigenvalues and eigenvectors. The function linalg.eig returns an eigenvalue array 'w' and a normalized eigenvalue
# matrix 'v' so that the column v[;, i] is the eigenvector corresponding to the eigenvalue w[i].

w, v = np.linalg.eig(a)

# Print returned arrays.

print('Eigenvalues: ')
print(w)
print(' ')
print('Normalized eigenvectors arranged as columns: ')
print(v)
