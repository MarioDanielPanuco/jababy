import numpy as np

# Define adjacency matrix A and degree matrix D
A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 0]])

D = np.linalg.matrix_power(A, 2)
print(f'D = {D}\n===\n')
# Compute Laplacian matrix L
L = D - A

# Find eigenvalues and eigenvectors of L
eigvals, eigvecs = np.linalg.eig(L)

print(f'|v| = {len(eigvals)}\neigenvals: {eigvals}\n')
print(f'|vec| = {len(eigvecs)}\neigenvector: {eigvecs}\n')

# Number of zero eigenvalues corresponds to number of connected components in graph
tolerance = 1e-10  # Choose a suitable tolerance value
num_connected_components = np.sum(np.isclose(eigvals, 0, atol=tolerance))
print("Number of connected components:", num_connected_components)