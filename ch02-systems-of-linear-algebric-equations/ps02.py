'''

'''

import numpy as np

def classify_matrix(name, matrix):
    det = np.linalg.det(matrix)
    
    # Heuristic for classification
    # We compare the absolute determinant to a threshold. 
    # For ill-conditioned, it's "close to zero" but not exactly zero.
    # Since floating point arithmetic isn't exact, we use a small epsilon.
    
    classification = ""
    if abs(det) < 1e-10:
        classification = "Singular"
    elif abs(det) < 1.0: # Arbitrary threshold relative to typical integer entries
        # A more robust check involves the condition number, but the problem asks to use determinant.
        # If det is small but non-zero, it's likely ill-conditioned.
        classification = "Ill-conditioned"
    else:
        classification = "Well-conditioned"
        
    print(f"Matrix ({name}):")
    print(matrix)
    print(f"Determinant: {det:.4f}")
    print(f"Classification: {classification}")
    print("-" * 30)

# Define the matrices
A_a = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])

A_b = np.array([
    [2.11, -0.80, 1.72],
    [-1.84, 3.03, 1.29],
    [-1.57, 5.25, 4.30]
])

A_c = np.array([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
])

A_d = np.array([
    [4, 3, -1],
    [7, -2, 3],
    [5, -18, 13]
])

# Evaluate and classify
print("Evaluating matrices based on determinant:\n")
classify_matrix('a', A_a)
classify_matrix('b', A_b)
classify_matrix('c', A_c)
classify_matrix('d', A_d)
