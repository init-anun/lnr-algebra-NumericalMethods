import numpy as np


class SolutionToProblem1:
    def classify_matrix(self, A, name):
        A = np.array(A, dtype=float)
        det = np.linalg.det(A)
        cond = np.linalg.cond(A)
        
        print(f"--- Matrix {name} ---")
        print("Matrix:\n", A)
        print(f"Determinant     : {det:.10f}")
        print(f"Condition number: {cond:.2f}")
        
        if abs(det) < 1e-8:
            print("→ **Singular** (Determinant is zero)")
        elif cond > 1000:
            print("→ **Ill-conditioned**")
        else:
            print("→ **Well-conditioned**")
        print()


# (a)
A = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
]

# (b)
B = [
    [2.11, -0.80, 1.72],
    [-1.84, 3.03, 1.29],
    [-1.57, 5.25, 4.30]
]

# (c)
C = [
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
]

# (d) 
D = [
    [4, 3, -1],
    [7, -2, 3],
     [5, -18, 13]
]



solver = SolutionToProblem1()

solver.classify_matrix(A, "(a)")
solver.classify_matrix(B, "(b)")
solver.classify_matrix(C, "(c)")
solver.classify_matrix(D, "(d)")

