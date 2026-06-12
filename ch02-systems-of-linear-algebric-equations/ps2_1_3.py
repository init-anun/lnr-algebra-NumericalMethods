import numpy as np
from fractions import Fraction

class Solution:
    def method(self,L, U, b):
        L = np.array(L, dtype=float)
        U = np.array(U, dtype=float)
        b = np.array(b, dtype=float)

        A = L @ U
        x = np.linalg.solve(A, b)
        return x.tolist()   

# Example usage:

# Exact fractions
L = [
    [1, 0, 0],
    [Fraction(3,2), 1, 0],
    [Fraction(1,2), Fraction(11,13), 1]
]

U = [
    [2, -3, -1],
    [0, Fraction(13,2), Fraction(-7,2)],
    [0, 0, Fraction(32,13)]
]


b = [1, -1, 2]


solution = Solution()
result = solution.method(L, U, b)
print(result)
