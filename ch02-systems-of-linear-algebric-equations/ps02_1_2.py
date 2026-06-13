import numpy as np

class Solution:
    def solver(self, L, U):
        L = np.array(L, dtype="float")
        U = np.array(U, dtype="float")

        A = L @ U
        det = np.linalg.det(A)

        print("A = LU ")
        print(A)
        print(f"|A| = {det:.10f}")


L_a = [
    [1, 0, 0],
    [1, 1, 0],
    [1, 5/3, 1]
]

U_a = [
    [1, 2, 4],
    [0, 3, 21],
    [0, 0, 0]
]


L_b = [
    [2, 0, 0],
    [-1, 1, 0],
    [1, -3, 1]
]

U_b = [
    [2, -1, 1],
    [0, 1, -3],
    [0, 0, 1]
]



# calling the solutions

solution = Solution()
print("2 (a)")
solution.solver(L_a, U_a)
print("2(b)")
solution.solver(L_b, U_b)




