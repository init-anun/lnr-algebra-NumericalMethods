'''
    Using Gaussian Eliminatin, solve the following system of linear equations:
        2x - 3y - z = 3
        3x + 2y - 5z = -9
        2x + 4y -1z = -5
'''

import numpy as np

class Solution:

    def gaussElimination(self, A, b)-> np.ndarray:

        n = len(A)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        # form augmented matrix
        Ab = np.hstack([A, b.reshape(-1, 1)])

        print("Augmented Matrix:", Ab)

        # forward elimination
        for i in range(0, n-1):
            for j in range(i + 1, n):
                factor = Ab[j][i] / Ab[i][i]

                for k in range(i, n + 1):
                    Ab[j][k] = Ab[j][k] - factor * Ab[i][k]
        

        print("Upper Triangular Matrix:", Ab)        

        # backward substitution
        x = np.zeros(n)
        x[n-1] = Ab[n-1][n] / Ab[n-1][n-1]

        for j in range(n-2, -1, -1):
            sum = 0
            for i in range(j + 1, n):
                sum += Ab[j][i] * x[i]
            x[j] = (Ab[j][n] - sum) / Ab[j][j]

        return x

if __name__ == "__main__":
    A = [[2, -3, -1], [3, 2, -5], [2, 4, -1]]
    b = [3, -9, -5]

    solution = Solution()
    result = solution.gaussElimination(A, b)
    print("Solution:", result)