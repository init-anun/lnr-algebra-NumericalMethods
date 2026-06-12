'''
    Using Gaussian Eliminatin, solve the following system of linear equations:
        2x - 3y - z = 1
        3x + 2y - 5z = -9
        2x + 4y -1z = -5
'''

import numpy as np

class Solution:

    def gaussElimination(self, A, b):

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
        
        for i in range(n - 1, -1, -1):