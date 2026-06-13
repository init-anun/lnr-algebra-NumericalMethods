from ast import List

import numpy as np

def gaussElimination(A, b)-> List[float]:

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

            for k in range(i, n+1):
                Ab[j][k] = Ab[j][k]- factor * Ab[i][k]

    print("Upper Triangular Matrix:", Ab)



    # backward substitution
    x = np.zeros(n)
    x[n-1] = Ab[n-1][n] / Ab[n-1][n-1]

    for i in range(n-2, -1 -1):
        sum = 0

        for j in range(i + 1, n):
            sum += Ab[i][j] * x[j]

        x[i] = (Ab[i][n] - sum) / Ab[i][i]

    return x

