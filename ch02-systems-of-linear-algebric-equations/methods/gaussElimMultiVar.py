import numpy as np


def gaussElimination(self, A, b)-> np.ndarray:

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        
        # form augmented matrix
        augmented_matrix = np.hstack([A, b])

        print("Augmented Matrix:\n", augmented_matrix)

        n = len(augmented_matrix)
        m = len(augmented_matrix[0])


        # forward elimination
        for i in range(0, n-1):
            for j in range(i + 1, n):
                factor = augmented_matrix[j][i] / augmented_matrix[i][i]

                for k in range(i, m):
                    augmented_matrix[j][k] = augmented_matrix[j][k] - factor * augmented_matrix[i][k]
        

        print("Upper Triangular Matrix:\n", augmented_matrix)        

        # backward substitution
        x = np.zeros((n, m-n))

        for col in range(n, m):
            x[n-1][col-n] = augmented_matrix[n-1][col] / augmented_matrix[n-1][n-1]

            for j in range(n-2, -1, -1):
                sum = 0
                for i in range(j + 1, n):
                    sum += augmented_matrix[j][i] * x[i][col-n]
                x[j][col-n] = (augmented_matrix[j][col] - sum) / augmented_matrix[j][j]

        return x
