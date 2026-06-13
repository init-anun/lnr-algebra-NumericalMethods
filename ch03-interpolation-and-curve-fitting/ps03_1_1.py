'''
    Given the data points:

        x	-1.2	0.3	1.1
        y	-5.76	-5.61	-3.69

    determine y at x = 0 using 
    (a) Neville’s method and 
    (b) Lagrange’s method.

'''


import numpy as np

# Data points
x = np.array([-1.2, 0.3, 1.1])
y = np.array([-5.76, -5.61, -3.69])

x_target = 0.0

# --------------------------------------------------
# Neville's Method
# --------------------------------------------------
n = len(x)

Q = np.zeros((n, n))
Q[:, 0] = y

for i in range(1, n):
    for j in range(1, i + 1):
        Q[i, j] = (
            (x_target - x[i - j]) * Q[i, j - 1]
            - (x_target - x[i]) * Q[i - 1, j - 1]
        ) / (x[i] - x[i - j])

neville_result = Q[n - 1, n - 1]

print("Neville Table:")
print(Q)
print("\nNeville Result:")
print(f"y({x_target}) = {neville_result:.6f}")

# --------------------------------------------------
# Lagrange Method
# --------------------------------------------------
lagrange_result = 0.0

for i in range(n):
    term = y[i]

    for j in range(n):
        if i != j:
            term *= (x_target - x[j]) / (x[i] - x[j])

    lagrange_result += term

print("\nLagrange Result:")
print(f"y({x_target}) = {lagrange_result:.6f}")