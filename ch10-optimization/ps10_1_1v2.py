import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the symbol
sigma = sp.symbols('sigma', real=True)

# Define the function psi (ignoring constant C as it doesn't affect the minimum location)
# psi = C * (27 - 18*sigma + 2*sigma**2) * exp(-sigma/3)
f = (27 - 18*sigma + 2*sigma**2) * sp.exp(-sigma/3)

# 1. Find the derivative
f_prime = sp.diff(f, sigma)

# 2. Solve for critical points (f' = 0)
critical_points = sp.solve(f_prime, sigma)

print("Critical points (exact):")
for point in critical_points:
    print(point)

print("\nCritical points (numerical):")
numerical_points = [point.evalf() for point in critical_points]
for point in numerical_points:
    print(point)

# 3. Verify minimum using Second Derivative Test
f_double_prime = sp.diff(f_prime, sigma)

print("\nVerifying nature of critical points:")
min_sigma = None
for point in critical_points:
    # Evaluate second derivative at the critical point
    val = f_double_prime.subs(sigma, point)
    if val > 0:
        print(f"sigma = {point} is a MINIMUM (f'' > 0)")
        min_sigma = point
    else:
        print(f"sigma = {point} is a MAXIMUM (f'' < 0)")

# 4. Plotting to visualize
sigma_vals = np.linspace(0, 15, 400)
# Convert sympy function to a numpy-compatible function
f_func = sp.lambdify(sigma, f, 'numpy')
psi_vals = f_func(sigma_vals)

plt.figure(figsize=(10, 6))
plt.plot(sigma_vals, psi_vals, label=r'$\psi(\sigma) \propto (27 - 18\sigma + 2\sigma^2)e^{-\sigma/3}$')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(float(min_sigma.evalf()), color='red', linestyle='--', label=f'Minimum at σ ≈ {min_sigma.evalf():.2f}')
plt.title('Hydrogen Atom Wave Function (Radial Part)')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$\psi$ (arbitrary units)')
plt.legend()
plt.grid(True)
plt.show()