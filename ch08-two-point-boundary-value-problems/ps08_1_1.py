'''

'''


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ode_system(x, Y):
    """
    Convert second-order ODE to system of first-order ODEs
    y'' + y' - y = 0
    
    Let y1 = y, y2 = y'
    Then: y1' = y2
          y2' = y - y2
    """
    y1, y2 = Y
    dy1dx = y2
    dy2dx = y1 - y2  # Rearranged from y'' + y' - y = 0
    return [dy1dx, dy2dx]

def solve_ivp_with_initial_derivative(y0_prime):
    """Solve the IVP with given y'(0)"""
    sol = solve_ivp(ode_system, [0, 1], [0, y0_prime], 
                    t_eval=np.linspace(0, 1, 100), method='RK45')
    return sol

def analytical_solution(x, y0_prime):
    """
    Analytical solution of y'' + y' - y = 0
    Characteristic equation: r^2 + r - 1 = 0
    Roots: r1 = (-1 + sqrt(5))/2, r2 = (-1 - sqrt(5))/2
    """
    r1 = (-1 + np.sqrt(5)) / 2
    r2 = (-1 - np.sqrt(5)) / 2
    
    # Using initial conditions y(0) = 0, y'(0) = alpha
    # y(x) = alpha/sqrt(5) * (e^(r1*x) - e^(r2*x))
    alpha = y0_prime
    return (alpha / np.sqrt(5)) * (np.exp(r1 * x) - np.exp(r2 * x))

# Main calculation
print("=" * 70)
print("SHOOTING METHOD FOR LINEAR ODE")
print("=" * 70)

# Given information
y0_prime_given = 1.0
y1_given = 0.741028
y1_target = 1.0

print(f"\nGiven:")
print(f"  y(0) = 0")
print(f"  y'(0) = {y0_prime_given}")
print(f"  y(1) = {y1_given}")
print(f"\nTarget:")
print(f"  y(1) = {y1_target}")
print(f"  Find: y'(0) = ?")

# Calculate the required y'(0) using linearity
alpha = y1_target / y1_given

print("\n" + "=" * 70)
print("CALCULATION USING LINEARITY")
print("=" * 70)
print(f"\nSince the ODE is linear:")
print(f"  y'(0)_new / y'(0)_old = y(1)_new / y(1)_old")
print(f"  y'(0)_new = y'(0)_old × (y(1)_new / y(1)_old)")
print(f"  y'(0)_new = {y0_prime_given} × ({y1_target} / {y1_given})")
print(f"  y'(0)_new = {alpha:.6f}")

# Verify numerically
print("\n" + "=" * 70)
print("NUMERICAL VERIFICATION")
print("=" * 70)

# Solve with original y'(0) = 1
sol1 = solve_ivp_with_initial_derivative(y0_prime_given)
y1_at_1_numerical = sol1.y[0, -1]

# Solve with new y'(0) = alpha
sol2 = solve_ivp_with_initial_derivative(alpha)
y2_at_1_numerical = sol2.y[0, -1]

print(f"\nWith y'(0) = {y0_prime_given}:")
print(f"  y(1) = {y1_at_1_numerical:.6f} (given: {y1_given})")

print(f"\nWith y'(0) = {alpha:.6f}:")
print(f"  y(1) = {y2_at_1_numerical:.6f} (target: {y1_target})")
print(f"  Error: {abs(y2_at_1_numerical - y1_target):.2e}")

# Verify with analytical solution
print("\n" + "=" * 70)
print("ANALYTICAL VERIFICATION")
print("=" * 70)

x_analytical = np.linspace(0, 1, 100)
y_analytical_1 = analytical_solution(x_analytical, y0_prime_given)
y_analytical_2 = analytical_solution(x_analytical, alpha)

y1_at_1_analytical = analytical_solution(1.0, y0_prime_given)
y2_at_1_analytical = analytical_solution(1.0, alpha)

print(f"\nAnalytical solution at x=1:")
print(f"  With y'(0) = {y0_prime_given}: y(1) = {y1_at_1_analytical:.6f}")
print(f"  With y'(0) = {alpha:.6f}: y(1) = {y2_at_1_analytical:.6f}")

# Plot results
plt.figure(figsize=(12, 5))

# Plot 1: Comparison of solutions
plt.subplot(1, 2, 1)
plt.plot(x_analytical, y_analytical_1, 'b-', linewidth=2, label=f"y'(0) = {y0_prime_given}")
plt.plot(x_analytical, y_analytical_2, 'r-', linewidth=2, label=f"y'(0) = {alpha:.4f}")
plt.plot(1, y1_given, 'bo', markersize=10, label='Given point (1, 0.741028)')
plt.plot(1, y1_target, 'r*', markersize=15, label='Target point (1, 1)')
plt.xlabel('x', fontsize=11)
plt.ylabel('y(x)', fontsize=11)
plt.title('Solutions with Different Initial Derivatives', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Phase portrait
plt.subplot(1, 2, 2)
plt.plot(sol1.y[0], sol1.y[1], 'b-', linewidth=2, label=f"y'(0) = {y0_prime_given}")
plt.plot(sol2.y[0], sol2.y[1], 'r-', linewidth=2, label=f"y'(0) = {alpha:.4f}")
plt.xlabel('y(x)', fontsize=11)
plt.ylabel("y'(x)", fontsize=11)
plt.title('Phase Portrait', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final answer
print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)
print(f"\nThe value of y'(0) that results in y(1) = 1 is:")
print(f"  y'(0) = {alpha:.6f}")
print(f"\nRounded to 5 decimal places: y'(0) = {alpha:.5f}")
print("=" * 70)
