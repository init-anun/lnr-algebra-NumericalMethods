'''
    Given the values of f(x) at the points x, x − h1, and x + h2, where h1 != h2, 
    determine the finite difference approximation for f (x). What is the order of the
    truncation error?


'''



import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Symbolic Derivation using Taylor Series
# =============================================================================
print("="*70)
print("SYMBOLIC DERIVATION OF FINITE DIFFERENCE APPROXIMATION")
print("="*70)

# Define symbols
x, h1, h2 = sp.symbols('x h1 h2', real=True, positive=True)
f = sp.Function('f')

# Taylor expansions around x
f_x = f(x)
f_x_minus_h1 = f(x - h1)
f_x_plus_h2 = f(x + h2)

# Expand using Taylor series up to 3rd order
# f(x - h1) = f(x) - h1*f'(x) + h1^2/2*f''(x) - h1^3/6*f'''(x) + O(h^4)
# f(x + h2) = f(x) + h2*f'(x) + h2^2/2*f''(x) + h2^3/6*f'''(x) + O(h^4)

# Define unknown coefficients A, B, C for: f'(x) ≈ A*f(x) + B*f(x-h1) + C*f(x+h2)
A, B, C = sp.symbols('A B C')

# Taylor series expansions (keeping terms up to 2nd derivative for solving)
# We'll use f0, f1, f2 to represent f(x), f'(x), f''(x)
f0, f1, f2, f3 = sp.symbols('f0 f1 f2 f3')

# Taylor expansions in terms of derivatives at x
taylor_minus = f0 - h1*f1 + (h1**2/2)*f2 - (h1**3/6)*f3
taylor_plus = f0 + h2*f1 + (h2**2/2)*f2 + (h2**3/6)*f3

# Linear combination: A*f(x) + B*f(x-h1) + C*f(x+h2)
linear_combo = A*f0 + B*taylor_minus + C*taylor_plus

# Collect coefficients
coeff_f0 = sp.simplify(sp.diff(linear_combo, f0))
coeff_f1 = sp.simplify(sp.diff(linear_combo, f1))
coeff_f2 = sp.simplify(sp.diff(linear_combo, f2))

print("\nSetting up system of equations:")
print(f"1. Coefficient of f(x):    {coeff_f0} = 0")
print(f"2. Coefficient of f'(x):   {coeff_f1} = 1")
print(f"3. Coefficient of f''(x):  {coeff_f2} = 0")

# Solve the system
solution = sp.solve([coeff_f0, coeff_f1 - 1, coeff_f2], [A, B, C])
print("\nSolution for coefficients:")
print(f"A = {sp.simplify(solution[A])}")
print(f"B = {sp.simplify(solution[B])}")
print(f"C = {sp.simplify(solution[C])}")

# Simplify the coefficients
A_sol = sp.simplify(solution[A])
B_sol = sp.simplify(solution[B])
C_sol = sp.simplify(solution[C])

print("\n" + "="*70)
print("FINITE DIFFERENCE FORMULA:")
print("="*70)
print(f"f'(x) ≈ {A_sol}·f(x) + ({B_sol})·f(x-h₁) + ({C_sol})·f(x+h₂)")

# Write with common denominator
common_denom = h1 * h2 * (h1 + h2)
print(f"\nOr equivalently:")
print(f"f'(x) ≈ [h₁²·f(x+h₂) + (h₂²-h₁²)·f(x) - h₂²·f(x-h₁)] / [h₁·h₂·(h₁+h₂)]")

# =============================================================================
# Part 2: Truncation Error Analysis
# =============================================================================
print("\n" + "="*70)
print("TRUNCATION ERROR ANALYSIS")
print("="*70)

# Calculate the coefficient of f'''(x) in the error term
coeff_f3 = sp.simplify(A*0 + B*(-h1**3/6) + C*(h2**3/6))
error_coeff = sp.simplify(coeff_f3.subs(solution))

print(f"\nCoefficient of f'''(x) in truncation error: {error_coeff}")
print(f"Simplified: {sp.factor(error_coeff)}")

# If h1 = h2 = h, what happens?
error_equal_h = sp.simplify(error_coeff.subs({h1: sp.Symbol('h'), h2: sp.Symbol('h')}))
print(f"\nWhen h₁ = h₂ = h: Error coefficient = {error_equal_h}")

# Order of accuracy
print("\n" + "="*70)
print("ORDER OF ACCURACY:")
print("="*70)
print("Since the error term is proportional to h₁·h₂,")
print("if h₁, h₂ = O(h), then the error is O(h²)")
print("Therefore, this is a SECOND-ORDER accurate approximation.")

# =============================================================================
# Part 3: Numerical Verification
# =============================================================================
print("\n" + "="*70)
print("NUMERICAL VERIFICATION")
print("="*70)

def finite_diff_approx(f, x, h1, h2):
    """
    Compute the finite difference approximation for f'(x)
    using points x, x-h1, and x+h2
    """
    A = (h2 - h1) / (h1 * h2)
    B = -h2 / (h1 * (h1 + h2))
    C = h1 / (h2 * (h1 + h2))
    
    return A * f(x) + B * f(x - h1) + C * f(x + h2)

# Test function: f(x) = sin(x), f'(x) = cos(x)
def test_func(x):
    return np.sin(x)

def exact_derivative(x):
    return np.cos(x)

# Test at x = 1.0
x_test = 1.0
exact = exact_derivative(x_test)

print(f"\nTest function: f(x) = sin(x)")
print(f"Test point: x = {x_test}")
print(f"Exact derivative: f'({x_test}) = cos({x_test}) = {exact:.10f}")

# Test with different step sizes
h_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]
errors = []
ratios = []

print("\n" + "-"*70)
print(f"{'h':<10} {'h1':<10} {'h2':<10} {'Approximation':<15} {'Error':<15} {'Ratio':<10}")
print("-"*70)

prev_error = None
for h in h_values:
    h1_val = h
    h2_val = 1.5 * h  # Use different h1 and h2
    
    approx = finite_diff_approx(test_func, x_test, h1_val, h2_val)
    error = abs(approx - exact)
    errors.append(error)
    
    if prev_error is not None:
        ratio = prev_error / error
        ratios.append(ratio)
        print(f"{h:<10.5f} {h1_val:<10.5f} {h2_val:<10.5f} {approx:<15.10f} {error:<15.2e} {ratio:<10.2f}")
    else:
        print(f"{h:<10.5f} {h1_val:<10.5f} {h2_val:<10.5f} {approx:<15.10f} {error:<15.2e} {'--':<10}")
    
    prev_error = error

# Calculate observed order of convergence
if len(ratios) > 0:
    avg_ratio = np.mean(ratios)
    observed_order = np.log2(avg_ratio)
    print("-"*70)
    print(f"\nAverage error ratio (as h is halved): {avg_ratio:.2f}")
    print(f"Observed order of convergence: {observed_order:.2f}")
    print(f"Expected order of convergence: 2.00 (second-order)")

# =============================================================================
# Part 4: Visualization
# =============================================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Create a plot showing convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Error vs step size (log-log)
axes[0].loglog(h_values, errors, 'bo-', linewidth=2, markersize=8)
axes[0].loglog(h_values, [errors[0] * (h/h_values[0])**2 for h in h_values], 
               'r--', linewidth=2, label='O(h²) reference')
axes[0].set_xlabel('Step size h', fontsize=11)
axes[0].set_ylabel('Absolute Error', fontsize=11)
axes[0].set_title('Convergence of Finite Difference Approximation', fontsize=12)
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)

# Plot 2: Approximation quality
h_test = 0.5
x_range = np.linspace(-2, 2, 100)
exact_deriv = exact_derivative(x_range)
approx_deriv = [finite_diff_approx(test_func, x, h_test, 1.5*h_test) for x in x_range]

axes[1].plot(x_range, exact_deriv, 'b-', linewidth=2, label='Exact: cos(x)')
axes[1].plot(x_range, approx_deriv, 'r--', linewidth=2, label=f'Approximation (h={h_test})')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].axvline(0, color='black', linewidth=0.5)
axes[1].set_xlabel('x', fontsize=11)
axes[1].set_ylabel("f'(x)", fontsize=11)
axes[1].set_title('Approximation vs Exact Derivative', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# Part 5: Special Case - Central Difference
# =============================================================================
print("\n" + "="*70)
print("SPECIAL CASE: h₁ = h₂ = h (Central Difference)")
print("="*70)

h_sym = sp.Symbol('h', positive=True)
A_central = sp.simplify(A_sol.subs({h1: h_sym, h2: h_sym}))
B_central = sp.simplify(B_sol.subs({h1: h_sym, h2: h_sym}))
C_central = sp.simplify(C_sol.subs({h1: h_sym, h2: h_sym}))

print(f"\nWhen h₁ = h₂ = h:")
print(f"A = {A_central}")
print(f"B = {B_central}")
print(f"C = {C_central}")
print(f"\nFormula: f'(x) ≈ {C_central}·f(x+h) + {B_central}·f(x-h)")
print(f"Simplified: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)")
print("\nThis is the standard central difference formula!")

error_central = sp.simplify(error_coeff.subs({h1: h_sym, h2: h_sym}))
print(f"\nTruncation error: {error_central}·f'''(x) = h²/6·f'''(x)")