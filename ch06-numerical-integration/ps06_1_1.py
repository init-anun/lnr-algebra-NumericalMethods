'''
    

'''

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """The integrand function: ln(1 + tan(x))"""
    return np.log(1 + np.tan(x))

def recursive_trapezoidal(a, b, max_iter=10, tol=1e-10):
    """
    Recursive trapezoidal rule for numerical integration.
    
    Parameters:
    -----------
    a, b : float
        Integration limits
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    T : list
        List of approximations at each iteration
    """
    T = []
    h = b - a
    
    # First approximation (n=1)
    T0 = (h / 2) * (f(a) + f(b))
    T.append(T0)
    
    print("Recursive Trapezoidal Rule Results:")
    print("=" * 70)
    print(f"{'Iter':<5} | {'n':<5} | {'h':<12} | {'Approximation':<15} | {'Error':<15}")
    print("-" * 70)
    print(f"{0:<5} | {1:<5} | {h:<12.6f} | {T0:<15.10f} | {'N/A':<15}")
    
    for k in range(1, max_iter):
        n = 2**k  # Number of intervals
        h = (b - a) / n
        
        # Sum of function values at new points (odd indices)
        sum_new = 0
        for i in range(1, n, 2):
            x_new = a + i * h
            sum_new += f(x_new)
        
        # Recursive formula: T_{2n} = T_n/2 + h * sum(new points)
        T_new = T[-1] / 2 + h * sum_new
        T.append(T_new)
        
        error = abs(T_new - T[-2]) if k > 0 else float('inf')
        print(f"{k:<5} | {n:<5} | {h:<12.6f} | {T_new:<15.10f} | {error:<15.2e}")
        
        if error < tol:
            print("-" * 70)
            print(f"Converged after {k} iterations!")
            break
    
    return T

def verify_symmetry():
    """Verify the symmetry property f(x) + f(π/4 - x) = ln(2)"""
    print("\n" + "=" * 70)
    print("Verifying Symmetry Property: f(x) + f(π/4 - x) = ln(2)")
    print("=" * 70)
    
    x_values = np.linspace(0, np.pi/4, 6)
    print(f"{'x':<15} | {'f(x)':<15} | {'f(π/4-x)':<15} | {'Sum':<15} | {'ln(2)':<15}")
    print("-" * 70)
    
    for x in x_values:
        fx = f(x)
        f_complement = f(np.pi/4 - x)
        total = fx + f_complement
        print(f"{x:<15.6f} | {fx:<15.10f} | {f_complement:<15.10f} | {total:<15.10f} | {np.log(2):<15.10f}")

# Main execution
if __name__ == "__main__":
    a = 0
    b = np.pi / 4
    
    # Run recursive trapezoidal rule
    T = recursive_trapezoidal(a, b, max_iter=6)
    
    # Exact value
    exact = (np.pi / 8) * np.log(2)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Exact value:           {exact:.10f}")
    print(f"Numerical (n=64):      {T[-1]:.10f}")
    print(f"Absolute error:        {abs(T[-1] - exact):.2e}")
    print(f"Relative error:        {abs(T[-1] - exact)/exact:.2e}")
    
    # Verify symmetry
    verify_symmetry()
    
    # Plot convergence
    print("\n" + "=" * 70)
    print("CONVERGENCE PLOT")
    print("=" * 70)
    
    n_values = [2**k for k in range(len(T))]
    errors = [abs(t - exact) for t in T]
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Error vs n
    plt.subplot(1, 2, 1)
    plt.semilogy(n_values, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of intervals (n)', fontsize=11)
    plt.ylabel('Absolute Error', fontsize=11)
    plt.title('Convergence of Recursive Trapezoidal Rule', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(n_values)
    
    # Plot 2: Approximation vs exact
    plt.subplot(1, 2, 2)
    plt.axhline(y=exact, color='r', linestyle='--', linewidth=2, label=f'Exact = {exact:.6f}')
    plt.plot(n_values, T, 'bo-', linewidth=2, markersize=8, label='Numerical')
    plt.xlabel('Number of intervals (n)', fontsize=11)
    plt.ylabel('Approximation', fontsize=11)
    plt.title('Approach to Exact Value', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(n_values)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate why trapezoidal rule is exact
    print("\n" + "=" * 70)
    print("WHY IS THE TRAPEZOIDAL RULE EXACT?")
    print("=" * 70)
    print("""
The trapezoidal rule with n intervals is:
    T_n = h/2 [f(0) + 2f(h) + 2f(2h) + ... + 2f((n-1)h) + f(π/4)]

Due to the symmetry f(x) + f(π/4-x) = ln(2), we can pair terms:
    f(0) + f(π/4) = ln(2)
    f(h) + f(π/4-h) = ln(2)
    f(2h) + f(π/4-2h) = ln(2)
    ...

When we sum all pairs and multiply by h/2, we get:
    T_n = (π/4)/n * n/2 * ln(2) = π/8 * ln(2)

This is EXACTLY the analytical value, independent of n!
Therefore, the error is zero (up to machine precision).
    """)
