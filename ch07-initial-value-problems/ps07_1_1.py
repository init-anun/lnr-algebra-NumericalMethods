'''
    

'''
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """Define the ODE: y' = x^2 - 4y"""
    return x**2 - 4*y

def analytical_solution(x):
    """Exact analytical solution"""
    return (x**2)/4 - x/8 + 1/32 + (31/32)*np.exp(-4*x)

def runge_kutta_2nd_order(x0, y0, h, n_steps):
    """
    Second-order Runge-Kutta method (Heun's method)
    
    Parameters:
    -----------
    x0 : float - initial x value
    y0 : float - initial y value
    h : float - step size
    n_steps : int - number of steps
    
    Returns:
    --------
    x_values : array of x values
    y_values : array of y values
    """
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    print("Runge-Kutta 2nd Order Method Results:")
    print("=" * 70)
    print(f"{'Step':<5} | {'x':<10} | {'y':<15} | {'k1':<15} | {'k2':<15}")
    print("-" * 70)
    
    for i in range(n_steps):
        # Calculate k1 and k2
        k1 = h * f(x, y)
        k2 = h * f(x + h, y + k1)
        
        # Update y
        y_new = y + 0.5 * (k1 + k2)
        x_new = x + h
        
        print(f"{i:<5} | {x:<10.6f} | {y:<15.8f} | {k1:<15.8f} | {k2:<15.8f}")
        
        # Store values
        x_values.append(x_new)
        y_values.append(y_new)
        
        # Update for next iteration
        x = x_new
        y = y_new
    
    print("-" * 70)
    print(f"{n_steps:<5} | {x:<10.6f} | {y:<15.8f}")
    print("=" * 70)
    
    return np.array(x_values), np.array(y_values)

# Main calculation
if __name__ == "__main__":
    # Initial conditions
    x0 = 0
    y0 = 1
    
    # We want y(0.03) using 2 steps
    x_final = 0.03
    n_steps = 2
    h = x_final / n_steps
    
    print(f"Initial conditions: y({x0}) = {y0}")
    print(f"Goal: Compute y({x_final}) using {n_steps} steps")
    print(f"Step size h = {h}")
    print()
    
    # Apply Runge-Kutta method
    x_rk, y_rk = runge_kutta_2nd_order(x0, y0, h, n_steps)
    
    # Calculate exact solution
    y_exact = analytical_solution(x_final)
    y_numerical = y_rk[-1]
    
    # Calculate error
    absolute_error = abs(y_numerical - y_exact)
    relative_error = (absolute_error / abs(y_exact)) * 100
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH ANALYTICAL SOLUTION")
    print("=" * 70)
    print(f"Numerical solution (RK2): y({x_final}) = {y_numerical:.8f}")
    print(f"Exact solution:           y({x_final}) = {y_exact:.8f}")
    print(f"Absolute error:           {absolute_error:.8f}")
    print(f"Relative error:           {relative_error:.6f}%")
    print("=" * 70)
    
    # Plot the results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Solution comparison
    plt.subplot(1, 2, 1)
    x_fine = np.linspace(0, x_final, 100)
    y_fine = analytical_solution(x_fine)
    plt.plot(x_fine, y_fine, 'b-', linewidth=2, label='Exact solution')
    plt.plot(x_rk, y_rk, 'ro-', markersize=8, linewidth=2, label='RK2 approximation')
    plt.xlabel('x', fontsize=11)
    plt.ylabel('y', fontsize=11)
    plt.title('Solution Comparison', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error visualization
    plt.subplot(1, 2, 2)
    error_at_points = [abs(y_rk[i] - analytical_solution(x_rk[i])) for i in range(len(x_rk))]
    plt.plot(x_rk, error_at_points, 'gs-', markersize=8, linewidth=2)
    plt.xlabel('x', fontsize=11)
    plt.ylabel('Absolute Error', fontsize=11)
    plt.title('Error at Each Step', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison at intermediate points
    print("\nDetailed Comparison at All Points:")
    print("=" * 70)
    print(f"{'x':<10} | {'RK2':<15} | {'Exact':<15} | {'Error':<15}")
    print("-" * 70)
    for i in range(len(x_rk)):
        exact_val = analytical_solution(x_rk[i])
        err = abs(y_rk[i] - exact_val)
        print(f"{x_rk[i]:<10.6f} | {y_rk[i]:<15.8f} | {exact_val:<15.8f} | {err:<15.2e}")
    print("=" * 70)
