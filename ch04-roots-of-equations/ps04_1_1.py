'''
    Use the Newton-Raphson method and a four-function calculator (+ − × ÷ operations only) 
    to compute 3√75, cube root of 75, with four significant figure accuracy

'''

'''
    


'''
# Newton-Raphson Method for cube root of 75

N = 75

# Initial guess
x = 4.0

print("Iteration\tApproximation")

for i in range(10):
    print(f"{i}\t\t{x:.10f}")

    x_new = (2 * x + N / (x * x)) / 3

    # Stop when 4 significant figure accuracy is reached
    if abs(x_new - x) < 0.00005:
        x = x_new
        break

    x = x_new

print("\nCube root of 75 =", round(x, 4))