'''


'''


import numpy as np
import scipy.linalg

def main():
    # Define the matrices A and B
    A = np.array([
        [7, 3, 1],
        [3, 9, 6],
        [1, 6, 8]
    ], dtype=float)

    B = np.array([
        [4, 0, 0],
        [0, 9, 0],
        [0, 0, 4]
    ], dtype=float)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # 1. Calculate B^(1/2) and B^(-1/2)
    # Since B is diagonal, we just take the sqrt of the diagonal elements
    sqrt_B_diag = np.sqrt(np.diag(B))
    inv_sqrt_B_diag = 1.0 / sqrt_B_diag
    
    # Create diagonal matrices for the transformation
    # B_inv_sqrt = diag(1/2, 1/3, 1/2)
    B_inv_sqrt = np.diag(inv_sqrt_B_diag)
    
    # 2. Calculate H = B^(-1/2) * A * B^(-1/2)
    H = B_inv_sqrt @ A @ B_inv_sqrt
    
    print("\n" + "="*30)
    print("Standard Form Matrix H:")
    print("="*30)
    print(H)
    
    # 3. Define the relationship between x and z
    # z = B^(1/2) * x
    print("\n" + "="*30)
    print("Relationship between x and z:")
    print("="*30)
    print("z = B^(1/2) * x")
    print(f"z1 = {sqrt_B_diag[0]} * x1")
    print(f"z2 = {sqrt_B_diag[1]} * x2")
    print(f"z3 = {sqrt_B_diag[2]} * x3")
    
    # 4. Verification (Optional but recommended)
    # Calculate eigenvalues of the original generalized problem
    eig_vals_gen, eig_vecs_gen = scipy.linalg.eig(A, B)
    
    # Calculate eigenvalues of the new standard problem
    eig_vals_std, eig_vecs_std = np.linalg.eig(H)
    
    print("\n" + "="*30)
    print("Verification of Eigenvalues:")
    print("="*30)
    print(f"Eigenvalues from generalized problem (Ax = lBx): {np.sort(eig_vals_gen.real)}")
    print(f"Eigenvalues from standard problem (Hz = lz):     {np.sort(eig_vals_std)}")
    
    # Check if eigenvectors satisfy the relationship x = B^(-1/2) * z
    # Let's check the first eigenvector
    x_1 = eig_vecs_gen[:, 0].real
    z_1 = eig_vecs_std[:, 0].real
    
    # Normalize for comparison (eigenvectors are unique up to a scalar)
    x_1 = x_1 / np.linalg.norm(x_1)
    z_1_transformed = B_inv_sqrt @ z_1
    z_1_transformed = z_1_transformed / np.linalg.norm(z_1_transformed)
    
    # Check if they are close (allowing for sign flip)
    match = np.allclose(x_1, z_1_transformed) or np.allclose(x_1, -z_1_transformed)
    print(f"\nEigenvector relationship check (x = B^(-1/2)z): {'Passed' if match else 'Failed'}")

if __name__ == "__main__":
    main()
