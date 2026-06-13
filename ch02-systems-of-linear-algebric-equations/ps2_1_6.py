'''
    problen set 3
    question 1.6
    solve the equations Ax = b by Gauss Eliminationm, where
    A = [
            [0, 0, 2, 1, 2],
            [0, 1, 0, 2, -1],
            [1, 2, 0, -2, 0],
            [0, 0, 0, -1, 1],
            [0, 1, -1, 1, -1]
        ]

    b = [1, 1, -4, -2, -1]

'''
import sys
from methods.gaussElim import gaussElimination

if __name__ == "__main__":

    A = [
            [0, 0, 2, 1, 2],
            [0, 1, 0, 2, -1],
            [1, 2, 0, -2, 0],
            [0, 0, 0, -1, 1],
            [0, 1, -1, 1, -1]
        ]

    b = [1, 1, -4, -2, -1]

    # arrange the equations in such a way that the first equation has a non-zero coefficient for x[1][1]
    A[0], A[2] = A[2], A[0]
    b[0], b[2] = b[2], b[0]
    
    
    
    
    
    
    
    
    
    solution = gaussElimination(A,b)

    print("Solution: \n", solution)
