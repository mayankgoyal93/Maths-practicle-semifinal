
import numpy as np

lst = [1, 2, 3, 4]
vector_1 = np.array(lst)        
print("Horizontal vector:",vector_1)



#creating matrix and finding the transpose


import numpy as np

r = int(input("Rows: "))
c = int(input("Cols: "))
print("Enter elements row by row (space separated):")
entries = []
for i in range(r):
    row = list(map(float, input(f"Row {i+1}: ").strip().split()))
    if len(row) != c:
        raise ValueError("Wrong number of columns")
    entries.append(row)

M = np.array(entries)
print("Matrix ", M)
print("Transpose", M.T)

#generate the matrix into echelon form and finds its rank

import numpy as np
rows = int(input("Enter the number of rows: "))
cols = int(input("Enter the number of columns: "))
print("Enter the entries, separated by spaces (row-wise):")
entries = list(map(int, input().split()))

# Step 2: Create and display the matrix
matrix = np.array(entries).reshape(rows, cols)
print("Matrix X is as follows:")
print(matrix)

# Step 3: Find the rank using numpy
rank = np.linalg.matrix_rank(matrix)
print("The Rank of the Matrix is:", rank)



#Find cofactors, determinant, adjoint and inverse of a matrix.
import numpy as np

# Step 1: Input matrix size and elements
n = int(input("Enter the size of your square matrix (for example, 3 for 3x3): "))
print("Enter the matrix elements row-wise, separated by spaces:")
elements = list(map(float, input().split()))
A = np.array(elements).reshape(n, n)
print("\nYour matrix:")
print(A)

# Step 2: Determinant
determinant = np.linalg.det(A)
print("\nDeterminant of the matrix:", determinant)

# Step 3: Inverse
if np.isclose(determinant, 0):
    print("\nMatrix is singular and does not have an inverse or cofactors/adjoint.")
else:
    inverse = np.linalg.inv(A)
    print("\nInverse of the matrix:")
    print(inverse)

    # Step 4: Cofactor matrix
    # The cofactor matrix is the transpose of the adjugate, which can be calculated as follows:
    # If A is invertible: cofactor = inv(A).T * det(A)
    cofactor = np.transpose(inverse) * determinant
    print("\nCofactor matrix:")
    print(cofactor)

    # Step 5: Adjoint matrix
    adjoint = np.transpose(cofactor)
    print("\nAdjoint matrix:")
    print(adjoint)
    
    
    
    
    #Solve a system of Homogeneous and non-homogeneous equations using Gauss elimination method.
    import numpy as np

# Step 1: Enter the size of your system
n = int(input("Enter the number of equations (and variables): "))

# Step 2: Enter the coefficient matrix A and the constant vector b
print("Enter the coefficients row-wise (each row separated by spaces):")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A)

print("Enter the constants (right side) as one line, separated by spaces:")
b = list(map(float, input().split()))
b = np.array(b)

# Step 3: Solve the system
try:
    x = np.linalg.solve(A, b)
    print("Solution (x vector) of the non-homogeneous system:")
    print(x)
except np.linalg.LinAlgError:
    print("System does not have a unique solution.")
    
    
    
    
    
    
    #Solve a system of Homogeneous equations using the Gauss Jordan method
    import numpy as np

# Step 1: Input the coefficient matrix
n = int(input("Enter number of variables: "))
print("Enter the coefficients for each row, separated by spaces:")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A)

# Step 2: Create zero vector for the constants (since the system is homogeneous)
b = np.zeros(n)

# Step 3: Form the augmented matrix
aug = np.hstack((A, b.reshape(-1,1)))

print("Starting Augmented Matrix:")
print(aug)

# Step 4: Gauss-Jordan Elimination to Reduced Row Echelon Form
def gauss_jordan(m):
    m = m.astype(float) # Work in float for division
    rows, cols = m.shape
    for i in range(rows):
        # Make the diagonal contain all 1's
        m[i] = m[i] / m[i,i] if m[i,i] != 0 else m[i]
        for j in range(rows):
            if j != i and m[j,i] != 0:
                m[j] = m[j] - m[j,i] * m[i]
    return m

rref = gauss_jordan(aug)

print("Matrix in Reduced Row Echelon Form ")
print(rref)
print("The solution set is given by the last column.")












#Generate basis of column space, null space, row space and left null space of a matrix space.
import numpy as np
from sympy import Matrix

# Input the matrix
print("Enter the matrix elements row-wise (space separated):")
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))
elements = list(map(float, input(f"Enter {rows * cols} elements: ").split()))
A = np.array(elements).reshape(rows, cols)

print("\nMatrix A:")
print(A)

# Convert to sympy matrix for space calculations
M = Matrix(A)

# Basis of column space (pivot columns of A)
pivot_columns = M.columnspace()
print("\nBasis for Column Space:")
for vec in pivot_columns:
    print(np.array(vec).astype(np.float64).flatten())

# Basis of null space (solutions to Ax=0)
null_space = M.nullspace()
print("\nBasis for Null Space:")
if not null_space:
    print("Trivial solution only.")
else:
    for vec in null_space:
        print(np.array(vec).astype(np.float64).flatten())

# Basis of row space (pivot rows of A)
row_space = M.rowspace()
print("\nBasis for Row Space:")
for vec in row_space:
    print(np.array(vec).astype(np.float64).flatten())

# Basis of left null space (null space of A^T)
left_null_space = M.T.nullspace()
print("\nBasis for Left Null Space:")
if not left_null_space:
    print("Trivial solution only.")
else:
    for vec in left_null_space:
        print(np.array(vec).astype(np.float64).flatten())

# Solving system of homogeneous equations Ax=0 using Gauss-Jordan elimination
augmented = M.row_join(Matrix.zeros(rows, 1))
rref, pivots = augmented.rref()
print("\nReduced Row Echelon Form (Gauss-Jordan):")
print(np.array(rref).astype(np.float64))

print("\nThe solutions correspond to the null space vectors above (free variables).")










#Check the linear dependence of vectors. Generate a linear combination of given vectors of Rⁿ / matrices of the same size and find the transition matrix of given matrix space