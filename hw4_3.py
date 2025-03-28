import numpy as np
import matplotlib.pyplot as plt

# Given parameters
p = 1e-5
t_min, t_max = -0.1, 0.1

# Define the analytical solution
def analytical_solution(t, p):
    return t / np.sqrt(p + t**2)

# Second-order finite difference method for y''(t)
def finite_difference_method(t_min, t_max, p, h):
    t_values = np.arange(t_min, t_max + h, h)
    N = len(t_values)
    
    # Initialize solution vector
    y_numerical = np.zeros(N)
    
    # Boundary conditions
    y_numerical[0] = -0.1 / np.sqrt(p + 0.01)
    y_numerical[-1] = 0.1 / np.sqrt(p + 0.01)
    
    # Construct coefficient matrix A and right-hand side vector b
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    # Fill the matrix for interior points
    for i in range(1, N-1):
        t_n = t_values[i]
        A[i, i-1] = 1 / h**2
        A[i, i] = -2 / h**2 + 3 * p / (p + t_n**2)**2
        A[i, i+1] = 1 / h**2
        b[i] = 0  # Right-hand side is zero
    
    # Enforce boundary conditions in the system
    A[0, 0] = A[-1, -1] = 1
    b[0] = y_numerical[0]
    b[-1] = y_numerical[-1]
    
    # Solve the linear system
    y_numerical = np.linalg.solve(A, b)
    
    return t_values, y_numerical

# Determine appropriate mesh size by resolving boundary layer at t=0 (eyeballing)
h = 0.0002  # Chosen based on boundary layer resolution (eyeballing)
t_values, y_numerical = finite_difference_method(t_min, t_max, p, h)

# Exact solution
y_exact = analytical_solution(t_values, p)

# Plot the solutions
plt.figure(figsize=(8, 6))
plt.plot(t_values, y_numerical, 'bo-', label='Numerical Solution (FD)')
plt.plot(t_values, y_exact, 'r-', label='Analytical Solution')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'Finite Difference Solution vs Analytical Solution (h={h})')
plt.legend()
plt.grid()
plt.show()