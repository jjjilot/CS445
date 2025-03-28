import numpy as np
import time

# ==============================================================================
# RESULTS
# Forward Euler took 0.782013 ms to achieve E ≤ 10^-4
# Backward Euler took 1.778126 ms to achieve E ≤ 10^-4

# Forward Euler is more efficient
# ==============================================================================

# Parameters
k = 100
L = 1
y0 = L / 2  # initial condition
t_end = 1
target_error = 1e-4
newton_tol = 1e-9

# Exact solution function
def exact_solution(t):
    return L / (1 + np.exp(-k * t))

# Forward Euler Method
def forward_euler(N):
    h = 1 / N
    y = np.full(N+1, y0)
    for n in range(N):
        y[n+1] = y[n] + h * (k / L) * y[n] * (L - y[n])
    return y

# Backward Euler Method with Newton's Method for implicit solving
def backward_euler(N):
    h = 1 / N
    y = np.full(N+1, y0)
    for n in range(N):
        # Newton method to solve implicit equation for y[n+1]
        y_new = y[n]  # initial guess
        for _ in range(10):  # max 10 iterations
            f = y_new - y[n] - h * (k / L) * y_new * (L - y_new)
            df = 1 - h * (k / L) * (L - 2 * y_new)
            y_new -= f / df
            if abs(f) < newton_tol:
                break
        y[n+1] = y_new
    return y

# Error Calculation
def calculate_error(y, N):
    h = 1 / N
    t_vals = np.linspace(0, t_end, N+1)
    exact_vals = exact_solution(t_vals)
    error = np.sum(np.abs(y - exact_vals) * h)
    return error

# Time Measurement and Error Calculation for Forward Euler with dynamic N
def forward_euler_time():
    N = 10  # Start with a small N
    while True:
        start_time = time.time()
        y_fe = forward_euler(N)
        error_fe = calculate_error(y_fe, N)
        if error_fe <= target_error:
            break
        N *= 2  # Double N to halve step size h
    end_time = time.time()
    return end_time - start_time, N

# Time Measurement and Error Calculation for Backward Euler with dynamic N
def backward_euler_time():
    N = 10  # Start with a small N
    while True:
        start_time = time.time()
        y_be = backward_euler(N)
        error_be = calculate_error(y_be, N)
        if error_be <= target_error:
            break
        N *= 2  # Double N to halve step size h
    end_time = time.time()
    return end_time - start_time, N

# Main
fe_time, fe_steps = forward_euler_time()
be_time, be_steps = backward_euler_time()
# Display results
print(f"Forward Euler took {fe_time * 1000:.6f} ms for {fe_steps} steps to achieve E ≤ 10^-4")
print(f"Backward Euler took {be_time * 1000:.6f} ms for {be_steps} steps to achieve E ≤ 10^-4")
# Print superior method
if fe_time < be_time:
    print("Forward Euler is more efficient.")
else:
    print("Backward Euler is more efficient.")