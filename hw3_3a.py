"""
Homework 3
Problem 3

Solving IVP: y' = -λ(y^2 - [sin(t) + 1]^2) + cos(t)
             y(0) = 1
             0 <= t <= 1
             With RK3
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos

#    Solving IVP with RK3
#    Given Butcher Table:
#     0  |  0   0   0
#    1/3 | 1/3  0   0
#    2/3 |  0  2/3  0
#    ----|-------------
#        | 1/4  0  3/4

fig = plt.figure(figsize = (10,6))
fig.suptitle(r"Predator and Prey population ")

""" h = .01 """

# Parameters
h = 0.01
y0 = 1
t0 = 0
t_end = 1
lam = 1

# Define the function f(t, y)
def f(t, y, lam):
    return -lam * (y**2 - (sin(t) + 1)**2) + cos(t)

# RK3 parameters (from the given Butcher Table)
def rk3(f, y0, t0, t_end, h, lam):
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    
    for i in range(len(t_values)-1):
        t = t_values[i]
        y = y_values[i]
        
        k1 = h * f(t, y, lam)
        k2 = h * f(t + h/3, y + k1/3, lam)
        k3 = h * f(t + 2*h/3, y + 2*k2/3, lam)
        
        y_values[i+1] = y + (1/4) * k1 + (3/4) * k3
    
    return t_values, y_values

# Solve IVP
t_vals, y_vals = rk3(f, y0, t0, t_end, h, lam)

# Scatter plot for numerical solution
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(t_vals, y_vals, linestyle=':', linewidth=1, marker='o', markersize=3, 
         label="RK3 (h = 0.01)", color = 'tab:orange')
ax1.axis([0, 1, 1, 2])

# Labels and legend
plt.xlabel("t")
plt.ylabel("y(t)")
ax1.legend()

""" h = .001 """

# Parameters
h = 0.001
y0 = 1
t0 = 0
t_end = 1
lam = 1

# Solve IVP
t_vals, y_vals = rk3(f, y0, t0, t_end, h, lam)

# Scatter plot for numerical solution
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(t_vals, y_vals, linestyle='None', linewidth=1, marker='o', markersize=3, 
         label="RK3 (h = 0.001)", color = 'tab:orange')
ax2.axis([0, 1, 1, 2])

# Scatter plot exact solution
sol = np.sin(t_vals) + 1
ax2.plot(t_vals, sol, linestyle='--', linewidth=2, marker='', 
         label="Exact Solution (sin(t) + 1)", color = "tab:blue")

# Labels and legend
ax2.set_xlabel("t")
ax2.set_ylabel("y(t)")
ax2.legend()

# Show the figure
fig.tight_layout(pad=2.0)
plt.show()