"""
Homework 3
Problem 1

Solving IVP: y' = -λ(y^2 - [sin(t) + 1]^2) + cos(t)
             y(0) = 1
             0 <= t <= 1
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos

fig = plt.figure(figsize = (10,6))
fig.suptitle(r"Solving IVP: $y' = -\lambda(y^2 - [\sin(t) + 1]^2) + \cos(t)$" + "\n" + 
             r"$y(0) = 1$")

""" Part a """

# Forward Euler for IVP (h = .001)

b = 1
y0 = 1
h = .001 # time step
t = np.arange(0, b+h, h)
N = len(t)-1 # this is because we want N+1 TOTAL temporal points
lam = 100 # λ value

y = np.zeros(N+1)

# fill in my initial condition
y[0] = y0

def f(t, y):
    return -lam*(y**2 - (sin(t) + 1)**2) + cos(t)

# FE
for n in range(N):
    y[n+1] = y[n] + h * f(t[n], y[n])
    
# Plot results
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(
    t, y, linestyle=":", linewidth=1, marker=".", markersize=3, 
    label="Forward Euler (h = .001)", color = "tab:orange"
    )
ax1.axis([0, 1, 1, 2])
plt.xlabel("t")
plt.ylabel("y(t)")
ax1.legend()

# Forward Euler for IVP (h = .02)

h = .02 # time step
t = np.arange(0, b+h, h)
N = len(t)-1 # this is because we want N+1 TOTAL temporal points

y = np.zeros(N+1)

# fill in my initial condition
y[0] = y0

# FE
for n in range(N):
    y[n+1] = y[n] + h * f(t[n], y[n])

# Plot results
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(
    t, y, linestyle=":", linewidth=1, marker=".", 
    label="Forward Euler (h = .02)", color = "tab:orange"
    )
ax2.axis([0, .5, -10, 5])
plt.xlabel("t")
plt.ylabel("y(t)")
ax2.legend()

""" Part b """

# Backward Euler for the IVP

TOL = 10e-7 # tolerance for Newton's method
h = .1 # time step
t = np.arange(0, b+h, h)
N = len(t)-1 # this is because we want N+1 TOTAL temporal points

y = np.zeros(N+1)

# fill in my initial condition
y[0] = y0

def df_dy(t, y):
    return -2*lam*y

y_new = 0
y_newplus1 = 0

# BE

for n in range(1, N+1):

    y_new = y[n-1]
    # do one step of Newton:
    y_newplus1 = y_new - (y_new - y[n-1] - h*f(t[n], 
                          y_new))/(1 - h*df_dy(t[n], y_new))
    
    diff = abs(y_new - y_newplus1) # difference between netwon iterates

    iter = 0
    while diff > TOL and iter < 10:
        # DO NEWTON 
        y_new = y_newplus1
        y_newplus1 = y_new - (y_new - y[n-1] - h*f(t[n], 
                              y_new))/(1 - h*df_dy(t[n], y_new))
        diff = abs(y_new - y_newplus1)
        # print(f"diff: {diff}")
        iter += 1
      
   # return 
    y[n] = y_newplus1
    
# Scatter plot exact solution
ax3 = fig.add_subplot(2, 2, 3)
sol = np.sin(t) + 1
ax3.plot(t, sol, linestyle='-', linewidth=1, marker='.', 
         label="Exact Solution (sin(t) + 1)", color = "tab:blue")
    
# Scatter plot for numerical solution
ax3.plot(t, y, linestyle=':', linewidth=1, marker='o', markersize=3, 
         label="Backward Euler (h = 0.01)", color = 'tab:orange')
ax3.axis([0, 1, 1, 2])

# Labels and legend
plt.xlabel("t")
plt.ylabel("y(t)")
ax3.legend()

# Show the figure
plt.axis([0, 1, 1, 2])
fig.tight_layout(pad=2.0)
plt.show()
