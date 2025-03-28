"""
Homework 3
Problem 4
Solving Predator-Prey Model:
y_1' = 0.25 y_1 - 0.01 y_1 y_2
y_2' = -2y_2 + 0.01 y_1 y_2
"""

import numpy as np
import matplotlib.pyplot as plt

#    Solving IVP with RK3
#    Given Butcher Table:
#     0  |  0   0   0
#    1/3 | 1/3  0   0
#    2/3 |  0  2/3  0
#    ----|-------------
#        | 1/4  0  3/4

fig = plt.figure(figsize = (10,6))
fig.suptitle("Prey population increases exponentially" + "\n" + 
             "as predator population decreases (but not to 0)")

""" h = .01 """

# Parameters
h = 0.01
y0 = 10
t0 = 0
t_end = 10

# Define the function f(t, y) for y_1
def f1(t, y1, y2):
    return .25*y1 - .01*y1*y2

# Define the function f(t, y) for y_2
def f2(t, y1, y2):
    return -2*y2 + .01*y1*y2

# RK3 parameters (from the given Butcher Table)
def rk3(f1, f2, y0, t0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    y_1values = np.zeros_like(t_values)
    y_2values = np.zeros_like(t_values)
    y_1values[0] = y0
    y_2values[0] = y0

    for i in range(len(t_values) - 1):
        t = t_values[i]
        y1 = y_1values[i]
        y2 = y_2values[i]

        # RK3 step for y1
        k1_1 = h * f1(t, y1, y2)
        k1_2 = h * f2(t, y1, y2)

        k2_1 = h * f1(t + h/3, y1 + k1_1/3, y2 + k1_2/3)
        k2_2 = h * f2(t + h/3, y1 + k1_1/3, y2 + k1_2/3)

        k3_1 = h * f1(t + 2*h/3, y1 + 2*k2_1/3, y2 + 2*k2_2/3)
        k3_2 = h * f2(t + 2*h/3, y1 + 2*k2_1/3, y2 + 2*k2_2/3)

        # Update y1 and y2
        y1_next = y1 + (1/4) * k1_1 + (3/4) * k3_1
        y2_next = y2 + (1/4) * k1_2 + (3/4) * k3_2

        # Store results
        y_1values[i+1] = y1_next
        y_2values[i+1] = y2_next

    return t_values, y_1values, y_2values

# Solve IVP
t_vals, y1_vals, y2_vals = rk3(f1, f2, y0, t0, t_end, h)

# Scatter plot for numerical solution
plt.plot(t_vals, y1_vals, linestyle=':', linewidth=1, marker='o', markersize=1, 
         label="Prey Population", color = 'tab:orange')

plt.plot(t_vals, y2_vals, linestyle=':', linewidth=1, marker='o', markersize=1, 
         label="Predator Population", color = 'tab:blue')

# Labels and legend
plt.xlabel("Time (t)")
plt.ylabel("Population")
fig.legend()
fig.tight_layout(pad=2.0)
plt.show()

# Run with t_end = 100 to plot phase space
fig = plt.figure(figsize = (10,6))
fig.suptitle("Averages of both predator and prey populations remain stable" + "\n" 
            + "despite oscillations in population")

t_end = 100
t_vals, y1_vals, y2_vals = rk3(f1, f2, y0, t0, t_end, h)

plt.plot(y1_vals, y2_vals, linestyle=':', linewidth=1, marker='o', markersize=1, 
         label="Predator Pop, Prey Pop", color = 'tab:orange')

# Labels and legend
plt.xlabel("Prey Population")
plt.ylabel("Predator Population")
fig.legend()
fig.tight_layout(pad=2.0)

plt.show()