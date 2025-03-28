import numpy as np
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10
beta = 8 / 3
rho = 28

# Initial conditions
y0 = np.array([0.0, 1.0, 0.0])
h = 0.01
T = 100
N = int(T / h)

# Lorenz system derivatives
def lorenz(y):
    y1, y2, y3 = y
    dy1 = sigma * (y2 - y1)
    dy2 = y1 * (rho - y3) - y2
    dy3 = y1 * y2 - beta * y3
    return np.array([dy1, dy2, dy3])

# Forward Euler for first step
y_vals = np.zeros((N + 1, 3))
y_vals[0] = y0

y_vals[1] = y_vals[0] + h * lorenz(y_vals[0])  # First step

# 2-step Adams-Bashforth method
for i in range(1, N):
    y_vals[i + 1] = y_vals[i] + (3/2) * h * lorenz(y_vals[i]) - (1/2) * h * lorenz(y_vals[i - 1])

# 3D phase-space plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(y_vals[:, 0], y_vals[:, 1], y_vals[:, 2], color="tab:blue", linewidth=0.5)
ax.set_xlabel("y1")
ax.set_ylabel("y2")
ax.set_zlabel("y3")
ax.set_title("Lorenz Attractor (Forward Euler + Adams-Bashforth)")
plt.show()