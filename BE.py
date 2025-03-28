import numpy as np
import matplotlib.pyplot as plt

# Backward Euler for the IVP
# y' = -5ty^2 + 5/t - 1/t^2
# y(1) = y0, where we take y0 = 1
# on the time domain t in [1, b]
# note that the exact solution is y(t) = 1/t

TOL = 1e-10 # tolerance for Newton's method
b = 30
y0 = 1
h = 10 # time step
t = np.arange(0, b+h, h)
N = len(t)-1 # this is because we want N+1 TOTAL temporal points
# to check, we should have that h = (b-1)/N

y = [0] * (N+1)

# fill in my initial condition
y[0] = y0

def f(t, y):
    return -5*t*y**2 + 5/t - 1/t**2

def df_dy(t, y):
    return -10*t*y

y_new = 0 # don't use y_new = [0]
y_newplus1 = 0


for n in range(2, N): # N time steps of BE becasue we already know intial condition

    # at each time step I need to solve g = 0 using Newton' method

    # y[n]: solution at the current time step
    # y[n-1]: solution at the previous time step
    # y_new: previous netwon iterate for y[n]
    # y_newplus1: updated newton iterate for y[n]

    # SET IC FOR NETWON'S method to be solution at previous time level
    y_new = y[n-1]
    # do one step of Newton:
    y_newplus1 = y_new - (y_new - y[n-1] - h*f(t[n], 
                        y_new))/(1 - h*df_dy(t[n], y_new))
    
    diff = abs(y_new - y_newplus1) # difference between netwon iterates

    
    while diff > TOL:
        # DO NEWTON 
        y_new = y_newplus1
        y_newplus1 = y_new - (y_new - y[n-1] - h*f(t[n], 
                            y_new))/(1 - h*df_dy(t[n], y_new))
        diff = abs(y_new - y_newplus1)
        # print(f"diff: {diff}")
      
   # return 
    y[n] = y_newplus1

tfine = np.arange(1, b+.001, .001)
print(tfine)
yexact = 1 / tfine

# First plot: scatter plot for numerical solution
plt.plot(t, y, linestyle=':', linewidth=4, label="numerical", marker='o')

# Second plot: exact solution
plt.plot(tfine, yexact, label="exact")

# Labels and legend
plt.xlabel("time")
plt.ylabel("y")
plt.legend()

# Show the figure
plt.show()