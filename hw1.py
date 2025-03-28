import matplotlib.pyplot as plt

y1 = 1
y2 = 0
dy1 = 1
dy2 = 101
y1h1_arr = [y1]
y2h1_arr = [y2]
t_arr1 = [0]
t = 0
N = 1
h1 = .01
h2 = .05

while t <= N:
    dy1 = y1
    dy2 = -100 * (y2-y1) + y1
    
    y1 += h1 * dy1
    y2 += h1 * dy2
    t += h1
    
    t_arr1.append(t)
    y1h1_arr.append(y1)
    y2h1_arr.append(y2)
    
y1 = 1
y2 = 0
dy1 = 1
dy2 = 101
y1h2_arr = [y1]
y2h2_arr = [y2]
t_arr2 = [0]
t = 0

while t <= N:
    dy1 = y1
    dy2 = -100 * (y2-y1) + y1
    
    y1 += h2 * dy1
    y2 += h2 * dy2
    t += h2
    
    t_arr2.append(t)
    y1h2_arr.append(y1)
    y2h2_arr.append(y2)
    
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(t_arr1, y1h1_arr, label='y1')
ax1.plot(t_arr1, y2h1_arr, label='y2', linestyle=":")
ax2.plot(t_arr2, y1h2_arr, label='y1')
ax2.plot(t_arr2, y2h2_arr, label='y2', linestyle=":")
ax1.set_xlabel('t')
ax1.set_ylabel('y vals')
ax1.set_title('h = .01')
ax1.legend()
ax2.set_xlabel('t')
ax2.set_ylabel('y vals')
ax2.set_title('h = .05')
ax2.legend()

plt.axis([0, 1, 0, 3])
plt.show()
