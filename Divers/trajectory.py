import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from IPython.display import HTML

fig = plt.figure()
ax1 = fig.add_subplot(projection="3d")

# t = np.linspace(0, 80, 300)
# x = np.cos(2 * np.pi * t / 10.)
# y = np.sin(2 * np.pi * t / 10.)
# z = 10 * t

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(-1, 1)


def animate(t):
    x = np.cos(t)
    y = 1 / 2 * np.sin(t / 10) + 1 / 2
    z = np.sin(t)
    ax1.scatter(x, y, z)
    f = open('../Data/testdata.csv', 'a')
    f.write("217,{},{},{},0,0,0,1,True\n".format(x, y, z))
    f.close()


ani = animation.FuncAnimation(fig, animate)

plt.show()
