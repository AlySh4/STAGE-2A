from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg

#
# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
# x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.8))
#
# def bonjour(theta):
#
#     u = np.sin(x)*np.sin(theta)
#     v = np.cos(y)*np.sin(theta)
#     w = np.sin(y)*np.sin(theta)
#     return x,y,z,u,v,w
#
# Q = ax.quiver(*bonjour(0))
#
# def ani(i):
#     global Q
#     Q.remove()
#     Q = ax.quiver(*bonjour(i))
#
# ani = FuncAnimation(fig, ani, frames=np.linspace(0,2*np.pi,200), interval=50)
#
# plt.show()

#
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
#
# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
#
# X, Y, Z = np.meshgrid(np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.8))
#
# def get_arrow(theta):
#     x = np.cos(theta)
#     y = np.sin(theta)
#     z = 0
#     u = np.sin(2*theta)
#     v = np.sin(3*theta)
#     w = np.cos(3*theta)
#     return x,y,z,u,v,w
#
# quiver = ax.quiver(*get_arrow(0))
#
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(-2, 2)
#
# def update(theta):
#     global quiver
#     quiver.remove()
#     quiver = ax.quiver(*get_arrow(theta))
#
# ani = FuncAnimation(fig, update, frames=np.linspace(0,2*np.pi,200), interval=50)
# plt.show()

################uniquement pour la visualisation
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

X, Y, Z = np.meshgrid(np.arange(0, 1, 0.2),
                      np.arange(-2, 2, 0.2),
                      np.arange(-2, 2, 0.2))

u = 1 - X
v = -Y * (1 - X)
w = -Z

ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)
print(type(pg.glColor((4, 50 * 1.3))))
plt.show()
