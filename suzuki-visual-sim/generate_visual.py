# generate_visual.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

N = 1200
steps = 200

phi = (1 + np.sqrt(5)) / 2

# 初期粒子
pos = np.random.uniform(-5,5,(N,3))
vel = np.random.normal(0,0.2,(N,3))


def control_field(x):
    r = np.linalg.norm(x,axis=1)
    field = np.zeros_like(x)

    field[:,0] = np.sin(r*phi)
    field[:,1] = np.cos(r*phi)
    field[:,2] = np.sin(r)

    return field*0.02


fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121,projection='3d')
ax2 = fig.add_subplot(122,projection='3d')

ax1.set_title("Uncontrolled Chaos")
ax2.set_title("Suzuki Controlled Emergence")

sc1 = ax1.scatter([],[],[],s=2)
sc2 = ax2.scatter([],[],[],s=2)

pos2 = pos.copy()
vel2 = vel.copy()


def update(frame):

    global pos,vel,pos2,vel2

    # chaos
    vel += np.random.normal(0,0.01,(N,3))
    pos += vel

    # controlled
    vel2 += control_field(pos2)
    pos2 += vel2

    ax1.clear()
    ax2.clear()

    ax1.set_xlim(-10,10)
    ax1.set_ylim(-10,10)
    ax1.set_zlim(-10,10)

    ax2.set_xlim(-10,10)
    ax2.set_ylim(-10,10)
    ax2.set_zlim(-10,10)

    ax1.set_title("Uncontrolled Chaos")
    ax2.set_title("Suzuki Controlled Emergence")

    ax1.scatter(pos[:,0],pos[:,1],pos[:,2],s=2)
    ax2.scatter(pos2[:,0],pos2[:,1],pos2[:,2],s=2)

ani = FuncAnimation(fig,update,frames=steps)

ani.save("suzuki_emergence.mp4",fps=30)
ani.save("suzuki_emergence.gif",fps=30)

print("simulation complete")
