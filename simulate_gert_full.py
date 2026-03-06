import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

# ======================
# ComplexGERT
# ======================

class ComplexGERT:

    def __init__(self):
        self.G = complex(0.1,0)
        self.E = 0.1
        self.R = 0.1

    def control(self,e):

        g = complex(0.95*(self.G + self.R*self.E + e),0)

        e_new = 0.5*(self.E + cmath.tanh(g).real)

        r = 0.1*max(min(g.real/max(e_new,1e-8),1),-1)

        self.G = g
        self.E = e_new
        self.R = r

        return self.G.real*0.236


# ======================
# simulation
# ======================

steps = 600

model = ComplexGERT()

G=[]
E=[]
R=[]
I=[]

for t in range(steps):

    e = 0.5*np.sin(t*0.05)

    model.control(e)

    G.append(model.G.real)
    E.append(model.E)
    R.append(model.R)
    I.append(e)


os.makedirs("output",exist_ok=True)

# ======================
# 1 state convergence
# ======================

fig,ax = plt.subplots()

line1, = ax.plot([],[])
line2, = ax.plot([],[])
line3, = ax.plot([],[])

ax.set_xlim(0,steps)
ax.set_ylim(min(G)-1,max(G)+1)

ax.set_title("Control Convergence")

def update1(frame):

    line1.set_data(range(frame),G[:frame])
    line2.set_data(range(frame),E[:frame])
    line3.set_data(range(frame),I[:frame])

    return line1,line2,line3

ani = FuncAnimation(fig,update1,frames=steps)

ani.save("output/convergence.mp4",fps=30)


# ======================
# 2 phase attractor
# ======================

fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection="3d")

def update2(frame):

    ax2.clear()

    ax2.plot(G[:frame],E[:frame],R[:frame])

    ax2.set_title("3D Attractor")

ani2 = FuncAnimation(fig2,update2,frames=steps)

ani2.save("output/attractor.mp4",fps=30)


# ======================
# 3 stability heatmap
# ======================

grid=50

heat=np.zeros((grid,grid))

for i,g in enumerate(np.linspace(-1,1,grid)):
    for j,e in enumerate(np.linspace(-1,1,grid)):

        m=ComplexGERT()

        m.G=g
        m.E=e

        for t in range(30):

            m.control(0.2)

        heat[i,j]=abs(m.G.real)


fig3,ax3=plt.subplots()

c=ax3.imshow(heat)

fig3.colorbar(c)

ax3.set_title("Control Stability Map")

plt.savefig("output/stability_map.png")

print("videos generated")
