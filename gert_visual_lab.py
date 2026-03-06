import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("output",exist_ok=True)

# ======================
# ComplexGERT
# ======================

class ComplexGERT:

    def __init__(self,G=0.1,E=0.1,R=0.1):
        self.G=complex(G,0)
        self.E=E
        self.R=R

    def step(self,e):

        g=complex(0.95*(self.G+self.R*self.E+e),0)

        e_new=0.5*(self.E+cmath.tanh(g).real)

        r=0.1*max(min(g.real/max(e_new,1e-8),1),-1)

        self.G=g
        self.E=e_new
        self.R=r

        return self.G.real


# ======================
# simulation
# ======================

steps=700

model=ComplexGERT()

G=[]
E=[]
R=[]
I=[]

for t in range(steps):

    inp=0.5*np.sin(t*0.05)

    model.step(inp)

    G.append(model.G.real)
    E.append(model.E)
    R.append(model.R)
    I.append(inp)


# ======================
# 1 convergence animation
# ======================

fig,ax=plt.subplots()

line1,=ax.plot([],[])
line2,=ax.plot([],[])
line3,=ax.plot([],[])

ax.set_xlim(0,steps)
ax.set_ylim(min(G)-1,max(G)+1)
ax.set_title("Control Convergence")

def update(frame):

    line1.set_data(range(frame),G[:frame])
    line2.set_data(range(frame),E[:frame])
    line3.set_data(range(frame),I[:frame])

    return line1,line2,line3

ani=FuncAnimation(fig,update,frames=steps)
ani.save("output/convergence.mp4",fps=30)


# ======================
# 2 attractor animation
# ======================

fig2=plt.figure()
ax2=fig2.add_subplot(111,projection="3d")

def update2(frame):

    ax2.clear()
    ax2.plot(G[:frame],E[:frame],R[:frame])
    ax2.set_title("3D Attractor")

ani2=FuncAnimation(fig2,update2,frames=steps)
ani2.save("output/attractor.mp4",fps=30)


# ======================
# 3 stability heatmap
# ======================

grid=60
heat=np.zeros((grid,grid))

g_vals=np.linspace(-1,1,grid)
e_vals=np.linspace(-1,1,grid)

for i,g in enumerate(g_vals):
    for j,e in enumerate(e_vals):

        m=ComplexGERT(g,e)

        for _ in range(40):
            m.step(0.2)

        heat[i,j]=abs(m.G.real)

plt.figure()
plt.imshow(heat,extent=[-1,1,-1,1],origin="lower")
plt.title("Control Stability Map")
plt.colorbar()
plt.savefig("output/stability_map.png")


# ======================
# 4 Lyapunov map
# ======================

def lyapunov(G0,E0):

    m1=ComplexGERT(G0,E0)
    m2=ComplexGERT(G0+1e-6,E0)

    d0=1e-6
    total=0

    for i in range(40):

        m1.step(0.2)
        m2.step(0.2)

        d=abs(m1.G.real-m2.G.real)

        if d==0:
            d=1e-8

        total+=np.log(abs(d/d0))

        d0=d

    return total/40


lyap=np.zeros((grid,grid))

for i,g in enumerate(g_vals):
    for j,e in enumerate(e_vals):

        lyap[i,j]=lyapunov(g,e)

plt.figure()
plt.imshow(lyap,extent=[-1,1,-1,1],origin="lower")
plt.title("Lyapunov Map")
plt.colorbar()
plt.savefig("output/lyapunov_map.png")


# ======================
# 5 chaos boundary
# ======================

chaos=(lyap>0).astype(int)

plt.figure()
plt.imshow(chaos,extent=[-1,1,-1,1],origin="lower")
plt.title("Chaos Boundary")
plt.savefig("output/chaos_boundary.png")

print("All visuals generated")
