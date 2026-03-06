import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

os.makedirs("output", exist_ok=True)

# =========================
# ComplexGERT
# =========================

class ComplexGERT:

    def __init__(self):
        self.G = complex(0.1,0)
        self.E = 0.1
        self.R = 0.1

    def control(self,e):

        g = complex(0.95*(self.G + self.R*self.E + e),0)

        e_new = 0.5*(self.E + cmath.tanh(g).real)

        r = 0.1 * max(min(g.real/max(e_new,1e-8),1),-1)

        self.G = g
        self.E = e_new
        self.R = r

        return self.G.real


# =========================
# Simulation
# =========================

steps = 500
model = ComplexGERT()

G=[]
E=[]
I=[]

for t in range(steps):

    inp = 0.5*np.sin(t*0.05)

    model.control(inp)

    G.append(model.G.real)
    E.append(model.E)
    I.append(inp)


# =========================
# 1 Control Response Video
# =========================

fig,ax = plt.subplots()

line1, = ax.plot([],[],label="G")
line2, = ax.plot([],[],label="E")
line3, = ax.plot([],[],label="Input")

ax.set_xlim(0,steps)
ax.set_ylim(min(G)-1,max(G)+1)

ax.legend()
ax.set_title("Control Stabilization")

def update(frame):

    line1.set_data(range(frame),G[:frame])
    line2.set_data(range(frame),E[:frame])
    line3.set_data(range(frame),I[:frame])

    return line1,line2,line3

ani = FuncAnimation(fig,update,frames=steps)

ani.save("output/control_response.mp4",fps=30)



# =========================
# 2 Phase Attractor
# =========================

plt.figure()

plt.plot(G,E)

plt.title("Phase Attractor (G vs E)")

plt.xlabel("G")
plt.ylabel("E")

plt.savefig("output/phase_attractor.png")



# =========================
# 3 Stability Map
# =========================

grid=50
heat=np.zeros((grid,grid))

vals=np.linspace(-1,1,grid)

for i,g in enumerate(vals):
    for j,e in enumerate(vals):

        m=ComplexGERT()

        m.G=complex(g,0)
        m.E=e

        for t in range(30):
            m.control(0.2)

        heat[i,j]=abs(m.G.real)

plt.figure()

plt.imshow(heat,extent=[-1,1,-1,1],origin="lower")

plt.title("Stability Region")

plt.colorbar()

plt.savefig("output/stability_map.png")

print("visuals generated")
