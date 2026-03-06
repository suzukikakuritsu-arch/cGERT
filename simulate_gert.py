import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# =========================
# ComplexGERT Model
# =========================

class ComplexGERT:

    def __init__(self):
        self.G = complex(0.1, 0)
        self.E = 0.1
        self.R = 0.1

    def control(self, e):

        g = complex(0.95*(self.G + self.R*self.E + e), 0)

        e_new = 0.5*(self.E + cmath.tanh(g).real)

        r = 0.1 * max(min((g.real/max(e_new,1e-8)),1),-1)

        self.G = g
        self.E = e_new
        self.R = r

        return self.G.real*0.236


# =========================
# Simulation
# =========================

steps = 500

model = ComplexGERT()

G_list = []
E_list = []
R_list = []
input_list = []

for t in range(steps):

    # 外乱入力
    e = 0.5*np.sin(t*0.05)

    out = model.control(e)

    G_list.append(model.G.real)
    E_list.append(model.E)
    R_list.append(model.R)
    input_list.append(e)

# =========================
# Visualization
# =========================

fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title("State Convergence")
ax1.set_xlabel("time")
ax1.set_ylabel("value")

ax2.set_title("Phase Space (G vs E)")
ax2.set_xlabel("G")
ax2.set_ylabel("E")

line_G, = ax1.plot([], [])
line_E, = ax1.plot([], [])
line_in, = ax1.plot([], [])

phase, = ax2.plot([], [])

def update(frame):

    line_G.set_data(range(frame), G_list[:frame])
    line_E.set_data(range(frame), E_list[:frame])
    line_in.set_data(range(frame), input_list[:frame])

    ax1.set_xlim(0, steps)
    ax1.set_ylim(min(G_list)-1, max(G_list)+1)

    phase.set_data(G_list[:frame], E_list[:frame])

    ax2.set_xlim(min(G_list)-1, max(G_list)+1)
    ax2.set_ylim(min(E_list)-1, max(E_list)+1)

    return line_G,line_E,line_in,phase

ani = FuncAnimation(fig, update, frames=steps, interval=30)

os.makedirs("output",exist_ok=True)

ani.save("output/gert_simulation.mp4", fps=30)

print("video saved: output/gert_simulation.mp4")
