import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math

if torch.cuda.is_available():
    torch.set_default_device("cuda")

#T
a = 4.31e-1
b = 2.17e-8

# NK cell-related rates
c1 = 3.5e-6  # NK cell tumor cell kill rate [cells^-1]
c2 = 1.0e-7  # NK cell inactivation rate by tumor cells [cells^-1 day^-1]
d1 = 1.0e-6  # Dendritic cell priming NK cells rate [cells^-1]
d2 = 4.0e-6  # NK cell dendritic cell kill rate [cells^-1]
d3 = 1.0e-4  # Tumor cells priming dendritic cells rate [Estimate]
e = 4.12e-2  # Death rate of NK cell [day^-1]

# CD8+ T cell-related rates
f1 = 1.0e-8  # CD8+ T cell dendritic cells kill rate [cells^-1]
f2 = 0.01    # Dendritic cells priming CD8+ T cell rate [cells^-1]
g = 2.4e-2   # Death rate of dendritic cells [cells^-1]
h = 3.42e-10 # CD8+ T cell inactivation rate by tumor cells [cells^-1 day^-1]
i = 2.0e-2   # Death rate of CD8+ T cells [day^-1]

# Dendritic cell tumor cell interaction
j = 1.0e-7   # Dendritic cell tumor cell kill rate [cells^-1]
k = 1.0e-7   # NK cell tumor cell kill rate [cells^-1]

# Cell sources
s1 = 1.3e4   # Source of NK cells [cells^-1]
s2 = 4.8e2   # Source of dendritic cells [cells^-1]

# Regulatory function
u = 1.80e-8  # Regulatory function by NK cells of CD8+ T cells [cell^-2 day^-1]

#k
KT = 9e-2
KN = 6e-2
KD = 6e-2
KL = 6e-2

#v
vL = 1e6
vM = 1

#UNDEFINED
g1 = 0
h1 = 0
r1 = 0
pI = 0
gI = 0
d4 = 0
d5 = 0
vI = 0

#temp
g1 = 1e-2
h1 = 1.0
r1 = 1e-3
pI = 1e-2
gI = 1.0
d4 = 1e-1
d5 = 1e-1
vI = 1.0


T0 = 100
N0 = 1
D0 = 1
L0 = 1
M0 = 0
I0 = 0

#UNDEFINED
g1 = 0
h1 = 0.01
r1 = 0
pI = 0
gI = 0
u = 0
KT = 0
KN = 0
KD = 0
KL = 0
vL = 0
vM = 0
d4 = 0
d5 = 0
d3 = 0
vI = 0

t_initial = 0
t_final = 10

def ode(t, Y):
    T = Y[:, 0:1]
    N = Y[:, 1:2]
    D = Y[:, 2:3]
    L = Y[:, 3:4]
    M = Y[:, 4:5]
    I = Y[:, 5:6]

    dT_dt = dde.grad.jacobian(Y, t, i=0)
    dN_dt = dde.grad.jacobian(Y, t, i=1)
    dD_dt = dde.grad.jacobian(Y, t, i=2)
    dL_dt = dde.grad.jacobian(Y, t, i=3)
    dM_dt = dde.grad.jacobian(Y, t, i=4)
    dI_dt = dde.grad.jacobian(Y, t, i=5)

    Tumor     = dT_dt - ( a * T * (1 - b * T) - (c1 * N + j * D + k * L) * T - KT * (1 - torch.exp(-M)) * T )
    NatKill   = dN_dt - (s1 + ((g1 * N * T**2) / (h1 + T**2)) - (c2 * T - d1 * D) * N - KN * (1 - torch.exp(-M)) * N - e * N)
    Dendritic = dD_dt - (s2 - (f1 * L + d2 * N - d3 * T) * D - KD * (1 - torch.exp(-M)) * D - g * D)
    Cytotoxic = dL_dt - ((f2 * D * T - h * L * T - u * N * L**2 + r1 * N * T - KL * (1 - torch.exp(-M)) * L - i * L + vL))
    Chemo     = dM_dt - (vM - d4 * M)
    Immuno    = dI_dt - (vI - d5 * I)

    return[Tumor, NatKill, Dendritic, Cytotoxic, Chemo, Immuno]

geom = dde.geometry.TimeDomain(t_initial, t_final)

def boundary(x, on_initial):
    return on_initial

ic_T = dde.icbc.IC(geom, lambda x: T0, boundary, component=0)
ic_N = dde.icbc.IC(geom, lambda x: N0, boundary, component=1)
ic_D = dde.icbc.IC(geom, lambda x: D0, boundary, component=2)
ic_L = dde.icbc.IC(geom, lambda x: L0, boundary, component=3)
ic_M = dde.icbc.IC(geom, lambda x: M0, boundary, component=4)
ic_I = dde.icbc.IC(geom, lambda x: I0, boundary, component=5)

data = dde.data.PDE(geom, ode, [], 512*12, 6, num_test=512*12)

neurons = 64
layers = 8
layer_size = [1] + [neurons] * layers + [6]

def output_transform(t, y):
    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    y3 = y[:, 2:3]
    y4 = y[:, 3:4]
    y5 = y[:, 4:5]
    y6 = y[:, 5:6]
    
    return torch.cat([y1 * torch.tanh(t) + T0, y2 * torch.tanh(t) + N0, y3 * torch.tanh(t) + D0, y4 * torch.tanh(t) + L0, y5 * torch.tanh(t) + M0, y6 * torch.tanh(t) + I0, ], dim=1)

def input_transform(t):
    return torch.cat(
        [
            torch.sin(t),
        ],
        dim=1,
    )


activation = "tanh"
initialiser = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initialiser)

net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=10000, display_every = 10)

# Most backends except jax can have a second fine tuning of the solution
model.compile("L-BFGS")
losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#dde.utils.external.plot_loss_history(losshistory)
#plt.show()

# Generate predictions
t_array = np.linspace(t_initial, t_final, 2048)
pinn_pred = model.predict(t_array.reshape(-1, 1))

T_pinn = pinn_pred[:, 0]
N_pinn = pinn_pred[:, 1]
D_pinn = pinn_pred[:, 2]
L_pinn = pinn_pred[:, 3]
M_pinn = pinn_pred[:, 4]
I_pinn = pinn_pred[:, 5]

# Setup subplots
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
axs = axs.flatten()

variables = [
    (T_pinn, r"$T(t)$ Tumor cells", "green"),
    (N_pinn, r"$N(t)$ NK cells", "blue"),
    (D_pinn, r"$D(t)$ Dendritic cells", "red"),
    (L_pinn, r"$L(t)$ CD8$^+$ T cells", "orange"),
    (M_pinn, r"$M(t)$ Chemotherapy", "purple"),
    (I_pinn, r"$I(t)$ Immunotherapy", "brown")
]

for ax, (values, label, color) in zip(axs, variables):
    ax.plot(t_array, values, color=color, label=label)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Population / Concentration")
    ax.set_title(label)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.suptitle("PINN Solution: Immune-Tumor Dynamics", fontsize=16, y=1.02)
plt.show()
