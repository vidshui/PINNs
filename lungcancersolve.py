import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

def immune_model(t, y):
    T, N, D, L, M, I = y

    dT_dt = ( a * T * (1 - b * T) - (c1 * N + j * D + k * L) * T - KT * (1 - np.exp(-M)) * T )
    dN_dt = (s1 + ((g1 * N * T**2) / (h1 + T**2)) - (c2 * T - d1 * D) * N - KN * (1 - np.exp(-M)) * N - e * N)
    dD_dt = (s2 - (f1 * L + d2 * N - d3 * T) * D - KD * (1 - np.exp(-M)) * D - g * D)
    dL_dt = (f2 * D * T - h * L * T - u * N * L**2 + r1 * N * T - KL * (1 - np.exp(-M)) * L - i * L + vL)
    dM_dt = (vM - d4 * M)
    dI_dt = (vI - d5 * I)

    return [dT_dt, dN_dt, dD_dt, dL_dt, dM_dt, dI_dt]

y0 = [100, 1, 1, 1, 0, 0]

# Time span
t_span = (0, 100)
t_eval = np.linspace(*t_span, 500)

# Solve
sol = solve_ivp(lambda t, y: immune_model(t, y), t_span, y0, t_eval=t_eval)

# Plot results
labels = ['Tumor (T)', 'Natural Killer (N)', 'Dendritic (D)', 
          'Cytotoxic (L)', 'Chemo (M)', 'Immuno (I)']

fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # 3 rows, 2 columns
axs = axs.flatten()  # Flatten to easily index

for i in range(6):
    axs[i].plot(sol.t, sol.y[i], label=labels[i], color='tab:blue')
    axs[i].set_title(labels[i])
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('Concentration')
    axs[i].grid(True)
    axs[i].legend()

plt.tight_layout()
plt.show()
