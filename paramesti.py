import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math

torch.set_default_device("cpu")

import torch

I = torch.tensor(10.0, requires_grad=True, dtype=torch.float32)
r_1 = torch.tensor(1.2, requires_grad=True, dtype=torch.float32)
m_1 = torch.tensor(0.008, requires_grad=True, dtype=torch.float32)
h_1 = torch.tensor(0.006, requires_grad=True, dtype=torch.float32)

r_2 = torch.tensor(0.06, requires_grad=True, dtype=torch.float32)
h_2 = torch.tensor(0.005, requires_grad=True, dtype=torch.float32)
m_2 = torch.tensor(0.01, requires_grad=True, dtype=torch.float32)
c1  = torch.tensor(0.0005, requires_grad=True, dtype=torch.float32)

r_3 = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
h_3 = torch.tensor(0.00, requires_grad=True, dtype=torch.float32)
m_3 = torch.tensor(0.01, requires_grad=True, dtype=torch.float32)
c2  = torch.tensor(0.00, requires_grad=True, dtype=torch.float32)

n00 = 140 #0.9 Steady state
n10 = 40 #0.5 Steady state
n20 = 1
n30 = 1

t_initial = 0
t_final = 10

def ode(t, Y):
    n0 = Y[:, 0:1]
    n1 = Y[:, 1:2]
    n2 = Y[:, 2:3]
    n3 = Y[:, 3:4]

    dn0_dt = dde.grad.jacobian(Y, t, i=0)
    dn1_dt = dde.grad.jacobian(Y, t, i=1)
    dn2_dt = dde.grad.jacobian(Y, t, i=2)
    dn3_dt = dde.grad.jacobian(Y, t, i=3)

    first  = dn0_dt - I - math.e * n0 + (n1*n0*r_1)
    second = dn1_dt - n1 * n0 * r_1 + n1 * m_1 + (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) + (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1)
    third  = dn2_dt - (n2 * n1 * r_2)/(1 + h_2 * r_2 * n1) - n2 * m_2 - n2 * n3 * c1
    fourth = dn3_dt - (n3 * n1 * r_3)/(1 + h_3 * r_3 * n1) - n3 * m_3 - n2 * n3 * c2

    #first = dn0_dt - n0
    #second = dn1_dt + n1
    #third = dn2_dt - n2
    #fourth = dn3_dt + n3

    return [first, second, third, fourth]

geom = dde.geometry.TimeDomain(t_initial, t_final)

def boundary(x, on_initial):
    return on_initial

ic_n0 = dde.icbc.IC(geom, lambda x: n00, boundary, component=0)
ic_n1 = dde.icbc.IC(geom, lambda x: n10, boundary, component=1)
ic_n2 = dde.icbc.IC(geom, lambda x: n20, boundary, component=2)
ic_n3 = dde.icbc.IC(geom, lambda x: n30, boundary, component=3)

data = dde.data.PDE(geom, ode, [ic_n0, ic_n1, ic_n2, ic_n3], num_domain=1000, num_boundary=2000)

neurons = 40
layers = 5
layer_size = [1] + [neurons] * layers + [4]

activation = "tanh"
initialiser = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initialiser)

model = dde.Model(data, net)

optimizer = torch.optim.Adam([I, r_1, m_1, h_1, r_2, h_2, m_2, c1, r_3, h_3, m_3, c2], lr=0.001)

model.compile(optimizer)

losshistory, train_state = model.train(iterations=3000, display_every=50)

dde.utils.external.plot_loss_history(losshistory)
plt.show()

t_array = np.linspace(t_initial, t_final, 2048)

pinn_pred = model.predict(t_array.reshape(-1, 1))

n0_pinn = pinn_pred[:, 0:1]
n1_pinn = pinn_pred[:, 1:2]
n2_pinn = pinn_pred[:, 2:3]
n3_pinn = pinn_pred[:, 3:4]

t_test = np.array([[t_initial]])  # Check only at initial time
predicted_initial = model.predict(t_test)
print("Predicted Initial Conditions:", predicted_initial)


plt.plot(t_array, n0_pinn.flatten(), color="green", label=r"$x(t)$ PINNs, basal")
plt.plot(t_array, n1_pinn.flatten(), color="blue", label=r"$y(t)$ PINNs, prey")
plt.plot(t_array, n2_pinn.flatten(), color="red", label=r"$y(t)$ PINNs, predator 1")
plt.plot(t_array, n3_pinn.flatten(), color="orange", label=r"$y(t)$ PINNs, predator 2")


plt.legend()
plt.ylabel(r"population")
plt.xlabel(r"$t$")
plt.title("Lotka-Volterra numerical solution using PINNs method")
plt.show()

print(f"Estimated I: {I.item()}")
print(f"Estimated r_1: {r_1.item()}")
print(f"Estimated m_1: {m_1.item()}")
print(f"Estimated h_1: {h_1.item()}")
print(f"Estimated r_2: {r_2.item()}")
print(f"Estimated h_2: {h_2.item()}")
print(f"Estimated m_2: {m_2.item()}")
print(f"Estimated c1: {c1.item()}")
print(f"Estimated r_3: {r_3.item()}")
print(f"Estimated h_3: {h_3.item()}")
print(f"Estimated m_3: {m_3.item()}")
print(f"Estimated c2: {c2.item()}")