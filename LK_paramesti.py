import torch
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
import math
import re


if torch.cuda.is_available():
    torch.set_default_device("cuda")

#alpha = 2/3 #reprodcution rate of x
#beta = 4/3 #loss of x cause of y
#gamma = 1 #gain of y cause of x
#delta = 1 #death rate of y

#alpha = torch.tensor(1, requires_grad=True, dtype=torch.float32)
#beta = torch.tensor(1, requires_grad=True, dtype=torch.float32)
#gamma = torch.tensor(0.25, requires_grad=True, dtype=torch.float32)
#delta = torch.tensor(1, requires_grad=True, dtype=torch.float32)

alpha = dde.Variable(0.7)
beta = dde.Variable(1.1)
gamma = dde.Variable(0.8)
delta = dde.Variable(0.8)

x0 = 0.9 #0.9 Steady state
y0 = 0.5 #0.5 Steady state

t_initial = 0
t_final = 10

def ode(t, Y):
    x = Y[:, 0:1]
    y = Y[:, 1:2]
    dx_dt = dde.grad.jacobian(Y, t, i=0)
    dy_dt = dde.grad.jacobian(Y, t, i=1)
    return [
    dx_dt - torch.abs(alpha) * x + torch.abs(beta) * x * y,
    dy_dt + torch.abs(gamma) * y - torch.abs(delta) * x * y
]


geom = dde.geometry.TimeDomain(t_initial, t_final)

def boundary(_, on_initial):
    return on_initial

ic_x = dde.icbc.IC(geom, lambda x: x0, boundary, component=0)
ic_y = dde.icbc.IC(geom, lambda x: y0, boundary, component=1)

data = dde.data.PDE(geom, ode, [ic_x, ic_y], num_domain=512, num_boundary=2)

neurons = 64
layers = 3
layer_size = [1] + [neurons] * layers + [2]

activation = "tanh"
initialiser = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initialiser)

model = dde.Model(data, net)

external_trainable_variables = [alpha, beta, gamma, delta]

model.compile("adam", lr=0.001, external_trainable_variables=external_trainable_variables)

fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([alpha, beta, gamma, delta], period=100, filename=fnamevar)

losshistory, train_state = model.train(iterations=10000, display_every=100,callbacks=[variable])

#losshistory, train_state = model.train(iterations=10000, display_every=100)

dde.utils.external.plot_loss_history(losshistory)
plt.show()

t_array = np.linspace(t_initial, t_final, 100)
pinn_pred = model.predict(t_array.reshape(-1, 1))
x_pinn = pinn_pred[:, 0:1]
y_pinn = pinn_pred[:, 1:2]
plt.plot(t_array, x_pinn, color="green", label=r"$x(t)$ PINNs, prey")
plt.plot(t_array, y_pinn, color="blue", label=r"$y(t)$ PINNs, predator")
plt.legend()
plt.ylabel(r"population")
plt.xlabel(r"$t$")
plt.title("Lotka-Volterra numerical solution using PINNs method")
plt.show()

print(f"Estimated alpha: {alpha.item()}")
print(f"Estimated beta: {beta.item()}")
print(f"Estimated gamma: {gamma.item()}")
print(f"Estimated delta: {delta.item()}")

# Plots
# reopen saved data using callbacks in fnamevar
lines = open(fnamevar, "r").readlines()
# read output data in fnamevar (this line is a long story...)
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

l, c = Chat.shape


plt.plot(range(l), Chat[:, 0], label="alpha", linestyle="-", color="r")
plt.plot(range(l), Chat[:, 1], label="beta", linestyle="-", color="g")
plt.plot(range(l), Chat[:, 2], label="gamma", linestyle="-", color="b")
plt.plot(range(l), Chat[:, 3], label="delta", linestyle="-", color="m")

plt.axhline(y=alpha.item(), color="r", linestyle="--", label="Estimated alpha")
plt.axhline(y=beta.item(), color="g", linestyle="--", label="Estimated beta")
plt.axhline(y=gamma.item(), color="b", linestyle="--", label="Estimated gamma")
plt.axhline(y=delta.item(), color="m", linestyle="--", label="Estimated delta")

plt.legend(loc="upper right", fontsize=10)
plt.ylabel("absolute value")
plt.xlabel("Epoch")
plt.show()
