import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Define the PDE ──────────────────────────────────────────────
def burgers_equation(x, y):
    """
    Burgers: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
    x[:, 0] = spatial coordinate
    x[:, 1] = time coordinate
    y        = u(x, t) predicted by the network
    """
    nu = 0.01 / np.pi  # viscosity (standard benchmark value)

    dy_x = dde.grad.jacobian(y, x, i=0, j=0)   # ∂u/∂x
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)   # ∂u/∂t
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)   # ∂²u/∂x²

    # Burgers residual: should be 0 everywhere
    return dy_t + y * dy_x - nu * dy_xx


# ── 2. Define the Domain ───────────────────────────────────────────
# x ∈ [-1, 1],  t ∈ [0, 1]
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# ── 3. Boundary Conditions (u = 0 at x = -1 and x = 1) ────────────
def boundary(x, on_boundary):
    return on_boundary

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary)


# ── 4. Initial Condition (u(x, 0) = -sin(πx)) ─────────────────────
def initial_condition(x):
    return -np.sin(np.pi * x[:, 0:1])

ic = dde.icbc.IC(geomtime, initial_condition, lambda x, on_initial: on_initial)


# ── 5. Build the Problem ───────────────────────────────────────────
data = dde.data.TimePDE(
    geomtime,
    burgers_equation,
    [bc, ic],
    num_domain=2000,       # collocation points inside domain
    num_boundary=100,      # points on spatial boundaries
    num_initial=200,       # points at t=0
    num_test=1000
)


# ── 6. Define the Neural Network ───────────────────────────────────
# Input: [x, t]  →  Output: u(x, t)
net = dde.nn.FNN(
    layer_sizes=[2, 64, 64, 64, 1],   # 3 hidden layers, 64 neurons each
    activation="tanh",                 # tanh works best for PINNs
    kernel_initializer="Glorot normal"
)


# ── 7. Create Model and Train ──────────────────────────────────────
model = dde.Model(data, net)

# Phase 1: Quick Adam optimization
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=10000)

# Phase 2: Fine-tune with L-BFGS (second-order optimizer)
model.compile("L-BFGS")
losshistory, train_state = model.train()


# ── 8. Visualize the Solution ──────────────────────────────────────
X = np.linspace(-1, 1, 256)
T = np.linspace(0, 1, 100)
XX, TT = np.meshgrid(X, T)

# Predict u(x,t) over the entire grid
XT = np.column_stack([XX.ravel(), TT.ravel()])
u_pred = model.predict(XT).reshape(100, 256)

plt.figure(figsize=(10, 5))
plt.contourf(XX, TT, u_pred, levels=100, cmap="RdBu_r")
plt.colorbar(label="u(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Burgers' Equation — PINN Solution (DeepXDE)")
plt.show()
