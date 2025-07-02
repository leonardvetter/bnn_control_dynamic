import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#Import the classes
from control_system_sim import SMD_System_Sim
from control_system_sim import PID_Controller

# Try to import torchbnn, install if missing (may fail if no internet)
try:
    import torchbnn as bnn
except ImportError:
    raise ImportError("torchbnn is required but not installed in this environment.")


# ---------- Initialize System and Controller ----------
system = SMD_System_Sim(1.0, 1.0, 0.1)  #Initialize the system for given constants
pid   = PID_Controller(kp=10.0, ki=2.0, kd=2, max_signal=100.0) #Initialize the pid controller for given constants (Would later be interesting to vary)

# ---------- Generate an artificial dataset from the simulation ----------
# X = (x_t,u_t) is the position and control input at time t
# Y = x_{t+1} is the position 
X_list, Y_list = [], []
rng = np.random.default_rng(42)
for step in np.linspace(0, 100, 50):         # 50 different step sizes, where we do a step up at time 10 and then step down at time 30
    t, u, x, r = pid.run_simulation(system, duration=50.0, commands= [[10,step],[30,-step]])
    # create (x_t, u_t) -> x_{t+1}
    X_list.append(np.stack([x[:-1], u[:-1]], axis=1))
    Y_list.append(x[1:])

X = np.concatenate(X_list, axis=0)
Y = np.concatenate(Y_list, axis=0)

# train/test split of 80/20 and permute
perm = rng.permutation(len(X))
train_size = int(0.8 * len(X))
train_idx, test_idx = perm[:train_size], perm[train_size:]

X_train, Y_train = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(Y[train_idx], dtype=torch.float32).unsqueeze(1)
X_test,  Y_test  = torch.tensor(X[test_idx], dtype=torch.float32),  torch.tensor(Y[test_idx], dtype=torch.float32).unsqueeze(1)

# ---------- Define Bayesian NN ----------
class BNN(nn.Module):
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=in_dim,out_features = hidden),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=hidden,out_features = hidden),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,in_features=hidden,out_features = 1)
        )

    def forward(self, x):
        return self.net(x)


bnn_model = BNN()
optimizer = torch.optim.Adam(bnn_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ---------- Training ----------
epochs = 20000
#epochs = 500
bnn_model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    preds = bnn_model(X_train)
    loss = loss_fn(preds, Y_train)
    # add KL term for Bayesian layers
    kl = sum(m.kl_loss() for m in bnn_model.modules() if hasattr(m, "kl_loss"))
    total_loss = loss + 1e-6 * kl
    total_loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} - MSE: {loss.item():.4e}")

# ---------- Evaluate with uncertainty ----------
bnn_model.eval()
with torch.no_grad():
    # simulate one unseen step = 0.8 m
    t_eval, u_eval, x_eval, r = pid.run_simulation(system, duration=100.0, commands= [[0,70], [20,30], [40,50],[60,10],[80,90]])
    inputs = torch.tensor(np.stack([x_eval[:-1], u_eval[:-1]], axis=1), dtype=torch.float32)

    # Monte Carlo sampling
    mc_samples = 100
    preds_samples = torch.stack([bnn_model(inputs).squeeze(1) for _ in range(mc_samples)], dim=0)
    mean_pred = preds_samples.mean(0).numpy()
    std_pred  = preds_samples.std(0).numpy()

# ---------- Plot ----------
plt.figure()
time = t_eval[1:]  # x_{t+1} timestamps
plt.plot(time, x_eval[1:], label="true position")
plt.plot(time, mean_pred, label="BNN mean prediction")
plt.fill_between(time, mean_pred-2*std_pred, mean_pred+2*std_pred, alpha=0.3, label="±2σ interval")
plt.xlabel("time [s]")
plt.ylabel("position")
plt.legend()
plt.tight_layout()
plt.show()