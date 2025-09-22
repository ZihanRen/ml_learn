import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# Generate toy PDE data
# --------------------------
def generate_data(n_samples=1000, n_points=32):
    xs = torch.linspace(0, 1, n_points).unsqueeze(0).repeat(n_samples, 1)
    # forcing function f(x) = random combination of sinusoids
    coeffs = torch.randn(n_samples, 3)
    f = (coeffs[:,0:1]*torch.sin(2*3.14*xs) +
         coeffs[:,1:2]*torch.sin(4*3.14*xs) +
         coeffs[:,2:3]*torch.sin(6*3.14*xs))
    # "solution" u(x) = smoothed version of f(x) (imitating PDE)
    u = torch.cumsum(f, dim=1) / n_points
    return f, u

n_points = 32
f_train, u_train = generate_data(2000, n_points)
f_test, u_test = generate_data(200, n_points)

# --------------------------
# Standard NN (vector → vector)
# --------------------------
class StandardNN(nn.Module):
    def __init__(self, n_points):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_points, 64),
            nn.ReLU(),
            nn.Linear(64, n_points)
        )
    def forward(self, x):
        return self.net(x)

# --------------------------
# Neural Operator (Fourier layer style)
# --------------------------
class SimpleFNO1D(nn.Module):
    def __init__(self, modes=8, width=32):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(1, width)  # lift to high-dim

        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, n_points]
        x = x.unsqueeze(-1)   # [batch, n_points, 1]
        x = self.fc0(x)       # [batch, n_points, width]
        x = x.permute(0,2,1)  # [batch, width, n_points]
        x = torch.relu(self.conv(x))  # convolution = local integral operator
        x = x.permute(0,2,1)  # [batch, n_points, width]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)   # [batch, n_points]
        return x

# --------------------------
# Training loop
# --------------------------
def train(model, f_train, u_train, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model(f_train)
        loss = loss_fn(out, u_train)
        loss.backward()
        optimizer.step()
        if (ep+1) % 5 == 0:
            print(f"Epoch {ep+1}, Loss = {loss.item():.4f}")

# --------------------------
# Run demo
# --------------------------
print("Training Standard NN:")
nn_model = StandardNN(n_points)
train(nn_model, f_train, u_train)

print("\nTraining Neural Operator style:")
no_model = SimpleFNO1D()
train(no_model, f_train, u_train)

# --------------------------
# Test generalization to finer grid (NO should adapt, NN fails)
# --------------------------
f_fine, u_fine = generate_data(1, n_points=64)  # finer discretization
try:
    out_nn = nn_model(f_fine)  # <-- will fail (wrong input size)
except Exception as e:
    print("\nStandard NN failed on finer grid:", e)

out_no = no_model(f_fine)  # works fine
print("\nNeural Operator output shape on finer grid:", out_no.shape)
