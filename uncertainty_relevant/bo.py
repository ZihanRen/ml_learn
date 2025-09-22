#%%
import numpy as np
import scipy.stats
import torch
from botorch import acquisition, sampling
from botorch.acquisition.objective import ScalarizedPosteriorTransform, ConstrainedMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm.notebook import trange

# device should be mac m1
device = "cpu"
torch.set_default_device(device)
# Set default tensor type to float32 for MPS compatibility
torch.set_default_dtype(torch.float32)

def array(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
# %%
default_a = 15.0 
default_b = 2.0 
default_noise = 0.2

def example_1d_function(
    x: torch.Tensor, a: float = default_a, b: float = default_b
) -> torch.Tensor:
    """Ground truth function.""" 
    return torch.sin(a * x) + b * x**2

def example_noisy_1d_function(
    x: torch.Tensor, a: float = default_a, b: float = default_b, noise: float = default_noise
) -> torch.Tensor:
    """Noisy ground truth observations."""
    return example_1d_function(x, a, b) + noise * torch.randn_like(x)



x_init = torch.rand(2, dtype=torch.float32)
y_init = example_noisy_1d_function(x_init)

# Create a grid of x values for evaluation
x_grid = torch.linspace(0, 1, 100, dtype=torch.float32)

gp = SingleTaskGP(x_init[:, None], y_init[:, None])
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

gp_posterior = gp.posterior(x_grid)
mean = gp_posterior.mean.detach()[:, 0]
std = gp_posterior.variance.detach()[:, 0] ** 0.5

#%% acquision function
ucb = acquisition.UpperConfidenceBound(gp, beta=4)
# We'll explain this indexing later!
ucb_value = ucb(x_grid[:, None, None])

i_max = ucb_value.argmax()
x_candidate = x_grid[i_max]
value_candidate = ucb_value[i_max]
# %%
