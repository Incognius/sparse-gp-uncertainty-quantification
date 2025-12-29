import torch
import gpytorch
import numpy as np
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ResidualGPWrapper:
    def __init__(self, num_inducing=500):
        self.num_inducing = num_inducing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

    def get_inducing_points(self):
        """Extract current locations of inducing points."""
        if self.model is not None:
            return self.model.variational_strategy.inducing_points.detach().cpu().numpy()
        return None

    def fit(self, X, y, epochs=100, track_inducing=False):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        inducing_idx = np.random.choice(X.shape[0], self.num_inducing, replace=False)
        inducing_points = X_tensor[inducing_idx, :].clone()

        self.model = SparseGPModel(inducing_points).to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y_tensor.size(0))
        
        history = []
        print(f"Starting GP Training on {self.device}...")
        for i in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            # Capture snapshots for visualization
            if track_inducing and (i % 10 == 0 or i == epochs - 1):
                history.append(self.get_inducing_points())
                
            if (i + 1) % 20 == 0:
                print(f"Epoch {i+1}/{epochs} - Loss: {loss.item():.4f}")
        
        return history

    def predict(self, X):
        self.model.eval()
        self.likelihood.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_tensor))
            return observed_pred.mean.cpu().numpy(), observed_pred.stddev.cpu().numpy()