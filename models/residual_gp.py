import torch
import gpytorch
import numpy as np
from sklearn.preprocessing import StandardScaler
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.constraints import Interval

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
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
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=Interval(1e-4, 0.4)
        ).to(self.device)
        self.scaler = StandardScaler()

    def fit(self, X, y, epochs=100, track_inducing=False):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        # Initialize inducing points
        inducing_idx = np.random.choice(X.shape[0], self.num_inducing, replace=False)
        inducing_points = X_tensor[inducing_idx, :].clone()

        self.model = SparseGPModel(inducing_points).to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y_tensor.size(0))

        history = []
        print(f"Training GP on {self.device}...")
        
        for i in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            # Track inducing points for visualization
            if track_inducing and (i % 10 == 0 or i == epochs - 1):
                points = self.model.variational_strategy.inducing_points.detach().cpu().numpy()
                history.append(points)
                
            if (i + 1) % 20 == 0:
                print(f"Epoch {i+1}/{epochs} - Loss: {loss.item():.4f}")
        
        if track_inducing:
            return history

    def predict(self, X):
        self.model.eval()
        self.likelihood.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_tensor))
            return observed_pred.mean.cpu().numpy(), observed_pred.stddev.cpu().numpy()
        

    def save(self, path):
        print(f"Saving GP model to {path}...")
        state = {
            'model_state': self.model.state_dict(),
            'likelihood_state': self.likelihood.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'inducing_points': self.model.variational_strategy.inducing_points
        }
        torch.save(state, path)

    def load(self, path):
        print(f"Loading GP model from {path}...")
        state = torch.load(path, map_location=self.device)
        
        # Reinitialize model with saved inducing points
        inducing_points = state['inducing_points'].to(self.device)
        self.model = SparseGPModel(inducing_points).to(self.device)
        self.model.load_state_dict(state['model_state'])
        
        self.likelihood.load_state_dict(state['likelihood_state'])
        
        # Restore scaler
        self.scaler.mean_ = state['scaler_mean']
        self.scaler.scale_ = state['scaler_scale']
        
        self.model.eval()
        self.likelihood.eval()    