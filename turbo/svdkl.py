import torch
import torch.nn.functional as F
import torch.nn as nn
import gpytorch
import math

class VariationalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters
        pass

    def forward(self, input):
        weight = self.weight_mu + torch.randn_like(self.weight_mu) * F.softplus(self.weight_rho)
        bias = self.bias_mu + torch.randn_like(self.bias_mu) * F.softplus(self.bias_rho)
        return F.linear(input, weight, bias)

class VariationalNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] 
        for i in range(hidden_dims):
            dims.append(input_dim)
        dims.append(output_dim)
        for i in range(len(dims) - 1):
            self.layers.append(VariationalLayer(dims[i], dims[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing=64, inducing_points=None):
        # Use the simplest form of GP model, exact inference
        inducing_points = torch.randn(num_inducing, 1) if inducing_points is None else inducing_points
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GaussianProcessLayer, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, input_dim, hidden_dims, num_inducing=64, inducing_points=None):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor(input_dim, hidden_dims, 1)
        self.gp_layer = GaussianProcessLayer(num_inducing=num_inducing, inducing_points=inducing_points)
    

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.unsqueeze(-1)
        res = self.gp_layer(features)
        return res