import torch
import torch.nn.functional as F
import torch.nn as nn
import gpytorch
import math
from tqdm import tqdm

class VariationalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #hard coding the dtype to torch.float64 to prevent 32 vs 64 bit errors
        #what systems use 32 bit floats?
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).to(torch.float64))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).to(torch.float64))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).to(torch.float64))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).to(torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters
        pass

    def forward(self, input):
        weight = self.weight_mu + torch.randn_like(self.weight_mu) * F.softplus(self.weight_rho)
        bias = self.bias_mu + torch.randn_like(self.bias_mu) * F.softplus(self.bias_rho)
        return F.linear(input, weight, bias)

class VariationalNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dtype=torch.float64):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] 
        for _ in range(hidden_dims):
            dims.append(input_dim)
        dims.append(output_dim)
        for i in range(len(dims) - 1):
            self.layers.append(VariationalLayer(dims[i], dims[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, num_inducing=64, inducing_points=None, dtype=torch.float64):
        inducing_points = torch.randn(num_inducing, input_dim, dtype=dtype) if inducing_points is None else inducing_points
        
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
        #print("x dtype:", x.dtype) 
        mean = self.mean_module(x)
        #print("mean dtype:", mean.dtype)
        covar = self.covar_module(x)
        #print("covar dtype:", covar.dtype)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, input_dim, hidden_dims, output_dim, num_inducing=64, inducing_points=None, dtype=torch.float64):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor(input_dim, hidden_dims, output_dim, dtype=dtype)
        self.gp_layer = GaussianProcessLayer(output_dim, num_inducing=num_inducing, inducing_points=inducing_points, dtype=dtype)
        self.gp_layer = self.gp_layer.double()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.double()
        res = self.gp_layer(features)
        return res
    
    def get_kernel(self):
        return self.gp_layer.covar_module


'''Turns the dkl model into a kernel for a gp model'''
class DKLKernel(gpytorch.kernels.Kernel):
    def __init__(self, dkl_model):
        super().__init__()
        self.dkl_model = dkl_model
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False):
        if diag:
            return self.dkl_model.get_kernel()(x1, x2, diag=True, last_dim_is_batch=last_dim_is_batch)
        else:
            features1 = self.dkl_model.feature_extractor(x1)
            features2 = self.dkl_model.feature_extractor(x2)
            return self.dkl_model.get_kernel()(features1, features2, diag=False, last_dim_is_batch=last_dim_is_batch)

    def get_lengthscale(self):
        return self.dkl_model.gp_layer.covar_module.base_kernel.lengthscale
    
'''class for the gp that uses the dkl kernel'''
class CombinedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dkl_model):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = DKLKernel(dkl_model)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.matmul(covar_x.transpose(-1, -2))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(train_loader, model, likelihood, optimizer, mll, dtype):
    model.train()
    likelihood.train()
    
    minibatch_iter = tqdm(train_loader, disable=True)
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data = data.to(dtype)
            target = target.to(dtype)
            #print("data dtype:", data.dtype)
            #print("target dtype:", target.dtype)
            print("data shape:", data.shape)    

            optimizer.zero_grad()
            
            output = model(data)
            print("output", output)
            print("output mean:", output.mean)
            print("output variance:", output.variance)
            print("Target:", target)
            #print("output dtype:", output.mean.dtype)
            loss = -mll(output, target.to(dtype))
            loss = loss.mean()
            loss.backward()
            optimizer.step()

def test_regression(test_loader, model, likelihood, dtype):
    model.eval()
    likelihood.eval()

    mse = 0
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # Get predictive distribution
            output = likelihood(model(data.to(dtype)))
            # Get mean prediction
            pred_mean = output.mean
            # Calculate MSE for this batch
            mse += ((pred_mean - target.to(dtype)) ** 2).mean().item()
    
    # Calculate average MSE across all batches
    mse /= len(test_loader)
    #print(f'Test set: Average MSE: {mse:.4f}')
    return mse
