###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math
import sys
import os
from copy import deepcopy

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
import numpy as np
import torch
from tqdm import tqdm


from .gp import train_gp
from .turbo_1 import Turbo1
from .utils import from_unit_cube, latin_hypercube, to_unit_cube
from .svdkl import DKLModel, VariationalNet


class TurboDKL(Turbo1):
    """The TuRBO-m algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points *FOR EACH TRUST REGION* (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    n_trust_regions : Number of trust regions
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo5 = TurboM(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, n_trust_regions=5)
        turbo5.optimize()  # Run optimization
        X, fX = turbo5.X, turbo5.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        n_trust_regions,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):
        self.n_trust_regions = n_trust_regions
        super().__init__(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=max_evals,
            batch_size=batch_size,
            verbose=verbose,
            use_ard=use_ard,
            max_cholesky_size=max_cholesky_size,
            n_training_steps=n_training_steps,
            min_cuda=min_cuda,
            device=device,
            dtype=dtype,
        )

        self.succtol = 3
        self.failtol = max(5, self.dim)

        # Very basic input checks
        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough trust regions to do initial evaluations"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.n_trust_regions, dtype=int)
        self.succcount = np.zeros(self.n_trust_regions, dtype=int)
        self.length = self.length_init * np.ones(self.n_trust_regions)

    def _adjust_length(self, fX_next, i):
        assert i >= 0 and i <= self.n_trust_regions - 1

        fX_min = self.fX[self._idx[:, 0] == i, 0].min()  # Target value
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates from samples from all trust regions."""
        assert X_cand.shape == (self.n_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (self.n_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (self.n_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next, idx_next
    
    

    def optimize(self):
        """Run the full optimization process."""
        # Create initial points for each TR
        for i in range(self.n_trust_regions):
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.array([[self.f(x)] for x in X_init])

            # Update budget and set as initial data for this TR
            self.X = np.vstack((self.X, X_init))
            self.fX = np.vstack((self.fX, fX_init))
            self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
            self.n_evals += self.n_init

            if self.verbose:
                fbest = fX_init.min()
                print(f"TR-{i} starting from: {fbest:.4}")
                sys.stdout.flush()

        """Initialize the DKL model"""
         # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.from_numpy(self.X).float()
        fX_tensor = torch.from_numpy(self.fX).float()
        # Create data loader
        # Split the dataset into training and testing sets
        dataset_size = len(X_tensor)
        indices = list(range(dataset_size))
        split = dataset_size // 2
        train_indices, test_indices = indices[:split], indices[split:]
        # Create data loaders
        train_dataset = TensorDataset(X_tensor[train_indices], fX_tensor[train_indices])
        test_dataset = TensorDataset(X_tensor[test_indices], fX_tensor[test_indices])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
         # have to switch from grid variational layer
        model = DKLModel(VariationalNet, self.dim, grid_bounds=(-10., 10.))
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        n_epochs = 1
        lr = 0.1
        optimizer = SGD([
            {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
            {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
            {'params': model.gp_layer.variational_parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
        #have to figure out where this is used
        scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
        
        '''Define train'''
        def train(train_loader, model, likelihood, optimizer, mll, epoch):
            model.train()
            likelihood.train()
            
            minibatch_iter = tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
            with gpytorch.settings.num_likelihood_samples(8):
                for data, target in minibatch_iter:
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    
                    print("Data shape:", data.shape)
                    print("Target shape:", target.shape)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    print("Output", output)
                    print("output shape:", output.shape)
                    print("target shape:", target.shape)
                    try:
                        loss = -mll(output, target)
                        loss.backward()
                        optimizer.step()
                        minibatch_iter.set_postfix(loss=loss.item())
                    except RuntimeError as e:
                        print("Error in loss calculation:")
                        print(e)
                        break

        '''Define test'''
        def test_regression():
            model.eval()
            likelihood.eval()

            mse = 0
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for data, target in test_loader:
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    # Get predictive distribution
                    output = likelihood(model(data))
                    # Get mean prediction
                    pred_mean = output.mean
                    # Calculate MSE for this batch
                    mse += ((pred_mean - target) ** 2).mean().item()
            
            # Calculate average MSE across all batches
            mse /= len(test_loader)
            print(f'Test set: Average MSE: {mse:.4f}')

            return mse
        # Train the model
        for epoch in range(n_epochs):
            with gpytorch.settings.use_toeplitz(False):
                train(train_loader, model, likelihood, optimizer, mll, epoch)
                #test
            scheduler.step()
            state_dict = model.state_dict()
            likelihood_state_dict = likelihood.state_dict()
            torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_turbo_checkpoint.dat')


        if torch.cuda.is_available():
            model = model.cuda()
            likelihood = likelihood.cuda()
        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals:

            # Generate candidates from each TR
            X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim))
            y_cand = np.inf * np.ones((self.n_trust_regions, self.n_cand, self.batch_size))

            '''This loop has runtime cost. Trains gp for all TR's '''
            for i in range(self.n_trust_regions):
                idx = np.where(self._idx == i)[0]  # Extract all "active" indices

                # Get the points, values the active values
                X = deepcopy(self.X[idx, :])
                X = to_unit_cube(X, self.lb, self.ub)

                # Get the values from the standardized data
                fX = deepcopy(self.fX[idx, 0].ravel())

                # Don't retrain the model if the training data hasn't changed
                n_training_steps = 0 if self.hypers[i] else self.n_training_steps

                # Create new candidates
                X_cand[i, :, :], y_cand[i, :, :], self.hypers[i] = self._create_candidates(
                    X, fX, length=self.length[i], n_training_steps=n_training_steps, hypers=self.hypers[i]
                )

            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            # Undo the warping
            X_next = from_unit_cube(X_next, self.lb, self.ub)

            # Evaluate batch
            fX_next = np.array([[self.f(x)] for x in X_next])

            # Update trust regions
            for i in range(self.n_trust_regions):
                idx_i = np.where(idx_next == i)[0]
                if len(idx_i) > 0:
                    self.hypers[i] = {}  # Remove model hypers
                    fX_i = fX_next[idx_i]

                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min()
                        print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, i)

            # Update budget and append data
            self.n_evals += self.batch_size
            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))
            self._idx = np.vstack((self._idx, deepcopy(idx_next)))

            # Check if any TR needs to be restarted
            for i in range(self.n_trust_regions):
                if self.length[i] < self.length_min:  # Restart trust region if converged
                    idx_i = self._idx[:, 0] == i

                    if self.verbose:
                        n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                        print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                        sys.stdout.flush()

                    # Reset length and counters, remove old data from trust region
                    self.length[i] = self.length_init
                    self.succcount[i] = 0
                    self.failcount[i] = 0
                    self._idx[idx_i, 0] = -1  # Remove points from trust region
                    self.hypers[i] = {}  # Remove model hypers

                    # Create a new initial design
                    X_init = latin_hypercube(self.n_init, self.dim)
                    X_init = from_unit_cube(X_init, self.lb, self.ub)
                    fX_init = np.array([[self.f(x)] for x in X_init])

                    # Print progress
                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min()
                        print(f"{n_evals}) TR-{i} is restarting from: : {fbest:.4}")
                        sys.stdout.flush()

                    # Append data to local history
                    self.X = np.vstack((self.X, X_init))
                    self.fX = np.vstack((self.fX, fX_init))
                    self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                    self.n_evals += self.n_init
