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
from copy import deepcopy

import gpytorch
import numpy as np
import torch

from .gp import train_gp
from .turbo_1 import Turbo1
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

"""Make a class for TR's. It should have a run counter, min counter(used for 
both min and history)"""
class TRTracker:
    def __init__(self, runNum=0, minList=[float('inf')]):
        self.runNum = runNum
        self.minList = minList

class TurboO(Turbo1):
    """The TuRBO-O algorithm. Generates the initial LHS sample points in the 
    beginning to make sure no TR's have hyper-plane redundancy. Second optimization
    being the screen on which TR's are even selected to create candidates, to
    further optimize runtime

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

    turbo5 = TurboO(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, n_trust_regions=5)
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
        tr_evaluation_percentage,
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
        '''percentage of the tr's that get evaluated per cycle (ranked according to UCB)'''
        self.tr_evaluation_percentage=tr_evaluation_percentage
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
        """Added an arrau of TR's to keep track of them and their run number and min values"""
        self.succtol = 3
        self.failtol = max(5, self.dim)
        self.lhs_points = None
        self.TR_array = [TRTracker() for _ in range(self.n_trust_regions)]
        self.TR_EvalCtr = 0

        # Very basic input checks
        assert n_trust_regions > 1 and isinstance(max_evals, int)
        assert max_evals > n_trust_regions * n_init, "Not enough trust regions to do initial evaluations"
        assert max_evals > batch_size, "Not enough evaluations to do a single batch"

        # Remember the hypers for trust regions we don't sample from
        self.hypers = [{} for _ in range(self.n_trust_regions)]

        # Initialize parameters
        self._start()
    
    def _start(self):
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
        """Select candidates from samples from all selected trust regions this cycle"""
        num_trust_regions = int(self.tr_evaluation_percentage * self.n_trust_regions)
        assert X_cand.shape == (num_trust_regions, self.n_cand, self.dim)
        assert y_cand.shape == (num_trust_regions, self.n_cand, self.batch_size)
        assert X_cand.min() >= 0.0 and X_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))

        X_next = np.zeros((self.batch_size, self.dim))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)
        for k in range(self.batch_size):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (num_trust_regions, self.n_cand))
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            X_next[k, :] = deepcopy(X_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(y_cand[i, j, k])  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return X_next, idx_next    
    
    def optimize(self):
        """Run the full optimization process."""

        """Make an array of lhs sample points for the whole duration of the program (used like a list)"""
        self.lhs_points = latin_hypercube(2*self.n_init*self.n_trust_regions, self.dim)       
        self.lhs_points = from_unit_cube(self.lhs_points, self.lb, self.ub)

        """Put them into the array here"""
        for i in range(self.n_trust_regions):
            TR = self.TR_array[i]
            X_init = self.lhs_points[:self.n_init]
            self.lhs_points = self.lhs_points[self.n_init:]
            fX_init = np.array([[self.f(x)] for x in X_init])

            # Update budget and set as initial data for this TR
            self.X = np.vstack((self.X, X_init))
            self.fX = np.vstack((self.fX, fX_init))
            TR.minList = [fX_init.min()]
            TR.runNum += 1
            self.TR_EvalCtr += 1
            self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
            self.n_evals += self.n_init

            if self.verbose:
                fbest = fX_init.min()
                fbest = float(fbest)
                print(f"TR-{i} starting from: {fbest:.4f}")
                sys.stdout.flush()

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals:

            """"Use UCB and newly-added trackers to select the TR's for 
            generating and selecting candidates"""
            ucb_values = []
            ucb_hashmap = {}
            for i in range(self.n_trust_regions):
                #make a hashmap for ucb and index, make an array of uct values, sort list, and 
                # then use hashmap to then access the best TR's
                """Add exploration weight later"""
                ucb_value = self.TR_array[i].minList[-1] + np.sqrt(self.TR_array[i].runNum / self.TR_EvalCtr)
                ucb_values.append(ucb_value)
                ucb_hashmap[ucb_value] = i

            print("UCB Values")
            print(ucb_values)
            #making an array to hold the best TR's, size based on the percentage variable and TR#
            ucb_values.sort()
            n_tr_to_eval = int(self.tr_evaluation_percentage * self.n_trust_regions)
            tr_to_eval = np.zeros(n_tr_to_eval, dtype=int)

            # Select the TRs with the highest UCB values
            for i in range(n_tr_to_eval):
                tr_index = ucb_hashmap[ucb_values[i]]
                tr_to_eval[i] = tr_index
            
            #print("TR's to evaluate")
            #print(tr_to_eval)
            
            # Generate candidates from each TR
            X_cand = np.zeros((n_tr_to_eval, self.n_cand, self.dim))
            y_cand = np.inf * np.ones((n_tr_to_eval, self.n_cand, self.batch_size))

            '''This loop has runtime cost. Trains gp for all TR's '''
            for i in range(n_tr_to_eval):
                tr_index = tr_to_eval[i]
                #print("TR Index")
                #print(tr_index)
                idx = np.where(self._idx == tr_index)[0]  # Extract all "active" indices

                # Get the points, values the active values
                X = deepcopy(self.X[idx, :])
                X = to_unit_cube(X, self.lb, self.ub)

                # Get the values from the standardized data
                fX = deepcopy(self.fX[idx, 0].ravel())

                # Don't retrain the model if the training data hasn't changed
                n_training_steps = 0 if self.hypers[tr_index] else self.n_training_steps

                # Create new candidates
                X_cand[i, :, :], y_cand[i, :, :], self.hypers[tr_index] = self._create_candidates(
                    X, fX, length=self.length[tr_index], n_training_steps=n_training_steps, hypers=self.hypers[tr_index]
                )

            # Select the next candidates
            X_next, idx_next = self._select_candidates(X_cand, y_cand)
            assert X_next.min() >= 0.0 and X_next.max() <= 1.0

            # Undo the warping
            X_next = from_unit_cube(X_next, self.lb, self.ub)

            # Evaluate batch
            fX_next = np.array([[self.f(x)] for x in X_next])

            """ Only update the TR's that were selected for evaluation"""
            # Update trust regions
            for tr_index in tr_to_eval:
                idx_i = np.where(idx_next == tr_index)[0]
                if len(idx_i) > 0:
                    self.hypers[tr_index] = {}  # Remove model hypers
                    fX_i = fX_next[idx_i]
                    self.TR_array[tr_index].minList.append(fX_i.min())
                    self.TR_array[tr_index].runNum += 1
                    print(f"Min value for TR-{tr_index}: {fX_i.min()}")
                    if self.verbose and fX_i.min() < self.fX.min() - 1e-3 * math.fabs(self.fX.min()):
                        n_evals, fbest = self.n_evals, fX_i.min()
                        print(f"{n_evals}) New best @ TR-{tr_index}: {fbest:.4}")
                        sys.stdout.flush()
                    self._adjust_length(fX_i, tr_index)

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

                    #make new TR object and add it to the array
                    self.TR_array[i] = TRTracker()

                    # Add new points to the trust region
                    X_init = self.lhs_points[:self.n_init]
                    self.lhs_points = self.lhs_points[self.n_init:]
                    fX_init = np.array([[self.f(x)] for x in X_init])
                    
                    # Print progress
                    if self.verbose:
                        n_evals, fbest = self.n_evals, fX_init.min()
                        print(f"{n_evals}) TR-{i} is restarting from: : {fbest:.4}")
                        sys.stdout.flush()

                    # Update budget and set as initial data for this TR
                    self.X = np.vstack((self.X, X_init))
                    self.fX = np.vstack((self.fX, fX_init))
                    self._idx = np.vstack((self._idx, i * np.ones((self.n_init, 1), dtype=int)))
                    self.n_evals += self.n_init
                    self.TR_array[i].minList = [fX_init.min()]
