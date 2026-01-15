import os
import random
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# PyTorch imports
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, lr_scheduler
import torch.optim as optim

# Scikit-learn and other scientific libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from numba import jit, prange, set_num_threads

# Imports for KernelQR
from cvxopt import solvers, matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Plotting imports
import matplotlib.pyplot as plt
import statsmodels.api as sm
from joblib import Parallel, delayed

# =============================================================================
# 0. Utility Functions and Global Settings
# =============================================================================

def set_seed(s=42):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# =============================================================================
# 1. Model Implementations
# =============================================================================

class TorchModelBase(nn.Module):
    """Base class for PyTorch models to handle data scaling and prediction."""
    def __init__(self, X_mean, X_std, y_mean, y_std):
        super().__init__()
        # Register scaling parameters as non-trainable buffers
        self.register_buffer('X_mean', torch.FloatTensor(X_mean))
        self.register_buffer('X_std', torch.FloatTensor(X_std))
        self.register_buffer('y_mean', torch.FloatTensor(y_mean))
        self.register_buffer('y_std', torch.FloatTensor(y_std))

    def predict(self, X_raw):
        """Scales input, predicts, and un-scales output."""
        self.eval()
        with torch.no_grad():
            X_scaled = (torch.FloatTensor(X_raw).to(self.X_mean.device) - self.X_mean) / self.X_std
            pred_scaled = self.forward(X_scaled)
            pred_raw = pred_scaled * self.y_std + self.y_mean
        return pred_raw.cpu().numpy()

class NQNet_arch(TorchModelBase):
    """NQ-Net architecture ensuring non-crossing quantiles by modeling gaps.
       value_head 和 delta_head 共用同一个 feature extractor。
    """
    def __init__(self, n_in, n_out, width_vec,
                 X_mean, X_std, y_mean, y_std, **kwargs):
        super().__init__(X_mean, X_std, y_mean, y_std)

        # Shared feature extractor
        layers = []
        in_dim = n_in
        for out_dim in width_vec:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.feature_extractor = nn.Sequential(*layers)

        # Two heads
        self.value_head = nn.Linear(in_dim, 1)
        self.delta_head = nn.Linear(in_dim, n_out)

        # Activation to ensure gaps are non-negative
        self.gap_activation = lambda x: torch.nn.functional.elu(x) + 1

    def forward(self, x_scaled):
        h = self.feature_extractor(x_scaled)
        value = self.value_head(h)               # (batch, 1)
        gaps_raw = self.delta_head(h)            # (batch, n_out)
        gaps = self.gap_activation(gaps_raw)     # gaps >= 0

        # Cumulatively sum the non-negative gaps to ensure monotonicity
        cumsum_gaps = torch.cumsum(gaps, dim=1)
        # Center the cumulated gaps
        centered_gaps = cumsum_gaps - cumsum_gaps.mean(dim=1, keepdim=True)
        return value + centered_gaps             # (batch, n_out)

class NQNet:
    """Wrapper class for NQNet training and prediction."""
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
    def fit(self, X, y, **kwargs):
        self.model = _fit_torch_model(NQNet_arch, X, y.reshape(-1, 1), self.quantiles, **kwargs)
    def predict(self, X):
        return self.model.predict(X)

class QuantileNet_arch(TorchModelBase):
    """Standard Deep Quantile Regression architecture (VDQR) without non-crossing constraints."""
    def __init__(self, n_in, n_out, width_vec, X_mean, X_std, y_mean, y_std, **kwargs):
        super().__init__(X_mean, X_std, y_mean, y_std)
        layers = []
        in_dim = n_in
        for out_dim in width_vec:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x_scaled):
        # Directly output one value for each quantile
        return self.net(x_scaled)

class QuantileNet:
    """Wrapper for the unconstrained QuantileNet (VDQR)."""
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
    def fit(self, X, y, **kwargs):
        self.model = _fit_torch_model(QuantileNet_arch, X, y.reshape(-1, 1), self.quantiles, **kwargs)
    def predict(self, X):
        return self.model.predict(X)

class DQR_arch(TorchModelBase):
    """DQR architecture from the original paper, using softplus to enforce monotonicity."""
    def __init__(self, n_in, n_out, width_vec, X_mean, X_std, y_mean, y_std, **kwargs):
        super().__init__(X_mean, X_std, y_mean, y_std)
        layers = []; in_dim = n_in
        for out_dim in width_vec:
            layers.append(nn.Linear(in_dim, out_dim)); layers.append(nn.ReLU()); in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_out))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x_scaled):
        fout = self.net(x_scaled)
        # First quantile is predicted directly, subsequent ones are cumulative sums of softplus-activated outputs
        return torch.cat(
            (fout[:, 0].unsqueeze(1),
             fout[:, 0].unsqueeze(1) + torch.cumsum(self.softplus(fout[:, 1:]), dim=1)),
            dim=1
        )

class DQR:
    """Wrapper for the DQR model."""
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
    def fit(self, X, y, **kwargs):
        self.model = _fit_torch_model(DQR_arch, X, y.reshape(-1, 1), self.quantiles, **kwargs)
    def predict(self, X):
        return self.model.predict(X)

def _fit_torch_model(model_class, X, y, quantiles,
                     nepochs=1000, lr=5e-4, batch_size=64, val_pct=0.15,
                     patience=50, verbose=False, desc="Training NN"):
    """A generic training loop for PyTorch quantile regression models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data splitting and scaling
    indices = np.arange(X.shape[0]); np.random.shuffle(indices)
    train_cutoff = int(np.round(len(indices) * (1 - val_pct)))
    train_indices, val_indices = indices[:train_cutoff], indices[train_cutoff:]
    X_train, X_val = X[train_indices], X[val_indices]; y_train, y_val = y[train_indices], y[val_indices]
    X_mean = X_train.mean(axis=0, keepdims=True); X_std = X_train.std(axis=0, keepdims=True); X_std[X_std == 0] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True); y_std = y_train.std(axis=0, keepdims=True)
    
    # Create tensors and data loaders
    X_train_t = torch.FloatTensor((X_train - X_mean) / X_std).to(device)
    y_train_t = torch.FloatTensor((y_train - y_mean) / y_std).to(device)
    X_val_t = torch.FloatTensor((X_val - X_mean) / X_std).to(device)
    y_val_t = torch.FloatTensor((y_val - y_mean) / y_std).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    # Model initialization
    n_features, n_quantiles = X.shape[1], len(quantiles); width_vec = [256, 256]
    model_arch = model_class(n_in=n_features, n_out=n_quantiles, width_vec=width_vec,
                             X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std).to(device)
    optimizer = Adam(model_arch.parameters(), lr=lr); scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    # Quantile loss function
    t_quantiles = torch.FloatTensor(quantiles).to(device)
    def quantile_loss_fn(preds_scaled, target_scaled):
        error = target_scaled - preds_scaled
        return torch.mean(torch.max((t_quantiles - 1) * error, t_quantiles * error))
    
    best_val_loss = float('inf'); num_bad_epochs = 0; best_model_state = model_arch.state_dict()
    
    # Use a quieter progress bar for replications
    use_tqdm = 'replication_idx' not in locals() and 'replication_idx' not in globals()
    epoch_iterator = tqdm(range(nepochs), desc=desc, leave=False, bar_format='{l_bar}{bar:20}{r_bar}') if use_tqdm else range(nepochs)

    # Training loop
    for epoch in epoch_iterator:
        model_arch.train()
        for xb_scaled, yb_scaled in train_loader:
            optimizer.zero_grad()
            preds_scaled = model_arch(xb_scaled)
            loss = quantile_loss_fn(preds_scaled, yb_scaled)
            if torch.isnan(loss):
                warnings.warn('NaNs encountered in training model loss.')
                break
            loss.backward()
            optimizer.step()
        if torch.isnan(loss):
            break
        
        # Validation and early stopping
        model_arch.eval()
        with torch.no_grad():
            val_preds_scaled = model_arch(X_val_t)
            val_loss = quantile_loss_fn(val_preds_scaled, y_val_t).item()
        
        if use_tqdm:
            epoch_iterator.set_postfix(val_loss=f"{val_loss:.4f}")

        if num_bad_epochs > patience:
            if verbose:
                print(f'Validation loss did not improve for {patience} epochs. Decreasing learning rate.')
            scheduler.step()
            num_bad_epochs = 0
        
        if epoch == 0 or val_loss < best_val_loss:
            if verbose:
                print(f'\t New best val_loss: {val_loss:.6f} on epoch {epoch + 1}')
            best_val_loss = val_loss
            best_model_state = model_arch.state_dict()
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            
        if optimizer.param_groups[0]['lr'] < 1e-6:
            if verbose:
                print("Learning rate is too low. Stopping training.")
            break
            
    model_arch.load_state_dict(best_model_state)
    return model_arch

class ReLU2(nn.Module):
    """ReQU activation: ReLU squared."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return nn.functional.relu(x).pow(2)
    
class DQRP_arch(nn.Module):
    """DQRP architecture that takes quantiles as input."""
    def __init__(self, X_means, X_stds, y_mean, y_std, width_vec=None, activation='ReQU'):
        super(DQRP_arch, self).__init__()
        self.register_buffer('X_means', torch.FloatTensor(X_means))
        self.register_buffer('X_stds', torch.FloatTensor(X_stds))
        self.register_buffer('y_mean', torch.FloatTensor(y_mean))
        self.register_buffer('y_std', torch.FloatTensor(y_std))
        self.n_in = X_means.shape[1]+1
        self.y_dim = 1 if len(y_mean.shape) == 1 else y_mean.shape[1]
        self.width_vec = width_vec if width_vec is not None else [self.n_in, 200, 200, 1]
        self.activation = activation

        modules = []
        if self.activation.lower() =='requ':
            for i in range(len(self.width_vec) - 2):
                modules.append(nn.Sequential(nn.Linear(self.width_vec[i],self.width_vec[i+1]), ReLU2()))
        elif self.activation.lower() =='relu':
            for i in range(len(self.width_vec) - 2):
                modules.append(nn.Sequential(nn.Linear(self.width_vec[i],self.width_vec[i+1]), nn.ReLU()))
        else:
            raise ValueError("Activation must be 'requ' or 'relu'")

        self.net = nn.Sequential(*modules, nn.Linear(self.width_vec[-2], self.width_vec[-1]))

    def forward(self, x, u):
        z = torch.cat((x, u), dim=1)
        output = self.net(z)
        return output
    
    def predict(self, X, quantiles):
        self.eval()
        self.zero_grad()
        tX = autograd.Variable(torch.FloatTensor((X - self.X_means.cpu().numpy()) / self.X_stds.cpu().numpy()), requires_grad=False)
        size = X.shape[0]
        preds = torch.zeros([size, len(quantiles)])
        
        quantiles_t = torch.FloatTensor(quantiles) if not isinstance(quantiles, torch.Tensor) else quantiles
            
        for t in range(len(quantiles_t)):
            u_t = quantiles_t[t].repeat(size, 1).float()
            z = torch.cat((tX, u_t), dim=1)
            preds[:, t] = self.net(z).detach().squeeze() * self.y_std.cpu() + self.y_mean.cpu()
        return preds.numpy()

def fit_quantiles(X, y, width_vec=[256,256,256], activation='ReQU', penalty=None,
                  nepochs=100, val_pct=0.25,
                  batch_size=None, target_batch_pct=0.01,
                  min_batch_size=20, max_batch_size=100,
                  verbose=False, lr=1e-1, patience=5,
                  init_model=None, splits=None, 
                  clip_gradients=False, clip_value=5, desc="Training DQRP", **kwargs):
    """Training loop specifically for DQRP, which includes a monotonicity penalty."""
    if penalty is None:
        penalty = np.log(X.shape[0])
    if batch_size is None:
        batch_size = min(
            X.shape[0],
            max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct))))
        )

    Xmean = X.mean(axis=0, keepdims=True); Xstd = X.std(axis=0, keepdims=True); Xstd[Xstd == 0] = 1
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)
    
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int); np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices, validate_indices = indices[:train_cutoff], indices[train_cutoff:]
    else:
        train_indices, validate_indices = splits
        
    train_loader = DataLoader(TensorDataset(tX[train_indices], tY[train_indices]), batch_size=batch_size, shuffle=True)
    X_valid, Y_valid = tX[validate_indices], tY[validate_indices]
    
    model = DQRP_arch(Xmean, Xstd, ymean, ystd, width_vec=width_vec, activation=activation) if init_model is None else init_model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    best_loss = float('inf'); num_bad_epochs = 0

    def quantile_loss(yhat, y_true, u):
        z = y_true - yhat
        return torch.max(u * z, (u - 1) * z)

    use_tqdm = 'replication_idx' not in locals() and 'replication_idx' not in globals()
    epoch_iterator = tqdm(range(nepochs), desc=desc, leave=False, bar_format='{l_bar}{bar:20}{r_bar}') if use_tqdm else range(nepochs)

    for epoch in epoch_iterator:
        model.train()
        for batch_X, batch_Y in train_loader:
            batch_u = autograd.Variable(torch.rand(batch_Y.shape), requires_grad=True)
            batch_yhat = model(batch_X, batch_u)
            
            # Standard quantile loss
            loss = quantile_loss(batch_yhat, batch_Y.view_as(batch_yhat), batch_u.view_as(batch_yhat)).mean()
            
            # Gradient penalty for monotonicity
            grads = autograd.grad(outputs=batch_yhat, inputs=batch_u, grad_outputs=torch.ones_like(batch_yhat),
                                  retain_graph=True, create_graph=True, only_inputs=True)[0]
            loss_grad = penalty * torch.max(-grads, torch.zeros_like(grads)).mean()
            
            total_loss = loss + loss_grad
            optimizer.zero_grad()
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            total_loss.backward()
            optimizer.step()
            
            if np.isnan(total_loss.item()): 
                warnings.warn('NaNs encountered in training model.')
                break
        if np.isnan(total_loss.item()):
            break
        
        with torch.no_grad():
            model.eval()
            u_valid = torch.rand(Y_valid.shape)
            yhat = model(X_valid, u_valid)
            val_loss = quantile_loss(yhat, Y_valid.view_as(yhat), u_valid.view_as(yhat)).mean().item()

        if use_tqdm:
            epoch_iterator.set_postfix(val_loss=f"{val_loss:.4f}")

        if num_bad_epochs > patience:
            scheduler.step()
            num_bad_epochs = 0
        if epoch == 0 or val_loss <= best_loss:
            best_loss = val_loss
            num_bad_epochs = 0
        else: 
            num_bad_epochs += 1
    return model

class DQRP:
    """High-level wrapper for DQRP to maintain a consistent API."""
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
       
    def fit(self, X, y, width_vec=None, activation='ReQU', lr=0.1, epochs=100, **kwargs):
        if width_vec is None:
            width_vec = [X.shape[1] + 1, 200, 200, 1]
        self.model = fit_quantiles(X, y, width_vec=width_vec, activation=activation, lr=lr, nepochs=epochs, **kwargs)

    def predict(self, X):
        return self.model.predict(X, self.quantiles)

class DQRP_Adapter:
    """Adapter class to handle kwargs correctly for the DQRP model."""
    def __init__(self, quantiles):
        self.quantiles = np.array(quantiles)
        self.model = None

    def fit(self, X, y, **kwargs):
        self.dqr_p_model = DQRP(quantiles=self.quantiles)
        
        # Pop specific arguments for DQRP to avoid passing them down to other functions
        lr = kwargs.pop('dqr_p_lr', 0.01)
        nepochs = kwargs.pop('nepochs', 100)
        desc = kwargs.pop('desc', 'Training DQRP')

        # Fit the model with its specific arguments
        self.dqr_p_model.fit(X, y.reshape(-1, 1), 
                             lr=lr, 
                             epochs=nepochs,
                             desc=desc,
                             **kwargs)
    
    def predict(self, X):
        return self.dqr_p_model.predict(X)

class LinearQR:
    """
    Initializes the Linear Quantile Regression model.
    Uses statsmodels.api.QuantReg as the backend.
    """
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.models = [] # Will be populated with fitted models for each quantile

    def fit(self, X, y, **kwargs):
        """
        Fits a separate model for each quantile in parallel.
        """
        X_with_const = sm.add_constant(X)
        y_flat = y.ravel()

        def _fit_single_quantile(q, X_data, y_data):
            model = sm.QuantReg(y_data, X_data)
            result = model.fit(q=q)
            return result

        print(f"    - Fitting {len(self.quantiles)} quantiles in parallel using statsmodels...")
        self.models = Parallel(n_jobs=-1)(
            delayed(_fit_single_quantile)(q, X_with_const, y_flat) for q in self.quantiles
        )

    def predict(self, X):
        if not self.models:
            raise RuntimeError("The model has not been fitted yet. Call fit() before predict().")

        X_with_const = sm.add_constant(X)
        preds = np.zeros((X.shape[0], len(self.quantiles)))
        for i, model_result in enumerate(self.models):
            preds[:, i] = model_result.predict(X_with_const)
        return preds

@jit(nopython=True, parallel=True)
def find_quant(trainy, train_tree_node_ID, pred_tree_node_ID, qntl):
    """Numba-accelerated function to find quantiles within terminal leaf nodes."""
    npred = pred_tree_node_ID.shape[0]
    out = np.zeros((npred, qntl.size)) * np.nan
    for i in prange(npred):
        # Find training samples that fall into the same leaf nodes as the test sample
        idxs = np.where(train_tree_node_ID == pred_tree_node_ID[i, :])[0]
        if len(idxs) > 0: 
            out[i, :] = np.quantile(trainy[idxs], qntl)
    return out

class QuantileForest:
    """Quantile Regression Forest implementation."""
    def __init__(self, quantiles, nthreads=4, **kwargs):
        self.quantiles = np.array(quantiles)
        self.forest = RandomForestRegressor(n_jobs=nthreads, **kwargs)
        set_num_threads(nthreads)
        self.trainX = None
        self.trainy = None
    def fit(self, X, y, **kwargs): 
        self.trainX, self.trainy = X, y.ravel()
        self.forest.fit(self.trainX, self.trainy)
    def predict(self, X):
        train_leaf_nodes = self.forest.apply(self.trainX)
        pred_leaf_nodes = self.forest.apply(X)
        return find_quant(self.trainy, train_leaf_nodes, pred_leaf_nodes, self.quantiles)

class Kernel:
    """Wrapper for the Kernel Quantile Regression implementation."""
    def __init__(self, quantiles): 
        self.model = KernelQR_impl(quantiles=quantiles, C=1.0, gamma=1.0)
    def fit(self, X, y, **kwargs): 
        self.model.fit(X, y)
    def predict(self, X): 
        return self.model.predict(X)
    
class KernelQR_impl(RegressorMixin, BaseEstimator):
    """Kernel Quantile Regression implementation using a QP solver."""
    def __init__(self, quantiles=0.5, kernel_type="gaussian_rbf", C=1, gamma=1):
        self.C, self.quantiles, self.kernel_type, self.gamma = C, quantiles, kernel_type, gamma
    def kernel(self, X, Y): 
        return Matern(length_scale=self.gamma, nu=np.inf)(X, Y) if self.kernel_type == "gaussian_rbf" else rbf_kernel(X, Y, gamma=1 / self.gamma)
    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False)
        self.X_ = X
        self.y_ = y.ravel()
        quantiles = np.array([self.quantiles]) if np.isscalar(self.quantiles) else np.array(self.quantiles).flatten()
        K = self.kernel(self.X_, self.X_)
        Kmat, r, A, b, n = matrix(K), matrix(self.y_) * 1.0, matrix(np.ones(self.y_.size)).T, matrix(0.0), self.y_.size
        solvers.options['show_progress'] = False
        self.a_list, self.b_list = [], []
        for quantile in quantiles:
            G = matrix(np.vstack([np.eye(n), -np.eye(n)]))
            h = matrix(np.hstack([self.C * quantile * np.ones(n), self.C * (1 - quantile) * np.ones(n)]))
            sol = solvers.qp(P=Kmat, q=-r, G=G, h=h, A=A, b=b)
            a = np.array(sol["x"]).flatten()
            sv_indices = np.where((a > 1e-5) & (a < (self.C * quantile) - 1e-5))[0]
            offshift = sv_indices[0] if len(sv_indices) > 0 else np.argmin(np.abs(a - (self.C * quantile / 2)))
            bias = self.y_[offshift] - a.T @ K[:, offshift]
            self.a_list.append(a)
            self.b_list.append(bias)
        self.quantiles_ = quantiles
        return self
    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_', 'a_list', 'b_list', 'quantiles_'])
        X = check_array(X)
        K = self.kernel(self.X_, X)
        preds = [a.T @ K + b for a, b in zip(self.a_list, self.b_list)]
        return np.vstack(preds).T

# =============================================================================
# 3. Helper for crossing and (optional) plotting
# =============================================================================

def crossing_stats(pred_q):
    """
    pred_q: (n_sample, n_quantile)
    Returns:
        ncr: non-crossing rate
        avg_viol_per_sample: average number of (adjacent τ) decreases per sample
        max_violation: the minimum of all increments (negative value indicates crossing depth)
    """
    diffs = np.diff(pred_q, axis=1)  # (n, K-1)
    ncr = np.mean(np.all(diffs >= 0.0, axis=1))
    violations = diffs < 0.0
    avg_viol_per_sample = violations.sum(axis=1).mean()
    max_violation = diffs.min()
    return ncr, avg_viol_per_sample, max_violation


def plot_quantile_curves(pred_q, quantiles, title, num_examples=5):
    """Randomly sample several test examples and plot Q(τ|X_i) curves for visual inspection of crossings."""
    n_test, K = pred_q.shape
    idxs = np.random.choice(n_test, size=min(num_examples, n_test), replace=False)
    plt.figure(figsize=(8, 5))
    for i in idxs:
        plt.plot(quantiles, pred_q[i, :], marker='o', alpha=0.7)
    plt.xlabel(r'Quantile ($\tau$)')
    plt.ylabel('Predicted Y')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. Main Experiment Loop (Refactored for Replications and Detailed Metrics)
# =============================================================================

def run_single_experiment(cfg, replication_idx=0, show_plot=False):
    """
    Runs a single full experiment for a given seed.
    Returns:
        1. summary_df: Pandas DataFrame with aggregated metrics (ATE, NCR, etc.)
        2. all_preds: Dictionary of raw predictions
        3. quantile_details_list: List of dictionaries containing metrics for EACH quantile
    """
    set_seed(cfg.seed)
    
    # Suppress verbose tqdm progress bars inside replications
    global replication_id 
    replication_id = replication_idx

    try:
        data_df = pd.read_csv(cfg.data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{cfg.data_path}'.")
        return None, None, None
        
    y_col_name, t_col_name = data_df.columns[0], data_df.columns[1]
    y = data_df[y_col_name].values
    t = data_df[t_col_name].values
    X = data_df.drop(columns=[y_col_name, t_col_name]).values

    ALL_QUANTILES = np.round(np.arange(0.05, 1.0, 0.05), 2)
    
    model_factory = {
        'VDQR': QuantileNet, 'NQNet': NQNet, 'DQR': DQR, 'DQRP': DQRP_Adapter, 
        'Linear': LinearQR, 'Forest': QuantileForest, 'Kernel': Kernel
    }
    paper_order = ['Linear','VDQR', 'NQNet', 'DQR', 'DQRP', 'Kernel','Forest' ]
    
    all_preds = {} # Store all raw predictions for possible plotting
    
    # DataFrame for overall summary
    summary_df = pd.DataFrame(columns=[
        'ATE', 'SD_CATE', 'NCR', 'Avg_Violations', 'Max_Violation',
        'PI_Width', 'Coverage', 'Time (s)'
    ])
    
    # List to store detailed metrics for each quantile
    quantile_details_list = []

    # Data splitting
    indices = np.arange(X.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size, random_state=cfg.seed)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    t_train, t_test = t[train_indices], t[test_indices]
    
    # Split by treatment group for training
    X0_train, y0_train = X_train[t_train == 0], y_train[t_train == 0]
    X1_train, y1_train = X_train[t_train == 1], y_train[t_train == 1]
    
    # Prepare output directory
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # Main loop over models
    for name in paper_order:
        if name not in model_factory:
            continue
        
        # Skip Kernel method if dataset is too large
        if name == 'Kernel' and (X0_train.shape[0] > cfg.kernel_max_samples or X1_train.shape[0] > cfg.kernel_max_samples):
            print(f"--- [Seed {cfg.seed}] Skipping {name}: Training set size too large.")
            continue
        
        # Start timing
        start_time = time.time()
            
        ModelClass = model_factory[name]
        print(f"--- [Seed {cfg.seed}] Running: {name} ...")
        
        # Instantiate models for control and treated groups
        model0 = ModelClass(quantiles=ALL_QUANTILES)
        model1 = ModelClass(quantiles=ALL_QUANTILES)
        
        # Set model-specific fitting parameters
        fit_params = {}
        desc_t0 = f"Training {name} (T=0, Seed {cfg.seed})"
        desc_t1 = f"Training {name} (T=1, Seed {cfg.seed})"

        if name in ['NQNet', 'VDQR', 'DQR']:
            fit_params = {'nepochs': cfg.nn_epochs, 'lr': cfg.nn_lr, 'patience': cfg.nn_patience,
                          'batch_size': cfg.nn_batch_size, 'val_pct': cfg.nn_val_pct, 'desc': desc_t0}
        elif name == 'DQRP':
            fit_params = {'nepochs': cfg.nn_epochs, 'dqr_p_lr': cfg.dqr_p_lr, 'desc': desc_t0}

        # Fit models
        model0.fit(X0_train, y0_train, **fit_params)
        if 'desc' in fit_params:
            fit_params['desc'] = desc_t1
        model1.fit(X1_train, y1_train, **fit_params)
        
        # Predict on the test set
        pred_q0 = model0.predict(X_test)
        pred_q1 = model1.predict(X_test)
        
        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Store predictions for visualization
        all_preds[name] = (pred_q0, pred_q1)
        
        # --- Calculate Overall Metrics ---
        est_cate = np.mean(pred_q1 - pred_q0, axis=1)
        est_ate = np.mean(est_cate)
        sd_cate = np.std(est_cate)
        
        # Crossing statistics
        ncr_0, avg_viol_0, max_viol_0 = crossing_stats(pred_q0)
        ncr_1, avg_viol_1, max_viol_1 = crossing_stats(pred_q1)
        ncr = (ncr_0 + ncr_1) / 2.0
        avg_viol = (avg_viol_0 + avg_viol_1) / 2.0
        max_viol = min(max_viol_0, max_viol_1)

        # Prediction Interval (PI) Width
        pi_width_0 = pred_q0[:, -1] - pred_q0[:, 0]
        pi_width_1 = pred_q1[:, -1] - pred_q1[:, 0]
        avg_pi_width = np.mean(np.concatenate([pi_width_0, pi_width_1]))
        
        # Factual Coverage
        covered = np.zeros_like(y_test, dtype=float)
        t0_indices = np.where(t_test == 0)[0]
        lower_0, upper_0 = pred_q0[t0_indices, 0], pred_q0[t0_indices, -1]
        covered[t0_indices] = (y_test[t0_indices] >= lower_0) & (y_test[t0_indices] <= upper_0)
        t1_indices = np.where(t_test == 1)[0]
        lower_1, upper_1 = pred_q1[t1_indices, 0], pred_q1[t1_indices, -1]
        covered[t1_indices] = (y_test[t1_indices] >= lower_1) & (y_test[t1_indices] <= upper_1)
        factual_coverage = np.mean(covered)

        # Add results to summary DataFrame
        summary_df.loc[name] = [
            est_ate, sd_cate, ncr, avg_viol, max_viol,
            avg_pi_width, factual_coverage, elapsed_time
        ]

        # --- Calculate Detailed Metrics per Quantile ---
        for k, q_val in enumerate(ALL_QUANTILES):
            diff_k = pred_q1[:, k] - pred_q0[:, k]
            ate_k = np.mean(diff_k)
            sd_cate_k = np.std(diff_k)
            mean_y0_k = np.mean(pred_q0[:, k])
            mean_y1_k = np.mean(pred_q1[:, k])
            
            quantile_details_list.append({
                'Replication': replication_idx,
                'Method': name,
                'Quantile_Index': k,
                'Quantile_Value': q_val,
                'Estimated_ATE': ate_k,
                'Estimated_SD_CATE': sd_cate_k,
                'Mean_Y0': mean_y0_k,
                'Mean_Y1': mean_y1_k
            })

        
        
        if show_plot and name in ['VDQR', 'NQNet', 'DQR', 'DQRP']:
            print(f"    - Plotting random quantile curves for {name} (T=0 and T=1) ...")
            plot_quantile_curves(pred_q0, ALL_QUANTILES,
                                 title=f'{name} T=0: sample quantile curves')
            plot_quantile_curves(pred_q1, ALL_QUANTILES,
                                 title=f'{name} T=1: sample quantile curves')

        # Print results
        print(f"    - Done in: {elapsed_time:.2f} seconds")
        print(f"    - Metrics: ATE={est_ate:.4f}, SD_CATE={sd_cate:.4f}, "
              f"NCR={ncr:.4f}, PI_Width={avg_pi_width:.4f}, Coverage={factual_coverage:.4f}, "
              f"AvgViol={avg_viol:.3f}, MaxViol={max_viol:.4f}")

    return summary_df, all_preds, quantile_details_list

# =============================================================================
# 5. Replications
# =============================================================================

def run_replications(cfg):
    """
    Runs the experiment multiple times with different seeds and aggregates the results.
    """
    all_summary_results = []
    all_quantile_details = [] # List to collect detailed quantile metrics from all runs
    
    print("\n" + "="*80)
    print(f"Starting Replications: {cfg.num_replications} runs")
    print(f"Dataset: {os.path.basename(cfg.data_path)}")
    print(f"Output Directory: {cfg.output_dir}")
    print("="*80 + "\n")

    initial_seed = cfg.seed

    for i in range(cfg.num_replications):
        current_seed = initial_seed + i
        cfg.seed = current_seed
        print(f"\n===== Running Replication {i+1}/{cfg.num_replications} with Seed {current_seed} =====\n")
        
        show_plot_on_this_run = (i == cfg.num_replications - 1)
        single_run_df, preds, single_run_details = run_single_experiment(
            cfg, replication_idx=i, show_plot=show_plot_on_this_run
        )
        
        if single_run_df is not None:
            all_summary_results.append(single_run_df)
            all_quantile_details.extend(single_run_details)
            print(f"===== Replication {i+1}/{cfg.num_replications} (Seed {current_seed}) Completed =====")

    if not all_summary_results:
        print("No results were generated. Exiting.")
        return

    combined_df = pd.concat(all_summary_results)
    summary_stats = combined_df.groupby(combined_df.index).agg(['mean', 'std'])
    
    metrics_list = ['ATE', 'SD_CATE', 'NCR', 'Avg_Violations', 'Max_Violation',
                    'PI_Width', 'Coverage', 'Time (s)']
    
    formatted_report = pd.DataFrame(index=summary_stats.index)

    for metric in metrics_list:
        mean_col = summary_stats[(metric, 'mean')]
        std_col = summary_stats[(metric, 'std')]
        if metric == 'Time (s)':
            formatted_report[metric] = mean_col.map('{:.2f}'.format) + " (" + std_col.map('{:.2f}'.format) + ")"
        else:
            formatted_report[metric] = mean_col.map('{:.4f}'.format) + " (" + std_col.map('{:.4f}'.format) + ")"
            
    paper_order = ['Linear','VDQR', 'NQNet', 'DQR', 'DQRP', 'Kernel','Forest' ]
    existing_models = [model for model in paper_order if model in formatted_report.index]
    formatted_report = formatted_report.reindex(existing_models)

    print("\n" + "="*80)
    print("           >>> FINAL AGGREGATED RESULTS <<<")
    print("="*80)
    print(f"Summary of {cfg.num_replications} replications (Seeds: {initial_seed} to {initial_seed + cfg.num_replications - 1})")
    print("Format: Mean (Std)")
    print("-" * 80)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150):
        print(formatted_report)

    print("="*80)
    
    sanitized_data_name = os.path.basename(cfg.data_path).replace('.csv', '')
    final_csv_path = os.path.join(cfg.output_dir, f"ACIC_{cfg.num_replications}_runs_summary.csv")
    formatted_report.to_csv(final_csv_path)
    print(f"\nFinal aggregated summary saved to: {final_csv_path}")

    if all_quantile_details:
        detailed_df = pd.DataFrame(all_quantile_details)
        detailed_csv_path = os.path.join(cfg.output_dir, f"ACIC_detailed_quantile_metrics.csv")
        detailed_df.to_csv(detailed_csv_path, index=False)
        print(f"Detailed quantile metrics (for plotting) saved to: {detailed_csv_path}")
        print("Columns in detailed file: ", list(detailed_df.columns))

# =============================================================================
# 6. Script Execution
# =============================================================================
if __name__ == "__main__":
    class Config:
        def __init__(self):
            # --- Main Settings ---
            self.data_path = 'dataset/ACIC2019.csv'
            self.output_dir = './ACIC_results'
            self.seed = 42
            self.num_replications = 20

            # --- Data Split and Model Hyperparameters ---
            self.test_size = 0.25
            self.nn_epochs = 100
            self.nn_patience = 20
            self.nn_batch_size = 128
            self.nn_val_pct = 0.20
            self.nn_lr = 5e-4
            self.dqr_p_lr = 0.001
            self.kernel_max_samples = 1500

    config = Config()
    run_replications(config)
