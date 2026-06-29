import os
import random
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
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from numba import jit, prange, set_num_threads
from scipy.stats import norm, t as student_t
import argparse
from multiprocessing import Pool

# Imports for KernelQR
from cvxopt import solvers, matrix
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# =============================================================================
# 0. Utility Functions and Global Settings
# =============================================================================

def set_seed(s=42):
    """Sets the random seed for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# =============================================================================
# 1. Model Implementations
# =============================================================================

# --- Original Torch Models ---

class TorchModelBase(nn.Module):
    def __init__(self, X_mean, X_std, y_mean, y_std):
        super().__init__()
        self.register_buffer('X_mean', torch.FloatTensor(X_mean))
        self.register_buffer('X_std', torch.FloatTensor(X_std))
        self.register_buffer('y_mean', torch.FloatTensor(y_mean))
        self.register_buffer('y_std', torch.FloatTensor(y_std))

    def predict(self, X_raw):
        self.eval()
        with torch.no_grad():
            X_scaled = (torch.FloatTensor(X_raw).to(self.X_mean.device) - self.X_mean) / self.X_std
            pred_scaled = self.forward(X_scaled)
            pred_raw = pred_scaled * self.y_std + self.y_mean
        return pred_raw.cpu().numpy()

class NQNet_arch(TorchModelBase):
    def __init__(self, n_in, n_out, width_vec, X_mean, X_std, y_mean, y_std, activation='ELU', **kwargs):
        super().__init__(X_mean, X_std, y_mean, y_std)
        layers = []
        in_dim = n_in
        for out_dim in width_vec:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        
        self.value_net = nn.Sequential(*layers, nn.Linear(in_dim, 1))
        self.delta_net = nn.Sequential(*layers, nn.Linear(in_dim, n_out))
        
        self.gap_activation = lambda x: torch.nn.functional.elu(x) + 1

    def forward(self, x_scaled):
        value = self.value_net(x_scaled)
        gaps = self.gap_activation(self.delta_net(x_scaled))
        cumsum_gaps = torch.cumsum(gaps, dim=1)
        centered_gaps = cumsum_gaps - cumsum_gaps.mean(dim=1, keepdim=True)
        return value + centered_gaps

class NQNet:
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
    def fit(self, X, y, **kwargs):
        self.model = _fit_torch_model(NQNet_arch, X, y.reshape(-1, 1), self.quantiles, **kwargs)
    def predict(self, X):
        return self.model.predict(X)

class QuantileNet_arch(TorchModelBase):
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
        return self.net(x_scaled)

class QuantileNet:
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
    def fit(self, X, y, **kwargs):
        self.model = _fit_torch_model(QuantileNet_arch, X, y.reshape(-1, 1), self.quantiles, **kwargs)
    def predict(self, X):
        return self.model.predict(X)

class DQR_arch(TorchModelBase):
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
        return torch.cat((fout[:, 0].unsqueeze(1), fout[:, 0].unsqueeze(1) + torch.cumsum(self.softplus(fout[:, 1:]), dim=1)), dim=1)

class DQR:
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
    def fit(self, X, y, **kwargs):
        self.model = _fit_torch_model(DQR_arch, X, y.reshape(-1, 1), self.quantiles, **kwargs)
    def predict(self, X):
        return self.model.predict(X)

def _fit_torch_model(model_class, X, y, quantiles,
                     nepochs=1000, lr=5e-4, batch_size=64, val_pct=0.15,
                     patience=50, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = np.arange(X.shape[0]); np.random.shuffle(indices)
    train_cutoff = int(np.round(len(indices) * (1 - val_pct)))
    train_indices, val_indices = indices[:train_cutoff], indices[train_cutoff:]
    X_train, X_val = X[train_indices], X[val_indices]; y_train, y_val = y[train_indices], y[val_indices]
    X_mean = X_train.mean(axis=0, keepdims=True); X_std = X_train.std(axis=0, keepdims=True); X_std[X_std == 0] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True); y_std = y_train.std(axis=0, keepdims=True)
    X_train_t = torch.FloatTensor((X_train - X_mean) / X_std).to(device)
    y_train_t = torch.FloatTensor((y_train - y_mean) / y_std).to(device)
    X_val_t = torch.FloatTensor((X_val - X_mean) / X_std).to(device)
    y_val_t = torch.FloatTensor((y_val - y_mean) / y_std).to(device)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    n_features, n_quantiles = X.shape[1], len(quantiles); width_vec = [128, 128]
    model_arch = model_class(n_in=n_features, n_out=n_quantiles, width_vec=width_vec,
                             X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std).to(device)
    optimizer = Adam(model_arch.parameters(), lr=lr); scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    t_quantiles = torch.FloatTensor(quantiles).to(device)
    def quantile_loss_fn(preds_scaled, target_scaled):
        error = target_scaled - preds_scaled
        return torch.mean(torch.max((t_quantiles - 1) * error, t_quantiles * error))
    best_val_loss = float('inf'); num_bad_epochs = 0; best_model_state = model_arch.state_dict()
    for epoch in range(nepochs):
        model_arch.train()
        for xb_scaled, yb_scaled in train_loader:
            optimizer.zero_grad(); preds_scaled = model_arch(xb_scaled); loss = quantile_loss_fn(preds_scaled, yb_scaled)
            if torch.isnan(loss): warnings.warn('NaNs encountered in training model loss.'); break
            loss.backward(); optimizer.step()
        if torch.isnan(loss): break
        model_arch.eval()
        with torch.no_grad(): val_preds_scaled = model_arch(X_val_t); val_loss = quantile_loss_fn(val_preds_scaled, y_val_t).item()
        if num_bad_epochs > patience:
            if verbose: print(f'Validation loss did not improve for {patience} epochs. Decreasing learning rate.')
            scheduler.step(); num_bad_epochs = 0
        if epoch == 0 or val_loss < best_val_loss:
            if verbose: print(f'\t New best val_loss: {val_loss:.6f} on epoch {epoch + 1}')
            best_val_loss = val_loss; best_model_state = model_arch.state_dict(); num_bad_epochs = 0
        else:
            num_bad_epochs += 1
        if optimizer.param_groups[0]['lr'] < 1e-6:
            if verbose: print("Learning rate is too low. Stopping training."); break
    model_arch.load_state_dict(best_model_state); return model_arch

# --- DQRP Implementation ---

class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return nn.functional.relu(x).pow(2)
    
class DQRP_arch(nn.Module):
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
        
        if not isinstance(quantiles, torch.Tensor):
            quantiles_t = torch.FloatTensor(quantiles)
        else:
            quantiles_t = quantiles
            
        for t in range(len(quantiles_t)):
            u_t = quantiles_t[t].repeat(size, 1).float()
            z = torch.cat((tX, u_t), dim=1)
            preds[:, t] = self.net(z).detach().squeeze() * self.y_std.cpu() + self.y_mean.cpu()
        return preds.numpy()


class DQRP:
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.model = None
       
    def fit(self, X, y, width_vec=None, activation='ReQU', lr=0.1, epochs=100, **kwargs):
        if width_vec is None:
            width_vec = [X.shape[1] + 1, 200, 200, 1]
        self.model = fit_quantiles(X, y, width_vec=width_vec, activation=activation, lr=lr, nepochs=epochs, **kwargs)

    def predict(self, X, quantiles):
        return self.model.predict(X, quantiles)


def fit_quantiles(X, y, width_vec=[256,256,256], activation='ReQU', penalty=None,
                    nepochs=100, val_pct=0.25,
                    batch_size=None, target_batch_pct=0.01,
                    min_batch_size=20, max_batch_size=100,
                    verbose=False, lr=1e-1, weight_decay=0.0, patience=5,
                    init_model=None, splits=None, 
                    clip_gradients=False, clip_value=5, **kwargs):
    if penalty is None: penalty = np.log(X.shape[0])
    if batch_size is None:
        batch_size = min(X.shape[0], max(min_batch_size, min(max_batch_size, int(np.round(X.shape[0]*target_batch_pct)))))

    Xmean = X.mean(axis=0, keepdims=True); Xstd = X.std(axis=0, keepdims=True); Xstd[Xstd == 0] = 1
    ymean, ystd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    tX = autograd.Variable(torch.FloatTensor((X - Xmean) / Xstd), requires_grad=False)
    tY = autograd.Variable(torch.FloatTensor((y - ymean) / ystd), requires_grad=False)
    
    if splits is None:
        indices = np.arange(X.shape[0], dtype=int); np.random.shuffle(indices)
        train_cutoff = int(np.round(len(indices)*(1-val_pct)))
        train_indices, validate_indices = indices[:train_cutoff], indices[train_cutoff:]
    else: train_indices, validate_indices = splits
        
    train_dataset = TensorDataset(tX[train_indices], tY[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_valid, Y_valid = tX[validate_indices], tY[validate_indices]
    
    model = DQRP_arch(Xmean, Xstd, ymean, ystd, width_vec=width_vec, activation=activation) if init_model is None else init_model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    #train_losses, val_losses = np.zeros(nepochs), np.zeros(nepochs), 
    best_loss = float('inf'); num_bad_epochs = 0

    def quantile_loss(yhat, y_true, u):
        z = y_true - yhat
        return torch.max(u * z, (u - 1) * z)

    for epoch in range(nepochs):
        train_loss = 0.0; model.train()
        for batch_X, batch_Y in train_loader:
            batch_u = autograd.Variable(torch.rand(batch_Y.shape), requires_grad=True)
            batch_yhat = model(batch_X, batch_u)
            loss = quantile_loss(batch_yhat, batch_Y.view_as(batch_yhat), batch_u.view_as(batch_yhat)).mean()
            grads = autograd.grad(outputs=batch_yhat, inputs=batch_u, grad_outputs=torch.ones_like(batch_yhat),
                                  retain_graph=True, create_graph=True, only_inputs=True)[0]
            loss_grad = penalty * torch.max(-grads, torch.zeros_like(grads)).mean()
            total_loss = loss + loss_grad
            optimizer.zero_grad()
            if clip_gradients: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            if np.isnan(total_loss.item()): warnings.warn('NaNs encountered in training model.'); break
        if np.isnan(train_loss): break
        
        with torch.no_grad():
            model.eval()
            u_valid = torch.rand(Y_valid.shape)
            yhat = model(X_valid, u_valid)
            val_loss = quantile_loss(yhat, Y_valid.view_as(yhat), u_valid.view_as(yhat)).mean().item()

        if num_bad_epochs > patience:
            scheduler.step(); num_bad_epochs = 0
        if epoch == 0 or val_loss <= best_loss:
            best_loss = val_loss; num_bad_epochs = 0
        else: num_bad_epochs += 1
    return model


class DQRP_Adapter:
    def __init__(self, quantiles):
        self.quantiles = np.array(quantiles)
        self.model = None

    def fit(self, X, y, **kwargs):
        self.dqr_p_model = DQRP(quantiles=self.quantiles)
        self.dqr_p_model.fit(X, y.reshape(-1, 1), 
                             lr=kwargs.get('dqr_p_lr', 0.01), 
                             epochs=kwargs.get('nepochs', 100))
    
    def predict(self, X):
        return self.dqr_p_model.predict(X, self.quantiles)


# --- Other Models (Linear, Forest, Kernel) ---
class LinearQR:
    def __init__(self, quantiles): self.quantiles, self.models = quantiles, []
    def fit(self, X, y, **kwargs):
        y = y.ravel()
        for q in self.quantiles:
            model = QuantileRegressor(quantile=q, alpha=0, solver='highs'); model.fit(X, y)
            self.models.append(model)
    def predict(self, X):
        preds = np.zeros((X.shape[0], len(self.quantiles)))
        for i, model in enumerate(self.models): preds[:, i] = model.predict(X)
        return preds


@jit(nopython=True, parallel=True)
def find_quant(trainy, train_tree_node_ID, pred_tree_node_ID, qntl):
    npred = pred_tree_node_ID.shape[0]; out = np.zeros((npred, qntl.size)) * np.nan
    for i in prange(npred):
        idxs = np.where(train_tree_node_ID == pred_tree_node_ID[i, :])[0]
        if len(idxs) > 0: out[i, :] = np.quantile(trainy[idxs], qntl)
    return out

class QuantileForest:
    def __init__(self, quantiles, nthreads=4, **kwargs):
        self.quantiles = np.array(quantiles); self.forest = RandomForestRegressor(n_jobs=nthreads, **kwargs)
        set_num_threads(nthreads); self.trainX = None; self.trainy = None
    def fit(self, X, y): self.trainX, self.trainy = X, y.ravel(); self.forest.fit(self.trainX, self.trainy)
    def predict(self, X):
        train_leaf_nodes = self.forest.apply(self.trainX); pred_leaf_nodes = self.forest.apply(X)
        return find_quant(self.trainy, train_leaf_nodes, pred_leaf_nodes, self.quantiles)

class Kernel:
    def __init__(self, quantiles): self.model = KernelQR_impl(quantiles=quantiles, C=1.0, gamma=1.0)
    def fit(self, X, y, **kwargs): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)
    
class KernelQR_impl(RegressorMixin, BaseEstimator):
    def __init__(self, quantiles=0.5, kernel_type="gaussian_rbf", C=1, gamma=1):
        self.C, self.quantiles, self.kernel_type, self.gamma = C, quantiles, kernel_type, gamma
    def kernel(self, X, Y): return Matern(length_scale=self.gamma, nu=np.inf)(X, Y) if self.kernel_type == "gaussian_rbf" else rbf_kernel(X, Y, gamma=1 / self.gamma)
    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=False); self.X_ = X; self.y_ = y.ravel()
        quantiles = np.array([self.quantiles]) if np.isscalar(self.quantiles) else np.array(self.quantiles).flatten()
        K = self.kernel(self.X_, self.X_); Kmat, r, A, b, n = matrix(K), matrix(self.y_) * 1.0, matrix(np.ones(self.y_.size)).T, matrix(0.0), self.y_.size
        solvers.options['show_progress'] = False; self.a_list, self.b_list = [], []
        for quantile in quantiles:
            G = matrix(np.vstack([np.eye(n), -np.eye(n)])); h = matrix(np.hstack([self.C * quantile * np.ones(n), self.C * (1 - quantile) * np.ones(n)]))
            sol = solvers.qp(P=Kmat, q=-r, G=G, h=h, A=A, b=b); a = np.array(sol["x"]).flatten()
            sv_indices = np.where((a > 1e-5) & (a < (self.C * quantile) - 1e-5))[0]
            offshift = sv_indices[0] if len(sv_indices) > 0 else np.argmin(np.abs(a - (self.C * quantile / 2)))
            bias = self.y_[offshift] - a.T @ K[:, offshift]; self.a_list.append(a); self.b_list.append(bias)
        self.quantiles_ = quantiles; return self
    def predict(self, X):
        check_is_fitted(self, ['X_', 'y_', 'a_list', 'b_list', 'quantiles_']); X = check_array(X); K = self.kernel(self.X_, X)
        preds = [a.T @ K + b for a, b in zip(self.a_list, self.b_list)]; return np.vstack(preds).T


# =============================================================================
# 2. Evaluation Metrics Implementation
# =============================================================================
def calculate_metrics(pred_q0, pred_q1, true_q0, true_q1, true_cate, true_qte, all_quantiles):
    metrics = {}
    est_cate = (pred_q1 - pred_q0).mean(axis=1)
    metrics['PEHE'] = np.sqrt(np.mean((est_cate - true_cate)**2))
    w1_dist_0 = np.mean(np.abs(pred_q0 - true_q0)); w1_dist_1 = np.mean(np.abs(pred_q1 - true_q1))
    metrics['W1-dist'] = (w1_dist_0 + w1_dist_1) / 2.0
    ncr_0 = np.mean(np.all(np.diff(pred_q0, axis=1) >= -1e-6, axis=1)); ncr_1 = np.mean(np.all(np.diff(pred_q1, axis=1) >= -1e-6, axis=1))
    metrics['NCR'] = (ncr_0 + ncr_1) / 2.0
    idx_lower = np.where(np.isclose(all_quantiles, 0.05))[0][0]; idx_upper = np.where(np.isclose(all_quantiles, 0.95))[0][0]
    pi_width_0 = pred_q0[:, idx_upper] - pred_q0[:, idx_lower]; pi_width_1 = pred_q1[:, idx_upper] - pred_q1[:, idx_lower]
    metrics['PI Width'] = np.mean(np.concatenate([pi_width_0, pi_width_1]))
    pred_qte = pred_q1 - pred_q0
    metrics['IMSE-QTE'] = np.mean((pred_qte - true_qte)**2)
    return metrics

def calculate_coverage(all_pred_q0, all_pred_q1, all_quantiles, y_test, t_test):
    idx_lower = np.where(np.isclose(all_quantiles, 0.05))[0][0]; idx_upper = np.where(np.isclose(all_quantiles, 0.95))[0][0]
    pred_q0_pi = all_pred_q0[:, [idx_lower, idx_upper]]; pred_q1_pi = all_pred_q1[:, [idx_lower, idx_upper]]
    covered = np.zeros(len(y_test))
    for j in range(len(y_test)):
        lower, upper = (pred_q0_pi[j, 0], pred_q0_pi[j, 1]) if t_test[j] == 0 else (pred_q1_pi[j, 0], pred_q1_pi[j, 1])
        if lower > upper: lower, upper = upper, lower
        if y_test[j] >= lower and y_test[j] <= upper: covered[j] = 1
    return np.mean(covered)

# =============================================================================
# 3. Main Experiment Loop
# =============================================================================

def main(cfg):
    set_seed(cfg.seed)
    
    try:
        train_raw = np.load(os.path.join(cfg.data_dir, cfg.train_data_file))
        test_raw = np.load(os.path.join(cfg.data_dir, cfg.test_data_file))
    except FileNotFoundError:
        print(f"Error: Make sure '{cfg.train_data_file}' and '{cfg.test_data_file}' are in '{cfg.data_dir}'.")
        return

    xs = np.concatenate([train_raw['x'], test_raw['x']], axis=0); ys = np.concatenate([train_raw['yf'], test_raw['yf']], axis=0)
    ts = np.concatenate([train_raw['t'], test_raw['t']], axis=0); mu0s = np.concatenate([train_raw['mu0'], test_raw['mu0']], axis=0)
    mu1s = np.concatenate([train_raw['mu1'], test_raw['mu1']], axis=0)
    ALL_QUANTILES = np.round(np.arange(0.05, 1.0, 0.05), 2)
    
    # 'DQR*' is the internal label for VDQR, the unconstrained deep quantile network.
    model_factory = {'DQR*': QuantileNet, 'NQNet': NQNet, 'DQR': DQR, 'DQRP': DQRP_Adapter,
                     'Linear': LinearQR, 'Forest': QuantileForest, 'Kernel': Kernel}
    results = {name: [] for name in list(model_factory.keys()) + ['VDQR+R']}

    for i in tqdm(range(cfg.replications), desc="Replications"):
        X, y, t, mu0, mu1 = xs[:, :, i], ys[:, i], ts[:, i], mu0s[:, i], mu1s[:, i]
        true_cate = mu1 - mu0; z_scores = norm.ppf(ALL_QUANTILES)
        true_q0 = mu0[:, np.newaxis] + z_scores; true_q1 = mu1[:, np.newaxis] + z_scores
        true_qte = true_q1 - true_q0
        
        indices = np.arange(X.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size, random_state=42 + i)
        X_train, X_test = X[train_indices], X[test_indices]; y_train, y_test = y[train_indices], y[test_indices]
        t_train, t_test = t[train_indices], t[test_indices]
        
        if i == 0:
            print("\n" + "="*50 + "\nData & Hyperparameters\n" + "="*50)
            print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}")
            print(f"Train/Test Split: {len(train_indices)}/{len(test_indices)}")
            print(f"NN epochs: {cfg.nn_epochs}, LR (most NNs): {cfg.nn_lr}, LR (DQRP): {cfg.dqr_p_lr}")
            print("="*50 + "\n")

        X0_train, y0_train = X_train[t_train == 0], y_train[t_train == 0]
        X1_train, y1_train = X_train[t_train == 1], y_train[t_train == 1]
        
        current_replication_results = {}
        
        for name, ModelClass in model_factory.items():
            model0 = ModelClass(quantiles=ALL_QUANTILES)
            model1 = ModelClass(quantiles=ALL_QUANTILES)
            
            fit_params = {}
            if name in ['NQNet', 'DQR*', 'DQR']:
                fit_params = {'nepochs': cfg.nn_epochs, 'lr': cfg.nn_lr, 'patience': cfg.nn_patience,
                              'batch_size': cfg.nn_batch_size, 'val_pct': cfg.nn_val_pct}
            elif name == 'DQRP':
                fit_params = {'nepochs': cfg.nn_epochs, 'dqr_p_lr': cfg.dqr_p_lr}

            model0.fit(X0_train, y0_train, **fit_params)
            model1.fit(X1_train, y1_train, **fit_params)
            
            pred_q0 = model0.predict(X_test)
            pred_q1 = model1.predict(X_test)
            
            metrics = calculate_metrics(pred_q0, pred_q1, true_q0[test_indices], true_q1[test_indices], 
                                        true_cate[test_indices], true_qte[test_indices], ALL_QUANTILES)
            metrics['Coverage'] = calculate_coverage(pred_q0, pred_q1, ALL_QUANTILES, y_test, t_test)
            current_replication_results[name] = metrics

            # VDQR + rearrangement (Chernozhukov et al., 2010): sort each predicted
            # quantile vector ascending, restoring monotonicity from the same VDQR fit.
            if name == 'DQR*':
                p0r, p1r = np.sort(pred_q0, axis=1), np.sort(pred_q1, axis=1)
                metrics_r = calculate_metrics(p0r, p1r, true_q0[test_indices], true_q1[test_indices],
                                              true_cate[test_indices], true_qte[test_indices], ALL_QUANTILES)
                metrics_r['Coverage'] = calculate_coverage(p0r, p1r, ALL_QUANTILES, y_test, t_test)
                current_replication_results['VDQR+R'] = metrics_r

        for name, metrics in current_replication_results.items(): results[name].append(metrics)

    print("\n" + "="*80 + f"\nFinal Results (mean (std) over {cfg.replications} replications)\n" + "="*80)
    summary_df = pd.DataFrame(columns=['PEHE', 'IMSE-QTE', 'W1-dist', 'PI Width', 'Coverage', 'NCR'])
    
    for name, res_list in results.items():
        df = pd.DataFrame(res_list); mean_vals, std_vals = df.mean(), df.std()
        
        # format each cell as 'mean (std)'; NCR as a percentage
        summary_df.loc[name] = [
            f"{mean_vals['PEHE']:.3f} ({std_vals['PEHE']:.3f})",
            f"{mean_vals['IMSE-QTE']:.3f} ({std_vals['IMSE-QTE']:.3f})",
            f"{mean_vals['W1-dist']:.3f} ({std_vals['W1-dist']:.3f})",
            f"{mean_vals['PI Width']:.3f} ({std_vals['PI Width']:.3f})",
            f"{mean_vals['Coverage']:.3f} ({std_vals['Coverage']:.3f})",
            f"{mean_vals['NCR']*100:.2f}% ({std_vals['NCR']*100:.2f}%)"
        ]
    
    paper_order = ['NQNet', 'DQR', 'DQR*', 'VDQR+R', 'DQRP', 'Linear', 'Forest', 'Kernel']
    summary_df = summary_df.reindex([o for o in paper_order if o in summary_df.index])
    summary_df = summary_df.rename(index={'DQR*': 'VDQR'})
    print(summary_df.to_string())
    
    if not os.path.exists(cfg.output_dir): os.makedirs(cfg.output_dir)
    output_path = os.path.join(cfg.output_dir, "ihdp_results.csv")
    summary_df.to_csv(output_path)
    print(f"\nResults saved to {output_path}")

# =============================================================================
# 4. Script Execution
# =============================================================================


# =============================================================================
# 5. Non-Gaussian distributional descriptors (Table S10)
# =============================================================================

ALL_Q = np.round(np.arange(0.05, 1.0, 0.05), 2)         # 19 quantile levels
TAUS_NP = ALL_Q
FP = dict(nepochs=200, lr=5e-4, patience=20, batch_size=64, val_pct=0.20)
S_LN = 0.5                                              # log-normal log-sd
NOISES = ['normal', 't3', 'lognorm']
METHODS_NG = ['NQ-Net', 'VDQR', 'VDQR+R', 'DQR', 'DQRP']


def descriptors(q):
    """mean, sd, skewness, excess kurtosis read off the K quantile values."""
    m = q.mean(1); c = q - m[:, None]
    var = np.maximum((c ** 2).mean(1), 1e-12); sd = np.sqrt(var)
    skew = (c ** 3).mean(1) / sd ** 3
    kurt = (c ** 4).mean(1) / var ** 2 - 3.0
    return m, sd, skew, kurt


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def to_np(p):
    return p.detach().numpy() if hasattr(p, 'detach') else np.asarray(p)


def es_lower(Q, beta=0.10):
    """(1/beta) int_0^beta Q(u) du; Q piecewise-linear on the tau grid, flat below tau_0."""
    g = TAUS_NP
    integral = Q[:, 0] * g[0]
    for k in range(len(g) - 1):
        a, b = g[k], g[k + 1]
        if b <= beta:
            integral += 0.5 * (Q[:, k] + Q[:, k + 1]) * (b - a)
        elif a < beta < b:
            w = (beta - a) / (b - a); Qb = Q[:, k] * (1 - w) + Q[:, k + 1] * w
            integral += 0.5 * (Q[:, k] + Qb) * (beta - a); break
        else:
            break
    return integral / beta


def es_upper(Q, beta=0.10):
    """(1/beta) int_{1-beta}^1 Q(u) du; flat above tau_{K-1}."""
    g = TAUS_NP; start = 1 - beta
    integral = Q[:, -1] * (1 - g[-1])
    for k in range(len(g) - 1, 0, -1):
        a, b = g[k - 1], g[k]
        if a >= start:
            integral += 0.5 * (Q[:, k - 1] + Q[:, k]) * (b - a)
        elif a < start < b:
            w = (start - a) / (b - a); Qs = Q[:, k - 1] * (1 - w) + Q[:, k] * w
            integral += 0.5 * (Qs + Q[:, k]) * (b - start); break
        else:
            break
    return integral / beta


def _ln_const(s):
    m = np.exp(s * s / 2.0); sd = np.sqrt((np.exp(s * s) - 1.0) * np.exp(s * s))
    return m, sd


def noise_sample(kind, n, rng):
    if kind == 'normal':
        return rng.standard_normal(n)
    if kind == 't3':
        return rng.standard_t(3, n) / np.sqrt(3.0)
    if kind == 'lognorm':
        m, sd = _ln_const(S_LN); return (np.exp(S_LN * rng.standard_normal(n)) - m) / sd
    raise ValueError(kind)


def noise_quantile(kind, taus):
    if kind == 'normal':
        return norm.ppf(taus)
    if kind == 't3':
        return student_t.ppf(taus, 3) / np.sqrt(3.0)
    if kind == 'lognorm':
        m, sd = _ln_const(S_LN); return (np.exp(S_LN * norm.ppf(taus)) - m) / sd
    raise ValueError(kind)


def panel_te(p0, p1, tq0, tq1, true_cate, true_qte, y_test, t_test):
    d = {}
    est_cate = (p1 - p0).mean(1)
    d['PEHE'] = float(np.sqrt(np.mean((est_cate - true_cate) ** 2)))
    d['IMSE-QTE'] = float(np.mean(((p1 - p0) - true_qte) ** 2))
    d['W1'] = float(0.5 * (np.mean(np.abs(p0 - tq0)) + np.mean(np.abs(p1 - tq1))))

    def ncr(p): return float(np.mean(np.all(np.diff(p, 1) >= -1e-6, 1)))
    def sev(p): return float(np.mean(np.sum(np.maximum(0.0, -np.diff(p, 1)), 1)))
    d['NCR'] = 0.5 * (ncr(p0) + ncr(p1)); d['CrossSev'] = 0.5 * (sev(p0) + sev(p1))

    m0, s0, k0, u0 = descriptors(p0); m1, s1, k1, u1 = descriptors(p1)
    tm0, ts0, tk0, tu0 = descriptors(tq0); tm1, ts1, tk1, tu1 = descriptors(tq1)
    d['mean'] = 0.5 * (rmse(m0, tm0) + rmse(m1, tm1)); d['sd'] = 0.5 * (rmse(s0, ts0) + rmse(s1, ts1))
    d['skew'] = 0.5 * (rmse(k0, tk0) + rmse(k1, tk1)); d['kurt'] = 0.5 * (rmse(u0, tu0) + rmse(u1, tu1))
    d['CVaR_low'] = 0.5 * (rmse(es_lower(p0), es_lower(tq0)) + rmse(es_lower(p1), es_lower(tq1)))
    d['CVaR_high'] = 0.5 * (rmse(es_upper(p0), es_upper(tq0)) + rmse(es_upper(p1), es_upper(tq1)))
    lo = np.where(t_test == 0, p0[:, 0], p1[:, 0]); hi = np.where(t_test == 0, p0[:, 18], p1[:, 18])
    sw = lo > hi; lo2 = np.where(sw, hi, lo); hi2 = np.where(sw, lo, hi)
    d['Coverage'] = float(np.mean((y_test >= lo2) & (y_test <= hi2)))
    return d


def run_nongauss_cell(task):
    """Full panel on IHDP surfaces with a standardized non-Gaussian outcome. One (noise, rep)."""
    kind, rep, xs, mu0s, mu1s, ts = task
    import torch
    torch.set_num_threads(1)
    rng = np.random.RandomState(1000 + rep)
    rows = []
    try:
        X = xs[:, :, rep]; mu0 = mu0s[:, rep]; mu1 = mu1s[:, rep]; t = ts[:, rep]
        eps = noise_sample(kind, X.shape[0], rng)
        y = np.where(t == 0, mu0, mu1) + eps
        idx = np.arange(X.shape[0]); tri, tei = train_test_split(idx, test_size=0.20, random_state=42 + rep)
        X0, y0 = X[tri][t[tri] == 0], y[tri][t[tri] == 0]
        X1, y1 = X[tri][t[tri] == 1], y[tri][t[tri] == 1]
        Xte = X[tei]; yte = y[tei]; tte = t[tei]
        Z = noise_quantile(kind, ALL_Q)
        tq0 = mu0[tei][:, None] + Z; tq1 = mu1[tei][:, None] + Z
        true_cate = (mu1 - mu0)[tei]; true_qte = (mu1 - mu0)[tei][:, None] + np.zeros_like(Z)

        def fit_pred(Cls, **kw):
            m0, m1 = Cls(quantiles=ALL_Q), Cls(quantiles=ALL_Q)
            m0.fit(X0, y0, **kw); m1.fit(X1, y1, **kw)
            return to_np(m0.predict(Xte)), to_np(m1.predict(Xte))

        nq0, nq1 = fit_pred(NQNet, **FP); v0, v1 = fit_pred(QuantileNet, **FP)
        dq0, dq1 = fit_pred(DQR, **FP); dp0, dp1 = fit_pred(DQRP_Adapter, nepochs=100, dqr_p_lr=0.01)
        preds = {'NQ-Net': (nq0, nq1), 'VDQR': (v0, v1), 'VDQR+R': (np.sort(v0, 1), np.sort(v1, 1)),
                 'DQR': (dq0, dq1), 'DQRP': (dp0, dp1)}
        for m, (p0, p1) in preds.items():
            d = panel_te(p0, p1, tq0, tq1, true_cate, true_qte, yte, tte)
            d.update(dict(model=kind, rep=rep, method=m)); rows.append(d)
    except Exception as ex:
        rows.append(dict(model=kind, rep=rep, method='ERROR', err=str(ex)))
    print(f"  done {kind} rep {rep}", flush=True)
    return rows


def run_nongauss(reps=100, workers=10, out='results/ihdp_descriptors_nongauss.csv'):
    """Table S10: RMSE against a non-Gaussian truth of the conditional mean, standard
    deviation, skewness and kurtosis (plus a treatment-effect/validity panel), on the
    IHDP response surfaces with standardized Student-t_3 / log-normal outcome noise."""
    set_seed(42)
    tr = np.load('data/ihdp_npci_1-1000.train.npz'); te = np.load('data/ihdp_npci_1-1000.test.npz')
    xs = np.concatenate([tr['x'], te['x']], 0); ts = np.concatenate([tr['t'], te['t']], 0)
    mu0s = np.concatenate([tr['mu0'], te['mu0']], 0); mu1s = np.concatenate([tr['mu1'], te['mu1']], 0)
    tasks = [(kind, rep, xs, mu0s, mu1s, ts) for kind in NOISES for rep in range(reps)]
    print(f"=== IHDP non-Gaussian descriptors (Table S10): {len(tasks)} cells, {workers} workers ===", flush=True)
    with Pool(workers) as pool:
        rows = [r for sub in pool.map(run_nongauss_cell, tasks) for r in sub]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    df.to_csv(out, index=False)
    nerr = int((df['method'] == 'ERROR').sum()) if 'method' in df else -1
    print(f"\nwrote {out}  ({len(df)} rows, {nerr} errors)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['table6', 'nongauss'], default='table6',
                    help="table6: IHDP treatment-effect comparison (Table 6); "
                         "nongauss: non-Gaussian distributional descriptors (Table S10)")
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--out', default='results/ihdp_descriptors_nongauss.csv')
    args = ap.parse_args()
    if args.mode == 'table6':
        class Config:
            data_dir = 'data'
            train_data_file = 'ihdp_npci_1-1000.train.npz'
            test_data_file = 'ihdp_npci_1-1000.test.npz'
            replications = 100
            output_dir = 'IHDP_results'
            seed = 42
            test_size = 0.20
            nn_epochs = 200
            nn_patience = 20
            nn_batch_size = 64
            nn_val_pct = 0.20
            nn_lr = 5e-4
            dqr_p_lr = 0.01
        main(Config())
    else:
        run_nongauss(args.reps, args.workers, args.out)
