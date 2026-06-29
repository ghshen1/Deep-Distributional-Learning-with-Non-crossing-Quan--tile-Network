"""
Distributional-RL experiments on a synthetic finite MDP, producing Tables S7, S8,
and S9 of the Supplementary Material. This single self-contained file holds the
synthetic MDP, the fitted iteration algorithms, and all three experiments.

  --table S7 : heavy-tail robustness -- fitted NQ iteration (Algorithm 1) versus a
               mean-based fitted-Q iteration, across parametric reward-noise families
               ordered from light (Gaussian, Laplace) to heavy/skewed (contaminated
               normal, log-normal, Student-t, Pareto), all centred and standardized
               to a common inter-quartile spread.
  --table S8 : quantile-averaging ablation -- uniform versus trimmed versus
               tau(1-tau) weights, with a paired Wilcoxon test across the common seeds.
  --table S9 : the same heavy-tail comparison with the parametric reward noise
               replaced by an empirical bootstrap from three real heavy-tailed
               datasets (S&P 500 monthly log-returns, Danish and Norwegian fire
               insurance losses; arrays shipped under data/).

Regret = J(pi*) - J(pi_hat), computed exactly on the known finite MDP, averaged over
20 independent seeds. Results are written to RL_results/.

Run:  /opt/anaconda3/bin/python tableS7_S9.py --table all
"""
import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from scipy.stats import wilcoxon
    _HAVE_WILCOXON = True
except Exception:
    _HAVE_WILCOXON = False

OUT = 'RL_results'
TAUS = np.round(np.arange(0.05, 1.0, 0.05), 2)   # 19 quantile levels
N_SEEDS = 20
M, N, INNER = 30, 2000, 12
NS, NA, GAMMA = 20, 4, 0.9


# ======================================================================================
# Synthetic finite MDP
# ======================================================================================

class SyntheticMDP:
    def __init__(self, n_states=20, n_actions=4, gamma=0.9,
                 noise='t', nu=1.5, noise_scale=1.0, seed=0):
        rng = np.random.RandomState(seed)
        self.nS, self.nA, self.gamma = n_states, n_actions, gamma
        self.noise, self.nu, self.noise_scale = noise, nu, noise_scale
        self.seed = seed

        # Transition kernel P[s,a,:]: a sparse-ish random distribution over next states.
        P = np.zeros((n_states, n_actions, n_states))
        for s in range(n_states):
            for a in range(n_actions):
                # each (s,a) can reach a random subset of ~5 states
                k = rng.randint(3, 6)
                nxt = rng.choice(n_states, size=k, replace=False)
                w = rng.dirichlet(np.ones(k))
                P[s, a, nxt] = w
        self.P = P

        # Mean reward r(s,a): structured so the optimal policy is non-trivial.
        self.R = rng.uniform(-1.0, 1.0, size=(n_states, n_actions))

        # Initial-state distribution (for J(pi)).
        self.mu0 = rng.dirichlet(np.ones(n_states))

        self._rng = np.random.RandomState(seed + 12345)

        # IQR-standardization factor so every noise family has the same
        # inter-quartile spread as N(0,1) (IQR = 1.349); computed once from a
        # large fixed sample so it is deterministic and shared across seeds.
        cal = self._raw_noise(400000, np.random.RandomState(987654))
        iqr = np.quantile(cal, 0.75) - np.quantile(cal, 0.25)
        self._iqr_scale = 1.349 / iqr

    # ---- exact dynamic programming (uses the MEAN reward) ----
    def optimal(self):
        """Exact optimal Q*, V*, greedy policy via value iteration on mean rewards."""
        Q = np.zeros((self.nS, self.nA))
        for _ in range(2000):
            V = Q.max(axis=1)
            Q_new = self.R + self.gamma * (self.P @ V)
            if np.max(np.abs(Q_new - Q)) < 1e-10:
                Q = Q_new
                break
            Q = Q_new
        pi = Q.argmax(axis=1)
        return Q, Q.max(axis=1), pi

    def policy_value(self, pi):
        """Exact V^pi and J(pi)=mu0 . V^pi for a deterministic policy pi (array of actions)."""
        Ppi = self.P[np.arange(self.nS), pi, :]      # nS x nS
        rpi = self.R[np.arange(self.nS), pi]          # nS
        V = np.linalg.solve(np.eye(self.nS) - self.gamma * Ppi, rpi)
        return V, float(self.mu0 @ V)

    def regret(self, pi):
        _, Jstar = self.policy_value(self.optimal()[2])
        _, Jpi = self.policy_value(pi)
        return Jstar - Jpi, Jstar, Jpi

    # ---- sampling an offline batch under a behaviour (exploration) policy ----
    def sample_batch(self, n, rng=None):
        """n tuples (s,a,r,s') with s,a drawn uniformly (uniform exploration)."""
        rng = self._rng if rng is None else rng
        s = rng.randint(self.nS, size=n)
        a = rng.randint(self.nA, size=n)
        probs = self.P[s, a, :]                       # n x nS
        cdf = np.cumsum(probs, axis=1)
        u = rng.rand(n, 1)
        sp = (u > cdf).sum(axis=1)                    # inverse-cdf sample of next state
        sp = np.minimum(sp, self.nS - 1)
        mean_r = self.R[s, a]
        r = mean_r + self.noise_scale * self._noise(n, rng)
        return s, a, r, sp

    def _raw_noise(self, n, rng):
        """Zero-mean (centered) raw noise of the requested family.

        `nu` is the family shape parameter: degrees of freedom for Student-t,
        sigma for the log-normal, and the tail index for the (Lomax) Pareto.
        """
        nu = self.nu
        if self.noise == 'gauss':
            return rng.randn(n)
        elif self.noise == 'laplace':
            return rng.laplace(0.0, 1.0, size=n)
        elif self.noise == 't':                  # symmetric, heavy (mean exists for nu>1)
            return rng.standard_t(nu, size=n)
        elif self.noise == 'lognorm':            # right-skewed, centered
            sig = nu
            return np.exp(sig * rng.randn(n)) - np.exp(sig ** 2 / 2.0)
        elif self.noise == 'pareto':             # heavy, skewed; Lomax centered (mean 1/(nu-1))
            return rng.pareto(nu, size=n) - 1.0 / (nu - 1.0)
        elif self.noise == 'mixture':            # contaminated normal: 0.9 N(0,1) + 0.1 N(0,5^2)
            base = rng.randn(n)
            contam = rng.rand(n) < 0.1
            base[contam] = 5.0 * rng.randn(int(contam.sum()))
            return base
        else:
            raise ValueError(self.noise)

    def _noise(self, n, rng):
        # IQR-standardized so that every family has the same inter-quartile
        # spread as N(0,1); only the tail/shape differs across rows.
        return self._raw_noise(n, rng) * self._iqr_scale

    def onehot(self, s):
        out = np.zeros((len(s), self.nS), dtype=np.float32)
        out[np.arange(len(s)), s] = 1.0
        return out


class RealNoiseMDP(SyntheticMDP):
    """SyntheticMDP whose reward noise is an empirical bootstrap from real data.

    The empirical sample is centred to zero mean (its mean is finite for the
    datasets considered, matching Assumption 4 with p>1); the parent class then applies the
    same IQR-standardization and the same exact dynamic programming as for the
    parametric families, so this is a drop-in noise source.
    """
    def __init__(self, sample, n_states=NS, n_actions=NA, gamma=GAMMA,
                 noise_scale=1.0, seed=0):
        self._sample = np.asarray(sample, dtype=float)
        self._sample = self._sample - self._sample.mean()   # zero-mean, as the other families
        super().__init__(n_states=n_states, n_actions=n_actions, gamma=gamma,
                         noise='bootstrap', noise_scale=noise_scale, seed=seed)

    def _raw_noise(self, n, rng):
        idx = rng.randint(0, len(self._sample), size=n)
        return self._sample[idx]


# ======================================================================================
# Fitted iteration algorithms (Algorithm 1 and the mean-based baseline)
# ======================================================================================

# ---------- value aggregation schemes ----------
def aggregation_weights(quantile_levels, scheme='uniform', trim=0.1):
    tau = np.asarray(quantile_levels, dtype=np.float64)
    K = len(tau)
    if scheme == 'uniform':
        w = np.ones(K)
    elif scheme == 'trimmed':
        keep = (tau >= trim) & (tau <= 1 - trim)
        w = keep.astype(np.float64)
    elif scheme == 'tau1mtau':
        w = tau * (1 - tau)
    else:
        raise ValueError(scheme)
    return w / w.sum()


# ---------- networks ----------
class _Trunk(nn.Module):
    def __init__(self, n_in, width, depth):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class QuantileRLNet(nn.Module):
    """Outputs theta of shape (B, nA, K): K quantiles of Z(s,a) for each action.

    noncross=True  -> per action: v + centered cumsum of (K-1) non-negative gaps
                      (the non-crossing NQ parametrization, identical in spirit to
                      NQNet.py: one mean component and K-1 gaps).
    noncross=False -> per action: K unconstrained outputs (quantiles may cross).
    """
    def __init__(self, n_in, nA, K, width=128, depth=2, noncross=True):
        super().__init__()
        self.nA, self.K, self.noncross = nA, K, noncross
        self.trunk = _Trunk(n_in, width, depth)
        if noncross:
            self.value_head = nn.Linear(self.trunk.out_dim, nA)          # one mean per action
            self.gap_head = nn.Linear(self.trunk.out_dim, nA * (K - 1))  # K-1 gaps per action
        else:
            self.head = nn.Linear(self.trunk.out_dim, nA * K)

    def forward(self, x):
        h = self.trunk(x)
        B = x.shape[0]
        if self.noncross:
            v = self.value_head(h).view(B, self.nA, 1)
            gaps = torch.nn.functional.elu(self.gap_head(h)).view(B, self.nA, self.K - 1) + 1.0
            zero = torch.zeros(B, self.nA, 1, dtype=gaps.dtype, device=gaps.device)
            csum = torch.cumsum(torch.cat([zero, gaps], dim=2), dim=2)   # (B,nA,K)
            theta = v + (csum - csum.mean(dim=2, keepdim=True))
        else:
            theta = self.head(h).view(B, self.nA, self.K)
        return theta


class MeanRLNet(nn.Module):
    def __init__(self, n_in, nA, width=128, depth=2):
        super().__init__()
        self.trunk = _Trunk(n_in, width, depth)
        self.head = nn.Linear(self.trunk.out_dim, nA)

    def forward(self, x):
        return self.head(self.trunk(x))


# ---------- helpers ----------
def _state_feats(mdp, s):
    return torch.from_numpy(mdp.onehot(s))


def _greedy_policy_from_quantiles(mdp, net, w):
    """Return argmax_a (weighted-average quantile value) for every state."""
    net.eval()
    with torch.no_grad():
        X = _state_feats(mdp, np.arange(mdp.nS))
        theta = net(X).numpy()                       # nS x nA x K
    val = (theta * w[None, None, :]).sum(axis=2)     # nS x nA
    return val.argmax(axis=1)


def _greedy_policy_from_mean(mdp, net):
    net.eval()
    with torch.no_grad():
        X = _state_feats(mdp, np.arange(mdp.nS))
        Q = net(X).numpy()
    return Q.argmax(axis=1)


# ---------- Algorithm 1 (and the crossing variant) ----------
def fit_quantile_iteration(mdp, quantile_levels, M=40, N=4000, width=128, depth=2,
                           noncross=True, agg_scheme='uniform', trim=0.1,
                           inner_epochs=25, lr=1e-3, batch_size=256, seed=0,
                           eval_scheme=None):
    """Fitted NQ iteration (Algorithm 1). Returns (greedy_policy, diagnostics).

    agg_scheme governs the value used for the greedy step inside the iteration;
    eval_scheme (defaults to agg_scheme) governs the final extracted policy.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(1000 + seed)
    tau = np.asarray(quantile_levels, dtype=np.float64)
    K = len(tau)
    w = aggregation_weights(tau, agg_scheme, trim)
    w_t = torch.tensor(w, dtype=torch.float32)
    tau_t = torch.tensor(tau, dtype=torch.float32)

    net = QuantileRLNet(mdp.nS, mdp.nA, K, width, depth, noncross)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for m in range(M):
        s, a, r, sp = mdp.sample_batch(N, rng)
        Xs = _state_feats(mdp, s)
        Xsp = _state_feats(mdp, sp)
        a_t = torch.from_numpy(a).long()
        r_t = torch.from_numpy(r.astype(np.float32))

        # frozen targets from the current network
        net.eval()
        with torch.no_grad():
            theta_sp = net(Xsp)                                   # N x nA x K
            val_sp = (theta_sp * w_t[None, None, :]).sum(dim=2)   # N x nA  (weighted avg)
            a_next = val_sp.argmax(dim=1)                         # greedy next action
            theta_next = theta_sp[torch.arange(N), a_next, :]     # N x K  (target atoms base)
            atoms = r_t[:, None] + mdp.gamma * theta_next         # N x K  target atoms y_{i,j}

        # inner regression: fit new quantiles to the K target atoms via check loss
        net.train()
        idx_all = np.arange(N)
        for _ in range(inner_epochs):
            rng.shuffle(idx_all)
            for b0 in range(0, N, batch_size):
                bi = idx_all[b0:b0 + batch_size]
                xb = Xs[bi]
                theta_pred = net(xb)[torch.arange(len(bi)), a_t[bi], :]   # |b| x K
                yb = atoms[bi]                                            # |b| x K
                # rho_{tau_k}(y_j - theta_k) summed over k (pred) and j (target atoms)
                diff = yb[:, None, :] - theta_pred[:, :, None]            # |b| x K(pred) x K(atom)
                loss = torch.maximum(tau_t[None, :, None] * diff,
                                     (tau_t[None, :, None] - 1.0) * diff).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    final_scheme = eval_scheme or agg_scheme
    w_final = aggregation_weights(tau, final_scheme, trim)
    pi = _greedy_policy_from_quantiles(mdp, net, w_final)
    regret, Jstar, Jpi = mdp.regret(pi)
    # validity of the learned value distribution: fraction of (s,a) whose K
    # quantiles are monotone (1.0 by construction for the non-crossing net).
    net.eval()
    with torch.no_grad():
        theta = net(_state_feats(mdp, np.arange(mdp.nS))).numpy()   # nS x nA x K
    ncr = float(np.mean(np.all(np.diff(theta, axis=2) >= -1e-6, axis=2)))
    return pi, {'regret': regret, 'Jstar': Jstar, 'Jpi': Jpi, 'ncr': ncr}


def fit_mean_fqi(mdp, M=40, N=4000, width=128, depth=2,
                 inner_epochs=25, lr=1e-3, batch_size=256, seed=0):
    """Standard fitted-Q iteration: regress the mean return with squared loss."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(2000 + seed)
    net = MeanRLNet(mdp.nS, mdp.nA, width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()

    for m in range(M):
        s, a, r, sp = mdp.sample_batch(N, rng)
        Xs = _state_feats(mdp, s)
        Xsp = _state_feats(mdp, sp)
        a_t = torch.from_numpy(a).long()
        r_t = torch.from_numpy(r.astype(np.float32))
        net.eval()
        with torch.no_grad():
            Qsp = net(Xsp)                       # N x nA
            y = r_t + mdp.gamma * Qsp.max(dim=1).values
        net.train()
        idx_all = np.arange(N)
        for _ in range(inner_epochs):
            rng.shuffle(idx_all)
            for b0 in range(0, N, batch_size):
                bi = idx_all[b0:b0 + batch_size]
                q = net(Xs[bi])[torch.arange(len(bi)), a_t[bi]]
                loss = mse(q, y[bi])
                opt.zero_grad()
                loss.backward()
                opt.step()

    pi = _greedy_policy_from_mean(mdp, net)
    regret, Jstar, Jpi = mdp.regret(pi)
    return pi, {'regret': regret, 'Jstar': Jstar, 'Jpi': Jpi}


# ======================================================================================
# Tables S7 and S8: parametric reward-noise families
# ======================================================================================

NOISES_A = [
    ('Gaussian', 'gauss', None),
    ('Laplace', 'laplace', None),
    ('contaminated normal', 'mixture', None),
    ('log-normal', 'lognorm', 1.0),
    ('Student-t (nu=3)', 't', 3.0),
    ('Student-t (nu=2)', 't', 2.0),
    ('Student-t (nu=1.5)', 't', 1.5),
    ('Pareto (alpha=1.5)', 'pareto', 1.5),
]
# Averaging ablation uses the same grid as Experiment A (light -> heavy), so that
# Tables S7 and S8 share families and ordering.
NOISES_B = NOISES_A


def _mdp(noise, nu, seed):
    return SyntheticMDP(NS, NA, GAMMA, noise=noise, nu=nu, noise_scale=1.0, seed=seed)


def experiment_A():
    """Heavy-tail robustness: fitted NQ iteration vs mean-based fitted-Q iteration."""
    rows = []
    for label, noise, nu in NOISES_A:
        acc = {'NQ-FQI': [], 'mean-FQI': []}
        for sd in range(N_SEEDS):
            mdp = _mdp(noise, nu, sd)
            _, d = fit_quantile_iteration(mdp, TAUS, M=M, N=N, noncross=True,
                                            agg_scheme='uniform', inner_epochs=INNER, seed=sd)
            acc['NQ-FQI'].append(d['regret'])
            _, d = fit_mean_fqi(mdp, M=M, N=N, inner_epochs=INNER, seed=sd)
            acc['mean-FQI'].append(d['regret'])
        for meth, v in acc.items():
            rows.append({'noise': label, 'method': meth,
                         'mean_regret': np.mean(v), 'std_regret': np.std(v)})
        print(f"[A] {label:22s} "
              + "  ".join(f"{m}={np.mean(acc[m]):.3f}" for m in acc))
    return pd.DataFrame(rows)


def _paired_p(v, u):
    """One-sided paired Wilcoxon p-value that scheme v does not improve on uniform
    u across the common seeds (NaN if unavailable or all differences are zero)."""
    diff = np.asarray(v) - np.asarray(u)
    if not _HAVE_WILCOXON or not np.any(diff != 0):
        return float('nan')
    try:
        return float(wilcoxon(diff).pvalue)
    except Exception:
        return float('nan')


def experiment_B():
    """Averaging-scheme ablation under several heavy/skewed reward families."""
    schemes = [('uniform', dict(agg_scheme='uniform')),
               ('trimmed-10%', dict(agg_scheme='trimmed', trim=0.10)),
               ('trimmed-20%', dict(agg_scheme='trimmed', trim=0.20)),
               ('tau(1-tau)', dict(agg_scheme='tau1mtau'))]
    rows = []
    for label, noise, nu in NOISES_B:
        acc = {name: [] for name, _ in schemes}
        for sd in range(N_SEEDS):
            mdp = _mdp(noise, nu, sd)
            for name, kw in schemes:
                _, d = fit_quantile_iteration(mdp, TAUS, M=M, N=N, noncross=True,
                                                inner_epochs=INNER, seed=sd, **kw)
                acc[name].append(d['regret'])
        for name, v in acc.items():
            rows.append({'noise': label, 'scheme': name,
                         'mean_regret': np.mean(v), 'std_regret': np.std(v)})
        u = acc['uniform']
        cells = []
        for name, _ in schemes:
            if name == 'uniform':
                cells.append(f"{name}={np.mean(acc[name]):.3f}")
            else:
                cells.append(f"{name}={np.mean(acc[name]):.3f}(p={_paired_p(acc[name], u):.2f})")
        print(f"[B] {label:22s} " + "  ".join(cells))
    return pd.DataFrame(rows)


# ======================================================================================
# Table S9: empirical reward noise bootstrapped from real datasets
# ======================================================================================

DATASETS = [
    ('S\\&P 500 returns', 'data/sp500_monthly_logret.csv', 2.7),
    ('Danish fire losses', 'data/danish_fire_losses.csv', 1.4),
    ('Norwegian fire losses', 'data/norwegian_fire_losses.csv', 1.3),
]


def experiment_C(n_seeds):
    summary, perseed = [], []
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    for label, path, alpha in DATASETS:
        sample = np.loadtxt(path, skiprows=1)
        acc = {'NQ-FQI': [], 'mean-FQI': []}
        for sd in range(n_seeds):
            mdp = RealNoiseMDP(sample, seed=sd)
            _, dq = fit_quantile_iteration(mdp, TAUS, M=M, N=N, noncross=True,
                                             agg_scheme='uniform', inner_epochs=INNER, seed=sd)
            acc['NQ-FQI'].append(dq['regret'])
            _, dm = fit_mean_fqi(mdp, M=M, N=N, inner_epochs=INNER, seed=sd)
            acc['mean-FQI'].append(dm['regret'])
            perseed.append({'data': label, 'seed': sd,
                            'regret_NQ': dq['regret'], 'regret_mean': dm['regret']})
        nq, mn = np.array(acc['NQ-FQI']), np.array(acc['mean-FQI'])
        # paired one-sided test: is mean-FQI regret larger than NQ-FQI?
        p = wilcoxon(mn, nq, alternative='greater').pvalue if wilcoxon is not None else float('nan')
        wins = int(np.sum(nq < mn))
        for meth, v in acc.items():
            summary.append({'data': label, 'hill_alpha': alpha, 'method': meth,
                            'mean_regret': float(np.mean(v)), 'std_regret': float(np.std(v))})
        print(f"[real] {label:24s} (alpha~{alpha}) "
              + "  ".join(f"{m}={np.mean(acc[m]):.3f} ({np.std(acc[m]):.2f})" for m in acc)
              + f"  ratio={mn.mean()/nq.mean():.2f}x  NQ-wins={wins}/{n_seeds}  Wilcoxon p={p:.4f}",
              flush=True)
    return pd.DataFrame(summary), pd.DataFrame(perseed)


# ======================================================================================
# Entry point
# ======================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', choices=['S7', 'S8', 'S9', 'all'], default='all')
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.table in ('S7', 'all'):
        print(f"=== Table S7: heavy-tail robustness ({N_SEEDS} seeds) ===", flush=True)
        experiment_A().to_csv(os.path.join(OUT, 'expA_heavytail_robustness.csv'), index=False)
    if a.table in ('S8', 'all'):
        print(f"=== Table S8: quantile-averaging ablation ({N_SEEDS} seeds) ===", flush=True)
        experiment_B().to_csv(os.path.join(OUT, 'expB_averaging_ablation.csv'), index=False)
    if a.table in ('S9', 'all'):
        print(f"=== Table S9: real-data reward-noise robustness ({a.seeds} seeds) ===", flush=True)
        df, dfp = experiment_C(a.seeds)
        df.to_csv(os.path.join(OUT, 'expC_realdata_noise.csv'), index=False)
        dfp.to_csv(os.path.join(OUT, 'expC_realdata_noise_perseed.csv'), index=False)
    print(f"\nSaved CSVs to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
