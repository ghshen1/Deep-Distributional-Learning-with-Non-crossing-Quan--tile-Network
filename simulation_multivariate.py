"""
Multivariate Scenario 1/2/3 simulations and the LaTeX table blocks they feed
(Tables 4/5 in the main text and Tables S4/S5/S6 in the Supplement). Self-contained
runner + aggregator.

The data generator (generate.py) and the released models (model/) are imported
unchanged. For each scenario we draw the coefficient vectors A, B once
(seeds 2024/2025), generate training data with gen_multi and the true test
quantiles with quant_multi, fit the seven methods in the column order of Tables
S4/S5 with the shared width/lr/epochs, and report the per-level L1/L2 errors
against the truth on a 10^4-point test set, together with the non-crossing rate
(NCR). NQ-Net is fitted first. Results are sharded as one block of 10 replications
per .npz (5 shards = R=50).

Run:
  python simulation_multivariate.py                 # rebuild table blocks from existing shards
  python simulation_multivariate.py --run --workers 10   # regenerate all s{1,2,3}_N{1024,2048,4096} shards, then tables
"""
import os, argparse, glob
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

K = 19
W, LR, EP = 256, 0.01, 100
# column order EXACTLY as in Tables S4/S5
METHODS = ['Linear', 'Forest', 'DQRP', 'Cheb-Net', 'DQR', 'NC-QR-DQN', 'NQ-Net']
# NQ-Net fitted first, then the rest
FIT_ORDER = ['NQ-Net', 'NC-QR-DQN', 'DQR', 'DQRP', 'Cheb-Net', 'Linear', 'Forest']
TWO_ARG_PREDICT = {'DQRP', 'Cheb-Net'}
CFG = {1: dict(d=6,  model='scenario1', error='scenario1', df=2),
       2: dict(d=15, model='scenario2', error='t',         df=3),
       3: dict(d=30, model='scenario3', error='sinex_t',   df=3)}
TAUS_NP = np.round(np.linspace(0.05, 0.95, 19), 2)
SEL = [0, 4, 9, 14, 18]                       # tau = 0.05,0.25,0.50,0.75,0.95 (main-text columns)
NS_DEFAULT = [1024, 2048, 4096]
N_SHARDS, REPS_PER_SHARD = 5, 10              # 5 shards x 10 reps = R=50


# ------------------------------ runner ------------------------------
def _to_np(p):
    return p.detach().numpy() if hasattr(p, 'detach') else np.asarray(p)


def _fit_model(name, xtr, ytr, d, TAUS):
    import torch  # noqa: F401 (models use torch)
    from model.RandomForestQR import QuantileForest
    from model.LinearQR import LinearQR
    from model.DQR import DQR
    from model.NQNet import NQNet
    from model.NC_QR_DQN import NC_QR_DQN
    from model.DQRP import DQRP
    from model.ChebNet import ChebNet
    xn, yn = xtr.numpy(), ytr.numpy()
    if name == 'Linear':
        m = LinearQR(TAUS); m.fit(xtr, ytr)
    elif name == 'Forest':
        m = QuantileForest(quantiles=TAUS, nthreads=1); m.fit(xtr, ytr)
    elif name == 'DQR':
        m = DQR(quantiles=TAUS); m.fit(xn, yn, width_vec=[d, W, W, W, K], lr=LR, epochs=EP)
    elif name == 'NQ-Net':
        m = NQNet(quantiles=TAUS); m.fit(X=xn, y=yn, width_vec=[d, W, W, W, K], activation='ELU', lr=LR, epochs=EP)
    elif name == 'DQRP':
        m = DQRP(quantiles=TAUS); m.fit(X=xn, y=yn, width_vec=[d + 1, W, W, W, 1], activation='ReQU', lr=LR, epochs=EP)
    elif name == 'Cheb-Net':
        m = ChebNet(quantiles=TAUS); m.fit(X=xn, y=yn, phi_layer=[d, W, W, 15], K_layer=[d, W, W, 1], K='q0', lr=LR, epochs=EP)
    elif name == 'NC-QR-DQN':
        m = NC_QR_DQN(quantiles=TAUS); m.fit(X=xn, y=yn, logit_layer=[d, W, W, W, K], factor_layer=[d, W, W, W, 2], activation='ReLU', epochs=EP, lr=LR)
    else:
        raise ValueError(name)
    return m


def _predict_model(name, m, x_test, TAUS):
    return _to_np(m.predict(x_test, TAUS) if name in TWO_ARG_PREDICT else m.predict(x_test))


def run_shard(task):
    """One (scenario, N, [rep_start, rep_end)) block -> results/s{S}_N{N}_{k}.npz."""
    scenario, N, rep_start, rep_end, out = task
    import torch
    torch.set_num_threads(1)
    from generate import gen_multi, quant_multi
    TAUS = torch.linspace(0.05, 0.95, 19).unsqueeze(1)
    c = CFG[scenario]; d = c['d']
    torch.manual_seed(2024); A = -1 * torch.randint(0, 3, [d, 1]) * torch.randn([d, 1])
    torch.manual_seed(2025); B = -1 * torch.randint(0, 3, [d, 1]) * torch.randn([d, 1])
    reps = list(range(rep_start, rep_end))
    L1 = np.full((len(reps), len(METHODS), K), np.nan)
    L2 = np.full((len(reps), len(METHODS), K), np.nan)
    NCR = np.full((len(reps), len(METHODS)), np.nan)
    for ri, r in enumerate(reps):
        torch.manual_seed(r); np.random.seed(r)
        data = gen_multi(A, B, size=N, d=d, model=c['model'], error=c['error'], df=c['df'])
        xtr, ytr = data[:][0], data[:][1]
        models = {}
        for name in FIT_ORDER:
            try:
                models[name] = _fit_model(name, xtr, ytr, d, TAUS)
            except Exception as e:
                print(f"  [s{scenario} N{N} rep {r} {name}] FIT FAILED: {e}", flush=True)
        x_test = torch.rand([10000, d])
        tq = _to_np(quant_multi(x_test, TAUS, A, B, model=c['model'], error=c['error'], df=c['df']))
        for mi, name in enumerate(METHODS):
            if name not in models:
                continue
            try:
                p = _predict_model(name, models[name], x_test, TAUS)
                L1[ri, mi] = np.abs(p - tq).mean(axis=0)
                L2[ri, mi] = ((p - tq) ** 2).mean(axis=0)
                NCR[ri, mi] = np.mean(np.all(np.diff(p, axis=1) >= -1e-6, axis=1))
            except Exception as e:
                print(f"  [s{scenario} N{N} rep {r} {name}] PREDICT FAILED: {e}", flush=True)
        print(f"scenario {scenario} N{N} rep {r} done", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, L1=L1, L2=L2, NCR=NCR, reps=np.array(reps), methods=np.array(METHODS))
    print(f"wrote {out}", flush=True)


def run_all(scenarios, NS, workers):
    from multiprocessing import Pool
    tasks = [(s, N, k * REPS_PER_SHARD, (k + 1) * REPS_PER_SHARD, f'results/s{s}_N{N}_{k}.npz')
             for s in scenarios for N in NS for k in range(N_SHARDS)]
    print(f"=== multivariate simulation: {len(tasks)} shards, {workers} workers ===", flush=True)
    with Pool(workers) as pool:
        pool.map(run_shard, tasks)


# ------------------------------ aggregation (table blocks) ------------------------------
def _load(scenario, N):
    files = sorted(glob.glob(f'results/s{scenario}_N{N}_*.npz'))
    if not files:
        raise FileNotFoundError(f'no results/s{scenario}_N{N}_*.npz (run with --run first)')
    L1 = np.concatenate([np.load(f, allow_pickle=True)['L1'] for f in files], axis=0)
    NCR = np.concatenate([np.load(f, allow_pickle=True)['NCR'] for f in files], axis=0)
    return L1.mean(axis=0), L1.std(axis=0), NCR          # (7,19),(7,19),(nreps,7)


def _supp_block(mean, std, N):
    """19 rows (one per tau), columns in METHODS order, 'mean (std)' -- Tables S4/S5/S6."""
    lines = []
    for k in range(19):
        cells = " & ".join(f"{mean[mi, k]:.2f} ({std[mi, k]:.2f})" for mi in range(len(METHODS)))
        lead = f"\\multirow{{19}}{{*}}{{{N}}} & {TAUS_NP[k]:.2f}            " if k == 0 else f"& {TAUS_NP[k]:.2f}            "
        lines.append(f"{lead}& {cells} \\\\")
    closing = "\\bottomrule" if N == 4096 else "\\midrule"
    return "\n".join(lines) + " " + closing


def _main_block(scenario, NS):
    """7 method rows x 5 selected tau columns, blocked by N -- main-text Tables 4/5."""
    lines = []
    for N in NS:
        mean, std, _ = _load(scenario, N)
        closing = '\\bottomrule' if N == NS[-1] else '\\midrule'
        for mi, m in enumerate(METHODS):
            cells = " & ".join(f"{mean[mi, k]:.2f} ({std[mi, k]:.2f})" for k in SEL)
            lead = f"\\multirow{{7}}{{*}}{{{N}}} & {m:9s}" if mi == 0 else f"& {m:9s}"
            end = " \\\\ " + closing if mi == len(METHODS) - 1 else " \\\\"
            lines.append(f"{lead} & {cells}{end}")
    return "\n".join(lines)


def build_tables(supp_scenarios=(1, 2), main_scenarios=(1, 2), NS=tuple(NS_DEFAULT)):
    os.makedirs('results', exist_ok=True)
    for s in supp_scenarios:
        for N in NS:
            mean, std, NCR = _load(s, N)
            print(f"\n===== Scenario {s}, N={N}, {NCR.shape[0]} reps =====")
            print(f"{'method':10s} {'mean L1 (over levels)':>22} {'mean NCR':>10}")
            for mi, m in enumerate(METHODS):
                print(f"{m:10s} {mean[mi].mean():>22.3f} {np.nanmean(NCR[:, mi]):>10.3f}")
            out = f'results/table_block_s{s}_N{N}.tex'
            open(out, 'w').write(_supp_block(mean, std, N) + "\n")
            print(f"-> wrote {out}")
    for s in main_scenarios:
        out = f'results/main_block_s{s}.tex'
        open(out, 'w').write(_main_block(s, list(NS)) + "\n")
        print(f"-> wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', help='regenerate all s{1,2,3}_N{*} shards before building tables')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.run:
        run_all([1, 2, 3], NS_DEFAULT, a.workers)
    build_tables()


if __name__ == "__main__":
    main()
