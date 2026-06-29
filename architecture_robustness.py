"""
Architecture-robustness study for Figure S4: the sensitivity of NQ-Net to the
(depth, width) choice over the paper's six simulation models, and the architecture
picked by a holdout-validation criterion that uses no knowledge of beta or d*.
Self-contained runner + plotter.

For each (depth, width) on the grid depths {2,3,4,5,6} x widths {64,128,256,512,1024}
we train NQ-Net with the paper's setup and record (i) the test error against the
true quantiles (L2 univariate / L1 multivariate, as in the paper tables) and (ii) a
holdout pinball (check) loss on an INDEPENDENT validation draw, used only for
architecture selection. Sharded one replication per file under results/.

  * blue box  = the architecture used in the paper ([128,128,128] univariate /
                [256,256,256] multivariate; depth 3, width 128/256).
  * gold star = the architecture selected by holdout validation (argmin of the
                median pinball loss on an independent validation set).

The released model/generator are imported unchanged (model.NQNet, generate).

Run:
  python architecture_robustness.py                 # plot Figure S4 from existing shards
  python architecture_robustness.py --run --reps 30 --workers 10   # regenerate shards, then plot
"""
import os, argparse, glob
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEPTHS = [2, 3, 4, 5, 6]; WIDTHS = [64, 128, 256, 512, 1024]; K = 19
N_VAL = 2000
# (kind, model, N, lr); error/df/d come from UNIV/MULTI below
MODELS = [
    ('univ', 'linear', 1024, 1e-3), ('univ', 'angle', 1024, 1e-3), ('univ', 'wave', 1024, 1e-3),
    ('multi', 'scenario1', 2048, 1e-2), ('multi', 'scenario2', 2048, 1e-2), ('multi', 'scenario3', 2048, 1e-2),
]
UNIV = {'angle': ('sinex_t', 2), 'wave': ('expx_t', 2), 'linear': ('sinex_t', 1)}
MULTI = {'scenario1': (6, 'scenario1', 2), 'scenario2': (15, 't', 3), 'scenario3': (30, 'sinex_t', 3)}


# ------------------------------- sweep (runner) -------------------------------
def _to_np(p):
    return p.detach().numpy() if hasattr(p, 'detach') else np.asarray(p)


def _check_loss(pred, y, taus_np):
    """Average pinball (check) loss over the 19 levels; the holdout selection criterion."""
    z = y[:, None] - pred
    return float(np.mean(np.maximum(taus_np[None, :] * z, (taus_np[None, :] - 1) * z)))


def _make_data(kind, model, N, seed, TAUS):
    """Train, test (with true quantiles), and an independent validation draw. The
    validation set is drawn AFTER the test inputs so the existing test errors are
    unaffected by its addition."""
    import torch
    from generate import gen_univ, quant_univ, gen_multi, quant_multi
    if kind == 'univ':
        err, df = UNIV[model]; d = 1
        torch.manual_seed(seed); np.random.seed(seed)
        data = gen_univ(model=model, size=N, error=err, df=df)
        Xtr, ytr = data[:][0].numpy(), data[:][1].numpy()
        Xte = torch.rand([10000, 1])
        tq = quant_univ(Xte, TAUS, model=model, error=err, df=df).numpy()
        vd = gen_univ(model=model, size=N_VAL, error=err, df=df)
        Xval, yval = vd[:][0].numpy(), vd[:][1].numpy().ravel()
        return Xtr, ytr, Xte.numpy(), tq, Xval, yval, d, 'l2'
    d, err, df = MULTI[model]
    torch.manual_seed(2024); A = -1 * torch.randint(0, 3, [d, 1]) * torch.randn([d, 1])
    torch.manual_seed(2025); B = -1 * torch.randint(0, 3, [d, 1]) * torch.randn([d, 1])
    torch.manual_seed(seed); np.random.seed(seed)
    data = gen_multi(A, B, size=N, d=d, model=model, error=err, df=df)
    Xtr, ytr = data[:][0].numpy(), data[:][1].numpy()
    Xte = torch.rand([10000, d])
    tq = quant_multi(Xte, TAUS, A, B, model=model, error=err, df=df).numpy()
    vd = gen_multi(A, B, size=N_VAL, d=d, model=model, error=err, df=df)
    Xval, yval = vd[:][0].numpy(), vd[:][1].numpy().ravel()
    return Xtr, ytr, Xte.numpy(), tq, Xval, yval, d, 'l1'


def _run_shard(task):
    """One (model, rep): sweep the 5x5 grid and write results/arch_<model>_<rep>.npz."""
    kind, model, N, lr, rep = task
    import torch
    torch.set_num_threads(1)
    from model.NQNet import NQNet
    TAUS = torch.linspace(0.05, 0.95, 19).unsqueeze(1); TAUS_NP = TAUS.numpy().ravel()
    Xtr, ytr, Xte, tq, Xval, yval, d, metric = _make_data(kind, model, N, rep, TAUS)
    grid = np.full((1, len(DEPTHS), len(WIDTHS)), np.nan)
    valgrid = np.full((1, len(DEPTHS), len(WIDTHS)), np.nan)
    for i, depth in enumerate(DEPTHS):
        for j, W in enumerate(WIDTHS):
            try:
                torch.manual_seed(1000 + rep)              # clean, fixed init per rep
                nq = NQNet(quantiles=TAUS)
                nq.fit(X=Xtr, y=ytr.reshape(-1, 1), width_vec=[d] + [W] * depth + [K],
                       activation='ELU', lr=lr, epochs=100)
                p = _to_np(nq.predict(Xte)); pv = _to_np(nq.predict(Xval))
                if np.isnan(p).any() or np.isnan(pv).any():
                    raise ValueError('NaN predictions')
                grid[0, i, j] = np.mean((p - tq) ** 2) if metric == 'l2' else np.mean(np.abs(p - tq))
                valgrid[0, i, j] = _check_loss(pv, yval, TAUS_NP)
            except Exception as e:
                print(f"  [{model} rep {rep} d{depth} W{W}] FAILED: {e}", flush=True)
    os.makedirs('results', exist_ok=True)
    np.savez(f'results/arch_{model}_{rep}.npz', grid=grid, valgrid=valgrid,
             reps=np.array([rep]), depths=np.array(DEPTHS), widths=np.array(WIDTHS))
    print(f"{model} N{N} rep {rep} done", flush=True)


def sweep(reps, workers):
    from multiprocessing import Pool
    tasks = [(kind, model, N, lr, r) for (kind, model, N, lr) in MODELS for r in range(reps)]
    print(f"=== architecture sweep: {len(MODELS)} models x {reps} reps = {len(tasks)} shards, {workers} workers ===", flush=True)
    with Pool(workers) as pool:
        pool.map(_run_shard, tasks)


# ------------------------------- plot -------------------------------
PANELS = [
    ('linear',    '(a) Univariate "Linear" model ($d=1$, $N=1024$)',          'test $L_2$ error', 128),
    ('angle',     '(b) Univariate "Angle" model ($d=1$, $N=1024$)',           'test $L_2$ error', 128),
    ('wave',      '(c) Univariate "Wave" model ($d=1$, $N=1024$)',            'test $L_2$ error', 128),
    ('scenario1', '(d) Multivariate "Scenario 1" model ($d=6$, $N=2048$)',    'test $L_1$ error', 256),
    ('scenario2', '(e) Multivariate "Scenario 2" model ($d=15$, $N=2048$)',   'test $L_1$ error', 256),
    ('scenario3', '(f) Multivariate "Scenario 3" model ($d=30$, $N=2048$)',   'test $L_1$ error', 256),
]


def _load_grids(key):
    files = sorted(glob.glob(f'results/arch_{key}_*.npz'))
    if not files:
        raise FileNotFoundError(f'no results/arch_{key}_*.npz found (run with --run first)')
    G = np.concatenate([np.load(f)['grid'] for f in files], axis=0)
    V = np.concatenate([np.load(f)['valgrid'] for f in files], axis=0)
    med = np.nanmedian(G, axis=0)
    vpick = np.unravel_index(np.nanargmin(np.nanmedian(V, axis=0)), med.shape)
    return med, vpick


def _panel(ax, grid, vpick, title, metric_label, paper_w):
    best = np.nanmin(grid)
    # Color scale spans the realistic range; diverged configs (>5x best) saturate
    # at the top so they do not compress the gradient. Cell numbers always shown.
    cap = 5.0 * best
    stable = grid[np.isfinite(grid) & (grid <= cap)]
    vmax = float(np.max(stable)) * 1.03 if stable.size else 1.7 * best
    pi, pj = DEPTHS.index(3), WIDTHS.index(paper_w)
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis_r', vmin=best, vmax=vmax)
    thr = best + 0.55 * (vmax - best)
    for i in range(len(DEPTHS)):
        for j in range(len(WIDTHS)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha='center', va='center',
                    color='white' if grid[i, j] > thr else 'black', fontsize=8)
    vi, vj = vpick
    ax.scatter([vj], [vi], marker='*', s=430, color='gold', edgecolor='black', lw=1.0, zorder=7)
    ax.add_patch(plt.Rectangle((pj - 0.45, pi - 0.45), 0.90, 0.90, fill=False,
                               edgecolor='#1f4fff', lw=3.2, zorder=6))
    ax.set_xticks(range(len(WIDTHS))); ax.set_xticklabels(WIDTHS)
    ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels(DEPTHS)
    ax.set_xlabel('width per hidden layer'); ax.set_ylabel('number of hidden layers (depth)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric_label, extend='max')
    return best, grid[pi, pj], (DEPTHS[vi], WIDTHS[vj]), grid[vi, vj]


def plot():
    fig, axes = plt.subplots(2, 3, figsize=(21, 12)); ax = axes.ravel(); info = []
    for k, (key, title, lab, pw) in enumerate(PANELS):
        grid, vpick = _load_grids(key)
        best, paper_err, vsel, vsel_err = _panel(ax[k], grid, vpick, title, lab, pw)
        info.append((key, best, paper_err, paper_err / best, vsel, vsel_err))
    fig.suptitle('NQ-Net median test error across (depth, width);  blue box = architecture used in the paper,  '
                 r'$\bigstar$ = holdout-validation pick', y=1.005, fontsize=13)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/architecture_robustness.png', dpi=200, bbox_inches='tight')
    print('wrote figures/architecture_robustness.png')
    for key, best, pe, ratio, vsel, ve in info:
        print(f'  {key:10s}: best={best:.4f}  paper-arch={pe:.4f} ({int((ratio-1)*100)}% over best)  '
              f'val-pick={vsel} err={ve:.4f} ({int((ve/best-1)*100)}% over best)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', help='regenerate the sweep shards before plotting')
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.run:
        sweep(a.reps, a.workers)
    plot()


if __name__ == "__main__":
    main()
