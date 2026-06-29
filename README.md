# Deep Distributional Learning with Non-crossing Quantile Network (NQ-Net)

This repository contains the implementation and experiment code for the paper
*Deep Distributional Learning with Non-crossing Quantile Network*
(Shen, G., Dai, R., Wu, G., Luo, S., Shi, C., and Zhu, H.).

The code reproduces the simulation studies, real-data analyses, the
distributional reinforcement-learning experiments, and the figures reported in
the paper and its supplementary material, including the causal-inference
applications on the IHDP and ACIC 2019 datasets.

## Requirements

Python 3.8 or later with NumPy, SciPy, pandas, PyTorch, scikit-learn, numba,
cvxopt, and matplotlib. A GPU is optional.

## Repository structure

- `model/` — the NQ-Net architecture and the baseline estimators (DQR, DQRP,
  NC-QR-DQN, Cheb-Net, and Linear / Forest / Kernel / Spline quantile regression).
- `generate.py` — data-generating utilities for the simulation studies.
- `data/` — input datasets: the IHDP and ACIC 2019 causal-inference benchmarks
  and the real heavy-tailed series used as reward noise (see *Datasets* below).
- `RL_results/`, `IHDP_results/`, `ACIC_results/`, `figures/` — output tables and
  figures.

## Reproducing the results

| Script | What it reproduces |
| --- | --- |
| `simulation_univariate.py` | Univariate simulation studies (estimation-error tables). |
| `simulation_multivariate.py` | Multivariate simulation studies (Scenarios 1–3; main-text and supplementary tables). |
| `realdata_IHDP.py` | IHDP analysis (`--mode table6`) and the non-Gaussian conditional-distribution descriptors (`--mode nongauss`). |
| `realdata_ACIC.py` | ACIC 2019 analysis (ATE / SD\_CATE / non-crossing rate / runtime). |
| `architecture_robustness.py` | Architecture-robustness study (NQ-Net error across network depth/width). |
| `kernel_time.py` | Computational-scalability comparison of the Kernel and Spline baselines. |
| `tableS7_S9.py` | Distributional reinforcement-learning experiments (`--table` selects `S7`, `S8`, `S9`, or `all`): heavy-tail robustness, the quantile-averaging ablation, and the real-data reward-noise comparison. |
| `figure1.py`, `figure3.py`, `figureS5.py` | The estimated-quantile, model, and AQTE-subplot figures. |

Each script runs with `python <script>.py`. The sharded simulation scripts
(`simulation_multivariate.py` and `architecture_robustness.py`) first regenerate
their intermediate result files with the `--run` flag and then build the
corresponding tables or figure; other scripts accept options such as `--mode`,
`--table`, and `--workers`, documented in their headers.

## Datasets

The IHDP, ACIC 2019, and real reward-noise series are read from `data/`. The
causal-inference benchmarks can also be downloaded here and placed under `data/`:

<https://www.dropbox.com/scl/fo/wwi7jbouh4v59c9fppimh/APeSlB8XjzSSHZFxGFv-0YU?rlkey=kcg0oviucoqfeuwlraz0upgic&st=epjl34jw&dl=0>

## Citation

If you use this code, please cite:

> Shen, G., Dai, R., Wu, G., Luo, S., Shi, C., and Zhu, H.
> *Deep Distributional Learning with Non-crossing Quantile Network.*
