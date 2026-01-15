# Deep Distributional Learning with Non-crossing Quantile Network (NQ-Net)

This repository contains the full implementation and experimental code for the paper:

**Deep Distributional Learning with Non-crossing Quantile Network.** Shen, G., Dai, R., Wu, G., Luo, S., Shi, C., & Zhu, H. `arXiv preprint 	arXiv:2504.08215`

The code reproduces all simulation studies, real-data analyses, and figures reported in the paper, including applications to causal inference using the IHDP and ACIC datasets.

---

## Repository Structure


├── dataset/                       # Preprocessed datasets

├── model/                         # Estimation models (NQ-Net and baselines)


├── simulation_univariate/         # Univariate simulation results

├── simulation_multivariate/       # Multivariate simulation results


├── ACIC_results/                  # Output results for ACIC real-data analysis

├── IHDP_results/                  # Output results for IHDP real-data analysis


├── simulate_univariate.py         # Univariate simulation experiments

├── simulation_multivariate.py     # Multivariate simulation experiments


├── realdata_IHDP.py               # IHDP causal inference analysis

├── realdata_ACIC.py               # ACIC causal inference analysis
│

├── generate.py                    # Utility functions for data generation


├── figure1.py                     # Code to reproduce Figure 1

├── figure3.py                     # Code to reproduce Figure 3

├── figure4.py                     # Code to reproduce Figure 4

└── kernel_time.py                 # Computational time comparison for kernel methods

---

## Requirements

The code is written in Python and requires the following main packages:

- Python >= 3.8
- NumPy
- SciPy
- PyTorch
- scikit-learn
- matplotlib

GPU acceleration is optional.

---

## Reproducing Simulation Results

### Univariate Simulations
To reproduce the univariate simulation studies reported in the paper, run:
```bash
python simulate_univariate.py
```


### Multivariate Simulations
To reproduce the multivariate simulation studies, run:
```bash
python simulation_multivariate.py
```

The results will be saved in the corresponding simulation_univariate/ and simulation_multivariate/ folders.

### Reproducing Real-Data Analyses

## Download Datasets
Please down the datasets of IDHP and ACIC via:

https://www.dropbox.com/scl/fo/a8hgr5d3yjyoigf7p68y7/AE5Xw3eDZNw0MCIYyWFkw4g?rlkey=jmbfvlyu1qmk95yzdu19s3kde&e=1&st=1n8a6c1o&dl=0

Then put them into a folder named  `datasets` under the working directory.

## IHDP Dataset
To reproduce the IHDP distributional treatment effect analysis:
```bash
python realdata_IHDP.py
```
Results will be saved to IHDP_results/.

## ACIC Dataset
To reproduce the ACIC analysis:
```bash
python realdata_ACIC.py
```
Results will be saved to ACIC_results/.

### Reproducing Figures
Each figure in the paper can be reproduced independently:

python figure1.py
python figure3.py
python figure4.py

The generated figures correspond exactly to those reported in the manuscript.

### Citation
If you use this code or find it helpful, please cite our paper:

Shen, G., Dai, R., Wu, G., Luo, S., Shi, C., & Zhu, H. (2025). Deep Distributional Learning with Non-crossing Quantile Network. arXiv preprint 	arXiv:2504.08215
