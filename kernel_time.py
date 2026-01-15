import time
import torch
import numpy as np
import pandas as pd
import os

from generate import gen_multi
from model.KernelQR import KernelQR
from model.SplineQR import SplineQR

# Parameter settings
taus = torch.linspace(0.05, 0.95, 19).unsqueeze(1)
model = 'linear'
error = 't'
df = 2
sizes = [256, 512, 1024]
ds = [1, 2, 3]
R = 3

#%%
results = []

for SIZE in sizes:
    for d in ds:
        torch.manual_seed(2024)
        A = -1 * torch.randint(0, 3, [d, 1]) * torch.randn([d, 1])
        torch.manual_seed(2025)
        B = -1 * torch.randint(0, 3, [d, 1]) * torch.randn([d, 1])

        print(f"\nTesting SIZE={SIZE}, d={d}")
        # KernelQR timing
        kernel_times = []
        for r in range(R):
            data_train = gen_multi(A, B, model=model, size=SIZE, error=error, df=df, d=d)
            x_train, y_train = data_train[:][0], data_train[:][1]
            kqr = KernelQR(quantiles=taus)
            start = time.time()
            kqr.fit(x_train, y_train)
            end = time.time()
            kernel_times.append(end - start)
            print(f"  KernelQR Rep {r+1}: {end-start:.2f} sec")
        kernel_mean = np.mean(kernel_times)
        kernel_std = np.std(kernel_times)

        results.append({
            'SIZE': SIZE,
            'd': d,
            'KernelQR_mean': round(kernel_mean, 2),
            'KernelQR_std': round(kernel_std, 2),
        })

#%%
df = pd.DataFrame(results)

# Combine mean and std into a single string
df['KernelQR_time'] = df.apply(
    lambda row: f"{row['KernelQR_mean']} ({row['KernelQR_std']})", axis=1
)

# Create pivot table: rows are d, columns are SIZE, values are mean (std)
pivot = df.pivot(index='d', columns='SIZE', values='KernelQR_time')


import re

# Function to round mean and std in the string to 2 decimals
def round_str(s):
    # s format: '0.123 (0.045)'
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if len(nums) == 2:
        mean = float(nums[0])
        std = float(nums[1])
        return f"{mean:.2f} ({std:.2f})"
    else:
        return s

pivot_2d = pivot.applymap(round_str)


# Print the summary table
print("\nSummary table (columns: SIZE, rows: d, values: mean (std)):")
print(pivot_2d)


# Save as csv
output_folder = './timing_summary'
os.makedirs(output_folder, exist_ok=True)
pivot_2d.to_csv(os.path.join(output_folder, 'kernel_timing_pivot.csv'))
