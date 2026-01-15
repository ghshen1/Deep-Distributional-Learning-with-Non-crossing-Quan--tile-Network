import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# =============================================================================
# 1. Customized subplot plotting function
# =============================================================================
def plot_aqte_subplots_custom(aqte_df, paper_order, output_dir, sanitized_data_name):
    """
    Create one subplot for each model (fixed 2x3 grid) and highlight that model.
    Adapted to the column names in ACIC_detailed_quantile_metrics.csv:
      - Method
      - Quantile_Value
      - Estimated_ATE
    """
    print(f"\nGenerating customized AQTE subplot figure (2x3 grid, dataset: {sanitized_data_name})...")

    # Check whether required columns exist
    needed_cols = {'Method', 'Quantile_Value', 'Estimated_ATE'}
    if not needed_cols.issubset(aqte_df.columns):
        print(f"Error: data missing required columns {needed_cols}, current columns are: {list(aqte_df.columns)}")
        return
    
    # Standardize column names: 'model' for method, 'quantile' for quantile, 'aqte' for effect
    df = aqte_df.rename(columns={
        'Method': 'model',
        'Quantile_Value': 'quantile',
        'Estimated_ATE': 'aqte'
    })

    # Get the list of models to plot (only those present in the data)
    models_to_plot = [m for m in paper_order if m in df['model'].unique()]
    if not models_to_plot:
        print("Warning: none of the specified models were found in the data; cannot generate plots.")
        return

    # --- User-configurable style parameters ---
    font_sizes = {
        'suptitle': 24,       # Figure title
        'subplot_title': 18,  # Subplot title
        'axis_label': 20,     # Shared axis labels
        'legend_title': 16,   # Legend title
        'legend_text': 14,    # Legend text
    }

    # --- Set styles ---
    colors = sns.color_palette("husl", len(models_to_plot))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    
    style_dict = {
        model: {'color': colors[i], 'ls': linestyles[i % len(linestyles)], 'marker': markers[i % len(markers)]}
        for i, model in enumerate(models_to_plot)
    }

    # --- Fix layout to a 2x3 grid ---
    nrows, ncols = 2, 3
    num_subplots = nrows * ncols
    
    if len(models_to_plot) > num_subplots:
        print(f"Warning: number of models ({len(models_to_plot)}) exceeds the capacity of the 2x3 grid ({num_subplots}).")
        print(f"Only the first {num_subplots} models will be plotted: {models_to_plot[:num_subplots]}")
        models_to_plot = models_to_plot[:num_subplots]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    # --- Compute mean and confidence interval for all models ---
    summary_data = {}
    all_q_lower = []
    all_q_upper = []
    for name in models_to_plot:
        model_data = df[df['model'] == name]
        # Aggregate multiple replications by quantile
        summary = model_data.groupby('quantile')['aqte'].agg(
            mean='mean',
            lower=lambda x: x.quantile(0.05),
            upper=lambda x: x.quantile(0.95)
        ).reset_index()
        summary_data[name] = summary
        all_q_lower.extend(summary['lower'])
        all_q_upper.extend(summary['upper'])


    plt.setp(axes, ylim=(-1.1, -0.5))

    # --- Iterate over each model and fill the corresponding subplot ---
    for i, highlighted_model in enumerate(models_to_plot):
        ax = axes[i]
        
        # 1. Plot background models (in gray)
        for background_model in models_to_plot:
            if background_model == highlighted_model:
                continue
            summary = summary_data[background_model]
            ax.plot(
                summary['quantile'], summary['mean'],
                color='lightgray', linestyle='-', linewidth=1.5, zorder=1
            )

        # 2. Highlight the current model
        summary = summary_data[highlighted_model]
        style = style_dict[highlighted_model]
        
        ax.plot(
            summary['quantile'], summary['mean'],
            marker=style['marker'],
            markersize=4,
            linestyle=style['ls'],
            color=style['color'],
            label=highlighted_model,
            linewidth=3.0,
            zorder=10
        )
        ax.fill_between(
            summary['quantile'], summary['lower'], summary['upper'],
            color=style['color'], alpha=0.2, zorder=5
        )

        ax.set_title(f'{highlighted_model}', fontsize=font_sizes['subplot_title'])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True)

    # Remove unused subplots
    for i in range(len(models_to_plot), len(axes)):
        fig.delaxes(axes[i])

    # --- Add shared labels and legend ---
    fig.supxlabel(r'Quantile ($\tau$)', fontsize=font_sizes['axis_label'], y=0.02)
    fig.supylabel('Average Treatment Effect', fontsize=font_sizes['axis_label'], x=0.04)
    # fig.suptitle('Average Quantile Treatment Effect (AQTE) Comparison',
    #              fontsize=font_sizes['suptitle'], y=1.0)

    # Create a shared legend
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    from collections import OrderedDict
    unique_handles_labels = OrderedDict(zip(labels, handles))
    order_dict = {label: i for i, label in enumerate(paper_order)}
    sorted_handles_labels = sorted(
        unique_handles_labels.items(),
        key=lambda item: order_dict.get(item[0], 99)
    )

    fig.legend(
        [item[1] for item in sorted_handles_labels],
        [item[0] for item in sorted_handles_labels],
        title='Model',
        bbox_to_anchor=(0.88, 0.5),
        loc='center left',
        fontsize=font_sizes['legend_text'],
        title_fontsize=font_sizes['legend_title']
    )

    plt.tight_layout(rect=[0.03, 0.03, 0.88, 0.99])

    # --- Save figure ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_path = os.path.join(output_dir, "figure4.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Customized AQTE subplot figure saved to: {plot_path}")
    plt.close()


# =============================================================================
# 2. Main script
# =============================================================================
if __name__ == '__main__':
    # Detailed file name corresponding to the simulation output
    csv_filename = "./ACIC_results/ACIC_detailed_quantile_metrics.csv"
    output_directory = "./"
    dataset_name = "ACIC"

    if not os.path.exists(csv_filename):
        print(f"Error: data file '{csv_filename}' not found.")
    else:
        print(f"Successfully loaded data file: {csv_filename}")
        
        aqte_results_df = pd.read_csv(csv_filename)
        # Order consistent with paper_order used in the simulation
        model_order = ['Linear', 'VDQR', 'NQNet', 'DQR', 'DQRP', 'Kernel', 'Forest']
        
        plot_aqte_subplots_custom(
            aqte_df=aqte_results_df,
            paper_order=model_order,
            output_dir=output_directory,
            sanitized_data_name=dataset_name
        )
