import json
import os
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_summary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def plot_metric_by_dataset(summary, metric, outdir):
    ensure_dir(outdir)
    datasets = list(summary.keys())
    models = sorted({m for d in datasets for m in summary[d].keys()})

    # Build data matrix: rows=datasets, cols=models
    values = []
    for d in datasets:
        row = []
        for m in models:
            v = summary[d].get(m, {}).get(metric)
            row.append(v if v is not None else float('nan'))
        values.append(row)

    # For each dataset create a grouped bar chart
    for i, d in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(models))
        y = values[i]
        ax.bar(x, [v if not math.isnan(v) else 0 for v in y], color='C0')
        # annotate N/A
        for xi, vv in zip(x, y):
            if math.isnan(vv):
                ax.text(xi, 0.02, 'N/A', ha='center', va='bottom', color='red')
            else:
                ax.text(xi, vv + (0.01 if abs(vv) < 1 else vv * 0.01), f"{vv:.3f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=25)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — {d}")
        plt.tight_layout()
        out_path = os.path.join(outdir, f"{d}_{metric}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def plot_comparison(summary, metric, outdir):
    ensure_dir(outdir)
    datasets = list(summary.keys())
    models = sorted({m for d in datasets for m in summary[d].keys()})

    # For each model get dataset values
    data = {m: [] for m in models}
    for d in datasets:
        for m in models:
            v = summary[d].get(m, {}).get(metric)
            data[m].append(v if v is not None else float('nan'))

    x = range(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = []
    start = - (len(models) - 1) / 2.0 * width
    for idx, m in enumerate(models):
        offsets.append(start + idx * width)

    for offset, m in zip(offsets, models):
        ys = [v if not math.isnan(v) else 0 for v in data[m]]
        ax.bar([xi + offset for xi in x], ys, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} comparison across datasets")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, f"comparison_{metric}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    summary_path = base / 'model_results_summary.json'
    outdir = base / 'visualizations'
    summary = load_summary(summary_path)


    # metrics to plot
    single_dataset_metrics = ['mAP50', 'precision', 'recall']
    # add more comparison metrics including mAP50-95 and training time
    comparison_metrics = ['mAP50', 'precision', 'recall', 'mAP50-95', 'training_time_seconds']

    for m in single_dataset_metrics:
        plot_metric_by_dataset(summary, m, outdir)

    for m in comparison_metrics:
        plot_comparison(summary, m, outdir)

    print(f"Saved visualizations to: {outdir}")


if __name__ == '__main__':
    main()
