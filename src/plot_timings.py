import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Active model configurations
CONFIGS = [
    {'name': 'Static Baseline LSTM'},
    {'name': 'Config A (ADWIN + Random)'},
    {'name': 'Config F (Dual Drift + Prioritized + A-GEM)'},
    {'name': 'Config G (Dual Drift + Random + A-GEM)'},
]

COLOR_MAP = {
    'Static Baseline LSTM': '#e74c3c',
    'Config A (ADWIN + Random)': '#3498db',
    'Config F (Dual Drift + Prioritized + A-GEM)': '#d35400',
    'Config G (Dual Drift + Random + A-GEM)': '#2ecc71',
}


def plot_batch_latencies():
    """Reads latency_ms metrics from active model runs and generates a comparison plot."""
    latency_summary = {}

    for conf in CONFIGS:
        cname = conf['name']
        safe = cname.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'and')
        pattern = os.path.join(DATA_DIR, f'{safe}*_metrics.csv')
        matched_files = glob.glob(pattern)

        seed_means = []
        for mf in matched_files:
            try:
                m_df = pd.read_csv(mf)
                if 'latency_ms' in m_df.columns:
                    valid_latencies = m_df['latency_ms'].dropna()
                    if len(valid_latencies) > 0:
                        seed_means.append(valid_latencies.mean())
            except Exception as e:
                print(f"Warning reading {mf}: {e}")

        if seed_means:
            latency_summary[cname] = float(np.mean(seed_means))

    if not latency_summary:
        print("[!] No latency metrics found in data directory.")
        return

    # Create dark-themed bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    names = list(latency_summary.keys())
    latencies = [latency_summary[k] for k in names]
    colors = [COLOR_MAP.get(k, '#aaaaaa') for k in names]

    bars = ax.bar(names, latencies, color=colors, width=0.55, zorder=2)

    max_val = max(latencies) if latencies else 1.0
    for bar, t in zip(bars, latencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.015,
            f'{t:.1f} ms',
            ha='center',
            va='bottom',
            color='white',
            fontsize=10,
            fontweight='bold'
        )

    ax.set_xlabel('Model Configuration', color='white', fontsize=12)
    ax.set_ylabel('Average Batch Latency (ms)', color='white', fontsize=12)
    ax.set_title('Average Batch Latency per Model (ms)', color='white',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', color='white', fontsize=9)
    ax.yaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7, zorder=1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    output_file = os.path.join(DATA_DIR, 'batch_latencies.png')
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"[SUCCESS] Latency plot saved to: {output_file}")
    for name, lat in latency_summary.items():
        print(f"  • {name}: {lat:.2f} ms")


if __name__ == '__main__':
    plot_batch_latencies()
