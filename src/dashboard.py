import streamlit as st
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(
    page_title="Drift-Aware LSTM Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------------------------------- #
#  Page Header                                                                  #
# --------------------------------------------------------------------------- #
st.title("🛡️ Drift-Aware Continual LSTM for Fraud Detection")
st.markdown(
    "Comparing a **Static Baseline** against two continual-learning "
    "configurations that detect and adapt to emerging fraud patterns."
)

# --------------------------------------------------------------------------- #
#  Data Loading                                                                 #
# --------------------------------------------------------------------------- #
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

PALETTE = {
    'Static_MLP_Baseline':                   '#f39c12',
    'Static_Baseline':                       '#e74c3c',
    'Config_A_ADWIN_and_Random':             '#3498db',
    'Config_B_Window_and_Balanced':          '#2ecc71',
    'Config_C_Dual_and_Dynamic_and_Residual':'#9b59b6',
    'Config_D_EWC_Continual_LSTM':           '#e67e22',
}
LABELS = {
    'Static_MLP_Baseline':                   'Static MLP Baseline',
    'Static_Baseline':                       'Static Baseline LSTM',
    'Config_A_ADWIN_and_Random':             'Config A  (ADWIN + Random)',
    'Config_B_Window_and_Balanced':          'Config B  (Window + Balanced)',
    'Config_C_Dual_and_Dynamic_and_Residual':'Config C (Continual LSTM)',
    'Config_D_EWC_Continual_LSTM':           'Config D (EWC Continual LSTM)',
}

@st.cache_data
def load_data():
    metric_files = glob.glob(os.path.join(data_dir, '*_metrics.csv'))
    drift_files  = glob.glob(os.path.join(data_dir, '*_drifts.csv'))

    if not metric_files:
        return None, None

    metrics = {}
    drifts  = {}

    for f in metric_files:
        key = os.path.basename(f).replace('_metrics.csv', '')
        # Only load Neural Network models defined in LABELS
        if key in LABELS:
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    metrics[key] = df
            except Exception:
                pass

    for f in drift_files:
        key = os.path.basename(f).replace('_drifts.csv', '')
        if key in LABELS:
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    raw = df.iloc[:, 0].dropna().tolist()
                    drifts[key] = [int(d) for d in raw]
                else:
                    drifts[key] = []
            except Exception:
                drifts[key] = []

    return metrics, drifts


metrics, drifts = load_data()

if not metrics:
    st.warning(
        "No data found. Run the simulation first:\n\n"
        "```\ncd src\npython continual_learner.py\n```"
    )
    st.stop()

names = list(metrics.keys())

# --------------------------------------------------------------------------- #
#  Sidebar                                                                      #
# --------------------------------------------------------------------------- #
st.sidebar.header("⚙️ Controls")
window_size     = st.sidebar.slider("Smoothing window (batches)", 5, 200, 50)
selected_metric = st.sidebar.selectbox(
    "Primary metric to plot",
    ['pr_auc', 'roc_auc', 'f1', 'recall', 'precision', 'accuracy', 'kappa'],
)
show_drift_lines = st.sidebar.checkbox("Show drift event markers", value=True)

# --------------------------------------------------------------------------- #
#  Champion / Best Model Banner                                                #
# --------------------------------------------------------------------------- #
best_file = os.path.join(data_dir, 'best_model_results.csv')
if os.path.exists(best_file):
    try:
        best_df = pd.read_csv(best_file)
        if not best_df.empty:
            b_rec = best_df.iloc[0].to_dict()
            st.markdown("---")
            st.subheader("🏆 Champion / Best Historical Model Benchmark")
            
            b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
            b_col1.metric("Champion Model", str(b_rec.get('config_name', 'N/A')))
            b_col2.metric("Best F1 Score", f"{float(b_rec.get('f1', 0.0)):.4f}")
            b_col3.metric("Best ROC-AUC", f"{float(b_rec.get('roc_auc', 0.0)):.4f}")
            b_col4.metric("Best PR-AUC", f"{float(b_rec.get('pr_auc', 0.0)):.4f}")
            b_col5.metric("Git Commit", str(b_rec.get('git_commit', 'N/A')))
            
            st.caption(f"📅 Best benchmark recorded at: {b_rec.get('timestamp', 'N/A')} using metric: {b_rec.get('primary_metric_used', 'f1').upper()}")
    except Exception as e:
        pass
st.markdown("---")
st.subheader("📈 Summary Statistics (last 100 batches)")

kpi_cols = st.columns(len(names))
for col, name in zip(kpi_cols, names):
    df   = metrics[name]
    tail = df.tail(100)
    label = LABELS.get(name, name)
    color = PALETTE.get(name, '#999999')

    avg_pr_auc = tail['pr_auc'].mean() if 'pr_auc' in tail else 0.0
    avg_roc_auc = tail['roc_auc'].mean() if 'roc_auc' in tail else 0.5
    avg_recall = tail['recall'].mean()
    avg_f1     = tail['f1'].mean()
    n_drifts   = len(drifts.get(name, []))

    with col:
        st.markdown(
            f"<h4 style='color:{color}'>{label}</h4>",
            unsafe_allow_html=True,
        )
        st.metric("Avg PR-AUC (last 100 batches)", f"{avg_pr_auc:.3f}")
        st.metric("Avg ROC-AUC (last 100)", f"{avg_roc_auc:.3f}")
        st.metric("Avg Recall (last 100)", f"{avg_recall:.3f}")
        st.metric("Drift events detected", n_drifts)

st.markdown("---")
st.subheader(f"📊 {selected_metric.capitalize()} over Time")

fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

for name in names:
    df    = metrics[name].copy()
    color = PALETTE.get(name, '#999999')
    label = LABELS.get(name, name)

    df['smoothed'] = df[selected_metric].rolling(
        window=window_size, min_periods=1
    ).mean()

    ax.plot(df['batch'], df['smoothed'], label=label, color=color,
            linewidth=1.8, alpha=0.9)

    if show_drift_lines:
        for d in drifts.get(name, []):
            ax.axvline(x=d, color=color, linestyle='--', alpha=0.4, linewidth=1)

ax.set_xlabel("Batch Number", color='white')
ax.set_ylabel(selected_metric.capitalize(), color='white')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('#444')
ax.spines['left'].set_color('#444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(facecolor='#1a1a2e', labelcolor='white')
fig.tight_layout()
st.pyplot(fig)

# --------------------------------------------------------------------------- #
#  Recall + F1 side-by-side                                                    #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("🔍 Recall vs F1 (fraud-focused)")

fig2, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=False)
fig2.patch.set_facecolor('#0e1117')

for ax, metric_name in zip(axes, ['recall', 'f1']):
    ax.set_facecolor('#0e1117')
    for name in names:
        df    = metrics[name].copy()
        color = PALETTE.get(name, '#999999')
        label = LABELS.get(name, name)

        df['smoothed'] = df[metric_name].rolling(
            window=window_size, min_periods=1
        ).mean()
        ax.plot(df['batch'], df['smoothed'], label=label,
                color=color, linewidth=1.6, alpha=0.9)

        if show_drift_lines:
            for d in drifts.get(name, []):
                ax.axvline(x=d, color=color, linestyle='--',
                           alpha=0.35, linewidth=1)

    ax.set_title(metric_name.capitalize(), color='white')
    ax.set_xlabel("Batch", color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

fig2.tight_layout()
st.pyplot(fig2)

# --------------------------------------------------------------------------- #
#  Comparison Heatmap                                                           #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("🗺️ Final Performance Heatmap (last 100 batches)")

heatmap_data = {}
for name in names:
    tail = metrics[name].tail(100)
    heatmap_data[LABELS.get(name, name)] = {
        'Recall':    round(tail['recall'].mean(), 4),
        'F1':        round(tail['f1'].mean(), 4),
        'Precision': round(tail['precision'].mean(), 4),
        'Accuracy':  round(tail['accuracy'].mean(), 4),
    }

hm_df = pd.DataFrame(heatmap_data).T

fig3, ax3 = plt.subplots(figsize=(10, 3))
fig3.patch.set_facecolor('#0e1117')
sns.heatmap(
    hm_df, annot=True, fmt='.3f', cmap='YlGn',
    linewidths=0.5, ax=ax3, cbar_kws={'shrink': 0.7}
)
ax3.set_title("Mean metric value over last 100 batches", color='white', pad=10)
ax3.tick_params(colors='white')
fig3.tight_layout()
st.pyplot(fig3)

# --------------------------------------------------------------------------- #
#  Drift Event Timeline                                                         #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("⚡ Drift Event Timeline")

drift_cols = st.columns(len(names))
for col, name in zip(drift_cols, names):
    label  = LABELS.get(name, name)
    color  = PALETTE.get(name, '#999999')
    events = drifts.get(name, [])

    with col:
        st.markdown(
            f"<h5 style='color:{color}'>{label}</h5>",
            unsafe_allow_html=True,
        )
        if events:
            drift_df = pd.DataFrame({'Drift at batch': events})
            st.dataframe(drift_df, hide_index=True, use_container_width=True)
        else:
            if name == 'Static_Baseline':
                st.info("No detector – expected.")
            else:
                st.warning("No drifts detected.")

# --------------------------------------------------------------------------- #
#  Dynamic Insights                                                             #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("💡 Insights")

def _fmt(val):
    return f"{val:.3f}"

# Build insight strings dynamically from the data
bl_recall = metrics.get('Static_Baseline', pd.DataFrame({'recall': [0]}))['recall'].mean()
ca_recall  = metrics.get('Config_A_ADWIN_and_Random', pd.DataFrame({'recall': [0]}))['recall'].mean()
cb_recall  = metrics.get('Config_B_Window_and_Balanced', pd.DataFrame({'recall': [0]}))['recall'].mean()

ca_drifts = len(drifts.get('Config_A_ADWIN_and_Random', []))
cb_drifts = len(drifts.get('Config_B_Window_and_Balanced', []))

best = max(
    [('Static Baseline', bl_recall),
     ('Config A', ca_recall),
     ('Config B', cb_recall)],
    key=lambda x: x[1]
)

st.markdown(f"""
| Config | Avg Recall | Drifts Detected |
|---|---|---|
| Static Baseline | {_fmt(bl_recall)} | 0 (no detector) |
| Config A (ADWIN + Random) | {_fmt(ca_recall)} | {ca_drifts} |
| Config B (Window + Balanced) | {_fmt(cb_recall)} | {cb_drifts} |

- **Best average recall**: **{best[0]}** ({_fmt(best[1])})
- **Static Baseline** has no drift detector, so it never adapts. Its recall degrades
  after each concept drift injection as the fraud pattern changes.
- **Config A (ADWIN + Random)** uses ADWIN on the F1 stream — sensitive to statistical
  mean shifts, fast to react, but random memory may forget older fraud patterns.
- **Config B (Window + Balanced)** uses a sliding-window recall monitor and a balanced
  memory buffer — more stable recovery with better retention of minority-class patterns.
- Drifts marked as **dashed vertical lines** on the charts above correspond to the exact
  batches where the learner detected a shift and triggered model adaptation.
""")