import streamlit as st
import pandas as pd
import glob
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Drift-Aware LSTM Dashboard",
    layout="wide"
)

# --------------------------------------------------------------------------- #
#  Page Header                                                                  #
# --------------------------------------------------------------------------- #
st.title("Drift-Aware Continual LSTM for Fraud Detection")
st.markdown(
    "Comparing a **Static Baseline** against continual-learning "
    "configurations that detect and adapt to emerging fraud patterns."
)

# --------------------------------------------------------------------------- #
#  Configuration                                                                #
# --------------------------------------------------------------------------- #
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
BASELINE = 'Static_Baseline_LSTM'

PALETTE = {
    'Static_Baseline_LSTM':                         '#e74c3c',
    'Config_A_ADWIN_and_Random':                    '#3498db',
    'Config_F_Dual_Drift_and_Prioritized_and_A-GEM': '#2ecc71',
}
LABELS = {
    'Static_Baseline_LSTM':                         'Static Baseline LSTM',
    'Config_A_ADWIN_and_Random':                    'Config A (ADWIN + Random)',
    'Config_F_Dual_Drift_and_Prioritized_and_A-GEM': 'Config F (Dual Drift + Prioritized + A-GEM)',
}
METRIC_OPTIONS = ['pr_auc', 'roc_auc', 'f1', 'f2', 'recall', 'precision',
                  'precision_at_1_percent', 'accuracy', 'kappa', 'savings']


def load_data():
    """Loads seed-0 metrics/drifts (canonical files) and any extra seed runs
    saved as <name>_seed{k}_metrics.csv."""
    metrics, drifts, seeds = {}, {}, {}

    for key in LABELS:
        f = os.path.join(data_dir, f'{key}_metrics.csv')
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        metrics[key] = df
        seeds[key] = [df]

        for sf in sorted(glob.glob(os.path.join(data_dir, f'{key}_seed*_metrics.csv'))):
            try:
                sdf = pd.read_csv(sf)
                if not sdf.empty:
                    seeds[key].append(sdf)
            except Exception:
                pass

        d_file = os.path.join(data_dir, f'{key}_drifts.csv')
        drifts[key] = []
        if os.path.exists(d_file):
            try:
                ddf = pd.read_csv(d_file)
                if not ddf.empty:
                    drifts[key] = [int(d) for d in ddf.iloc[:, 0].dropna().tolist()]
            except Exception:
                pass

    return metrics, drifts, seeds


metrics, drifts, seeds = load_data()

gt_file = os.path.join(data_dir, 'ground_truth_drifts.csv')
gt_drifts = []
if os.path.exists(gt_file):
    try:
        gt_drifts = [int(x) for x in pd.read_csv(gt_file)['batch'].tolist()]
    except Exception:
        gt_drifts = []

if not metrics:
    st.warning(
        "No data found. Run the simulation first:\n\n"
        "```\ncd src\npython continual_learner.py\n```"
    )
    st.stop()

all_names = list(metrics.keys())
n_seeds = min(len(v) for v in seeds.values())

# --------------------------------------------------------------------------- #
#  Sidebar                                                                      #
# --------------------------------------------------------------------------- #
st.sidebar.header("Controls")
names = st.sidebar.multiselect(
    "Models to display",
    options=all_names,
    default=all_names,
    format_func=lambda k: LABELS.get(k, k),
)
if not names:
    st.info("Select at least one model in the sidebar.")
    st.stop()

selected_metric = st.sidebar.selectbox("Primary metric to plot", METRIC_OPTIONS)
window_size = st.sidebar.slider("Smoothing window (batches)", 1, 200, 50)
tail_n = st.sidebar.slider("Summary window (last N batches)", 20, 500, 100, step=10)
recovery_w = st.sidebar.slider("Drift before/after window (batches)", 10, 200, 50, step=10)
show_drift_lines = st.sidebar.checkbox("Show drift event markers", value=True)
st.sidebar.caption(f"Runs per model: **{n_seeds}** (mean ± std shown when > 1)")

PLOT_LAYOUT = dict(template="plotly_dark", height=430, hovermode="x unified",
                   margin=dict(l=10, r=10, t=40, b=10),
                   legend=dict(orientation="h", y=-0.2))


def lab(k):
    return LABELS.get(k, k)


def smooth(series):
    return series.rolling(window=window_size, min_periods=1).mean()


def add_drift_lines(fig, key, row=None, col=None):
    if not show_drift_lines:
        return
    for d in drifts.get(key, []):
        kw = dict(x=d, line_dash="dash", line_color=PALETTE.get(key, '#999'),
                  opacity=0.45, line_width=1)
        if row is not None:
            fig.add_vline(row=row, col=col, **kw)
        else:
            fig.add_vline(**kw)


def add_gt_lines(fig, row=None, col=None):
    """Solid white lines = injected (ground-truth) drift points."""
    for d in gt_drifts:
        kw = dict(x=d, line_color='white', line_width=2, opacity=0.6,
                  annotation_text="true drift", annotation_position="top")
        if row is not None:
            fig.add_vline(row=row, col=col, **kw)
        else:
            fig.add_vline(**kw)


def tail_stats(key, metric):
    """Mean and std (across seeds) of the metric's mean over the last tail_n batches."""
    vals = []
    for sdf in seeds[key]:
        if metric in sdf:
            vals.append(sdf.tail(tail_n)[metric].mean())
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


# (Champion Banner removed)

# --------------------------------------------------------------------------- #
#  Summary Table (mean ± std, deltas vs baseline, best highlighted)             #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader(f"Summary Statistics (last {tail_n} batches)")

SUMMARY_COLS = [('pr_auc', 'PR-AUC'), ('roc_auc', 'ROC-AUC'), ('recall', 'Recall'),
                ('precision', 'Precision'), ('f1', 'F1'), ('savings', 'Savings/batch ($)')]

raw_rows, display_rows = {}, {}
for k in names:
    raw_rows[lab(k)] = {}
    display_rows[lab(k)] = {}
    for m, title in SUMMARY_COLS:
        mu, sd = tail_stats(k, m)
        raw_rows[lab(k)][title] = mu
        if m == 'savings':
            txt = f"${mu:,.0f}"
            if n_seeds > 1:
                txt += f" ± {sd:,.0f}"
        else:
            txt = f"{mu:.3f}"
            if n_seeds > 1:
                txt += f" ± {sd:.3f}"
        display_rows[lab(k)][title] = txt
    display_rows[lab(k)]['Drifts'] = len(drifts.get(k, []))

raw_df = pd.DataFrame(raw_rows).T
disp_df = pd.DataFrame(display_rows).T
best_per_col = raw_df.idxmax()


def highlight_best(col):
    if col.name not in best_per_col:
        return [''] * len(col)
    return ['background-color: rgba(46,204,113,0.30); font-weight: 700'
            if idx == best_per_col[col.name] else '' for idx in col.index]


st.dataframe(disp_df.style.apply(highlight_best, axis=0), use_container_width=True)
st.caption("Best value per metric. Values are mean ± std across independent seeds.")

# Delta vs baseline cards
if BASELINE in names and len(names) > 1:
    st.markdown(f"**Δ vs. {lab(BASELINE)}** (savings and PR-AUC)")
    others = [k for k in names if k != BASELINE]
    cols = st.columns(len(others))
    base_sav, _ = tail_stats(BASELINE, 'savings')
    base_pr, _ = tail_stats(BASELINE, 'pr_auc')
    for col, k in zip(cols, others):
        sav, _ = tail_stats(k, 'savings')
        pr, _ = tail_stats(k, 'pr_auc')
        with col:
            st.markdown(f"<h5 style='color:{PALETTE.get(k, '#999')}'>{lab(k)}</h5>",
                        unsafe_allow_html=True)
            st.metric("Savings / batch", f"${sav:,.0f}", f"{sav - base_sav:+,.0f}")
            st.metric("PR-AUC", f"{pr:.3f}", f"{pr - base_pr:+.3f}")

# --------------------------------------------------------------------------- #
#  Primary metric over time                                                     #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader(f"{selected_metric} over Time")

fig = go.Figure()
for k in names:
    df = metrics[k]
    color = PALETTE.get(k, '#999999')
    s = smooth(df[selected_metric])
    if n_seeds > 1:
        # mean ± std band across seeds (aligned on common length)
        L = min(len(x) for x in seeds[k])
        stack = np.vstack([smooth(x[selected_metric].iloc[:L]).values for x in seeds[k]])
        mu, sd = stack.mean(0), stack.std(0)
        x_axis = df['batch'].iloc[:L]
        fig.add_trace(go.Scatter(x=x_axis, y=mu + sd, mode='lines', line=dict(width=0),
                                 showlegend=False, hoverinfo='skip', legendgroup=k))
        fig.add_trace(go.Scatter(x=x_axis, y=mu - sd, mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(150,150,150,0.15)',
                                 showlegend=False, hoverinfo='skip', legendgroup=k))
        fig.add_trace(go.Scatter(x=x_axis, y=mu, name=lab(k), mode='lines',
                                 line=dict(color=color, width=2), legendgroup=k))
    else:
        fig.add_trace(go.Scatter(x=df['batch'], y=s, name=lab(k), mode='lines',
                                 line=dict(color=color, width=2)))
    add_drift_lines(fig, k)
add_gt_lines(fig)
fig.update_layout(xaxis_title="Batch", yaxis_title=selected_metric, **PLOT_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
#  Cumulative savings + adaptive threshold                                      #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Cumulative Savings & Decision Threshold")
left, right = st.columns(2)

with left:
    figc = go.Figure()
    for k in names:
        df = metrics[k]
        figc.add_trace(go.Scatter(x=df['batch'], y=df['savings'].cumsum(), name=lab(k),
                                  mode='lines', line=dict(color=PALETTE.get(k, '#999'), width=2)))
        add_drift_lines(figc, k)
    figc.update_layout(title="Cumulative savings ($)", xaxis_title="Batch",
                       yaxis_title="$", **PLOT_LAYOUT)
    st.plotly_chart(figc, use_container_width=True)

with right:
    figt = go.Figure()
    for k in names:
        df = metrics[k]
        if 'threshold' in df:
            figt.add_trace(go.Scatter(x=df['batch'], y=df['threshold'], name=lab(k),
                                      mode='lines', line=dict(color=PALETTE.get(k, '#999'),
                                                              width=1.8, shape='hv')))
    figt.update_layout(title="Decision threshold over time", xaxis_title="Batch",
                       yaxis_title="threshold", **PLOT_LAYOUT)
    st.plotly_chart(figt, use_container_width=True)

# --------------------------------------------------------------------------- #
#  Recall + F1                                                                  #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Recall vs F1 (fraud-focused)")

fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Recall", "F1"), shared_xaxes=False)
for ci, m in enumerate(['recall', 'f1'], start=1):
    for k in names:
        df = metrics[k]
        fig2.add_trace(
            go.Scatter(x=df['batch'], y=smooth(df[m]), name=lab(k), mode='lines',
                       line=dict(color=PALETTE.get(k, '#999'), width=1.8),
                       legendgroup=k, showlegend=(ci == 1)),
            row=1, col=ci)
        add_drift_lines(fig2, k, row=1, col=ci)
fig2.update_layout(**{**PLOT_LAYOUT, 'height': 400})
st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------- #
#  Post-drift recovery                                                          #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader(f"Drift Impact & Recovery ({selected_metric})")
st.caption(f"Mean of the metric over {recovery_w} batches before vs. after each detected drift.")

rec_rows = []
for k in names:
    df = metrics[k]
    for d in drifts.get(k, []):
        pre = df[selected_metric].iloc[max(0, d - recovery_w):d].mean()
        post = df[selected_metric].iloc[d + 1:d + 1 + recovery_w].mean()
        if np.isnan(pre) or np.isnan(post):
            continue
        rec_rows.append({'Model': lab(k), 'Drift batch': d, 'Before': round(pre, 4),
                         'After': round(post, 4), 'Δ (after − before)': round(post - pre, 4)})
if rec_rows:
    st.dataframe(pd.DataFrame(rec_rows), hide_index=True, use_container_width=True)
else:
    st.info("No drift events with enough surrounding data to analyse.")

# --------------------------------------------------------------------------- #
#  Drift detection quality (needs injected ground truth)                        #
# --------------------------------------------------------------------------- #
if gt_drifts:
    st.markdown("---")
    st.subheader("Drift Detection Quality (vs. injected ground truth)")
    st.caption(f"True drifts at batches {gt_drifts}. A detection counts if it fires within "
               f"150 batches after a true drift; anything else is a false alarm.")
    MAX_DELAY = 150
    det_rows = []
    for k in names:
        if k == BASELINE:
            continue
        dets = sorted(drifts.get(k, []))
        delays, missed = [], 0
        used = set()
        for g in gt_drifts:
            hits = [d for d in dets if g <= d <= g + MAX_DELAY and d not in used]
            if hits:
                used.add(hits[0])
                delays.append(hits[0] - g)
            else:
                missed += 1
        false_alarms = len([d for d in dets if d not in used])
        det_rows.append({
            'Model': lab(k),
            'Detected': f"{len(gt_drifts) - missed}/{len(gt_drifts)}",
            'Mean delay (batches)': round(float(np.mean(delays)), 1) if delays else None,
            'False alarms': false_alarms,
        })
    if det_rows:
        st.dataframe(pd.DataFrame(det_rows), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
#  Heatmap                                                                      #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader(f"Final Performance Heatmap (last {tail_n} batches)")

hm_cols = [('recall', 'Recall'), ('f1', 'F1'), ('f2', 'F2'), ('precision', 'Precision'),
           ('precision_at_1_percent', 'Prec@1%'), ('accuracy', 'Accuracy')]
hm = pd.DataFrame({lab(k): {t: tail_stats(k, m)[0] for m, t in hm_cols} for k in names}).T
fig3 = px.imshow(hm, text_auto='.3f', aspect='auto', color_continuous_scale='YlGn')
fig3.update_layout(template="plotly_dark", height=80 + 60 * len(names),
                   margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------- #
#  Drift Timeline                                                               #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Drift Event Timeline")

drift_cols = st.columns(len(names))
for col, k in zip(drift_cols, names):
    events = drifts.get(k, [])
    with col:
        st.markdown(f"<h5 style='color:{PALETTE.get(k, '#999')}'>{lab(k)}</h5>",
                    unsafe_allow_html=True)
        if events:
            st.dataframe(pd.DataFrame({'Drift at batch': events}),
                         hide_index=True, use_container_width=True)
        elif k == BASELINE:
            st.info("No detector – expected.")
        else:
            st.warning("No drifts detected.")

# --------------------------------------------------------------------------- #
#  Dynamic Insights                                                             #
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Insights")

STRATEGY = {
    'Static_Baseline_LSTM': "Non-adaptive sequential baseline",
    'Config_A_ADWIN_and_Random': "Statistical drift (ADWIN) + Random replay",
    'Config_F_Dual_Drift_and_Prioritized_and_A-GEM': "Multi-metric drift + A-GEM + Prioritized replay",
    'Config_G_Dual_Drift_and_Random_and_A-GEM': "Multi-metric drift + A-GEM + Random replay",
}

rows = []
for k in names:
    rows.append(f"| **{lab(k)}** | {raw_df.loc[lab(k), 'Recall']:.3f} | "
                f"{raw_df.loc[lab(k), 'PR-AUC']:.3f} | "
                f"${raw_df.loc[lab(k), 'Savings/batch ($)']:,.0f} | "
                f"{len(drifts.get(k, []))} | {STRATEGY.get(k, '')} |")
st.markdown("| Model | Recall | PR-AUC | Savings/batch | Drifts | Strategy |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows))

bullets = []
for title in ['Savings/batch ($)', 'PR-AUC', 'Recall', 'F1']:
    bullets.append(f"- **Best {title}:** {best_per_col[title]} "
                   f"({raw_df[title].max():,.3f})")
if BASELINE in names:
    base_label = lab(BASELINE)
    beat = [lab(k) for k in names if k != BASELINE
            and raw_df.loc[lab(k), 'Savings/batch ($)'] > raw_df.loc[base_label, 'Savings/batch ($)']]
    bullets.append(f"- **{len(beat)} of {len(names) - 1}** adaptive models beat the static baseline on savings.")
    worse_pr = [lab(k) for k in names if k != BASELINE
                and raw_df.loc[lab(k), 'PR-AUC'] < raw_df.loc[base_label, 'PR-AUC']]
    if worse_pr:
        bullets.append("- Lower PR-AUC than the static baseline: " + ", ".join(worse_pr))
total_drifts = sum(len(drifts.get(k, [])) for k in names if k != BASELINE)
if total_drifts <= len(names):
    bullets.append("- Very few drifts were detected, so adaptation is rarely exercised; "
                   "differences may be within seed noise.")
if n_seeds == 1:
    bullets.append("- Only one run per model; re-run the simulation with `NUM_SEEDS>1` "
                   "for confidence intervals.")
st.markdown("\n".join(bullets))