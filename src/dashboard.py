import streamlit as st
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Drift-Aware LSTM Dashboard", layout="wide")

st.title("🛡️ Drift-Aware Continual LSTM for Fraud Detection")
st.markdown("Visualizing model performance and concept drift adaptation over time.")

# Load data
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

@st.cache_data
def load_data():
    metric_files = glob.glob(os.path.join(data_dir, '*_metrics.csv'))
    drift_files = glob.glob(os.path.join(data_dir, '*_drifts.csv'))
    
    if not metric_files:
        return None, None
        
    metrics = {}
    drifts = {}
    
    for f in metric_files:
        name = os.path.basename(f).replace('_metrics.csv', '')
        metrics[name] = pd.read_csv(f)
        
    for f in drift_files:
        name = os.path.basename(f).replace('_drifts.csv', '')
        drifts[name] = pd.read_csv(f)['drift_batch'].tolist()
        
    return metrics, drifts

metrics, drifts = load_data()

if not metrics:
    st.warning("No data found. Please run the simulation first (`python src/continual_learner.py`).")
else:
    st.sidebar.header("Configuration")
    
    # Calculate rolling averages for smoother plots
    window_size = st.sidebar.slider("Smoothing Window (Batches)", min_value=10, max_value=200, value=50)
    
    selected_metric = st.sidebar.selectbox("Metric to visualize", ['f1', 'accuracy', 'precision', 'recall'])
    
    st.subheader(f"Performance over Time: {selected_metric.capitalize()}")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['red', 'blue', 'green']
    
    # We want to plot all models together for comparison
    names = list(metrics.keys())
    for i, name in enumerate(names):
        df = metrics[name].copy()
        
        # Smooth metric
        df['smoothed'] = df[selected_metric].rolling(window=window_size, min_periods=1).mean()
        
        sns.lineplot(data=df, x='batch', y='smoothed', label=name, color=colors[i%len(colors)], ax=ax, alpha=0.8)
        
        # Plot drift points (using drifts from the best performing model just to show where they happened,
        # or plot for each if they detect differently)
        if name in drifts and len(drifts[name]) > 0:
            for d in drifts[name]:
                 ax.axvline(x=d, color=colors[i%len(colors)], linestyle='--', alpha=0.3)
                 
    ax.set_xlabel("Batch Number (Time)")
    ax.set_ylabel(selected_metric.capitalize())
    ax.set_title(f"{selected_metric.capitalize()} Score over Time across Data Stream")
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📊 Infographics & Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    for i, name in enumerate(names):
        df = metrics[name]
        avg_f1 = df['f1'].mean()
        final_f1 = df['f1'].tail(100).mean() # Average over last 100 batches
        drift_count = len(drifts[name]) if name in drifts else 0
        
        with [col1, col2, col3][i]:
            st.metric(label=f"{name} Overall F1", value=f"{avg_f1:.3f}")
            st.metric(label=f"Final 100-Batch F1", value=f"{final_f1:.3f}")
            st.metric(label=f"Drifts Detected", value=drift_count)
            
    st.markdown("""
    ### 💡 Insights
    *   **Static Baseline**: Performance degrades permanently after the first concept drift because it never updates.
    *   **Config A (ADWIN + Random)**: Detects drift dynamically and updates, recovering performance. However, random sampling might forget older, rare fraud patterns.
    *   **Config B (Window + Balanced)**: Balances the memory, leading to more robust recovery and higher average F1 scores, especially on highly imbalanced fraud data.
    """)
