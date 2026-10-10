# Drift-Aware Continual LSTM for Emerging Fraud Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end continual learning framework designed to detect emerging fraudulent financial transactions in non-stationary streaming environments. The system actively monitors for **concept drift**, adapts deep sequence models dynamically, and mitigates **catastrophic forgetting** via memory replay, gradient projection, and elastic weight consolidation.

---

## Table of Contents
- [Overview](#-overview)
- [Key Features & Innovations](#-key-features--innovations)
- [System Architecture](#-system-architecture)
- [Continual Learning Strategies](#-continual-learning-strategies)
- [Drift Detection Mechanisms](#-drift-detection-mechanisms)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quickstart & Usage](#-quickstart--usage)
  - [1. Data Preprocessing](#1-data-preprocessing-optional-for-ieee-cis-dataset)
  - [2. Run Continual Learning Simulation](#2-run-continual-learning-simulation)
  - [3. Launch the Interactive Dashboard](#3-launch-the-interactive-dashboard)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Configuration Reference](#-configuration-reference)
- [License & Acknowledgements](#-license--acknowledgements)

---

## Overview

In financial fraud detection, fraudsters continuously alter their tactics to bypass security mechanisms, leading to **concept drift** (covariate shift and posterior distribution shifts). Traditional batch-trained models decay rapidly in production, while naive online fine-tuning causes **catastrophic forgetting** of previously learned legitimate and fraudulent patterns.

This project addresses both challenges by combining:
1. **Multi-Modal Deep Sequence Modeling**: Channel Attention (Squeeze-and-Excitation), 1D Convolutions, Multi-layer Bidirectional LSTM, and Temporal Multi-Head Attention with Residual Connections.
2. **Multi-Signal Concept Drift Detection**: Statistical and metric-based drift monitors (ADWIN + Sliding Window) detecting both sudden changes and gradual performance erosion.
3. **Continual Learning Regularization & Replay**: Prioritized and Class-Balanced Experience Replay buffers, Averaged Gradient Episodic Memory (A-GEM) gradient projection, and Dynamic Elastic Weight Consolidation (EWC).
4. **Adaptive Business Utility Optimization**: Dynamic classification thresholding calibrated for extreme class imbalance and financial cost-benefit tradeoffs.

---

## Key Features & Innovations

- **Hybrid Deep Architecture**:
  - **Entity Embeddings** for high-cardinality categorical features (card types, address, device IDs).
  - **Channel Attention (SE Block)** to weight feature relevance per transaction.
  - **Bidirectional LSTM + Multi-Head Self-Attention** for capturing long-term cardholder behavioral sequences.
  - **Binary Focal Cross-Entropy Loss** with class weighting to handle extreme fraud imbalance (~3.5% fraud rate).
- **Multi-Signal Dual Drift Detector**:
  - Monitors **F2-Score** and **PR-AUC** via **ADWIN** for gradual drift.
  - Monitors **Recall** via **Sliding Window** with cooldown mechanisms for abrupt shifts.
  - Estimates drift severity to dynamically modulate learning plasticity and regularization strength.
- **Catastrophic Forgetting Mitigation**:
  - **Random Reservoir Sampling**: Uniform random past replay.
  - **Class-Balanced Buffer**: Equalized 50/50 normal vs. fraud experience replay.
  - **Prioritized Experience Replay (PER)**: Error-weighted replay based on prediction loss ($|y - \hat{y}|^\alpha$) for hard sample retention.
  - **A-GEM (Averaged Gradient Episodic Memory)**: Projects parameter gradients to prevent interference with historical tasks ($g \cdot g_{ref} \ge 0$).
  - **Severity-Adaptive EWC**: Dynamic Fisher information regularizer ($\lambda \in [500, 8000]$).
- **Business-Aware Adaptive Thresholding**:
  - Dynamically tunes decision thresholds across a moving window to maximize net monetary savings ($\$500$ saved per True Positive vs. $\$10$ cost per False Positive + $F_{1.5}$ score).
- **High-Performance Execution**:
  - **Vectorized Replay Buffers** utilizing pure NumPy indexing instead of Python loops, speeding up A-GEM gradients.
  - **JIT Compilation (`@tf.function`)** and mixed-precision tensors to aggressively optimize batch latency in the live stream loop.
- **Interactive Streamlit & Plotly Dashboard**:
  - Real-time visualization of streaming metrics, drift triggers, cumulative savings, multi-seed variance bands, and multi-model benchmark comparisons.

---

## System Architecture

```
                                  Streaming Transactions
                                            │
                                            ▼
                           ┌──────────────────────────────────┐
                           │   Real-time Sequence Generator   │
                           │   (Card-level Rolling Window)    │
                           └────────────────┬─────────────────┘
                                            │ (X_t, y_t)
                                            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              Deep Continual BiLSTM Model                               │
 │                                                                                        │
 │  ┌───────────────────────┐   ┌───────────────────────────┐                            │
 │  │ Categorical Features  │   │   Continuous Features     │                            │
 │  │ (Card, Product, etc.) │   │  (Amount, Dist, Time, etc.)│                            │
 │  └──────────┬────────────┘   └─────────────┬─────────────┘                            │
 │             ▼                              │                                          │
 │  ┌───────────────────────┐                 │                                          │
 │  │   Entity Embeddings   │                 │                                          │
 │  └──────────┬────────────┘                 │                                          │
 │             └──────────────┬───────────────┘                                          │
 │                            ▼                                                          │
 │             ┌──────────────────────────────┐                                          │
 │             │ Squeeze-and-Excitation (SE)  │ ──► [Residual Skip Shortcut]             │
 │             └──────────────┬───────────────┘                     │                    │
 │                            ▼                                     │                    │
 │             ┌──────────────────────────────┐                     │                    │
 │             │  1D Convolution + Dropout    │                     │                    │
 │             └──────────────┬───────────────┘                     │                    │
 │                            ▼                                     │                    │
 │             ┌──────────────────────────────┐                     │                    │
 │             │ 2-Layer Bidirectional LSTM   │                     │                    │
 │             └──────────────┬───────────────┘                     │                    │
 │                            ▼                                     │                    │
 │             ┌──────────────────────────────┐                     │                    │
 │             │ Temporal Multi-Head Attention│                     │                    │
 │             └──────────────┬───────────────┘                     │                    │
 │                            ▼                                     │                    │
 │             ┌──────────────────────────────┐                     │                    │
 │             │ Global Pooling & Concat Head │ ◄───────────────────┘                    │
 │             └──────────────┬───────────────┘                                          │
 │                            ▼                                                          │
 │             ┌──────────────────────────────┐                                          │
 │             │ Sigmoid Fraud Prob Output ŷ  │                                          │
 │             └──────────────────────────────┘                                          │
 └────────────────────────────┬───────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌───────────────────┐           ┌──────────────────────────────┐
    │  Dual Drift       │           │  Replay Memory Buffer        │
    │  Detector         │           │  (Balanced / Prioritized)    │
    │  (ADWIN + Window) │           └──────────────┬───────────────┘
    └─────────┬─────────┘                          │
              │ Drift Trigger                      │ Memory Batch
              ▼                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                 Continual Adaptation Engine                  │
    │  • Cosine Decay Learning Rate Recovery                       │
    │  • A-GEM Gradient Projection: g' = g - (g·g_ref / ||g_ref||²)│
    │  • Dynamic EWC Penalty: L_total = L_focal + (λ/2) Σ F (θ-θ*)² │
    │  • Adaptive Cost-Benefit Threshold Optimization              │
    └──────────────────────────────────────────────────────────────┘
```

---

## Continual Learning Strategies

To guarantee fair evaluation, all models undergo **Offline Bootstrapping**: a 5-epoch training phase on the first 16,000 transactions (250 batches) before entering the strictly one-pass online streaming phase.

| Strategy | Description | Key Advantage |
| :--- | :--- | :--- |
| **Static Baseline** | Model trained on initial bootstrap data without updates. | Benchmark showing degradation when drift occurs. |
| **Config A (ADWIN + Random Replay)** | ADWIN detects drift and triggers fine-tuning on a mix of stream and random reservoir memory. | Baseline continual learning setup. |
| **Config F (Dual Drift + PER + A-GEM)** | Multi-signal drift detector with Prioritized Experience Replay and A-GEM gradient projection. | Maximum retention of hard cases with zero catastrophic interference. |
| **Config G (Dual Drift + Random + A-GEM)** | Dual drift detection combined with uniform reservoir replay and A-GEM optimization. | High plasticity with low memory sampling overhead. |

---

## Drift Detection Mechanisms

1. **ADWIN (Adaptive Windowing)**:
   - Statistically compares sub-windows of real-time metrics (**F2-score** and **PR-AUC**) with bounds computed via Hoeffding's inequality.
   - Automatically adapts window size according to the rate of change without requiring manual sliding window tuning.
2. **Sliding Window Recall Monitor**:
   - Computes rolling recall over a fixed-length window ($W=200$).
   - Flags an alert when relative drop exceeds threshold ($\Delta \ge 0.15$) with an enforced post-drift cooldown period ($50$ batches) to prevent oscillation.
3. **Drift Severity Estimation**:
   - Measures the magnitude of drop: $\text{Severity} = \max(1.0 - \text{Recall}, 1.0 - F_2)$.
   - Scales the EWC parameter $\lambda$ dynamically: lower $\lambda$ during severe drift to allow plasticity; higher $\lambda$ during mild drift to enforce stability.

---

## Project Structure

```text
├── data/                         # Processed datasets, ground-truth drift logs, and metric CSVs
│   ├── processed_train.parquet   # Processed transaction stream (optional/preprocessed)
│   ├── metadata.json             # Feature definitions and vocabulary cardinalities
│   └── ground_truth_drifts.csv   # Synthetically injected drift locations
├── src/
│   ├── models/
│   │   └── lstm_model.py         # Squeeze-and-Excitation BiLSTM with Transformer Attention
│   ├── drift_detection/
│   │   ├── adwin.py              # ADWIN adaptive sliding window implementation
│   │   ├── window.py             # Rolling window metric drop detector
│   │   └── dual.py               # 3-signal Dual Drift Detector with severity estimation
│   ├── memory/
│   │   ├── random.py             # Random Reservoir Replay Buffer
│   │   ├── balanced.py           # 50/50 Class-Balanced Replay Buffer
│   │   └── prioritized.py        # Priority-Weighted Replay Buffer (|y - ŷ|^α)
│   ├── realtime_generator.py     # High-performance transaction stream generator with card history
│   ├── data_generator.py         # Synthetic tabular data stream generator with drift injection
│   ├── preprocess_data.py        # Pipeline for IEEE-CIS / raw transaction datasets
│   ├── continual_learner.py      # Core simulation engine, A-GEM, EWC, and training orchestrator
│   ├── baselines.py              # Classical ML and Static MLP Baselines
│   ├── model_evaluator.py        # Checkpoint tracker and metric logging
│   ├── dashboard.py              # Interactive Streamlit & Plotly analytics dashboard
│   └── plot_timings.py           # Script to plot batch latencies across models
├── literature review/            # Literature review documents and papers
├── reports and ppts/             # Project reports and presentations
├── graphify-out/                 # Knowledge graph and project map outputs
├── papers_list.txt               # List of literature review papers
├── papers_summary.json           # Summary of reviewed papers
├── sequence_diagram.png          # System architecture sequence diagram
├── gantt_chart_8_weeks.png       # Project timeline and Gantt chart
├── END SEMESTER PROJECT Plan.pdf # Comprehensive project plan document
├── DL.pdf                        # Deep Learning reference document
├── chuma.py                      # Scratch script for Gantt chart generation
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## Installation

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- Recommended: CUDA-compatible GPU (for accelerated TensorFlow training)

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/LakshmiKanthan07/Drift-Aware-Continual-LSTM-for-Emerging-Fraud-Detection.git
cd Drift-Aware-Continual-LSTM-for-Emerging-Fraud-Detection

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Data Preprocessing (Optional for IEEE-CIS dataset)
If using the raw IEEE-CIS Fraud Detection dataset (`train_transaction.csv` and `train_identity.csv` placed in `data/`):
```bash
python src/preprocess_data.py
```
*Note: If no parquet file is supplied, the framework defaults to the streaming generator in `src/realtime_generator.py` or synthetic stream generation.*

### 2. Run Continual Learning Simulation
Execute the full benchmark suite across all configurations, multiple random seeds, and injected drift events:
```bash
python src/continual_learner.py
```

**Environment Variables Configuration**:
You can customize the run without editing code using environment variables:
```bash
# Number of seeds for statistical confidence (default: 3)
export NUM_SEEDS=3

# Injected drift toggle (1 = enable, 0 = pure stream)
export INJECT_DRIFT=1

# Drift magnitude multiplier (default: 1.0)
export DRIFT_MAGNITUDE=1.0

python src/continual_learner.py
```

### 3. Launch the Interactive Dashboard
Once simulation metrics are logged to `data/`, start the Streamlit web dashboard:
```bash
streamlit run src/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to inspect real-time metric curves, drift event alignments, economic savings, and ablation comparisons.

### 4. Analyze Execution Timings
To visualize and compare the batch processing latency across different model configurations:
```bash
python src/plot_timings.py
```
This generates a `batch_latencies.png` chart inside the `data/` directory.

---

## Evaluation & Metrics

The streaming evaluation pipeline logs batch-level performance metrics:

- **PR-AUC (Precision-Recall AUC)**: Primary metric for imbalanced fraud classification.
- **ROC-AUC**: Overall discriminatory capability across all operating points.
- **F1 & F2 Scores**: Harmonic mean of precision and recall ($F_2$ heavily weights recall to penalize missed fraud).
- **Precision @ Top 1%**: Precision among transactions with the top 1% highest risk scores.
- **Cohen's Kappa ($\kappa$)**: Inter-rater agreement accounting for chance.
- **Net Monetary Savings (\$USD)**:
  $$\text{Savings} = (\text{True Positives} \times \$500) - (\text{False Positives} \times \$10)$$
- **Catastrophic Forgetting Ratio**: Quantifies retention on pre-drift transaction distributions.

---

## Configuration Reference

Key hyperparameters defined in [continual_learner.py](file:///d:/Priyan/amrita/5th%20semester/deep-learning/project/src/continual_learner.py):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `FRAUD_RATIO` | `0.035` | Background fraud prevalence in the stream (~3.5%). |
| `BATCH_SIZE` | `64` | Number of sequential transactions per streaming batch. |
| `BASE_LR` | `0.001` | Base Adam learning rate during stationary streaming. |
| `DRIFT_LR` | `0.003` | Peak learning rate for post-drift cosine recovery. |
| `DRIFT_DECAY_STEPS` | `100` | Steps over which learning rate decays back to `BASE_LR`. |
| `EWC_LAMBDA_MIN` | `500.0` | Lower EWC regularizer bound (high plasticity on major drift). |
| `EWC_LAMBDA_MAX` | `8000.0` | Upper EWC regularizer bound (high stability on minor drift). |
| `PER_ALPHA` | `0.6` | Prioritization exponent for Prioritized Replay Buffer. |
| `FOCAL_GAMMA` | `2.0` | Focus parameter for Binary Focal Cross-Entropy loss. |
| `FOCAL_ALPHA` | `0.80` | Positive class weight for Focal Loss. |

---

## License & Acknowledgements

- **License**: Released under the [MIT License](LICENSE).
- **Research & References**:
  - IEEE-CIS Fraud Detection dataset (Vesta Corporation / Kaggle).
  - ADWIN: *Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing.*
  - A-GEM: *Chaudhry, A. et al. (2018). Efficient Lifelong Learning with A-GEM.*
  - EWC: *Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks.*