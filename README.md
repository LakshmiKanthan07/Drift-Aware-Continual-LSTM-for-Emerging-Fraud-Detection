# Drift-Aware Continual LSTM for Emerging Fraud Detection

This project implements a continual-learning-based LSTM system for detecting emerging fraudulent transaction patterns in a continuously changing data stream. 

Unlike traditional static fraud detection models that gradually become outdated, this system actively monitors for **concept drift** and adapts the LSTM model when significant changes occur in the transaction behaviors. It utilizes a **replay memory mechanism** (such as Class-Balanced Reservoir Sampling) to prevent *catastrophic forgetting*, ensuring the model learns new fraudulent behaviors while retaining knowledge of previously learned patterns.

## Features
*   **Synthetic Data Stream Generator**: Simulates a continuous stream of tabular transactions and injects concept drifts at random intervals.
*   **TensorFlow LSTM Model**: Deep learning architecture for sequence-based fraud detection.
*   **Drift Detection**:
    *   **ADWIN**: Statistical method for detecting changes in the stream.
    *   **Sliding Window**: Metric-based detection by monitoring F1-score drops.
*   **Replay Memory Strategies**:
    *   **Random Reservoir Sampling**: Basic random sampling of past experiences.
    *   **Class-Balanced Sampling**: Ensures normal and fraud transactions are equally represented during replay.
*   **Streamlit Web Dashboard**: Real-time visualization of model performance and drift events.

## Installation

```bash
# Clone the repository
git clone https://github.com/LakshmiKanthan07/Drift-Aware-Continual-LSTM-for-Emerging-Fraud-Detection.git
cd Drift-Aware-Continual-LSTM-for-Emerging-Fraud-Detection

# Create a virtual environment
python -m venv venv
# Activate the environment (Windows)
.\venv\Scripts\activate
# Activate the environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run the Continual Learning Simulation
This script will generate a synthetic data stream, train the models under different configurations, and log the performance metrics to CSV files.
```bash
python src/continual_learner.py
```

### 2. View the Dashboard
Once the simulation has finished and generated the metrics in the `data/` folder, run the Streamlit dashboard to visualize the results:
```bash
streamlit run src/dashboard.py
```

## Project Structure
```text
├── data/                  # Output directory for logged metrics and drifts
├── src/
│   ├── data_generator.py  # Synthetic data generation and drift simulation
│   ├── models/
│   │   └── lstm_model.py  # TensorFlow LSTM architecture
│   ├── drift_detection/
│   │   ├── adwin.py       # ADWIN implementation
│   │   └── window.py      # Sliding window metric implementation
│   ├── memory/
│   │   ├── random.py      # Random Reservoir Sampling buffer
│   │   └── balanced.py    # Class-Balanced Sampling buffer
│   ├── continual_learner.py # Orchestrates training and updates
│   └── dashboard.py       # Streamlit web application
└── requirements.txt
```
