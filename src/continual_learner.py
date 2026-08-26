import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_generator import StreamingDataGenerator
from models.lstm_model import build_lstm_model
from drift_detection.adwin import ADWINDriftDetector
from drift_detection.window import WindowDriftDetector
from memory.random import RandomReplayBuffer
from memory.balanced import BalancedReplayBuffer


# ---------------------------------------------------------------------------
# Class weight helper
# ---------------------------------------------------------------------------
FRAUD_RATIO = 0.03  # Must match StreamingDataGenerator fraud_ratio
CLASS_WEIGHT = {0: 1.0, 1: round(1.0 / FRAUD_RATIO)}  # {normal: 1, fraud: ~33}


class ContinualLearner:
    def __init__(self, config_name, detector_type, memory_type,
                 seq_len=5, num_features=10):
        self.config_name = config_name
        self.model = build_lstm_model(seq_len, num_features)

        if detector_type == 'adwin':
            self.detector = ADWINDriftDetector()
        elif detector_type == 'window':
            self.detector = WindowDriftDetector()
        else:
            self.detector = None  # Static baseline – no detection

        if memory_type == 'random':
            self.memory = RandomReplayBuffer()
        elif memory_type == 'balanced':
            self.memory = BalancedReplayBuffer()
        else:
            self.memory = None  # No memory replay

        self.metrics_log = []
        self.drift_events = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train(self, X, y):
        """Train one step with class weighting to handle fraud imbalance."""
        self.model.train_on_batch(X, y, class_weight=CLASS_WEIGHT)

    def _batch_metrics(self, preds_binary, y):
        """Compute accuracy, precision, recall, and F1 from a batch."""
        acc = float(np.mean(preds_binary == y))

        tp = int(np.sum((preds_binary == 1) & (y == 1)))
        fp = int(np.sum((preds_binary == 1) & (y == 0)))
        fn = int(np.sum((preds_binary == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return acc, precision, recall, f1

    # ------------------------------------------------------------------
    # Main streaming loop
    # ------------------------------------------------------------------

    def process_stream(self, generator, num_batches, batch_size=64,
                       drift_prob=0.002):
        """Processes a data stream, simulating continual learning."""

        print(f"\nStarting simulation for configuration: {self.config_name}")

        # ---------------------------------------------------------------
        # Bootstrap: train for 100 batches before streaming to give the
        # model a decent starting point (with class weights from the start).
        # ---------------------------------------------------------------
        print("Bootstrapping model...")
        for _ in range(100):
            X, y = generator.generate_batch(batch_size)
            self._train(X, y)
            if self.memory:
                for i in range(batch_size):
                    self.memory.add(X[i], y[i])

        # ---------------------------------------------------------------
        # Streaming phase
        # ---------------------------------------------------------------
        for batch_idx in tqdm(range(num_batches), desc=self.config_name):

            # Randomly inject a concept drift based on probability
            if random.random() < drift_prob and batch_idx > 100:
                generator.trigger_drift()

            X, y = generator.generate_batch(batch_size)

            # --- Predict ---
            preds = self.model.predict(X, verbose=0)
            preds_binary = (preds > 0.5).astype(int)

            acc, precision, recall, f1 = self._batch_metrics(preds_binary, y)

            self.metrics_log.append({
                'batch':     batch_idx,
                'accuracy':  acc,
                'precision': precision,
                'recall':    recall,
                'f1':        f1,
            })

            # --- Add to memory ---
            if self.memory:
                for i in range(batch_size):
                    self.memory.add(X[i], y[i])

            # --- Drift detection (fed per-batch recall / F1) ---
            drift_detected = False
            if self.detector:
                if isinstance(self.detector, ADWINDriftDetector):
                    # ADWIN tracks the per-batch F1 stream
                    if self.detector.update(f1):
                        drift_detected = True
                        self.detector.reset()
                elif isinstance(self.detector, WindowDriftDetector):
                    # Window detector tracks per-batch recall stream
                    if self.detector.update(recall):
                        drift_detected = True
                        self.detector.reset()

            # --- Adaptation on detected drift ---
            if drift_detected:
                print(f"\n  [!] Drift detected at batch {batch_idx}! "
                      f"(recall={recall:.3f}, f1={f1:.3f}) Adapting...")
                self.drift_events.append(batch_idx)

                if self.memory:
                    # Fine-tune on mixed memory + current batch (5 steps)
                    for _ in range(5):
                        X_mem, y_mem = self.memory.sample(batch_size * 2)
                        if len(X_mem) > 0:
                            X_train = np.vstack((X_mem, X))
                            y_train  = np.vstack((y_mem, y))
                            self._train(X_train, y_train)
                else:
                    # Naive fine-tune (no memory – static baseline never
                    # reaches here, but kept for completeness)
                    for _ in range(5):
                        self._train(X, y)

            # --- Standard online update (every batch) ---
            else:
                self._train(X, y)

        return pd.DataFrame(self.metrics_log), self.drift_events


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------

def run_simulation():
    # Simulation parameters
    NUM_TRANSACTIONS = 192_000   # Reduced for faster runs (~10-15 min)
    BATCH_SIZE       = 64
    NUM_BATCHES      = NUM_TRANSACTIONS // BATCH_SIZE   # = 3 000 batches
    DRIFT_PROB       = 0.002     # ~6 drifts expected per 3 000-batch run

    os.makedirs('data', exist_ok=True)

    configs = [
        {'name': 'Static Baseline',          'detector': None,     'memory': None},
        {'name': 'Config A (ADWIN + Random)', 'detector': 'adwin',  'memory': 'random'},
        {'name': 'Config B (Window + Balanced)', 'detector': 'window', 'memory': 'balanced'},
    ]

    for conf in configs:
        generator = StreamingDataGenerator(fraud_ratio=FRAUD_RATIO)
        learner   = ContinualLearner(
            config_name=conf['name'],
            detector_type=conf['detector'],
            memory_type=conf['memory'],
        )

        df, drifts = learner.process_stream(
            generator,
            num_batches=NUM_BATCHES,
            batch_size=BATCH_SIZE,
            drift_prob=DRIFT_PROB,
        )

        safe_name = (conf['name']
                     .replace(' ', '_')
                     .replace('(', '')
                     .replace(')', '')
                     .replace('+', 'and'))
        df.to_csv(f"data/{safe_name}_metrics.csv", index=False)
        pd.DataFrame({'drift_batch': drifts}).to_csv(
            f"data/{safe_name}_drifts.csv", index=False)

        n_drifts = len(drifts)
        print(f"\nFinished {conf['name']}. "
              f"Drifts detected: {n_drifts}. Saved to data/\n")


if __name__ == "__main__":
    run_simulation()
