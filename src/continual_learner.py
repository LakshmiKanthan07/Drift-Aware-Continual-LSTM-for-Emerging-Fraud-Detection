import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score, average_precision_score
from realtime_generator import RealtimeDataGenerator
from models.lstm_model import build_lstm_model
from drift_detection.adwin import ADWINDriftDetector
from drift_detection.window import WindowDriftDetector
from memory.random import RandomReplayBuffer
from memory.balanced import BalancedReplayBuffer


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
FRAUD_RATIO   = 0.035                          # Approx ratio for IEEE dataset
CLASS_WEIGHT  = {0: 1.0, 1: round(1.0 / FRAUD_RATIO)}  # {normal:1, fraud:~29}
BASE_LR       = 0.001
DRIFT_LR      = 0.005   # 5× base – fast adaptation immediately after drift


class ContinualLearner:
    def __init__(self, config_name, detector_type, memory_type,
                 seq_len=1, num_features=431):
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

    def _set_lr(self, lr: float):
        """Dynamically change the optimizer's learning rate."""
        self.model.optimizer.learning_rate = lr

    def _train(self, X, y, lr=None):
        """Train one step. Optionally set a custom LR."""
        if lr is not None:
            self._set_lr(lr)
        self.model.train_on_batch(X, y)

    def _batch_metrics(self, preds_binary, preds_prob, y):
        """Compute accuracy, precision, recall, f1, roc_auc, pr_auc, and kappa."""
        acc = float(np.mean(preds_binary == y))

        tp = int(np.sum((preds_binary == 1) & (y == 1)))
        fp = int(np.sum((preds_binary == 1) & (y == 0)))
        fn = int(np.sum((preds_binary == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        try:
            roc_auc = float(roc_auc_score(y, preds_prob))
        except ValueError:
            roc_auc = 0.5  # Neutral score if only one class is present

        try:
            pr_auc = float(average_precision_score(y, preds_prob))
        except ValueError:
            pr_auc = 0.0  # Undefined if only one class present

        try:
            kappa = float(cohen_kappa_score(y, preds_binary))
        except Exception:
            kappa = 0.0

        return acc, precision, recall, f1, roc_auc, pr_auc, kappa

    # ------------------------------------------------------------------
    # Main streaming loop
    # ------------------------------------------------------------------

    def process_stream(self, generator, num_batches, batch_size=64,
                       drift_prob=0.002, decision_thresh=0.25):
        """Processes a data stream, simulating continual learning."""

        print(f"\nStarting simulation for configuration: {self.config_name}")

        # ---------------------------------------------------------------
        # Bootstrap: train for 100 batches before streaming to give the
        # model a decent starting point (with class weights from the start).
        # ---------------------------------------------------------------
        print("Bootstrapping model...")
        for _ in range(250):  # Extended: 250 batches = 16,000 transactions
            X, y = generator.generate_batch(batch_size)
            self._train(X, y)
            if self.memory:
                for i in range(batch_size):
                    self.memory.add(X[i], y[i])

        # ---------------------------------------------------------------
        # Streaming phase
        # ---------------------------------------------------------------
        for batch_idx in tqdm(range(num_batches), desc=self.config_name):

            # Concept drift is naturally occurring in real-time data

            X, y = generator.generate_batch(batch_size)

            # --- Predict ---
            preds = self.model.predict(X, verbose=0)
            preds_binary = (preds > decision_thresh).astype(int)

            acc, precision, recall, f1, roc_auc, pr_auc, kappa = self._batch_metrics(preds_binary, preds, y)

            self.metrics_log.append({
                'batch':     batch_idx,
                'accuracy':  acc,
                'precision': precision,
                'recall':    recall,
                'f1':        f1,
                'roc_auc':   roc_auc,
                'pr_auc':    pr_auc,
                'kappa':     kappa,
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

                # --- Soft Memory Eviction ---
                # Instead of wiping all memory (catastrophic forgetting risk),
                # we only evict samples IF the buffer has grown very large.
                # This preserves some historical examples while making room
                # for new concept data.
                if self.memory and len(self.memory) > 500:
                    self.memory.clear()

                # Fine-tune at elevated LR on the current batch.
                # 10 steps give the model a strong push toward the new concept.
                for _ in range(10):
                    self._train(X, y, lr=DRIFT_LR)

                # Restore base LR for stable online learning
                self._set_lr(BASE_LR)

            # --- Standard online update (every non-drift batch) ---
            else:
                if self.memory and len(self.memory) >= batch_size:
                    X_mem, y_mem = self.memory.sample(batch_size)
                    if len(X_mem) > 0:
                        X_combined = np.concatenate((X, X_mem))
                        y_combined = np.concatenate((y, y_mem))
                        indices = np.arange(len(X_combined))
                        np.random.shuffle(indices)
                        X_combined = X_combined[indices]
                        y_combined = y_combined[indices]
                        self._train(X_combined, y_combined, lr=BASE_LR)
                    else:
                        self._train(X, y, lr=BASE_LR)
                else:
                    self._train(X, y, lr=BASE_LR)

        return pd.DataFrame(self.metrics_log), self.drift_events


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------

def run_simulation():
    # Simulation parameters
    NUM_TRANSACTIONS = 590_000   # Full IEEE dataset size
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
        generator = RealtimeDataGenerator()
        learner   = ContinualLearner(
            config_name=conf['name'],
            detector_type=conf['detector'],
            memory_type=conf['memory'],
            seq_len=generator.seq_len,
            num_features=generator.num_features
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
