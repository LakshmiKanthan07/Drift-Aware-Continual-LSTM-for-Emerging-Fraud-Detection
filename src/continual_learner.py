import os
import random
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score, average_precision_score, f1_score
from realtime_generator import RealtimeDataGenerator
from models.lstm_model import build_lstm_model
from drift_detection.adwin import ADWINDriftDetector
from drift_detection.window import WindowDriftDetector
from drift_detection.dual import DualDriftDetector
from memory.random import RandomReplayBuffer
from memory.balanced import BalancedReplayBuffer


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
FRAUD_RATIO   = 0.035                          # Approx ratio for IEEE dataset
CLASS_WEIGHT  = {0: 1.0, 1: round(1.0 / FRAUD_RATIO)}  # {normal:1, fraud:~29}
BASE_LR       = 0.001
DRIFT_LR      = 0.005   # 5× base – fast adaptation immediately after drift


class AdaptiveThreshold:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.y_true_cache = collections.deque(maxlen=window_size)
        self.y_prob_cache = collections.deque(maxlen=window_size)
        self.current_threshold = 0.25  # default starting point

    def add(self, y_true_batch, y_prob_batch):
        self.y_true_cache.extend(y_true_batch.flatten())
        self.y_prob_cache.extend(y_prob_batch.flatten())

    def update_threshold(self):
        """Finds the optimal threshold to maximize F1-score over recent history."""
        if len(self.y_true_cache) < 200:
            return self.current_threshold  # Not enough data yet

        y_true = np.array(self.y_true_cache)
        y_prob = np.array(self.y_prob_cache)
        
        # If no fraud in history, keep current threshold
        if np.sum(y_true) == 0:
            return self.current_threshold

        best_thresh = self.current_threshold
        best_f1 = 0.0

        for thresh in np.arange(0.10, 0.60, 0.05):
            preds = (y_prob > thresh).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        self.current_threshold = best_thresh
        return self.current_threshold


class ContinualLearner:
    def __init__(self, config_name, detector_type, memory_type,
                 seq_len, num_cont_features, cat_vocab_dict):
        self.config_name = config_name
        self.model = build_lstm_model(seq_len, num_cont_features, cat_vocab_dict)

        if detector_type == 'adwin':
            self.detector = ADWINDriftDetector()
        elif detector_type == 'window':
            self.detector = WindowDriftDetector()
        elif detector_type == 'dual':
            self.detector = DualDriftDetector()
        else:
            self.detector = None  # Static baseline – no detection

        if memory_type == 'random':
            self.memory = RandomReplayBuffer()
        elif memory_type == 'balanced':
            self.memory = BalancedReplayBuffer()
        else:
            self.memory = None  # No memory replay
            
        self.adaptive_threshold = AdaptiveThreshold(window_size=1000)
        self.metrics_log = []
        self.drift_events = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_lr(self, lr: float):
        """Dynamically change the optimizer's learning rate."""
        self.model.optimizer.learning_rate = lr

    def _train(self, X_dict, y, lr=None):
        """Train one step. Optionally set a custom LR."""
        if lr is not None:
            self._set_lr(lr)
        self.model.train_on_batch(X_dict, y)

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
                       drift_prob=0.002, use_adaptive_thresh=True):
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
                    x_i = {k: v[i] for k, v in X.items()}
                    self.memory.add(x_i, y[i])

        # ---------------------------------------------------------------
        # Streaming phase
        # ---------------------------------------------------------------
        for batch_idx in tqdm(range(num_batches), desc=self.config_name):

            X, y = generator.generate_batch(batch_size)

            # --- Predict ---
            preds = self.model.predict(X, verbose=0).ravel()
            y_flat = y.ravel()
            
            # --- Dynamic Threshold Optimization ---
            if use_adaptive_thresh:
                self.adaptive_threshold.add(y_flat, preds)
                # Re-tune threshold every 100 batches
                if batch_idx > 0 and batch_idx % 100 == 0:
                    self.adaptive_threshold.update_threshold()
                decision_thresh = self.adaptive_threshold.current_threshold
            else:
                decision_thresh = 0.25
                
            preds_binary = (preds > decision_thresh).astype(int)

            acc, precision, recall, f1, roc_auc, pr_auc, kappa = self._batch_metrics(preds_binary, preds, y_flat)

            self.metrics_log.append({
                'batch':     batch_idx,
                'accuracy':  acc,
                'precision': precision,
                'recall':    recall,
                'f1':        f1,
                'roc_auc':   roc_auc,
                'pr_auc':    pr_auc,
                'kappa':     kappa,
                'threshold': decision_thresh
            })

            # --- Add to memory ---
            if self.memory:
                for i in range(batch_size):
                    x_i = {k: v[i] for k, v in X.items()}
                    self.memory.add(x_i, y[i])

            # --- Drift detection ---
            drift_detected = False
            if self.detector:
                if isinstance(self.detector, ADWINDriftDetector):
                    if self.detector.update(f1):
                        drift_detected = True
                        self.detector.reset()
                elif isinstance(self.detector, WindowDriftDetector):
                    if self.detector.update(recall):
                        drift_detected = True
                        self.detector.reset()
                elif isinstance(self.detector, DualDriftDetector):
                    if self.detector.update(recall, f1):
                        drift_detected = True
                        self.detector.reset()

            # --- Adaptation on detected drift ---
            if drift_detected:
                print(f"\n  [!] Drift detected at batch {batch_idx}! "
                      f"(recall={recall:.3f}, f1={f1:.3f}) Adapting...")
                self.drift_events.append(batch_idx)

                if self.memory and len(self.memory.normal_buffer if isinstance(self.memory, BalancedReplayBuffer) else self.memory.buffer) > 500:
                    self.memory.clear()

                # Fine-tune at elevated LR on the current batch.
                for _ in range(10):
                    self._train(X, y, lr=DRIFT_LR)

                # Restore base LR for stable online learning
                self._set_lr(BASE_LR)

            # --- Standard online update (every non-drift batch) ---
            else:
                if self.memory and len(self.memory.normal_buffer if isinstance(self.memory, BalancedReplayBuffer) else self.memory.buffer) >= batch_size:
                    X_mem, y_mem = self.memory.sample(batch_size)
                    if len(y_mem) > 0:
                        X_combined = {k: np.concatenate((X[k], X_mem[k])) for k in X.keys()}
                        y_combined = np.concatenate((y, y_mem))
                        indices = np.arange(len(y_combined))
                        np.random.shuffle(indices)
                        X_combined = {k: X_combined[k][indices] for k in X_combined.keys()}
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
    NUM_TRANSACTIONS = 590_000
    BATCH_SIZE       = 64
    NUM_BATCHES      = NUM_TRANSACTIONS // BATCH_SIZE
    DRIFT_PROB       = 0.002

    os.makedirs('data', exist_ok=True)

    # Replaced trees with Static MLP Baseline
    configs = [
        {'name': 'Static MLP Baseline', 'is_classical': True},
        {'name': 'Static Baseline', 'detector': None, 'memory': None, 'is_classical': False},
        {'name': 'Config C (Dual + Dynamic + Residual)', 'detector': 'dual', 'memory': 'balanced', 'is_classical': False},
    ]

    for conf in configs:
        generator = RealtimeDataGenerator(seq_len=5) # Now using sequence length of 5!
        
        if conf.get('is_classical'):
            from baselines import StaticMLPBaseline
            learner = StaticMLPBaseline(
                seq_len=generator.seq_len,
                cat_vocab_dict=generator.cat_vocab,
                num_cont_features=generator.num_cont_features
            )
            df, drifts = learner.process_stream(
                generator,
                num_batches=NUM_BATCHES,
                batch_size=BATCH_SIZE
            )
        else:
            learner = ContinualLearner(
                config_name=conf['name'],
                detector_type=conf['detector'],
                memory_type=conf['memory'],
                seq_len=generator.seq_len,
                num_cont_features=generator.num_cont_features,
                cat_vocab_dict=generator.cat_vocab
            )

            df, drifts = learner.process_stream(
                generator,
                num_batches=NUM_BATCHES,
                batch_size=BATCH_SIZE,
                drift_prob=DRIFT_PROB,
                use_adaptive_thresh=True if conf['detector'] is not None else False
            )

        safe_name = (conf['name']
                     .replace(' ', '_')
                     .replace('(', '')
                     .replace(')', '')
                     .replace('+', 'and'))
        df.to_csv(f'data/{safe_name}_metrics.csv', index=False)
        pd.DataFrame(drifts).to_csv(f'data/{safe_name}_drifts.csv', index=False)

if __name__ == '__main__':
    run_simulation()
