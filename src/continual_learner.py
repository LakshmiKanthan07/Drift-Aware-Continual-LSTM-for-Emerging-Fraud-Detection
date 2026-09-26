import os
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score, average_precision_score, f1_score, fbeta_score
from realtime_generator import RealtimeDataGenerator
from models.lstm_model import build_lstm_model
from drift_detection.adwin import ADWINDriftDetector
from drift_detection.window import WindowDriftDetector
from drift_detection.dual import DualDriftDetector
from memory.random import RandomReplayBuffer
from memory.balanced import BalancedReplayBuffer
from memory.prioritized import PrioritizedReplayBuffer
from model_evaluator import BestModelTracker

FRAUD_RATIO   = 0.035
CLASS_WEIGHT  = {0: 1.0, 1: 100}   # Boosted from ~28 → 100 for severe imbalance
BASE_LR       = 0.001
DRIFT_LR      = 0.010
DRIFT_DECAY_STEPS = 100             # Doubled from 50 → 100 for smoother post-drift recovery

class AdaptiveThreshold:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.y_true_cache = collections.deque(maxlen=window_size)
        self.y_prob_cache = collections.deque(maxlen=window_size)
        self.current_threshold = 0.25

    def add(self, y_true_batch, y_prob_batch):
        self.y_true_cache.extend(y_true_batch.flatten())
        self.y_prob_cache.extend(y_prob_batch.flatten())

    def update_threshold(self):
        if len(self.y_true_cache) < 200:
            return self.current_threshold  

        y_true = np.array(self.y_true_cache)
        y_prob = np.array(self.y_prob_cache)
        
        if np.sum(y_true) == 0:
            return self.current_threshold

        best_thresh = self.current_threshold
        best_score = 0.0

        for thresh in np.arange(0.05, 0.60, 0.02):
            preds = (y_prob > thresh).astype(int)
            score = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = thresh
                
        self.current_threshold = best_thresh
        return self.current_threshold

class ContinualLearner:
    def __init__(self, config_name, detector_type, memory_type,
                 seq_len, num_cont_features, cat_vocab_dict, use_ewc=False):
        self.config_name = config_name
        self.use_ewc = use_ewc
        self.model = build_lstm_model(seq_len, num_cont_features, cat_vocab_dict)

        if detector_type == 'adwin':
            self.detector = ADWINDriftDetector()
        elif detector_type == 'window':
            self.detector = WindowDriftDetector()
        elif detector_type == 'dual':
            self.detector = DualDriftDetector()
        else:
            self.detector = None  

        if memory_type == 'random':
            self.memory = RandomReplayBuffer()
        elif memory_type == 'balanced':
            self.memory = BalancedReplayBuffer()
        elif memory_type == 'prioritized':
            self.memory = PrioritizedReplayBuffer(capacity=5000, fraud_ratio=0.5, alpha=0.6)
        else:
            self.memory = None
            
        self.adaptive_threshold = AdaptiveThreshold(window_size=500)  # Increased from 300 → 500
        self.metrics_log = []
        self.drift_events = []

    def _memory_size(self) -> int:
        """Returns total number of samples in the memory buffer, regardless of type."""
        if self.memory is None:
            return 0
        return self.memory.size

    def _set_lr(self, lr: float):
        self.model.optimizer.learning_rate = lr

    def _cosine_lr(self, step: int, total_steps: int) -> float:
        """Cosine decay from DRIFT_LR back to BASE_LR over total_steps."""
        progress = step / max(total_steps - 1, 1)
        return BASE_LR + 0.5 * (DRIFT_LR - BASE_LR) * (1 + np.cos(np.pi * progress))

    def _train(self, X_dict, y, lr=None, apply_ewc=False):
        if lr is not None:
            self._set_lr(lr)
        # Build per-sample class weights array
        sample_weights = np.where(y.ravel() == 1, CLASS_WEIGHT[1], CLASS_WEIGHT[0]).astype(np.float32)
        
        if not apply_ewc or not hasattr(self, 'fisher_matrix'):
            self.model.train_on_batch(X_dict, y, sample_weight=sample_weights)
        else:
            with tf.GradientTape() as tape:
                preds = self.model(X_dict, training=True)
                # Compute base loss
                base_loss = self.model.compiled_loss(y, preds, sample_weight=sample_weights)
                
                # Add EWC penalty
                ewc_loss = 0.0
                ewc_lambda = 5000.0  # FIM regularization strength
                for weight, fisher, opt_weight in zip(self.model.trainable_weights, self.fisher_matrix, self.opt_weights):
                    ewc_loss += tf.reduce_sum(fisher * tf.square(weight - opt_weight))
                
                total_loss = base_loss + (ewc_lambda / 2.0) * ewc_loss
                
            grads = tape.gradient(total_loss, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def _compute_fisher_fast(self, batch_size=64):
        if not self.memory or self._memory_size() == 0:
            return
            
        print("\n  [EWC] Computing Fisher Information Matrix...")
        self.opt_weights = [tf.identity(w) for w in self.model.trainable_weights]
        fisher_matrix = [tf.zeros_like(w) for w in self.model.trainable_weights]
        
        num_samples = min(self._memory_size(), 1000)
        X_mem, y_mem = self.memory.sample(num_samples)
        
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        num_batches = 0
        for i in range(0, num_samples, batch_size):
            X_batch = {k: v[i:i+batch_size] for k, v in X_mem.items()}
            y_batch = y_mem[i:i+batch_size]
            
            with tf.GradientTape() as tape:
                preds = self.model(X_batch, training=False)
                loss = bce(y_batch, preds) 
            
            grads = tape.gradient(loss, self.model.trainable_weights)
            for idx, g in enumerate(grads):
                if g is not None:
                    fisher_matrix[idx] += tf.square(g)
            num_batches += 1
            
        self.fisher_matrix = [f / float(max(1, num_batches)) for f in fisher_matrix]

    def _batch_metrics(self, preds_binary, preds_prob, y):
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
            roc_auc = 0.5

        try:
            pr_auc = float(average_precision_score(y, preds_prob))
        except ValueError:
            pr_auc = 0.0

        try:
            kappa = float(cohen_kappa_score(y, preds_binary))
        except Exception:
            kappa = 0.0

        return acc, precision, recall, f1, roc_auc, pr_auc, kappa

    def process_stream(self, generator, num_batches, batch_size=64,
                       drift_prob=0.002, use_adaptive_thresh=True):

        print(f"\nStarting simulation for configuration: {self.config_name}")
        print("Bootstrapping model...")
        for _ in range(1000):  # Increased from 500 → 1000 for better initial calibration
            X, y = generator.generate_batch(batch_size)
            self._train(X, y)
            if self.memory:
                for i in range(batch_size):
                    x_i = {k: v[i] for k, v in X.items()}
                    self.memory.add(x_i, y[i])
        for batch_idx in tqdm(range(num_batches), desc=self.config_name):

            X, y = generator.generate_batch(batch_size)

            preds = self.model(X, training=False).numpy().ravel()
            y_flat = y.ravel()
            
            if use_adaptive_thresh:
                self.adaptive_threshold.add(y_flat, preds)
                if batch_idx > 0 and batch_idx % 50 == 0:   # Update every 50 batches (was 100)
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

            # --- Drift detection (F2-biased: recall weighted 2×) ---
            f2 = fbeta_score(y_flat, preds_binary, beta=2.0, zero_division=0)
            drift_detected = False
            if self.detector:
                if isinstance(self.detector, ADWINDriftDetector):
                    if self.detector.update(f2):
                        drift_detected = True
                        self.detector.reset()
                elif isinstance(self.detector, WindowDriftDetector):
                    if self.detector.update(recall):
                        drift_detected = True
                        self.detector.reset()
                elif isinstance(self.detector, DualDriftDetector):
                    if self.detector.update(recall, f2):
                        drift_detected = True
                        self.detector.reset()

            # --- Adaptation on detected drift (cosine LR decay + active replay) ---
            if drift_detected:
                print(f"\n  [!] Drift detected at batch {batch_idx}! "
                      f"(recall={recall:.3f}, f1={f2:.3f}) Adapting...")
                self.drift_events.append(batch_idx)

                if self.memory and self._memory_size() > 500:
                    self.memory.partial_clear(clear_fraud=False)

                if self.use_ewc:
                    self._compute_fisher_fast(batch_size=batch_size)

                # Fine-tune with cosine LR decay + 50% replay / 50% current data
                for step in range(DRIFT_DECAY_STEPS):
                    cosine_lr = self._cosine_lr(step, DRIFT_DECAY_STEPS)
                    has_memory = (self.memory and self._memory_size() >= batch_size // 2)
                    if has_memory:
                        X_mem, y_mem = self.memory.sample(batch_size // 2)
                        if len(y_mem) > 0:
                            # 50% current, 50% replay
                            half = batch_size // 2
                            X_combined = {k: np.concatenate((X[k][:half], X_mem[k])) for k in X.keys()}
                            y_combined = np.concatenate((y[:half], y_mem))
                            indices = np.arange(len(y_combined))
                            np.random.shuffle(indices)
                            X_combined = {k: X_combined[k][indices] for k in X_combined.keys()}
                            y_combined = y_combined[indices]
                            self._train(X_combined, y_combined, lr=cosine_lr, apply_ewc=self.use_ewc)
                        else:
                            self._train(X, y, lr=cosine_lr, apply_ewc=self.use_ewc)
                    else:
                        self._train(X, y, lr=cosine_lr, apply_ewc=self.use_ewc)

                # Ensure LR is cleanly restored to BASE_LR
                self._set_lr(BASE_LR)

            # --- Standard online update (every non-drift batch) ---
            else:
                if self.memory and self._memory_size() >= batch_size:
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
    # Simulation parameters (Reduced to 200,000 transactions for optimal laptop stability & performance)
    NUM_TRANSACTIONS = 200_000
    BATCH_SIZE       = 64
    NUM_BATCHES      = NUM_TRANSACTIONS // BATCH_SIZE
    DRIFT_PROB       = 0.002

    os.makedirs('data', exist_ok=True)
    tracker = BestModelTracker(primary_metric='f1')

    configs = [
        {'name': 'Static MLP Baseline', 'is_classical': True},
        {'name': 'Static Baseline', 'detector': None, 'memory': None, 'is_classical': False},
        {'name': 'Config A (ADWIN + Prioritized)', 'detector': 'adwin', 'memory': 'prioritized', 'is_classical': False},
        {'name': 'Config B (Window + Balanced)', 'detector': 'window', 'memory': 'balanced', 'is_classical': False},
        {'name': 'Config C (Dual + Dynamic + Residual)', 'detector': 'dual', 'memory': 'balanced', 'is_classical': False},
        {'name': 'Config D (EWC Continual LSTM)', 'detector': 'dual', 'memory': 'balanced', 'is_classical': False, 'use_ewc': True},
    ]

    generator = RealtimeDataGenerator(seq_len=10)  # Increased from 5 → 10 for richer temporal context

    for conf in configs:
        generator.reset() # Reset generator for the new config
        
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
                cat_vocab_dict=generator.cat_vocab,
                use_ewc=conf.get('use_ewc', False)
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
        
        # Evaluate performance against stored champion / best model
        tracker.evaluate_and_update(conf['name'], df, rollback_on_failure=False)

        # Free memory between simulation configs to prevent OOM
        del learner
        import gc
        import tensorflow as tf
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == '__main__':
    run_simulation()
