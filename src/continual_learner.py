import os
import glob
import gc
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
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
CLASS_WEIGHT  = {0: 1.0, 1: 20.0}   # Adjusted to 20 for focal loss calibration
BASE_LR       = 0.001
DRIFT_LR      = 0.0015            # Lowered from 0.003: gentle drift LR to prevent shock
DRIFT_DECAY_STEPS = 200             # Doubled from 100 for smoother post-drift recovery

# EWC lambda bounds — dynamically scaled by drift severity
EWC_LAMBDA_MIN = 500.0    # Low regularization on severe drift  → allow plasticity
EWC_LAMBDA_MAX = 8000.0   # High regularization on mild drift   → prevent forgetting

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision (FP16) enabled on GPU.")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")
else:
    print("No GPU detected. Running on CPU using default float32.")

class AdaptiveThreshold:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.y_true_cache = collections.deque(maxlen=window_size)
        self.y_prob_cache = collections.deque(maxlen=window_size)
        self.current_threshold = 0.25

    def add(self, y_true_batch, y_prob_batch):
        self.y_true_cache.extend(y_true_batch.flatten())
        self.y_prob_cache.extend(y_prob_batch.flatten())

    def update_threshold(self, is_drift=False):
        min_samples = 80 if is_drift else 200
        if len(self.y_true_cache) < min_samples:
            return self.current_threshold  

        y_true = np.array(self.y_true_cache)
        y_prob = np.array(self.y_prob_cache)
        
        if np.sum(y_true) == 0:
            return self.current_threshold

        best_thresh = self.current_threshold
        best_score = -1e9

        # Rate-limit threshold movement: allow larger step (0.12) upon drift alert
        # so the model realigns immediately without lagging for 50 batches.
        MAX_STEP = 0.12 if is_drift else 0.06
        lo = max(0.10, self.current_threshold - MAX_STEP)
        hi = min(0.65, self.current_threshold + MAX_STEP)

        for thresh in np.arange(lo, hi + 1e-9, 0.02):
            preds = (y_prob > thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y_true == 1)))
            fp = int(np.sum((preds == 1) & (y_true == 0)))
            
            savings = tp * 500 - fp * 10
            f1_5 = fbeta_score(y_true, preds, beta=1.5, zero_division=0)
            combined_score = savings + 100 * f1_5
            if combined_score > best_score:
                best_score = combined_score
                best_thresh = thresh
                
        self.current_threshold = best_thresh
        return self.current_threshold

class ContinualLearner:
    def __init__(self, config_name, detector_type, memory_type,
                 seq_len, num_cont_features, cat_vocab_dict, use_ewc=False, use_agem=False):
        self.config_name = config_name
        self.use_ewc = use_ewc
        self.use_agem = use_agem
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
            
        self.adaptive_threshold = AdaptiveThreshold(window_size=500) 
        self.metrics_log  = []
        self.drift_events = []
        self._ewc_lambda  = (EWC_LAMBDA_MIN + EWC_LAMBDA_MAX) / 2.0  

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

    def _project_gradients(self, grads, grads_ref):
        # Convert any IndexedSlices (e.g. from Embedding layers) to dense tensors 
        # so that tf.function if/else branches return matching types
        dense_grads = [tf.convert_to_tensor(g) if g is not None else None for g in grads]
        dense_grads_ref = [tf.convert_to_tensor(g) if g is not None else None for g in grads_ref]
        
        flat_grads = tf.concat([tf.reshape(g, [-1]) for g in dense_grads if g is not None], axis=0)
        flat_grads_ref = tf.concat([tf.reshape(g, [-1]) for g in dense_grads_ref if g is not None], axis=0)
        
        dot_product = tf.reduce_sum(flat_grads * flat_grads_ref)
        if dot_product < 0:
            ref_norm_sq = tf.reduce_sum(flat_grads_ref * flat_grads_ref) + 1e-8
            proj_flat = flat_grads - (dot_product / ref_norm_sq) * flat_grads_ref
            
            projected_grads = []
            offset = 0
            for g in dense_grads:
                if g is not None:
                    shape = tf.shape(g)
                    size = tf.reduce_prod(shape)
                    g_proj = tf.reshape(proj_flat[offset:offset+size], shape)
                    projected_grads.append(g_proj)
                    offset += size
                else:
                    projected_grads.append(None)
            return projected_grads
        return dense_grads

    @tf.function(reduce_retracing=True)
    def _train_ewc_step(self, X_dict, y, sample_weights, fisher_matrix, opt_weights, ewc_lambda, freeze_features):
        with tf.GradientTape() as tape:
            preds = self.model(X_dict, training=True)
            base_loss = self.model.compiled_loss(y, preds, sample_weight=sample_weights)
            
            ewc_loss = 0.0
            for weight, fisher, opt_weight in zip(self.model.trainable_weights, fisher_matrix, opt_weights):
                name = weight.name.lower()
                multiplier = 2.0 if ('emb' in name or 'conv' in name) else (0.2 if 'dense' in name else 1.0)
                ewc_loss += tf.reduce_sum(fisher * tf.square(weight - opt_weight)) * multiplier
                
            total_loss = base_loss + (ewc_lambda / 2.0) * ewc_loss
            
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        grads = [tf.convert_to_tensor(g) if g is not None else None for g in grads]
        
        if freeze_features:
            masked_grads = []
            for g, w in zip(grads, self.model.trainable_weights):
                if g is None:
                    masked_grads.append(None)
                elif 'dense' in w.name.lower() or 'se_fc' in w.name.lower():
                    masked_grads.append(g)
                else:
                    masked_grads.append(tf.zeros_like(g))
            grads = masked_grads
            
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    @tf.function(reduce_retracing=True)
    def _train_agem_step(self, X_dict, y, sample_weights, X_ref, y_ref, ref_weights, freeze_features):
        # 1. Compute gradients for reference memory batch
        with tf.GradientTape() as tape_ref:
            preds_ref = self.model(X_ref, training=True)
            loss_ref = self.model.compiled_loss(y_ref, preds_ref, sample_weight=ref_weights)
        grads_ref = tape_ref.gradient(loss_ref, self.model.trainable_weights)
        
        # 2. Compute gradients for current batch
        with tf.GradientTape() as tape:
            preds = self.model(X_dict, training=True)
            loss = self.model.compiled_loss(y, preds, sample_weight=sample_weights)
        grads = tape.gradient(loss, self.model.trainable_weights)
        
        # 3. Project and apply
        grads_proj = self._project_gradients(grads, grads_ref)
        if freeze_features:
            masked_grads = []
            for g, w in zip(grads_proj, self.model.trainable_weights):
                if g is None:
                    masked_grads.append(None)
                elif 'dense' in w.name.lower() or 'se_fc' in w.name.lower():
                    masked_grads.append(g)
                else:
                    masked_grads.append(tf.zeros_like(g))
            grads_proj = masked_grads
            
        self.model.optimizer.apply_gradients(zip(grads_proj, self.model.trainable_weights))

    @tf.function(reduce_retracing=True)
    def _train_standard_step(self, X_dict, y, sample_weights, freeze_features):
        with tf.GradientTape() as tape:
            preds = self.model(X_dict, training=True)
            loss = self.model.compiled_loss(y, preds, sample_weight=sample_weights)
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads = [tf.convert_to_tensor(g) if g is not None else None for g in grads]
        
        if freeze_features:
            masked_grads = []
            for g, w in zip(grads, self.model.trainable_weights):
                if g is None:
                    masked_grads.append(None)
                elif 'dense' in w.name.lower() or 'se_fc' in w.name.lower():
                    masked_grads.append(g)
                else:
                    masked_grads.append(tf.zeros_like(g))
            grads = masked_grads
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def _train(self, X_dict, y, lr=None, apply_ewc=False, apply_agem=False, freeze_features=False):
        if lr is not None:
            self._set_lr(lr)
        # Remove redundant class weights; BinaryFocalCrossentropy handles it natively
        sample_weights = np.ones_like(y.ravel(), dtype=np.float32)
        
        freeze = tf.constant(freeze_features, dtype=tf.bool)
        
        if apply_ewc and hasattr(self, 'fisher_matrix'):
            self._train_ewc_step(
                X_dict, y, sample_weights, 
                self.fisher_matrix, self.opt_weights, 
                tf.constant(self._ewc_lambda, dtype=tf.float32),
                freeze
            )
            
        elif apply_agem and self.memory and self._memory_size() > 0:
            X_ref, y_ref = self.memory.sample(min(32, self._memory_size()))
            ref_weights = np.where(y_ref.ravel() == 1, CLASS_WEIGHT[1], CLASS_WEIGHT[0]).astype(np.float32)
            self._train_agem_step(X_dict, y, sample_weights, X_ref, y_ref, ref_weights, freeze)
            
        else:
            self._train_standard_step(X_dict, y, sample_weights, freeze)

    def _compute_fisher_fast(self, batch_size=64):
        if not self.memory or self._memory_size() == 0:
            return
            
        print("\n  [EWC] Computing Fisher Information Matrix...")
        self.opt_weights = [tf.identity(w) for w in self.model.trainable_weights]
        fisher_matrix = [tf.zeros_like(w) for w in self.model.trainable_weights]
        
        num_samples = min(self._memory_size(), 1000)
        X_mem, y_mem = self.memory.sample(num_samples)
        num_samples = len(y_mem)
        if num_samples == 0:
            return
        
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        num_batches = 0
        for i in range(0, num_samples, batch_size):
            X_batch = {k: v[i:i+batch_size] for k, v in X_mem.items()}
            y_batch = y_mem[i:i+batch_size]
            if len(y_batch) == 0:
                continue
            
            with tf.GradientTape() as tape:
                preds = self.model(X_batch, training=False)
                loss = bce(y_batch, preds) 
            
            grads = tape.gradient(loss, self.model.trainable_weights)
            for idx, g in enumerate(grads):
                if g is not None:
                    fisher_matrix[idx] += tf.square(g)
            num_batches += 1
            
        self.fisher_matrix = [f / float(max(1, num_batches)) for f in fisher_matrix]

    def _batch_metrics(self, preds_binary, preds_prob, y, batch_idx=0):
        acc = float(np.mean(preds_binary == y))

        tp = int(np.sum((preds_binary == 1) & (y == 1)))
        fp = int(np.sum((preds_binary == 1) & (y == 0)))
        fn = int(np.sum((preds_binary == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        f2 = fbeta_score(y, preds_binary, beta=2.0, zero_division=0)
        
        # Savings = True Positives * $500 (avg fraud) - False Positives * $10 (investigation cost)
        savings = float(tp * 500 - fp * 10)
        
        # Precision @ 1% Alert Rate
        k = max(1, int(len(y) * 0.01))
        top_k_indices = np.argsort(preds_prob)[-k:]
        top_k_y = y[top_k_indices]
        precision_at_k = float(np.mean(top_k_y)) if len(top_k_y) > 0 else 0.0

        if batch_idx % 5 == 0:
            try:
                roc_auc = float(roc_auc_score(y, preds_prob))
            except ValueError:
                roc_auc = 0.5
            try:
                pr_auc = float(average_precision_score(y, preds_prob))
            except ValueError:
                pr_auc = 0.0
            
            self._last_roc_auc = roc_auc
            self._last_pr_auc = pr_auc
        else:
            roc_auc = getattr(self, '_last_roc_auc', 0.5)
            pr_auc = getattr(self, '_last_pr_auc', 0.0)

        try:
            kappa = float(cohen_kappa_score(y, preds_binary))
        except Exception:
            kappa = 0.0

        return acc, precision, recall, f1, f2, roc_auc, pr_auc, kappa, savings, precision_at_k

    def _update_ewc_lambda(self, drift_severity: float) -> None:
        """Dynamically scale EWC lambda based on detected drift severity.
        
        Severe drift  → lower lambda (allow plasticity, adapt fast)
        Mild drift    → higher lambda (preserve old knowledge, avoid forgetting)
        """
        # severity: 0.0 (very mild) to 1.0 (complete failure)
        severity = float(np.clip(drift_severity, 0.0, 1.0))
        # Linearly interpolate: severe → min, mild → max
        self._ewc_lambda = EWC_LAMBDA_MAX - severity * (EWC_LAMBDA_MAX - EWC_LAMBDA_MIN)
        print(f"     [EWC] Dynamic lambda set to {self._ewc_lambda:.0f} "
              f"(drift severity={severity:.2f})")

    # Weights after bootstrapping, shared across configs for the same seed.
    _BOOTSTRAP_CACHE = {}

    def process_stream(self, generator, num_batches, batch_size=64,
                       drift_prob=0.002, use_adaptive_thresh=True, bootstrap_key=None,
                       label_delay=None):

        # Fraud labels arrive late in reality (chargebacks / investigations).
        # The model, memory, threshold tuner and drift detector only see a batch's
        # labels `label_delay` batches after it was scored.
        if label_delay is None:
            label_delay = int(os.environ.get('LABEL_DELAY', 5))
        pending = collections.deque()

        print(f"\nStarting simulation for configuration: {self.config_name}")
        cached = (ContinualLearner._BOOTSTRAP_CACHE.get(bootstrap_key)
                  if bootstrap_key is not None else None)
        print("Bootstrapping model..." if cached is None
              else "Bootstrapping (reusing cached weights)...")
        if cached is None:
            X_boot = {k: [] for k in generator.cat_cols + ['continuous']}
            y_boot = []
            
            for _ in range(250):
                X, y = generator.generate_batch(batch_size)
                for k in X:
                    X_boot[k].append(X[k])
                y_boot.append(y)
                
                if self.memory:
                    for i in range(batch_size):
                        x_i = {k: v[i] for k, v in X.items()}
                        self.memory.add(x_i, y[i])
                        
            X_train = {k: np.vstack(v) for k, v in X_boot.items()}
            y_train = np.vstack(y_boot)
            
            print(f"Training offline bootstrap for 5 epochs on {len(y_train)} samples...")
            self.model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=1)
                           
            del X_boot, y_boot, X_train, y_train
            import gc
            gc.collect()
            
            if bootstrap_key is not None:
                ContinualLearner._BOOTSTRAP_CACHE[bootstrap_key] = self.model.get_weights()
        else:
            for _ in range(250):
                X, y = generator.generate_batch(batch_size)
                if self.memory:
                    for i in range(batch_size):
                        x_i = {k: v[i] for k, v in X.items()}
                        self.memory.add(x_i, y[i])
            self.model.set_weights(cached)
            
        for batch_idx in tqdm(range(num_batches), desc=self.config_name):
            batch_start_time = time.perf_counter()

            X, y = generator.generate_batch(batch_size)

            preds = np.asarray(self.model.predict_on_batch(X)).ravel()
            y_flat = y.ravel()
            
            if use_adaptive_thresh:
                if batch_idx > 0 and batch_idx % 50 == 0:
                    self.adaptive_threshold.update_threshold()
                decision_thresh = self.adaptive_threshold.current_threshold
            else:
                decision_thresh = 0.25
                
            preds_binary = (preds > decision_thresh).astype(int)

            acc, precision, recall, f1, f2, roc_auc, pr_auc, kappa, savings, precision_at_k = self._batch_metrics(preds_binary, preds, y_flat, batch_idx=batch_idx)

            log_entry = {
                'batch':     batch_idx,
                'accuracy':  acc,
                'precision': precision,
                'recall':    recall,
                'f1':        f1,
                'f2':        f2,
                'roc_auc':   roc_auc,
                'pr_auc':    pr_auc,
                'kappa':     kappa,
                'savings':   savings,
                'precision_at_1_percent': precision_at_k,
                'threshold': decision_thresh
            }
            self.metrics_log.append(log_entry)

            # --- Delayed label feedback ---
            pending.append((X, y, y_flat, preds, recall, f2, pr_auc))
            if len(pending) <= label_delay:
                continue
            X, y, y_flat, preds, recall, f2, pr_auc = pending.popleft()
            if use_adaptive_thresh:
                self.adaptive_threshold.add(y_flat, preds)

            # --- Add to memory (with priority for PrioritizedReplayBuffer) ---
            if self.memory:
                for i in range(batch_size):
                    x_i = {k: v[i] for k, v in X.items()}
                    y_i = np.atleast_1d(y[i])
                    if isinstance(self.memory, PrioritizedReplayBuffer):
                        priority = abs(float(y_flat[i]) - float(preds[i])) + 1e-5
                        prb: PrioritizedReplayBuffer = self.memory
                        prb.add(x_i, y_i, priority=priority)
                    else:
                        self.memory.add(x_i, y_i)

            # --- Drift detection using 3 signals ---
            # f2 was already computed in _batch_metrics above; reuse it.
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
                    # Feed all 3 signals: recall (abrupt), f2 (gradual), pr_auc (surface)
                    if self.detector.update(recall, f2, pr_auc):
                        drift_detected = True
                        self.detector.reset()

            if drift_detected:
                severity = getattr(self.detector, 'last_drift_severity', 0.5)
                print(f"\n  [!] Drift detected at batch {batch_idx}! "
                      f"(recall={recall:.3f}, f2={f2:.3f}, pr_auc={pr_auc:.3f}, "
                      f"severity={severity:.2f}) Adapting...")
                self.drift_events.append(batch_idx)
                if self.use_ewc:
                    self._update_ewc_lambda(severity)
                    self._compute_fisher_fast(batch_size=batch_size)

                adaptation_steps = int(8 + 17 * severity)
                # When A-GEM is active, gradient projection protects prior memory;
                # allow the BiLSTM to adapt to new temporal fraud patterns without hard-freezing.
                freeze_feat = (not self.use_agem)
                for step in range(adaptation_steps):
                    cosine_lr = self._cosine_lr(step, adaptation_steps)
                    has_memory = (self.memory and self._memory_size() >= batch_size // 2)
                    if has_memory:
                        X_mem, y_mem = self.memory.sample(batch_size // 2)
                        if len(y_mem) > 0:
                            y_mem = np.reshape(y_mem, (-1, 1))
                            # 50% current, 50% replay
                            half = batch_size // 2
                            X_combined = {k: np.concatenate((X[k][:half], X_mem[k])) for k in X.keys()}
                            y_combined = np.concatenate((y[:half], y_mem))
                            indices = np.arange(len(y_combined))
                            np.random.shuffle(indices)
                            X_combined = {k: X_combined[k][indices] for k in X_combined.keys()}
                            y_combined = y_combined[indices]
                            self._train(X_combined, y_combined, lr=cosine_lr,
                                        apply_ewc=self.use_ewc, apply_agem=self.use_agem, freeze_features=freeze_feat)
                        else:
                            self._train(X, y, lr=cosine_lr, apply_ewc=self.use_ewc, apply_agem=self.use_agem, freeze_features=freeze_feat)
                    else:
                        self._train(X, y, lr=cosine_lr, apply_ewc=self.use_ewc, apply_agem=self.use_agem, freeze_features=freeze_feat)

                self._set_lr(BASE_LR)
                if use_adaptive_thresh:
                    self.adaptive_threshold.update_threshold(is_drift=True)

            else:
                if self.memory and self._memory_size() >= batch_size:
                    X_mem, y_mem = self.memory.sample(batch_size)
                    if len(y_mem) > 0:
                        y_mem = np.reshape(y_mem, (-1, 1))
                        X_combined = {k: np.concatenate((X[k], X_mem[k])) for k in X.keys()}
                        y_combined = np.concatenate((y, y_mem))
                        indices = np.arange(len(y_combined))
                        np.random.shuffle(indices)
                        X_combined = {k: X_combined[k][indices] for k in X_combined.keys()}
                        y_combined = y_combined[indices]
                        self._train(X_combined, y_combined, lr=BASE_LR, apply_agem=self.use_agem)
                    else:
                        self._train(X, y, lr=BASE_LR, apply_agem=self.use_agem)
                else:
                    self._train(X, y, lr=BASE_LR, apply_agem=self.use_agem)

            log_entry['latency_ms'] = (time.perf_counter() - batch_start_time) * 1000.0

        return pd.DataFrame(self.metrics_log), self.drift_events

    # ------------------------------------------------------------------
    # Online (per-transaction) processing
    # ------------------------------------------------------------------
    def process_stream_online(self, generator, num_transactions,
                              mini_batch_size=32, label_delay=5,
                              use_adaptive_thresh=True, bootstrap_key=None):
        """True online mode: score every transaction individually, accumulate
        into a mini-batch of `mini_batch_size` confirmed (label-delayed)
        transactions, then run drift detection + a single training step.

        Scoring latency is per-transaction (~3-5 ms each).
        Gradient updates stay stable because training always uses
        `mini_batch_size` samples (never a single noisy gradient).
        """
        # --- Bootstrap (reuse the same cache as process_stream) ---
        label_delay = int(label_delay)
        print(f"\nStarting ONLINE simulation for: {self.config_name}")
        cached = (ContinualLearner._BOOTSTRAP_CACHE.get(bootstrap_key)
                  if bootstrap_key is not None else None)
        print("Bootstrapping model..." if cached is None
              else "Bootstrapping (reusing cached weights)...")

        boot_txs = 250 * mini_batch_size
        if cached is None:
            X_boot = {k: [] for k in generator.cat_cols + ['continuous']}
            y_boot = []
            for _ in range(boot_txs):
                x_i, y_i = generator.generate_one()
                for k in x_i:
                    X_boot[k].append(x_i[k])
                y_boot.append(y_i)
                if self.memory:
                    self.memory.add({k: x_i[k][0] for k in x_i}, y_i[0])
            X_train = {k: np.concatenate(v, axis=0) for k, v in X_boot.items()}
            y_train = np.concatenate(y_boot, axis=0)
            print(f"Training offline bootstrap for 5 epochs on {len(y_train)} samples...")
            self.model.fit(X_train, y_train, epochs=5,
                           batch_size=mini_batch_size, verbose=1)
            del X_boot, y_boot, X_train, y_train
            import gc; gc.collect()
            if bootstrap_key is not None:
                ContinualLearner._BOOTSTRAP_CACHE[bootstrap_key] = self.model.get_weights()
        else:
            for _ in range(boot_txs):
                x_i, y_i = generator.generate_one()
                if self.memory:
                    self.memory.add({k: x_i[k][0] for k in x_i}, y_i[0])
            self.model.set_weights(cached)

        # --- Streaming loop ---
        pending      = collections.deque()   # label-delay queue
        train_buffer = []                    # confirmed transactions awaiting training

        for tx_idx in tqdm(range(num_transactions), desc=self.config_name):
            tx_start = time.perf_counter()

            # 1. Score this single transaction immediately
            x_dict, y = generator.generate_one()
            pred = float(np.asarray(self.model.predict_on_batch(x_dict)).ravel()[0])

            decision_thresh = self.adaptive_threshold.current_threshold
            pred_binary = int(pred > decision_thresh)

            self.metrics_log.append({
                'transaction':  tx_idx,
                'pred_prob':    pred,
                'pred_binary':  pred_binary,
                'threshold':    decision_thresh,
                'latency_ms':   (time.perf_counter() - tx_start) * 1000.0,
            })

            # 2. Label-delay queue — labels arrive `label_delay` transactions later
            pending.append((x_dict, y, pred))
            if len(pending) <= label_delay:
                continue
            x_d, y_d, pred_d = pending.popleft()
            train_buffer.append(({'continuous': x_d['continuous'][0],
                                   **{k: x_d[k][0] for k in x_d if k != 'continuous'}},
                                  y_d[0], pred_d))

            # 3. Every mini_batch_size confirmed transactions → train
            if len(train_buffer) < mini_batch_size:
                continue

            # Stack mini-batch
            X_mb = {
                'continuous': np.stack([t[0]['continuous'] for t in train_buffer], axis=0),
                **{k: np.stack([t[0][k] for t in train_buffer], axis=0)
                   for k in train_buffer[0][0] if k != 'continuous'}
            }
            y_mb    = np.concatenate([t[1] for t in train_buffer], axis=0)   # (N, 1)
            preds_mb = np.array([t[2] for t in train_buffer])                # (N,)
            y_flat   = y_mb.ravel()

            # Adaptive threshold update
            if use_adaptive_thresh:
                self.adaptive_threshold.add(y_flat, preds_mb)
                if tx_idx % (mini_batch_size * 50) == 0:
                    self.adaptive_threshold.update_threshold()

            # Add confirmed transactions to memory
            if self.memory:
                for i in range(len(y_flat)):
                    x_i = {k: X_mb[k][i] for k in X_mb}
                    y_i = np.atleast_1d(y_mb[i])
                    if isinstance(self.memory, PrioritizedReplayBuffer):
                        priority = abs(float(y_flat[i]) - float(preds_mb[i])) + 1e-5
                        prb: PrioritizedReplayBuffer = self.memory
                        prb.add(x_i, y_i, priority=priority)
                    else:
                        self.memory.add(x_i, y_i)

            # Compute mini-batch metrics for drift detector
            preds_bin_mb = (preds_mb > decision_thresh).astype(int)
            tp = int(np.sum((preds_bin_mb == 1) & (y_flat == 1)))
            fn = int(np.sum((preds_bin_mb == 0) & (y_flat == 1)))
            fp = int(np.sum((preds_bin_mb == 1) & (y_flat == 0)))
            recall_mb = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f2_mb     = fbeta_score(y_flat, preds_bin_mb, beta=2.0, zero_division=0)
            pr_auc_mb = (average_precision_score(y_flat, preds_mb)
                         if np.sum(y_flat) > 0 else 0.0)

            # Drift detection
            drift_detected = False
            if self.detector:
                if isinstance(self.detector, ADWINDriftDetector):
                    if self.detector.update(f2_mb):
                        drift_detected = True; self.detector.reset()
                elif isinstance(self.detector, WindowDriftDetector):
                    if self.detector.update(recall_mb):
                        drift_detected = True; self.detector.reset()
                elif isinstance(self.detector, DualDriftDetector):
                    if self.detector.update(recall_mb, f2_mb, pr_auc_mb):
                        drift_detected = True; self.detector.reset()

            # Train
            if drift_detected:
                severity = getattr(self.detector, 'last_drift_severity', 0.5)
                print(f"\n  [!] Online drift at tx {tx_idx}! "
                      f"(recall={recall_mb:.3f}, f2={f2_mb:.3f}, "
                      f"severity={severity:.2f}) Adapting...")
                self.drift_events.append(tx_idx)
                if self.use_ewc:
                    self._update_ewc_lambda(severity)
                    self._compute_fisher_fast(batch_size=mini_batch_size)

                adaptation_steps = int(8 + 17 * severity)
                freeze_feat = (not self.use_agem)
                for step in range(adaptation_steps):
                    cosine_lr = self._cosine_lr(step, adaptation_steps)
                    half = mini_batch_size // 2
                    has_mem = (self.memory and self._memory_size() >= half)
                    if has_mem:
                        X_mem, y_mem = self.memory.sample(half)
                        if len(y_mem) > 0:
                            y_mem = np.reshape(y_mem, (-1, 1))
                            X_comb = {k: np.concatenate((X_mb[k][:half], X_mem[k]))
                                      for k in X_mb.keys()}
                            y_comb = np.concatenate((y_mb[:half], y_mem))
                            idx_s  = np.random.permutation(len(y_comb))
                            X_comb = {k: X_comb[k][idx_s] for k in X_comb}
                            y_comb = y_comb[idx_s]
                            self._train(X_comb, y_comb, lr=cosine_lr,
                                        apply_ewc=self.use_ewc,
                                        apply_agem=self.use_agem,
                                        freeze_features=freeze_feat)
                            continue
                    self._train(X_mb, y_mb, lr=cosine_lr,
                                apply_ewc=self.use_ewc,
                                apply_agem=self.use_agem,
                                freeze_features=freeze_feat)
                self._set_lr(BASE_LR)
                if use_adaptive_thresh:
                    self.adaptive_threshold.update_threshold(is_drift=True)

            else:
                if self.memory and self._memory_size() >= mini_batch_size:
                    X_mem, y_mem = self.memory.sample(mini_batch_size)
                    if len(y_mem) > 0:
                        y_mem = np.reshape(y_mem, (-1, 1))
                        X_comb = {k: np.concatenate((X_mb[k], X_mem[k]))
                                  for k in X_mb.keys()}
                        y_comb = np.concatenate((y_mb, y_mem))
                        idx_s  = np.random.permutation(len(y_comb))
                        X_comb = {k: X_comb[k][idx_s] for k in X_comb}
                        y_comb = y_comb[idx_s]
                        self._train(X_comb, y_comb, lr=BASE_LR,
                                    apply_agem=self.use_agem)
                    else:
                        self._train(X_mb, y_mb, lr=BASE_LR)
                else:
                    self._train(X_mb, y_mb, lr=BASE_LR)

            train_buffer.clear()

        return pd.DataFrame(self.metrics_log), self.drift_events

def run_simulation():
    NUM_TRANSACTIONS = 40_000
    BATCH_SIZE       = 64
    NUM_BATCHES      = NUM_TRANSACTIONS // BATCH_SIZE
    DRIFT_PROB       = 0.005  

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    tracker = BestModelTracker(primary_metric='f1')

    configs = [
        {'name': 'Static Baseline LSTM',                        'detector': None,   'memory': None,        'is_classical': False, 'online': False},
        {'name': 'Config A (ADWIN + Random)',                   'detector': 'adwin','memory': 'random',     'is_classical': False, 'online': False},
        {'name': 'Config F (Dual Drift + Prioritized + A-GEM)', 'detector': 'dual', 'memory': 'prioritized','is_classical': False, 'use_agem': True, 'online': False},
    ]

    generator = RealtimeDataGenerator(seq_len=10)  
    execution_times = {}

    # Synthetic abrupt drift with KNOWN locations (evaluation-batch indices).
    # Set INJECT_DRIFT=0 to stream the raw data without injected drift.
    BOOTSTRAP_TX = 1000 * BATCH_SIZE
    gt_path = os.path.join(DATA_DIR, 'ground_truth_drifts.csv')
    if int(os.environ.get('INJECT_DRIFT', 1)):
        GT_DRIFT_BATCHES = [150, 300, 450]
        generator.set_drift([BOOTSTRAP_TX + b * BATCH_SIZE for b in GT_DRIFT_BATCHES],
                            magnitude=float(os.environ.get('DRIFT_MAGNITUDE', 3.0)))
        pd.DataFrame({'batch': GT_DRIFT_BATCHES}).to_csv(gt_path, index=False)
    elif os.path.exists(gt_path):
        os.remove(gt_path)

    # Number of independent runs per config (override with env var NUM_SEEDS).
    # Seed 0 -> <name>_metrics.csv / _drifts.csv (backward compatible).
    # Seed k>0 -> <name>_seed{k}_metrics.csv / _drifts.csv
    NUM_SEEDS = int(os.environ.get('NUM_SEEDS', 3))

    for conf in configs:
        safe_name = (conf['name']
                     .replace(' ', '_')
                     .replace('(', '')
                     .replace(')', '')
                     .replace('+', 'and'))
        seed_times = []

        for seed in range(NUM_SEEDS):
            tf.keras.utils.set_random_seed(seed)
            generator.reset() 
            
            start_time = time.time()
            
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
                    use_ewc=conf.get('use_ewc', False),
                    use_agem=conf.get('use_agem', False)
                )

                # Adaptive thresholding is applied to ALL models (including the static
                # baseline) so that savings differences are not caused by threshold tuning.
                if conf.get('online', False):
                    # True online mode: score one transaction at a time
                    df, drifts = learner.process_stream_online(
                        generator,
                        num_transactions=NUM_TRANSACTIONS,
                        mini_batch_size=32,
                        label_delay=5,
                        use_adaptive_thresh=True,
                        bootstrap_key=seed
                    )
                else:
                    df, drifts = learner.process_stream(
                        generator,
                        num_batches=NUM_BATCHES,
                        batch_size=BATCH_SIZE,
                        drift_prob=DRIFT_PROB,
                        use_adaptive_thresh=True,
                        bootstrap_key=seed
                    )
                
            seed_times.append(time.time() - start_time)

            suffix = '' if seed == 0 else f'_seed{seed}'
            metrics_file = os.path.join(DATA_DIR, f'{safe_name}{suffix}_metrics.csv')
            drifts_file = os.path.join(DATA_DIR, f'{safe_name}{suffix}_drifts.csv')
            df.to_csv(metrics_file, index=False)
            pd.DataFrame(drifts).to_csv(drifts_file, index=False)
            
            if seed == 0:
                tracker.evaluate_and_update(conf['name'], df, rollback_on_failure=False)

            del learner
            tf.keras.backend.clear_session()
            gc.collect()

        execution_times[conf['name']] = float(np.mean(seed_times))
        
        avg_latency = df['latency_ms'].mean() if 'latency_ms' in df.columns else 0.0
        print(f"\n[{conf['name']}] Average Batch Latency: {avg_latency:.2f} ms")
        
    COLOR_MAP = {
        'Static Baseline LSTM':                       '#e74c3c',
        'Config A (ADWIN + Random)':                  '#3498db',
        'Config F (Dual Drift + Prioritized + A-GEM)': '#2ecc71',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    names   = list(execution_times.keys())
    times   = list(execution_times.values())
    colors  = [COLOR_MAP.get(n, '#aaaaaa') for n in names]
    bars    = ax.bar(names, times, color=colors, width=0.55, zorder=2)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{t:.0f}s', ha='center', va='bottom', color='white',
                fontsize=10, fontweight='bold')

    ax.set_xlabel('Configuration', color='white', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', color='white', fontsize=12)
    ax.set_title('Execution Time per Configuration', color='white',
                 fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(colors='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', color='white', fontsize=9)
    ax.yaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7, zorder=1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'execution_times.png'), dpi=150)
    plt.close()

    # Dedicated plot for Average Batch Latency (ms)
    latency_summary = {}
    for conf in configs:
        cname = conf['name']
        safe = (cname.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'and'))
        pattern = os.path.join(DATA_DIR, f'{safe}*_metrics.csv')
        matched_files = glob.glob(pattern)
        seed_means = []
        for mf in matched_files:
            try:
                m_df = pd.read_csv(mf)
                if 'latency_ms' in m_df.columns:
                    v = m_df['latency_ms'].dropna()
                    if len(v) > 0:
                        seed_means.append(v.mean())
            except Exception:
                continue
        if seed_means:
            latency_summary[cname] = float(np.mean(seed_means))

    if latency_summary:
        fig_lat, ax_lat = plt.subplots(figsize=(12, 6))
        fig_lat.patch.set_facecolor('#0e1117')
        ax_lat.set_facecolor('#0e1117')

        l_names = list(latency_summary.keys())
        l_times = [latency_summary[k] for k in l_names]
        l_colors = [COLOR_MAP.get(k, '#aaaaaa') for k in l_names]
        l_bars = ax_lat.bar(l_names, l_times, color=l_colors, width=0.55, zorder=2)

        for bar, t in zip(l_bars, l_times):
            ax_lat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(l_times)*0.015,
                        f'{t:.1f} ms', ha='center', va='bottom', color='white',
                        fontsize=10, fontweight='bold')

        ax_lat.set_xlabel('Configuration', color='white', fontsize=12)
        ax_lat.set_ylabel('Average Batch Latency (ms)', color='white', fontsize=12)
        ax_lat.set_title('Average Batch Latency (ms) — Model Comparison', color='white',
                         fontsize=14, fontweight='bold', pad=12)
        ax_lat.tick_params(colors='white')
        ax_lat.set_xticklabels(l_names, rotation=25, ha='right', color='white', fontsize=9)
        ax_lat.yaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7, zorder=1)
        for spine in ax_lat.spines.values():
            spine.set_visible(False)

        fig_lat.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, 'batch_latencies.png'), dpi=150)
        plt.close()
        print(f"  Saved batch latency chart to {os.path.join(DATA_DIR, 'batch_latencies.png')}")

    metrics_summary = {}
    for conf in configs:
        cname = conf['name']
        safe = (cname.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'and'))
        mfile = os.path.join(DATA_DIR, f'{safe}_metrics.csv')
        if os.path.exists(mfile):
            df = pd.read_csv(mfile)
            tail = df.tail(100)
            # Online-mode CSVs only contain per-transaction scores (no batch
            # aggregate metrics), so guard against missing columns gracefully.
            required = {'pr_auc', 'roc_auc', 'recall', 'f1'}
            if not required.issubset(df.columns):
                continue
            metrics_summary[cname] = {
                'PR-AUC':  round(tail['pr_auc'].mean(),   3),
                'ROC-AUC': round(tail['roc_auc'].mean(),  3),
                'Recall':  round(tail['recall'].mean(),   3),
                'F1':      round(tail['f1'].mean(),       3),
            }

    if metrics_summary:
        metric_names = ['PR-AUC', 'ROC-AUC', 'Recall', 'F1']
        config_names = list(metrics_summary.keys())
        x = np.arange(len(metric_names))
        n_configs = len(config_names)
        total_width = 0.70
        bar_w = total_width / n_configs

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')

        for i, cname in enumerate(config_names):
            vals   = [metrics_summary[cname][m] for m in metric_names]
            color  = COLOR_MAP.get(cname, '#aaaaaa')
            offset = (i - n_configs / 2 + 0.5) * bar_w
            rects  = ax2.bar(x + offset, vals, bar_w * 0.92,
                             label=cname, color=color, zorder=2, alpha=0.92)
            for r, v in zip(rects, vals):
                ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.005,
                         f'{v:.2f}', ha='center', va='bottom',
                         color='white', fontsize=7, fontweight='bold')

        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names, color='white', fontsize=12, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.set_ylim(0, 1.12)
        ax2.yaxis.grid(True, color='#333333', linestyle='--', linewidth=0.7, zorder=1)
        ax2.set_ylabel('Score (last 100 batches)', color='white', fontsize=12)
        ax2.set_title('Multi-Metric Comparison — All Configurations',
                      color='white', fontsize=14, fontweight='bold', pad=12)
        ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8,
                   loc='upper right', framealpha=0.8)
        for spine in ax2.spines.values():
            spine.set_visible(False)

        fig2.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, 'metric_comparison_bar.png'), dpi=150)
        plt.close()
        print(f"\n[SUCCESS] Charts saved to {DATA_DIR}")

if __name__ == '__main__':
    run_simulation()