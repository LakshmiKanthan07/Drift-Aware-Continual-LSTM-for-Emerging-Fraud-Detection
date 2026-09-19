import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score, average_precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import lightgbm as lgb
from continual_learner import AdaptiveThreshold

class ClassicalBaselineLearner:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        
        # Initialize standard classical models
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        elif model_type == 'logistic_regression':
            # SGDClassifier with log_loss acts as an incremental logistic regression
            self.model = SGDClassifier(loss='log_loss', random_state=42, learning_rate='optimal')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.adaptive_threshold = AdaptiveThreshold(window_size=1000)
        self.metrics_log = []

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

    def process_stream(self, generator, num_batches, batch_size=64):
        print(f"\nStarting simulation for baseline: {self.model_type}")

        # ---------------------------------------------------------------
        # Bootstrap Phase: Collect first 250 batches to train the static model
        # ---------------------------------------------------------------
        print("Bootstrapping classical model...")
        X_bootstrap = []
        y_bootstrap = []
        for _ in range(250):
            X, y = generator.generate_batch(batch_size)
            # Flatten sequence dimension since classical models don't use it
            X_bootstrap.append(X.reshape(batch_size, -1))
            y_bootstrap.append(y.reshape(-1))

        X_train = np.vstack(X_bootstrap)
        y_train = np.concatenate(y_bootstrap)

        print(f"Training {self.model_type} on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

        # ---------------------------------------------------------------
        # Streaming Phase
        # ---------------------------------------------------------------
        for batch_idx in tqdm(range(num_batches), desc=self.model_type):
            X, y = generator.generate_batch(batch_size)
            
            # Flatten sequence dimension
            X_flat = X.reshape(batch_size, -1)
            y_flat = y.reshape(-1)

            # --- Predict ---
            if hasattr(self.model, "predict_proba"):
                preds_prob = self.model.predict_proba(X_flat)[:, 1]
            else:
                # for SGDClassifier log_loss, predict_proba is usually available
                # but let's be safe.
                preds_prob = self.model.predict_proba(X_flat)[:, 1]
            
            # --- Dynamic Threshold Optimization ---
            self.adaptive_threshold.add(y_flat, preds_prob)
            if batch_idx > 0 and batch_idx % 100 == 0:
                self.adaptive_threshold.update_threshold()
            decision_thresh = self.adaptive_threshold.current_threshold
                
            preds_binary = (preds_prob > decision_thresh).astype(int)

            acc, precision, recall, f1, roc_auc, pr_auc, kappa = self._batch_metrics(preds_binary, preds_prob, y_flat)

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

            # Incremental models can update on stream
            if self.model_type == 'logistic_regression':
                # Note: must pass classes so it knows binary setting
                self.model.partial_fit(X_flat, y_flat, classes=np.array([0, 1]))

        return pd.DataFrame(self.metrics_log), [] # return empty drift array
