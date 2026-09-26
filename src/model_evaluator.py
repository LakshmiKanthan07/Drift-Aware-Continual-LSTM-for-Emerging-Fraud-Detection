import os
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BEST_MODEL_FILE = os.path.join(DATA_DIR, 'best_model_results.csv')
HISTORY_FILE = os.path.join(DATA_DIR, 'model_comparison_history.csv')


def get_git_commit_hash():
    """Retrieve the current Git commit short hash if inside a git repository."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__)
        ).decode('utf-8').strip()
        return commit
    except Exception:
        return 'N/A'


def execute_git_rollback():
    """Roll back uncommitted code changes in working directory using git restore/reset."""
    try:
        print("\n[ROLLBACK TRIPPED] New model performance did not surpass best score!")
        print("Rolling back working directory code changes...")
        subprocess.check_call(
            ['git', 'restore', '.'],
            cwd=os.path.dirname(__file__)
        )
        print("[SUCCESS] Code successfully rolled back to last clean commit.\n")
        return True
    except Exception as e:
        print(f"[ERROR] Git rollback failed: {e}")
        return False


class BestModelTracker:
    def __init__(self, primary_metric='f1', data_dir=DATA_DIR):
        self.primary_metric = primary_metric.lower()
        self.data_dir = data_dir
        self.best_file = os.path.join(self.data_dir, 'best_model_results.csv')
        self.history_file = os.path.join(self.data_dir, 'model_comparison_history.csv')
        os.makedirs(self.data_dir, exist_ok=True)

    def get_best_record(self):
        """Load the current best model record from CSV if exists."""
        if os.path.exists(self.best_file):
            try:
                df = pd.read_csv(self.best_file)
                if not df.empty:
                    return df.iloc[0].to_dict()
            except Exception as e:
                print(f"Warning loading best model file: {e}")
        return None

    def evaluate_and_update(self, config_name, metrics_df, primary_metric=None, rollback_on_failure=False):
        """
        Evaluate current model metrics dataframe against stored best model metrics.
        
        Args:
            config_name (str): Name of the model configuration run.
            metrics_df (pd.DataFrame): DataFrame containing batch metric logs.
            primary_metric (str, optional): Metric to compare ('f1', 'roc_auc', 'pr_auc', 'recall', etc.).
            rollback_on_failure (bool): If True and performance is not better than best, trigger git rollback.

        Returns:
            dict: Evaluation result details (is_best, current_metrics, best_metrics, message).
        """
        eval_metric = (primary_metric or self.primary_metric).lower()
        
        if metrics_df is None or metrics_df.empty:
            print(f"[ERROR] Cannot evaluate empty metrics for {config_name}")
            return {'is_best': False, 'message': 'Empty metrics data'}

        # Calculate macro averages over batches (using tail 80% to focus on post-bootstrap stability if long)
        eval_df = metrics_df if len(metrics_df) < 50 else metrics_df.iloc[int(len(metrics_df) * 0.2):]

        summary_metrics = {
            'config_name': config_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'f1': float(eval_df['f1'].mean()) if 'f1' in eval_df else 0.0,
            'roc_auc': float(eval_df['roc_auc'].mean()) if 'roc_auc' in eval_df else 0.0,
            'pr_auc': float(eval_df['pr_auc'].mean()) if 'pr_auc' in eval_df else 0.0,
            'accuracy': float(eval_df['accuracy'].mean()) if 'accuracy' in eval_df else 0.0,
            'precision': float(eval_df['precision'].mean()) if 'precision' in eval_df else 0.0,
            'recall': float(eval_df['recall'].mean()) if 'recall' in eval_df else 0.0,
            'kappa': float(eval_df['kappa'].mean()) if 'kappa' in eval_df else 0.0,
            'git_commit': get_git_commit_hash(),
            'primary_metric_used': eval_metric
        }

        current_score = summary_metrics.get(eval_metric, 0.0)
        best_record = self.get_best_record()
        
        is_best = False
        if best_record is None:
            is_best = True
            best_score = 0.0
            reason = f"Initial benchmark established for {config_name} with {eval_metric.upper()} = {current_score:.4f}."
        else:
            best_score = float(best_record.get(eval_metric, 0.0))
            if current_score > best_score + 1e-6:
                is_best = True
                diff = current_score - best_score
                reason = f"NEW BEST MODEL! {config_name} beat previous best ({best_score:.4f}) by +{diff:.4f} ({eval_metric.upper()}: {current_score:.4f})."
            else:
                diff = best_score - current_score
                reason = f"Model {config_name} ({current_score:.4f}) did NOT surpass best ({best_score:.4f}) in {eval_metric.upper()} (diff: -{diff:.4f})."

        # Log entry to comparison history
        history_entry = summary_metrics.copy()
        history_entry['is_best'] = is_best
        history_entry['status'] = 'UPDATED_CHAMPION' if is_best else 'REJECTED'
        
        self._append_to_history(history_entry)

        if is_best:
            # Update best_model_results.csv
            best_df = pd.DataFrame([summary_metrics])
            best_df.to_csv(self.best_file, index=False)
            print(f"\n=======================================================")
            print(f"[CHAMPION] {reason}")
            print(f"Saved updated best model metrics to: {self.best_file}")
            print(f"=======================================================\n")
        else:
            print(f"\n-------------------------------------------------------")
            print(f"[INFO] {reason}")
            print(f"Best model remains: {best_record.get('config_name', 'Unknown')} ({eval_metric.upper()} = {best_score:.4f})")
            print(f"-------------------------------------------------------\n")
            if rollback_on_failure:
                execute_git_rollback()

        return {
            'is_best': is_best,
            'current_score': current_score,
            'best_score': best_score,
            'primary_metric': eval_metric,
            'summary_metrics': summary_metrics,
            'message': reason
        }

    def _append_to_history(self, entry_dict):
        """Append record to full model run history CSV."""
        df_entry = pd.DataFrame([entry_dict])
        if not os.path.exists(self.history_file):
            df_entry.to_csv(self.history_file, index=False)
        else:
            df_entry.to_csv(self.history_file, mode='a', header=False, index=False)


if __name__ == '__main__':
    # Standalone unit test for BestModelTracker
    print("Testing BestModelTracker...")
    tracker = BestModelTracker(primary_metric='f1')
    
    # Dummy simulation metrics 1
    dummy_df1 = pd.DataFrame({
        'batch': range(10),
        'f1': np.linspace(0.5, 0.7, 10),
        'roc_auc': np.linspace(0.6, 0.8, 10),
        'pr_auc': np.linspace(0.4, 0.6, 10),
        'accuracy': 0.9,
        'precision': 0.7,
        'recall': 0.7,
        'kappa': 0.5
    })
    
    res1 = tracker.evaluate_and_update("Test Config 1", dummy_df1)
    print("Run 1 Result:", res1['message'])
    
    # Dummy simulation metrics 2 (Better)
    dummy_df2 = pd.DataFrame({
        'batch': range(10),
        'f1': np.linspace(0.75, 0.85, 10),
        'roc_auc': np.linspace(0.8, 0.9, 10),
        'pr_auc': np.linspace(0.6, 0.8, 10),
        'accuracy': 0.95,
        'precision': 0.85,
        'recall': 0.85,
        'kappa': 0.75
    })
    
    res2 = tracker.evaluate_and_update("Test Config 2 (Superior)", dummy_df2)
    print("Run 2 Result:", res2['message'])
    
    # Dummy simulation metrics 3 (Worse)
    dummy_df3 = pd.DataFrame({
        'batch': range(10),
        'f1': np.linspace(0.4, 0.5, 10),
        'roc_auc': np.linspace(0.5, 0.6, 10),
        'pr_auc': np.linspace(0.3, 0.4, 10),
        'accuracy': 0.8,
        'precision': 0.5,
        'recall': 0.5,
        'kappa': 0.3
    })
    
    res3 = tracker.evaluate_and_update("Test Config 3 (Inferior)", dummy_df3)
    print("Run 3 Result:", res3['message'])
