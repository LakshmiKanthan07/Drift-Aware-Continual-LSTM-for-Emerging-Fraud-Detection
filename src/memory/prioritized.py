import random
import numpy as np


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for continual fraud detection.

    Maintains separate queues for normal and fraud samples with priority scoring.
    Priority = prediction_error (|y_true - y_pred|) so high-error (hard) samples
    are replayed more often, giving the model more training signal on difficult cases.

    Attributes:
        capacity (int): Maximum total buffer size.
        fraud_ratio (float): Fraction of sampled batch reserved for fraud samples.
        alpha (float): Priority exponent. alpha=0 → uniform; alpha=1 → fully prioritized.
    """

    def __init__(self, capacity=5000, fraud_ratio=0.5, alpha=0.6):
        self.capacity = capacity
        self.fraud_ratio = fraud_ratio
        self.alpha = alpha
        self.half_capacity = capacity // 2

        # Each entry: (x_dict, y)
        self.normal_buffer = []
        self.fraud_buffer = []
        
        # Priorities for fast sampling
        self.normal_priorities = np.zeros(self.half_capacity, dtype=np.float32)
        self.fraud_priorities = np.zeros(self.half_capacity, dtype=np.float32)

    def add(self, x_dict, y, priority=1.0):
        """
        Add a single sample with an associated priority score.

        Args:
            x_dict (dict): Feature dict for the sample.
            y (np.ndarray): Label array.
            priority (float): Higher → more likely to be replayed. Typically |y - y_hat|.
        """
        priority_val = float(priority) ** self.alpha
        entry = (x_dict, y)
        is_fraud = (y[0] == 1.0)
        
        if is_fraud:
            buffer = self.fraud_buffer
            priorities = self.fraud_priorities
        else:
            buffer = self.normal_buffer
            priorities = self.normal_priorities

        curr_len = len(buffer)
        if curr_len < self.half_capacity:
            buffer.append(entry)
            priorities[curr_len] = priority_val
        else:
            # Tournament selection: find lowest priority among 4 random entries to evict
            candidates = [random.randint(0, self.half_capacity - 1) for _ in range(4)]
            min_idx = min(candidates, key=lambda i: priorities[i])
            if priority_val >= priorities[min_idx]:
                buffer[min_idx] = entry
                priorities[min_idx] = priority_val

    def sample(self, batch_size):
        """
        Sample a batch weighted by priority scores.
        Guarantees fraud_ratio fraction of the batch is fraud samples.

        Returns:
            X_batch (dict): Stacked feature arrays.
            y_batch (np.ndarray): Labels.
        """
        n_fraud = int(batch_size * self.fraud_ratio)
        n_normal = batch_size - n_fraud

        def priority_sample(buf, priorities_arr, k):
            n = len(buf)
            if n == 0:
                return []
            
            p = priorities_arr[:n]
            p_sum = p.sum()
            
            if p_sum == 0:
                indices = np.random.choice(n, size=min(k, n), replace=False)
            else:
                p = p / p_sum
                indices = np.random.choice(n, size=min(k, n), replace=False, p=p)
                
            return [buf[i] for i in indices]

        fraud_samples = priority_sample(self.fraud_buffer, self.fraud_priorities, n_fraud)
        normal_samples = priority_sample(self.normal_buffer, self.normal_priorities, n_normal)

        # If not enough fraud, pad with extra normals
        shortfall = n_fraud - len(fraud_samples)
        if shortfall > 0:
            extra = priority_sample(self.normal_buffer, self.normal_priorities, shortfall + n_normal)
            normal_samples = extra

        batch = normal_samples + fraud_samples
        random.shuffle(batch)

        if not batch:
            return {}, np.array([])

        y_batch = np.array([item[1] for item in batch]).reshape(-1, 1)
        keys = batch[0][0].keys()
        X_batch = {k: np.array([item[0][k] for item in batch]) for k in keys}

        return X_batch, y_batch

    def update_priorities(self, x_dicts, y_trues, y_preds):
        """
        Re-add samples with updated priorities based on current prediction errors.
        Call this periodically to keep priority estimates fresh.

        Args:
            x_dicts (list[dict]): Feature dicts.
            y_trues (np.ndarray): True labels.
            y_preds (np.ndarray): Model predicted probabilities.
        """
        for x, yt, yp in zip(x_dicts, y_trues.ravel(), y_preds.ravel()):
            priority = abs(float(yt) - float(yp)) + 1e-5  # small epsilon to prevent zero priority
            self.add(x, np.array([yt]), priority=priority)

    def clear(self):
        """Clear all samples."""
        self.normal_buffer = []
        self.fraud_buffer = []

    def partial_clear(self, clear_fraud=False):
        """Clear normal buffer; optionally clear fraud buffer too."""
        self.normal_buffer = []
        if clear_fraud:
            self.fraud_buffer = []

    @property
    def size(self):
        return len(self.normal_buffer) + len(self.fraud_buffer)
