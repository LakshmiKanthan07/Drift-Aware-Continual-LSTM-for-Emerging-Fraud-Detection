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

        # Each entry: (x_dict, y, priority)
        self.normal_buffer = []
        self.fraud_buffer = []

    def add(self, x_dict, y, priority=1.0):
        """
        Add a single sample with an associated priority score.

        Args:
            x_dict (dict): Feature dict for the sample.
            y (np.ndarray): Label array.
            priority (float): Higher → more likely to be replayed. Typically |y - y_hat|.
        """
        entry = (x_dict, y, float(priority) ** self.alpha)
        is_fraud = (y[0] == 1.0)
        buffer = self.fraud_buffer if is_fraud else self.normal_buffer

        if len(buffer) < self.half_capacity:
            buffer.append(entry)
        else:
            # Replace lowest-priority existing sample with probability
            # proportional to new sample's priority
            idx = random.randint(0, self.half_capacity - 1)
            # Replace if new priority >= existing, else replace randomly with 50% prob
            if entry[2] >= buffer[idx][2] or random.random() < 0.5:
                buffer[idx] = entry

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

        def priority_sample(buf, k):
            if not buf:
                return []
            weights = np.array([entry[2] for entry in buf], dtype=np.float64)
            weights = weights / weights.sum()
            k = min(k, len(buf))
            indices = np.random.choice(len(buf), size=k, replace=False, p=weights)
            return [buf[i] for i in indices]

        fraud_samples = priority_sample(self.fraud_buffer, n_fraud)
        normal_samples = priority_sample(self.normal_buffer, n_normal)

        # If not enough fraud, pad with extra normals
        shortfall = n_fraud - len(fraud_samples)
        if shortfall > 0:
            extra = priority_sample(self.normal_buffer, shortfall + n_normal)
            normal_samples = extra

        batch = normal_samples + fraud_samples
        random.shuffle(batch)

        if not batch:
            return {}, np.array([])

        y_batch = np.array([item[1] for item in batch])
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
