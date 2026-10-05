import random
import numpy as np


class BalancedReplayBuffer:
    """
    Balanced Replay Buffer with recency-weighted sampling.
    
    Improvements over v1:
    - Recency weighting: newer samples are more likely to be replayed,
      preventing the model from over-anchoring on stale pre-drift fraud patterns.
    - Minimum fraud retention: fraud buffer is never randomly replaced if
      below a minimum retention threshold (no eviction when scarce).
    - Priority-aware fraud insertion: within the fraud buffer, replaces the
      oldest entry (index 0) instead of a random one, preserving recent fraud.
    """

    def __init__(self, capacity=5000, recency_weight: float = 2.0):
        """
        Args:
            capacity (int): Maximum total buffer size.
            recency_weight (float): Exponent controlling how strongly to
                favor newer samples. 1.0 = uniform, 2.0 = strong recency bias.
        """
        self.capacity       = capacity
        self.half_capacity  = capacity // 2
        self.recency_weight = recency_weight

        # Ordered lists; latest append is most recent
        self.normal_buffer: list = []
        self.fraud_buffer:  list = []

    def add(self, x_dict: dict, y) -> None:
        """
        Add a sample to the appropriate buffer.
        For normal samples: evict a random old entry when full.
        For fraud samples:  evict the oldest entry (index 0) when full,
                            preserving recent fraud patterns.
        """
        is_fraud = (float(y[0]) == 1.0)

        if is_fraud:
            if len(self.fraud_buffer) < self.half_capacity:
                self.fraud_buffer.append((x_dict, y))
            else:
                # Evict the oldest (FIFO-ish) to retain recent fraud
                self.fraud_buffer.pop(0)
                self.fraud_buffer.append((x_dict, y))
        else:
            if len(self.normal_buffer) < self.half_capacity:
                self.normal_buffer.append((x_dict, y))
            else:
                idx = random.randint(0, self.half_capacity - 1)
                self.normal_buffer[idx] = (x_dict, y)

    def _recency_sample(self, buffer: list, k: int) -> list:
        """
        Sample k items from buffer with recency weighting.
        Items closer to the end (more recent) get higher weight.
        """
        if not buffer:
            return []
        k = min(k, len(buffer))
        n = len(buffer)
        # Weight = position^recency_weight (linear index from 1..n)
        weights = np.arange(1, n + 1, dtype=np.float64) ** self.recency_weight
        weights /= weights.sum()
        indices = np.random.choice(n, size=k, replace=False, p=weights)
        return [buffer[i] for i in indices]

    def sample(self, batch_size: int):
        """
        Sample a balanced mini-batch with recency-weighted sampling.
        Returns (X_dict, y_batch).
        """
        half_batch = batch_size // 2

        normals = self._recency_sample(self.normal_buffer, half_batch)
        frauds  = self._recency_sample(self.fraud_buffer,  batch_size - len(normals))

        # Pad with normals if fraud buffer is thin
        remaining = batch_size - len(normals) - len(frauds)
        if remaining > 0:
            extra = self._recency_sample(self.normal_buffer, remaining)
            normals.extend(extra)

        batch = normals + frauds
        random.shuffle(batch)

        if not batch:
            return {}, np.array([])

        y_batch = np.array([item[1] for item in batch])
        keys    = batch[0][0].keys()
        X_batch = {k: np.array([item[0][k] for item in batch]) for k in keys}

        return X_batch, y_batch

    def clear(self) -> None:
        self.normal_buffer = []
        self.fraud_buffer  = []

    def partial_clear(self, clear_fraud: bool = False) -> None:
        """Clear normal buffer; optionally also clear fraud buffer."""
        self.normal_buffer = []
        if clear_fraud:
            self.fraud_buffer = []

    @property
    def size(self) -> int:
        return len(self.normal_buffer) + len(self.fraud_buffer)
