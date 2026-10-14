import collections
import numpy as np


class WindowDriftDetector:
    def __init__(self, window_size=50, min_window=25, threshold_drop=0.20, cooldown=50):
        """
        Detects drift by monitoring per-batch recall over a sliding window.

        Args:
            window_size:     Number of recent batches in the sliding window.
            min_window:      Minimum samples in window before testing for drift.
            threshold_drop:  Absolute recall drop from historical max to flag.
            cooldown:        Batches to wait after a drift before re-enabling detection.
        """
        self.window_size    = window_size
        self.min_window     = min_window
        self.threshold_drop = threshold_drop
        self.cooldown       = cooldown

        self.window         = collections.deque(maxlen=window_size)
        self.historical_max = 0.0
        self.drift_detected = False
        self._cooldown_left = 0          # Batches remaining in cooldown

    def update(self, batch_recall: float) -> bool:
        """
        batch_recall: recall score for the current batch (0.0 – 1.0).
        Returns True if a genuine drift is detected.
        """
        # Respect cooldown — do not re-trigger while adapting
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self.window.append(batch_recall)
            return False

        self.window.append(batch_recall)

        # Need at least min_window before making a decision
        if len(self.window) < self.min_window:
            return False

        current_recall = float(np.mean(self.window))

        # Update historical maximum
        if current_recall > self.historical_max:
            self.historical_max = current_recall

        # Check for significant drop from historical best
        if self.historical_max > 0 and (self.historical_max - current_recall) > self.threshold_drop:
            self.drift_detected = True
            return True

        self.drift_detected = False
        return False

    def reset(self):
        """Resets after a drift is handled; starts cooldown period."""
        self.window.clear()
        self.drift_detected  = False
        self._cooldown_left  = self.cooldown   # Engage cooldown
        # Keep historical_max so we don't immediately re-trigger
