import collections
import numpy as np


class WindowDriftDetector:
    def __init__(self, window_size=200, threshold_drop=0.15, cooldown=50):
        """
        Detects drift by monitoring per-batch recall over a sliding window.

        Improvements over v1:
        - Larger window (200 batches) → more stable mean recall estimate,
          preventing false alarms from natural batch-to-batch variance.
        - Higher threshold_drop (0.15) → only genuine large drops trigger drift,
          not normal noise in sparse-fraud batches.
        - Cooldown period (50 batches) → after a drift event, detection is
          suspended for 50 batches so the model can stabilize before the
          next alert. This prevents the 'chain-trigger' problem where one
          drift spawns dozens of consecutive re-adaptations.

        Args:
            window_size:     Number of recent batches in the sliding window.
            threshold_drop:  Absolute recall drop from historical max to flag.
            cooldown:        Batches to wait after a drift before re-enabling detection.
        """
        self.window_size    = window_size
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

        # Need a full window before making a decision
        if len(self.window) < self.window_size:
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
