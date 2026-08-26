import collections
import numpy as np


class WindowDriftDetector:
    def __init__(self, window_size=100, threshold_drop=0.08):
        """
        Detects drift by monitoring per-batch recall over a sliding window.
        Recall is a fraud-sensitive metric: even a small drop signals that
        the model is missing fraudulent transactions.

        If the current window's mean recall drops by `threshold_drop` compared
        to the historical maximum, a drift is flagged.

        Args:
            window_size:    Number of recent batches to keep in the window.
                            100 batches is enough to get a stable mean while
                            still reacting quickly to pattern shifts.
            threshold_drop: Recall drop (absolute) needed to trigger a drift.
                            0.08 is sensitive enough for recall without too many
                            false positives.
        """
        self.window_size = window_size
        self.threshold_drop = threshold_drop
        self.window = collections.deque(maxlen=window_size)

        self.historical_max = 0.0
        self.drift_detected = False

    def update(self, batch_recall: float) -> bool:
        """
        batch_recall: recall score for the current batch (0.0 – 1.0).
        Returns True if drift is detected.
        """
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
        """Resets after a drift is handled, keeping historical_max for continuity."""
        self.window.clear()
        # Keep historical_max so we don't immediately re-trigger on the next window
        self.drift_detected = False
