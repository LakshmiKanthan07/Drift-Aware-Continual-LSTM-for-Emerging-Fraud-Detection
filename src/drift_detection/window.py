import collections
import numpy as np

class WindowDriftDetector:
    def __init__(self, window_size=500, threshold_drop=0.15):
        """
        Detects drift by monitoring the accuracy/error over a sliding window.
        If the current window's performance drops by `threshold_drop` compared
        to the historical maximum, it flags a drift.
        """
        self.window_size = window_size
        self.threshold_drop = threshold_drop
        self.window = collections.deque(maxlen=window_size)
        
        self.historical_max = 0.0
        self.drift_detected = False
        self.warmup = True

    def update(self, is_correct):
        """
        is_correct: 1 if prediction was correct, 0 otherwise.
        """
        self.window.append(is_correct)
        
        if len(self.window) < self.window_size:
            self.warmup = True
            return False
            
        self.warmup = False
        current_acc = np.mean(self.window)
        
        # Update historical max
        if current_acc > self.historical_max:
            self.historical_max = current_acc
            
        # Check for drift
        if (self.historical_max - current_acc) > self.threshold_drop:
            self.drift_detected = True
            return True
            
        self.drift_detected = False
        return False

    def reset(self):
        """Resets after a drift is handled."""
        self.window.clear()
        self.historical_max = 0.0
        self.drift_detected = False
        self.warmup = True
