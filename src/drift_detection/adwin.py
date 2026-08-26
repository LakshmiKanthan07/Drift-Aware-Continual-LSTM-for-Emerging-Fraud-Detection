from river.drift import ADWIN


class ADWINDriftDetector:
    def __init__(self, delta=0.002):
        """
        Wrapper around River's ADWIN algorithm.
        Instead of per-sample error, we feed per-batch F1 scores.
        ADWIN detects when the mean of the F1 stream changes significantly,
        which is sensitive to fraud-pattern shifts even at low fraud ratios.
        """
        self.delta = delta
        self.detector = ADWIN(delta=delta)
        self.drift_detected = False

    def update(self, batch_f1: float) -> bool:
        """
        Updates the detector with the per-batch F1 score (0.0 – 1.0).
        Returns True if drift is detected.
        """
        self.detector.update(batch_f1)
        self.drift_detected = self.detector.drift_detected
        return self.drift_detected

    def reset(self):
        """Resets the detector after a drift event is handled."""
        self.detector = ADWIN(delta=self.delta)
        self.drift_detected = False
