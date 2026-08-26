from river.drift import ADWIN

class ADWINDriftDetector:
    def __init__(self, delta=0.002):
        """
        Wrapper around River's ADWIN algorithm.
        ADWIN tracks a stream of real numbers and detects changes in their mean.
        We can feed it the model's error (0 for correct, 1 for incorrect).
        """
        self.detector = ADWIN(delta=delta)
        self.drift_detected = False

    def update(self, error):
        """
        Updates the detector with a new error observation.
        Returns True if drift is detected.
        """
        self.detector.update(error)
        self.drift_detected = self.detector.drift_detected
        return self.drift_detected

    def reset(self):
        """Resets the detector after a drift."""
        self.detector = ADWIN(delta=self.detector.delta)
        self.drift_detected = False
