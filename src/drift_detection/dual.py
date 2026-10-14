from .adwin import ADWINDriftDetector
from .window import WindowDriftDetector


class DualDriftDetector:
    def __init__(self, adwin_delta=0.002, window_size=50, threshold_drop=0.20, cooldown=50):
        """
        Dual Drift Detector combining two complementary signals:
          - WindowDriftDetector on Recall (abrupt drop in fraud-catch rate)
          - ADWIN on F2-score (gradual distribution drift)

        Tracks drift severity for downstream dynamic adaptation.
        """
        self.adwin_f2 = ADWINDriftDetector(delta=adwin_delta)
        self.window   = WindowDriftDetector(window_size=window_size,
                                            min_window=25,
                                            threshold_drop=threshold_drop,
                                            cooldown=cooldown)
        self.last_drift_severity = 0.0

    def update(self, recall_val: float, f2_val: float, pr_auc_val: float = 0.0) -> bool:
        """
        Returns True if a genuine drift is detected by either the windowed recall
        drop (abrupt drift) or ADWIN on F2 (distribution drift).
        """
        drift_win = self.window.update(recall_val)
        drift_f2  = self.adwin_f2.update(f2_val)

        if drift_win or drift_f2:
            self.last_drift_severity = max(
                1.0 - recall_val,
                1.0 - f2_val,
            )
            return True

        self.last_drift_severity = 0.0
        return False

    def reset(self):
        self.adwin_f2.reset()
        self.window.reset()
