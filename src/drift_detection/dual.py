from .adwin import ADWINDriftDetector
from .window import WindowDriftDetector


class DualDriftDetector:
    def __init__(self, adwin_delta=0.002, window_size=60, threshold_drop=0.15, cooldown=30):
        """
        Upgraded Dual Drift Detector using 3 signals:
          - ADWIN on F2-score (gradual distribution drift)
          - ADWIN on PR-AUC (precision-recall surface degradation)
          - Window on Recall (abrupt drop in fraud-catch rate)

        Also tracks drift severity (how large the drop is) so EWC lambda
        can be dynamically scaled in response.
        """
        self.adwin_f2    = ADWINDriftDetector(delta=adwin_delta)
        self.adwin_prauc = ADWINDriftDetector(delta=adwin_delta)
        self.window      = WindowDriftDetector(window_size=window_size,
                                               min_window=20,
                                               threshold_drop=threshold_drop,
                                               cooldown=cooldown)
        self.last_drift_severity = 0.0

    def update(self, recall_val: float, f2_val: float, pr_auc_val: float = 0.0) -> bool:
        """
        Returns True if drift detected by any of the three detectors.
        Also stores drift severity for downstream dynamic lambda scaling.
        """
        drift_f2    = self.adwin_f2.update(f2_val)
        drift_prauc = self.adwin_prauc.update(pr_auc_val)
        drift_win   = self.window.update(recall_val)

        if drift_f2 or drift_prauc or drift_win:
            self.last_drift_severity = max(
                1.0 - recall_val,
                1.0 - f2_val,
            )
            return True

        self.last_drift_severity = 0.0
        return False

    def reset(self):
        self.adwin_f2.reset()
        self.adwin_prauc.reset()
        self.window.reset()
