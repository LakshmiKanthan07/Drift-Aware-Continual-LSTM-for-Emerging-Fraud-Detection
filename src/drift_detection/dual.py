from .adwin import ADWINDriftDetector
from .window import WindowDriftDetector

class DualDriftDetector:
    def __init__(self, adwin_delta=0.002, window_size=200, threshold_drop=0.15, cooldown=50):
        """
        Combines ADWIN (good for gradual drift, tracking F1) and 
        Window (good for abrupt drift, tracking Recall).
        If either triggers, a drift is flagged and both reset.
        """
        self.adwin = ADWINDriftDetector(delta=adwin_delta)
        self.window = WindowDriftDetector(window_size=window_size, threshold_drop=threshold_drop, cooldown=cooldown)
    
    def update(self, recall_val, f1_val):
        """
        Takes both recall and F1.
        Returns True if drift detected by either algorithm.
        """
        drift_adwin = self.adwin.update(f1_val)
        drift_window = self.window.update(recall_val)
        
        if drift_adwin or drift_window:
            return True
        return False
        
    def reset(self):
        self.adwin.reset()
        self.window.reset()
