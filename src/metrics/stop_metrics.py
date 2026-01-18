class MetricsTracker:
    def __init__(self, max_gradients=None, max_database_reaches=None, max_epochs=None):
        self.max_gradients = max_gradients
        self.max_database_reaches = max_database_reaches
        self.max_epochs = max_epochs

        self.gradient_count = 0
        self.database_reach_count = 0
        self.epoch_count = 0

        self.stop_reason = "NONE"

    def record_gradients(self, count=1):
        self.gradient_count += count
        
        if self.max_gradients is not None and self.gradient_count >= self.max_gradients:
            self.stop_reason = "GRADIENT_LIMIT"
            return True
        return False

    def record_database_reaches(self, count):
        self.database_reach_count += count
        
        if self.max_database_reaches is not None and self.database_reach_count >= self.max_database_reaches:
            self.stop_reason = "DATABASE_REACH_LIMIT"
            return True
        return False

    def record_epoch(self):
        self.epoch_count += 1
        
        if self.max_epochs is not None and self.epoch_count >= self.max_epochs:
            self.stop_reason = "EPOCH_LIMIT"
            return True
        return False

    def signal_optimizer_stop(self):
        if self.stop_reason == "NONE":
            self.stop_reason = "OPTIMIZER_SIGNAL"

    def should_stop(self):
        return self.stop_reason != "NONE"

    def get_summary(self):
        return {
            "gradient_count": self.gradient_count,
            "database_reach_count": self.database_reach_count,
            "epoch_count": self.epoch_count,
            "stop_reason": self.stop_reason,
        }