import csv
import os
from src.config import Config, CMAOptimizerConfig, GradientOptimizerConfig, LBFGSOptimizerConfig, BaseOptimizerConfig

class Log:
    def __init__(self, output_file: str = "train_log.csv"):
        self.number_of_samples: int = 0
        self.number_of_mini_batches: int = 0
        self.records = []
        
        logs_dir = os.path.join("reports", "logs")
        self.output_file = os.path.join(logs_dir, output_file)

        with open(self.output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["samples", "mini_batches", "train_loss"])
    
    def increment_number_of_samples(self, inc: int):
        self.number_of_samples += inc
    
    def increment_mini_batches(self, inc: int):
        self.number_of_mini_batches += inc

    def log(self, train_loss: float, save_interval: int):
        self.records.append((self.number_of_samples, self.number_of_mini_batches, train_loss))
        if self.number_of_mini_batches % save_interval == 0:
            self.save_to_csv()

    def save_to_csv(self):
        with open(self.output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.records)
        self.records.clear()
