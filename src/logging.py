import csv
import os


class Log:
    def __init__(self, output_file: str = "train_log.csv"):
        self.number_of_samples: int = 0  # This IS database_reaches
        self.number_of_mini_batches: int = 0
        self.gradient_count: int = 0
        self.records = []
        logs_dir = os.path.join("reports", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.output_file = os.path.join(logs_dir, output_file)

        with open(self.output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["samples", "mini_batches", "gradients", "train_loss", "lr"])

    def add_number_of_samples(self, num: int):
        self.number_of_samples += num
        self._increment_mini_batches()

    def add_gradients(self, count: int = 1):
        self.gradient_count += count

    def log(self, train_loss: float, save_interval: int, learning_rate: float = "none"):
        self.records.append(
            (
                self.number_of_samples,
                self.number_of_mini_batches,
                self.gradient_count,
                train_loss,
                learning_rate,
            )
        )
        if self.number_of_mini_batches % save_interval == 0:
            self.save_to_csv()

    def save_to_csv(self):
        with open(self.output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.records)
        self.records.clear()

    def _increment_mini_batches(self):
        self.number_of_mini_batches += 1
