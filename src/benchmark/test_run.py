from src.benchmark import BenchmarkRunner, StopCondition
from test_sgd import SimpleSGD


def main():
    dataset = "digits"  # Options: digits, cifar10, heart_disease, wine_quality
    stop_cond = StopCondition(max_epochs=10)

    print(f"Initializing Benchmark for dataset: {dataset}...")
    runner = BenchmarkRunner(
        dataset_name=dataset, stop_condition=stop_cond, batch_size=32, random_seed=42
    )

    print("Starting optimization run...")
    result = runner.run(
        optimizer_class=SimpleSGD, optimizer_name="My_Simple_SGD", lr=0.01
    )

    print("\n" + "=" * 30)
    print("BENCHMARK RESULTS")
    print("=" * 30)
    print(f"Optimizer:      {result.optimizer_name}")
    print(f"Final Accuracy: {result.final_accuracy:.2f}%")
    print(f"Final Loss:     {result.final_loss:.4f}")
    print(f"Total Steps:    {result.total_steps}")
    print(f"Stop Reason:    {result.stop_reason.name}")
    print("=" * 30)


if __name__ == "__main__":
    main()
