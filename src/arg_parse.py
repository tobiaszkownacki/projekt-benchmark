import argparse


def get_args(arguments):
    parser = argparse.ArgumentParser(
        description="Train a model with specified configurations."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["cifar10", "heart_disease", "wine_quality", "digits"],
        help="Name of the dataset to use for training",
    )
    parser.add_argument(
        "optimizer",
        type=str,
        choices=["adam", "adamw", "sgd", "rmsprop", "lbfgs", "cma-es"],
        help="Optimizer to use for training the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training the model (default: 16)",
    )
    parser.add_argument(
        "--reaching_count",
        type=int,
        default=5000,
        help="Number of times to reach the dataset during non-gradient training (default: 500)",
    )
    parser.add_argument(
        "--gradient_counter_stop",
        type=int,
        default=5000,
        help="Stop training after this many gradient updates (default: 500)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args(arguments[1:])
    return args
