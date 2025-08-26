import argparse
import random


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
        "--scheduler",
        type=str,
        choices=["none", "steplr", "exponentiallr", "reduceonplateau", "cosineannealinglr"],
        default="none",
        help="Learning rate scheduler to use (default: none)",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size for StepLR scheduler (default: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma for StepLR/ExponentialLR/ReduceLROnPlateau scheduler (default: 0.5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=1,
        help="Patience for ReduceLROnPlateau scheduler (default: 5)",
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
        default=500,
        help="Number of times to reach the dataset during non-gradient training (default: 500)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model (default: 10)",
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
        default=random.randrange(2**32),
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Every how many mini-batches logs are saved to file",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="If set, saves the trained weights into the /weights directory"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="If set, loads the weights into the model from a file with the given name. Weights are loaded from the /weights directory",
    )
    parser.add_argument(
        "--init_xavier",
        action="store_true",
        help="if xavier_uniform initialization"
    )

    args = parser.parse_args(arguments[1:])
    return args
