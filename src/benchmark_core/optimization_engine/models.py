from enum import Enum


class Operation(Enum):
    GRAD = 0
    LOSS = 1
    LOSS_GRAD = 2
    CONVERGED = 3
    ERROR = 4


class OperationResponse(Enum):
    OK = 0
    STOP = 1
