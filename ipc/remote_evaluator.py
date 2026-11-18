from multiprocessing.connection import Connection

from torch import Tensor

from models import Operation, OperationResponse


class RemoteEvaluator:
    def __init__(self, pipe: Connection, shared: dict[str, Tensor]):
        self.pipe = pipe
        self.shared = shared

    def _request(self, operation_code: Operation, params: Tensor) -> None:
        self.shared["params"].copy_(params)
        self.pipe.send(operation_code)
        response = self.pipe.recv()

        match response:
            case OperationResponse.STOP:
                raise EOFError()
            case OperationResponse.OK:
                pass
            case _:
                raise RuntimeError(f"Evaluation failed: {response}")

    @property
    def batch_size(self) -> int:
        return int(self.shared["batch_size"].item())

    @property
    def param_count(self) -> int:
        return int(self.shared["param_count"].item())

    def grad(self, params: Tensor) -> Tensor:
        self._request(Operation.GRAD, params)

        return self.shared["grad"].clone()

    def evaluate(self, params: Tensor) -> float:
        self._request(Operation.LOSS, params)

        return self.shared["loss"].item()

    def evaluate_with_grad(self, params: Tensor) -> tuple[float, Tensor]:
        self._request(Operation.LOSS_GRAD, params)

        return (
            self.shared["loss"].item(),
            self.shared["grad"].clone(),
        )
