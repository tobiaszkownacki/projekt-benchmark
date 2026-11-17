from multiprocessing.connection import Connection

from torch import Tensor

class RemoteEvaluator:
    def __init__(self, pipe: Connection, shared: dict[str, Tensor]):
        self.pipe = pipe
        self.shared = shared

    def _request(self, operation_code: int, params: Tensor) -> None:
        self.shared["params"].copy_(params)

        self.pipe.send(operation_code)
        response = self.pipe.recv()

        if response != "OK":
            raise RuntimeError(f"Evaluation failed: {response}")

    def grad(self, params: Tensor) -> Tensor:
        self._request(0, params)
        return self.shared["grad"].clone()

    def evaluate(self, params: Tensor) -> float:
        self._request(1, params)
        return self.shared["loss"].item()

    def evaluate_with_grad(self, params: Tensor) -> tuple[float, Tensor]:
        self._request(2, params)

        return (
            self.shared["loss"].item(),
            self.shared["grad"].clone(),
        )
