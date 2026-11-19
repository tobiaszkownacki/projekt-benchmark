from multiprocessing.connection import Connection

from torch import Tensor

from .evaluator_dtos import PyTorchTensorEvaluatorDto
from .evaluator_dtos.evaluator_dto import T
from .models import Operation, OperationResponse


class RemoteEvaluator:
    def __init__(self, pipe: Connection, shared: dict[str, Tensor]):
        self.pipe = pipe
        self.shared = shared
        self.type = PyTorchTensorEvaluatorDto

    def set_output_type(self, output_type: type[T]):
        self.type = output_type

    def _to_tensor(self, params: object) -> Tensor:
        return self.type(params).to(PyTorchTensorEvaluatorDto, device=self.shared["params"].device).data()

    def _from_tensor(self, tensor: Tensor) -> object:
        return PyTorchTensorEvaluatorDto(tensor).to(self.type).data()

    def _request(self, operation_code: Operation, params: object) -> None:
        self.shared["params"].copy_(self._to_tensor(params))
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

    def grad(self, params: object) -> object:
        self._request(Operation.GRAD, params)

        return self._from_tensor(self.shared["grad"].clone())

    def evaluate(self, params: object) -> float:
        self._request(Operation.LOSS, params)

        return self.shared["loss"].item()

    def evaluate_with_grad(self, params: object) -> tuple[float, object]:
        self._request(Operation.LOSS_GRAD, params)

        return (
            self.shared["loss"].item(),
            self._from_tensor(self.shared["grad"].clone()),
        )
