import os
import time
import traceback

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

import multiprocessing as mp
from multiprocessing.connection import Connection

from model_evaluator import ModelEvaluator
from adam_optimizer import AdamOptimizer

from enum import Enum

from remote_evaluator import RemoteEvaluator

from models import Operation, OperationResponse


class VerbosePipe:
    def __init__(self, conn: Connection):
        self._conn = conn

    def _tag(self) -> str:
        return f"[{mp.current_process().name} pid={os.getpid()}]"

    def send(self, obj):
        print(f"{self._tag()} PIPE send -> {obj}", flush=True)
        return self._conn.send(obj)

    def recv(self):
        obj = self._conn.recv()
        print(f"{self._tag()} PIPE recv <- {obj}", flush=True)
        return obj

    def close(self):
        return self._conn.close()


class VerboseTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data: torch.Tensor, key: str):
        instance = torch.Tensor._make_subclass(cls, data, data.requires_grad)
        instance._key = key
        return instance

    def _tag(self) -> str:
        return f"[{mp.current_process().name} pid={os.getpid()}]"

    def copy_(self, src, non_blocking: bool = False):
        res = super().copy_(src, non_blocking=non_blocking)
        print(
            f"{self._tag()} SHARED[{self._key!r}].copy_ -> {self.tolist()}",
            flush=True,
        )
        return res

    def __setitem__(self, idx, val):
        super().__setitem__(idx, val)
        print(
            f"{self._tag()} SHARED[{self._key!r}][{idx}] = {val} -> {self.tolist()}",
            flush=True,
        )


class VerboseSharedDict:
    def __init__(self, data: dict):
        self._data = data

    def _tag(self) -> str:
        return f"[{mp.current_process().name} pid={os.getpid()}]"

    def __getitem__(self, key: str):
        val = self._data[key]
        if isinstance(val, torch.Tensor):
            return VerboseTensor(val, key)
        return val

    def __setitem__(self, key: str, value):
        if key in self._data and isinstance(self._data[key], torch.Tensor):
            if isinstance(value, torch.Tensor):
                self._data[key].copy_(value)
            else:
                self._data[key][0] = value
        else:
            self._data[key] = value
        print(f"{self._tag()} SHARED[{key!r}] = {value}", flush=True)


def model_factory() -> nn.Module:
    return nn.Linear(1, 1)


def optimizer_worker(pipe: Connection, shared: dict, optimizer_config: dict):
    try:
        optimizer = AdamOptimizer(shared["params"].clone(), **optimizer_config)
        evaluator = RemoteEvaluator(pipe, shared)

        while True:
            converged = optimizer.step(evaluator)

            if converged:
                pipe.send(Operation.CONVERGED)
                break
    except EOFError:
        pass
    except Exception:
        try:
            pipe.send(("ERROR", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        pipe.close()


def main():
    device = torch.device("cpu")
    batch_size = 100
    max_epochs = 3
    optimizer_config = {"lr": 0.05}

    torch.manual_seed(42)

    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = 2 * x + 1

    dataset = TensorDataset(x, y)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = model_factory().to(device)
    criterion = nn.MSELoss()

    initial_params = parameters_to_vector(model.parameters()).detach().clone()

    gradient_count = 0
    database_reaches = 0
    step_count = 0
    epoch_count = 0
    avg_loss = float("nan")

    def metrics_callback(db_inc: int, grad_inc: int):
        nonlocal database_reaches, gradient_count
        database_reaches += db_inc
        gradient_count += grad_inc

    evaluator = ModelEvaluator(
        model=model,
        criterion=criterion,
        device=device,
        metrics_callback=metrics_callback,
    )

    shared = VerboseSharedDict(
        {
            "params": initial_params.share_memory_(),
            "grad": torch.zeros_like(initial_params).share_memory_(),
            "loss": torch.zeros(1, dtype=initial_params.dtype).share_memory_(),
            "batch_size": torch.tensor([min(batch_size, len(dataset))], dtype=torch.int).share_memory_(),
            "param_count": torch.tensor([evaluator.param_count], dtype=torch.int).share_memory_(),
        }
    )

    ctx = mp.get_context("spawn")

    raw_parent_opt, raw_child_opt = ctx.Pipe(duplex=True)
    parent_opt = VerbosePipe(raw_parent_opt)
    child_opt = VerbosePipe(raw_child_opt)

    opt_process = ctx.Process(
        target=optimizer_worker,
        args=(child_opt, shared, optimizer_config),
        name="optimizer",
    )

    start_time = time.time()

    opt_process.start()
    child_opt.close()

    try:
        converged = False
        msg = parent_opt.recv()

        for epoch in range(max_epochs):
            epoch_losses = []

            for inputs, targets in train_loader:
                evaluator.set_batch(inputs, targets)

                if int(shared["batch_size"].item()) != evaluator.batch_size:
                    shared["batch_size"][0] = evaluator.batch_size

                match msg:
                    case Operation.GRAD:
                        gradient = evaluator.grad(shared["params"])
                        shared["grad"].copy_(gradient)
                    case Operation.LOSS:
                        loss_val = evaluator.evaluate(shared["params"])
                        shared["loss"][0] = loss_val
                    case Operation.LOSS_GRAD:
                        loss_val, gradient = evaluator.evaluate_with_grad(shared["params"])
                        shared["loss"][0] = loss_val
                        shared["grad"].copy_(gradient)
                    case _:
                        raise RuntimeError(f"Unexpected optimizer reply: {msg}")

                parent_opt.send(OperationResponse.OK)
                msg = parent_opt.recv()

                if msg == Operation.CONVERGED:
                    converged = True

                step_count += 1

                vector_to_parameters(
                    shared["params"],
                    model.parameters(),
                )

                model.eval()

                with torch.no_grad():
                    outputs = model(inputs.to(device))
                    loss = criterion(
                        outputs,
                        targets.to(device),
                    )

                epoch_losses.append(loss.item())

                if converged:
                    break

            epoch_count += 1

            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)

                # print(
                #     f"Epoch {epoch_count:3d} | "
                #     f"Loss: {avg_loss:.8f} | "
                #     f"w: {model.weight.item():+.4f} | "
                #     f"b: {model.bias.item():+.4f} | "
                #     f"Grads: {gradient_count} | "
                #     f"DB: {database_reaches}",
                #     flush=True,
                # )

            if converged:
                break

        elapsed = time.time() - start_time

        print("\n--- FINAL RESULTS ---")
        print("Target: y = 2x + 1")
        print(f"Learned: y = {model.weight.item():.6f}x + {model.bias.item():.6f}")
        print(f"Final loss: {avg_loss:.8f}")
        print(f"Epochs: {epoch_count}; steps: {step_count}")
        print(f"Gradient computations: {gradient_count}")
        print(f"Database reaches: {database_reaches}")
        print(f"Time: {elapsed:.3f}s")

    finally:
        if opt_process.is_alive():
            try:
                parent_opt.send(OperationResponse.STOP)
            except (BrokenPipeError, EOFError, OSError):
                pass

        parent_opt.close()

        opt_process.join(timeout=5)

        if opt_process.is_alive():
            opt_process.terminate()
            opt_process.join()


if __name__ == "__main__":
    main()
