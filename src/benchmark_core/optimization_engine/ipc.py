import importlib.util
import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection

import torch
from torch import Tensor

from benchmark_core.optimization_engine.debugging import VerbosePipe, VerboseSharedDict
from benchmark_core.optimization_engine.evaluator import ModelEvaluator
from benchmark_core.optimization_engine.models import Operation, OperationResponse
from benchmark_core.optimization_engine.optimizer_protocols import BenchmarkableOptimizer
from benchmark_core.optimization_engine.remote_evaluator import RemoteEvaluator


def load_custom_optimizer(path: str):
    spec = importlib.util.spec_from_file_location("custom_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "step")
            and name not in ("BenchmarkOptimizer", "NumpyBenchmarkOptimizer", "CupyBenchmarkOptimizer")
        ):
            return obj
    raise ValueError(f"No optimizer class found in {path}")


def optimizer_worker(
    pipe: Connection,
    shared: dict,
    optimizer_class: type[BenchmarkableOptimizer] | str,
    optimizer_config: dict,
):
    try:
        if isinstance(optimizer_class, str):
            optimizer_class = load_custom_optimizer(optimizer_class)
        output_type = optimizer_class.get_output_type()
        evaluator = RemoteEvaluator(pipe, shared)
        evaluator.set_output_type(output_type)
        initial_params = evaluator._from_tensor(shared["params"].clone())
        optimizer = optimizer_class(initial_params, **optimizer_config)

        while True:
            converged = optimizer.step(evaluator)

            if converged:
                shared["params"].copy_(evaluator._to_tensor(optimizer.params))
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
        evaluator = None
        optimizer = None
        shared.clear()
        pipe.close()


def create_ipc_worker(
    initial_params: Tensor,
    initial_batch_size: int,
    param_count: int,
    optimizer_class: type[BenchmarkableOptimizer] | str,
    optimizer_config: dict,
    verbose_ipc: bool = False,
):
    raw_shared = {
        "params": initial_params.share_memory_(),
        "grad": torch.zeros_like(initial_params).share_memory_(),
        "loss": torch.zeros(1, dtype=initial_params.dtype).share_memory_(),
        "batch_size": torch.tensor([initial_batch_size], dtype=torch.int).share_memory_(),
        "param_count": torch.tensor([param_count], dtype=torch.int).share_memory_(),
    }
    shared = VerboseSharedDict(raw_shared) if verbose_ipc else raw_shared

    ctx = mp.get_context("spawn")
    raw_parent_opt, raw_child_opt = ctx.Pipe(duplex=True)
    parent_opt = VerbosePipe(raw_parent_opt) if verbose_ipc else raw_parent_opt
    child_opt = VerbosePipe(raw_child_opt) if verbose_ipc else raw_child_opt

    opt_process = ctx.Process(
        target=optimizer_worker,
        args=(child_opt, shared, optimizer_class, optimizer_config),
        name="optimizer",
    )
    return shared, parent_opt, child_opt, opt_process


def handle_ipc_operation(msg, parent_opt: Connection, evaluator: ModelEvaluator, shared):
    match msg:
        case Operation.GRAD:
            gradient = evaluator.grad(shared["params"])
            shared["grad"].copy_(gradient.detach())
        case Operation.LOSS:
            loss_val = evaluator.evaluate(shared["params"])
            shared["loss"][0] = loss_val
        case Operation.LOSS_GRAD:
            loss_val, gradient = evaluator.evaluate_with_grad(shared["params"])
            shared["loss"][0] = loss_val
            shared["grad"].copy_(gradient.detach())
        case _:
            raise RuntimeError(f"Unexpected optimizer reply: {msg}")
    parent_opt.send(OperationResponse.OK)

    return parent_opt.recv()


def shutdown_ipc_worker(parent_opt, opt_process: mp.Process):
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
