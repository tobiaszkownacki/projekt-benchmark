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
from remote_evaluator import RemoteEvaluator

GRAD = 0
LOSS = 1
LOSS_GRAD = 2
CONVERGED = 3

def model_factory() -> nn.Module:
    return nn.Linear(1, 1)

def optimizer_worker(pipe: Connection, shared, optimizer_config):
    try:
        optimizer = AdamOptimizer(shared["params"].clone(), **optimizer_config)
        evaluator = RemoteEvaluator(pipe, shared)
    
        while True:
            message = pipe.recv()
    
            if message == "STOP":
                break
            if message == "STEP":
                converged = optimizer.step(evaluator)
                shared["params"].copy_(optimizer.params)
                pipe.send(3 if converged else "STEP_DONE")
            else:
                raise ValueError(f"Unexpected optimizer command: {message}")
    except EOFError:
        pass
    except Exception:
        try:
            pipe.send(("ERROR", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        pipe.close()


def evaluator_worker(pipe, shared):
    torch.set_num_threads(1)

    model = model_factory().to("cpu")
    criterion = nn.MSELoss()
    evaluator = None

    def metrics_callback(db_inc: int, grad_inc: int):
        pass

    try:
        while True:
            message = pipe.recv()

            if message == "STOP":
                break

            if message == "BATCH":
                n = int(shared["n"].item())

                evaluator = ModelEvaluator(
                    model=model,
                    inputs=shared["inputs"][:n],
                    targets=shared["targets"][:n],
                    criterion=criterion,
                    device=torch.device("cpu"),
                    metrics_callback=metrics_callback
                )

                pipe.send("OK")
                continue

            if message not in (GRAD, LOSS, LOSS_GRAD):
                raise ValueError(f"Unknown evaluator opcode: {message}")

            if evaluator is None:
                raise RuntimeError(
                    "BATCH must be provided before evaluation"
                )

            if message == GRAD:
                gradient = evaluator.grad(shared["params"])
                shared["grad"].copy_(gradient)

            elif message == LOSS:
                loss = evaluator.evaluate(shared["params"])
                shared["loss"][0] = loss

            else:
                loss, gradient = evaluator.evaluate_with_grad(shared["params"])
                shared["loss"][0] = loss
                shared["grad"].copy_(gradient)

            pipe.send("OK")

    except EOFError:
        pass
    except Exception:
        try:
            pipe.send(("ERROR", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        pipe.close()

def expect_ok(message):
    if message != "OK":
        raise RuntimeError(f"Evaluator returned {message}")


def main():
    device = torch.device("cpu")
    batch_size = 100
    max_epochs = 200
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

    initial_params = parameters_to_vector(
        model.parameters()
    ).detach().clone()

    shared = {
        "params": initial_params.share_memory_(),
        "grad": torch.zeros_like(initial_params).share_memory_(),
        "loss": torch.zeros(
            1, dtype=initial_params.dtype
        ).share_memory_(),
        "inputs": torch.empty(
            batch_size, 1, dtype=x.dtype
        ).share_memory_(),
        "targets": torch.empty(
            batch_size, 1, dtype=y.dtype
        ).share_memory_(),
        "n": torch.zeros(
            1, dtype=torch.long
        ).share_memory_(),
    }

    ctx = mp.get_context("spawn")

    parent_opt, child_opt = ctx.Pipe(duplex=True)
    parent_eval, child_eval = ctx.Pipe(duplex=True)

    opt_process = ctx.Process(
        target=optimizer_worker,
        args=(child_opt, shared, optimizer_config),
        name="optimizer",
    )

    eval_process = ctx.Process(
        target=evaluator_worker,
        args=(child_eval, shared),
        name="evaluator",
    )


    gradient_count = 0
    database_reaches = 0
    step_count = 0
    epoch_count = 0
    avg_loss = float("nan")

    start_time = time.time()

    opt_process.start()
    eval_process.start()

    child_eval.close()
    child_opt.close()

    try:
        converged = False

        for epoch in range(max_epochs):
                epoch_losses = []
        
                for inputs, targets in train_loader:
                    n = inputs.size(0)
        
                    shared["inputs"][:n].copy_(inputs)
                    shared["targets"][:n].copy_(targets)
                    shared["n"][0] = n

                    parent_eval.send("BATCH")
                    expect_ok(parent_eval.recv())

                    parent_opt.send("STEP")
        
                    while True:
                        msg = parent_opt.recv()

                        if isinstance(msg, int) and msg in (GRAD, LOSS, LOSS_GRAD):
                            parent_eval.send(msg)
                            status = parent_eval.recv()

                            if status == "OK":
                                database_reaches += n
                                gradient_count += int(msg in (GRAD, LOSS_GRAD))

                            parent_opt.send(status)
                        elif msg == "STEP_DONE":
                            break
                        elif msg == CONVERGED:
                            converged = True
                            break
                        else:
                            raise RuntimeError(
                                f"Unexpected optimizer reply: {msg}"
                            )

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
        
                    print(
                        f"Epoch {epoch_count:3d} | "
                        f"Loss: {avg_loss:.8f} | "
                        f"w: {model.weight.item():+.4f} | "
                        f"b: {model.bias.item():+.4f} | "
                        f"Grads: {gradient_count} | "
                        f"DB: {database_reaches}",
                        flush=True,
                    )

                if converged:
                    break
        
        elapsed = time.time() - start_time
    
        print("\n--- FINAL RESULTS ---")
        print("Target: y = 2x + 1")
        print(
            f"Learned: y = "
            f"{model.weight.item():.6f}x + "
            f"{model.bias.item():.6f}"
        )
        print(f"Final loss: {avg_loss:.8f}")
        print(f"Epochs: {epoch_count}; steps: {step_count}")
        print(f"Gradient computations: {gradient_count}")
        print(f"Database reaches: {database_reaches}")
        print(f"Time: {elapsed:.3f}s")

    finally:
        for connection, process in (
            (parent_opt, opt_process),
            (parent_eval, eval_process),
        ):
            if process.is_alive():
                try:
                    connection.send("STOP")
                except (BrokenPipeError, EOFError, OSError):
                    pass

        parent_opt.close()
        parent_eval.close()

        for process in (opt_process, eval_process):
            process.join(timeout=5)

            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    main()