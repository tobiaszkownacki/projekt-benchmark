"""The queue message must carry every field the Athena worker subscripts.

AthenaWorker.start_job reads task["optimizer"] and task["dataset"] directly.
A message without them raises KeyError inside the worker's callback, which
nacks it to the dead-letter queue instead of retrying, so the loss is silent.
"""

import ast
import uuid
from pathlib import Path

import pytest
from app.services import outbox

WORKER = Path(__file__).resolve().parents[2] / "task_queue/services/workers/athena/athena_worker.py"


def _subscripted_keys(source: str, argument: str) -> set[str]:
    """String literals used as `argument[...]` anywhere in the module."""
    keys = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == argument
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


def _message() -> dict:
    return outbox.task_message(
        uuid.uuid4(),
        "ATHENA_WORKER_QUEUE",
        run_name="adam-wine-s7",
        dataset="wine",
        optimizer="adam",
    )


@pytest.mark.skipif(not WORKER.exists(), reason="worker module not in this checkout")
def test_message_covers_every_key_the_worker_subscripts():
    required = _subscripted_keys(WORKER.read_text(encoding="utf-8"), "task")
    assert required, "parsed no task[...] accesses; the worker may have been renamed"
    missing = required - set(_message())
    assert not missing, f"worker reads {sorted(missing)}, absent from the outbox message"


def test_optimizer_is_a_string_the_worker_can_split():
    # _build_optimizer_args splits on "," and rejects an empty selection.
    optimizer = _message()["optimizer"]
    assert isinstance(optimizer, str)
    assert [name for name in optimizer.split(",") if name.strip()]
