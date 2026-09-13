"""The queue message must carry every field the pipeline decoder subscripts.

JobDescription.from_message reads msg["optimizer"] and msg["dataset"] directly.
A message without them raises KeyError inside the consumer's callback, which
nacks it to the dead-letter queue instead of retrying, so the loss is silent.
"""

import ast
import uuid
from pathlib import Path

import pytest

from backend.services import outbox

DECODER = Path(__file__).resolve().parents[1] / "pipeline/executor.py"


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
        "athena_worker_queue",
        run_name="adam-wine-s7",
        dataset="wine",
        optimizer="adam",
    )


@pytest.mark.skipif(not DECODER.exists(), reason="pipeline executor not in this checkout")
def test_message_covers_every_key_the_decoder_subscripts():
    required = _subscripted_keys(DECODER.read_text(encoding="utf-8"), "msg")
    assert required, "parsed no msg[...] accesses; from_message may have been renamed"
    missing = required - set(_message())
    assert not missing, f"decoder reads {sorted(missing)}, absent from the outbox message"


def test_optimizer_is_a_string_the_worker_can_split():
    # from_message splits on "," and drops empty entries.
    optimizer = _message()["optimizer"]
    assert isinstance(optimizer, str)
    assert [name for name in optimizer.split(",") if name.strip()]
