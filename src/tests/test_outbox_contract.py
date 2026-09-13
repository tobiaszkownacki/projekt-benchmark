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
from pipeline.executor import JobDescription

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


def _message(optimizer: str = "adam") -> dict:
    return outbox.task_message(
        uuid.uuid4(),
        "athena_worker_queue",
        run_name="adam-wine-s7",
        dataset="wine",
        optimizer=optimizer,
    )


def test_message_covers_every_key_the_decoder_subscripts():
    # An existence check that skips would silence this contract the next time the
    # decoder moves; a missing decoder is a broken checkout, not a reason to pass.
    assert DECODER.exists(), f"{DECODER} is missing; the decoder moved and this contract stopped being checked"
    required = _subscripted_keys(DECODER.read_text(encoding="utf-8"), "msg")
    assert required, "parsed no msg[...] accesses; from_message may have been renamed"
    missing = required - set(_message())
    assert not missing, f"decoder reads {sorted(missing)}, absent from the outbox message"


def test_one_task_carries_exactly_one_optimizer():
    job = JobDescription.from_message(_message())
    assert job.optimizer == "adam"


def test_a_comma_in_a_user_supplied_name_stays_one_optimizer():
    # display_name reaches the message verbatim when a submission is not builtin,
    # so splitting the field would turn one run into several phantom optimizers.
    job = JobDescription.from_message(_message("Adam, wersja 2"))
    assert job.optimizer == "Adam, wersja 2"


def test_an_empty_optimizer_is_rejected_at_the_boundary():
    with pytest.raises(ValueError):
        JobDescription.from_message(_message("   "))
