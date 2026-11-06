"""The job message is a contract between two deployables that never import each other.

The producer is app.services.outbox.task_message and the only reader is
pipeline.executor.JobDescription.from_message. Nothing else pins their shapes
together, and the previous version of this file aimed at a worker module that a
refactor had already removed: it skipped itself and CI stayed green for weeks.
"""

import ast
import uuid
from pathlib import Path

import pytest
from app.services import outbox

from pipeline.executor import SCHEMA_VERSION, JobDescription

DECODER = Path(__file__).resolve().parents[2] / "pipeline/executor.py"

# Read with .get() rather than subscripted, so a round trip cannot notice them
# missing. Every one of them is optional by intent and listed here on purpose.
OPTIONAL_KEYS = {"schema_version", "webhook_token"}


def _message(**overrides) -> dict:
    message = outbox.task_message(
        uuid.uuid4(),
        run_name="adam-wine_quality-s11",
        dataset="wine_quality",
        model="mlp-2x32",
        optimizers=["adam"],
        seed=11,
        stop_condition={"max_epochs": 5, "max_gradient_count": 1000},
    )
    message.update(overrides)
    return message


def _read_keys(kind: str) -> set[str]:
    """String literals the decoder reads out of its message, by access style."""
    source = ast.parse(DECODER.read_text(encoding="utf-8"))
    decoder = next(
        node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "from_message"
    )
    keys = set()
    for node in ast.walk(decoder):
        if (
            kind == "subscript"
            and isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
        if (
            kind == "get"
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def test_the_decoder_is_where_this_test_thinks_it_is():
    assert DECODER.exists(), f"{DECODER} is missing; the decoder moved and this contract stopped being checked"


def test_a_produced_message_decodes_field_for_field():
    message = _message()
    job = JobDescription.from_message(message)

    assert job.task_id == message["task_id"]
    assert job.run_name == message["run_name"]
    assert job.dataset == message["dataset"]
    assert job.model == message["model"]
    assert job.optimizers == message["optimizers"]
    assert job.seed == message["seed"]
    assert job.stop_condition == message["stop_condition"]


def test_every_required_key_is_produced():
    assert _read_keys("subscript") <= set(_message())


def test_optional_keys_are_the_ones_this_contract_allows():
    assert _read_keys("get") <= OPTIONAL_KEYS


def test_the_two_sides_agree_on_the_schema_version():
    assert outbox.SCHEMA_VERSION == SCHEMA_VERSION


def test_a_message_from_an_older_schema_is_refused():
    with pytest.raises(ValueError, match="schema_version"):
        JobDescription.from_message(_message(schema_version=SCHEMA_VERSION - 1))


def test_a_name_with_a_comma_stays_one_optimizer():
    job = JobDescription.from_message(_message(optimizers=["Adam, wersja 2"]))

    assert job.optimizers == ["Adam, wersja 2"]


def test_two_optimizers_are_refused():
    with pytest.raises(ValueError, match="optimizers"):
        JobDescription.from_message(_message(optimizers=["adam", "sgd"]))


def test_a_run_without_a_budget_is_refused():
    with pytest.raises(ValueError, match="stop condition"):
        JobDescription.from_message(_message(stop_condition={}))
