import pytest

from pipeline.executor import is_retryable_submission_error


@pytest.mark.parametrize(
    "message",
    [
        "sbatch: error: QOSMaxSubmitJobPerUserLimit",
        "sbatch: error: QOSMaxSubmitJobPerAccountLimit",
        "sbatch: error: AssocMaxSubmitJobLimit",
    ],
)
def test_scheduler_admission_limits_are_retryable(message):
    assert is_retryable_submission_error(message)


def test_invalid_submission_is_not_retryable():
    assert not is_retryable_submission_error("sbatch: error: Invalid account or account/partition combination")
