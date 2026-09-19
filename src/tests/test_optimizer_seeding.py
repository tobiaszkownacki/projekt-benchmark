"""The caller's seed has to reach a sampler that carries its own generator.

These assertions are about reproducibility of a competition result, not about
CMA-ES: a leaderboard entry that cannot be re-run is not defensible, and a
sweep whose spread comes from the model initialisation alone reports a
variance it did not measure.
"""

import numpy as np
import pytest

from benchmark_core.seeding import resolve_optimizer_seed


def test_an_explicit_seed_is_passed_through():
    assert resolve_optimizer_seed({"seed": 7}) == 7


def test_the_same_caller_seed_resolves_to_the_same_optimizer_seed():
    np.random.seed(11)
    first = resolve_optimizer_seed({})
    np.random.seed(11)
    assert resolve_optimizer_seed({}) == first


def test_different_caller_seeds_resolve_to_different_optimizer_seeds():
    draws = set()
    for caller_seed in (11, 23, 42, 57, 71, 89, 101, 113):
        np.random.seed(caller_seed)
        draws.add(resolve_optimizer_seed({}))
    assert len(draws) == 8, "a sweep over eight seeds must sample eight times, not one"


@pytest.mark.parametrize("caller_seed", [0, 1, 2**31 - 2])
def test_the_resolved_seed_is_within_the_range_cma_accepts(caller_seed):
    np.random.seed(caller_seed)
    seed = resolve_optimizer_seed({})
    assert 1 <= seed < 2**31
