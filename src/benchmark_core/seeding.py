"""Where the random number generators of a benchmark run get their seed.

A run is reproducible only if every generator it draws from starts from the
caller's seed, and they are not seeded in one place. The runner seeds the
NumPy and PyTorch globals. An optimizer that carries its own generator --
CMA-ES does -- is reached by neither, and takes its seed as an option
instead.
"""

import numpy as np

_SEED_UPPER_BOUND = 2**31 - 1


def resolve_optimizer_seed(config: dict) -> int:
    """The seed for an optimizer whose sampler has its own generator.

    Pinning such a sampler to a constant makes every run over one model
    identical however the caller seeded NumPy, so a sweep over eight seeds
    measures one sample eight times and its spread is the model
    initialisation alone. Drawing the default from the global NumPy RNG keeps
    a seeded caller reproducible and gives an unseeded one a different draw
    each time.
    """
    seed = config.get("seed")
    if seed is not None:
        return int(seed)
    return int(np.random.randint(1, _SEED_UPPER_BOUND))
