"""Local CPU execution backend.

Thread limits are set here, in the package __init__, because they have to be in
place before NumPy or PyTorch load: the BLAS backends read these variables at
import time and ignore any later change. Putting them in a submodule is too
late, since importing any sibling module pulls NumPy in first.

The networks this backend runs have hundreds to tens of thousands of parameters
and batches of 32, so one forward pass is far cheaper than the cost of splitting
it across cores and gathering it back. Left at the default, a CMA-ES run on a
275-parameter model burned minutes at ~350% CPU; pinned to a single thread the
same run takes 2.2 seconds. Single-threaded execution also makes a run
reproducible, because the reduction order stops depending on how many cores
happened to be free.
"""

import os

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
