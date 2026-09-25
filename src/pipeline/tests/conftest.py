"""Test environment for the pipeline side.

The pipeline is imported by its top-level package names (pipeline, shared) both
in the containers and here, so src/ has to be on the path; it holds no
installable distribution of its own.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
