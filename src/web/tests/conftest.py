"""Test environment.

Settings are read once, when app.settings is first imported. Any test module
that touches the application pins them for the whole session, so the values have
to be correct here rather than inside a fixture -- otherwise the first import
wins and later tests connect to whatever the first one happened to configure.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://unused@127.0.0.1:1/unused"
)
os.environ.setdefault("SESSION_SECRET", "test-secret-not-used-in-production")
os.environ.setdefault("RUN_MIGRATIONS", "0")
