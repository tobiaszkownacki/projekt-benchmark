"""A job that dies before it can write any state of its own."""

import sys

if __name__ == "__main__":
    sys.exit(3)
