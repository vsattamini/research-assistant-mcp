#!/usr/bin/env python
"""Convenience wrapper to execute the test-suite.

Runs pytest programmatically and prints a friendly message on success. This avoids
needing users to remember the exact pytest invocation and makes CI scripts more
readable:

    python run_tests.py  # or ./run_tests.py
"""

import sys
from pathlib import Path
import pytest


def main() -> None:
    # Ensure we execute from repository root (location of this script)
    repo_root = Path(__file__).resolve().parent
    print("🔍 Running unit tests with pytest …")

    # Run pytest on the `tests` directory. Pass through any CLI args.
    exit_code = pytest.main([str(repo_root / "tests"), *sys.argv[1:]])

    if exit_code == 0:
        print("✅ All systems OK – test suite passed.")
    else:
        print("❌ Test failures detected.")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
