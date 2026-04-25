"""
conftest.py — pytest configuration for the CausalRepair test suite.

WHY THIS FILE EXISTS:
    The package layout is:
        CausalRepair/           <- repo root
          CausalRepair/         <- Python package root (THIS directory)
            models.py
            server/
              base_adapter.py
              mock_adapter.py
              code_repair_adapter.py
            tests/
              conftest.py       <- you are here
              test_base_adapter.py
              test_code_adapter.py

    When pytest runs from *anywhere*, it will find this conftest.py and
    execute it before any test file. This file adds the package root
    (CausalRepair/CausalRepair/) to sys.path so that:
        from models import CausalrepairObservation   # works
        from server.base_adapter import BaseAdapter   # works
        from server.mock_adapter import MockAdapter   # works

    Without this, pytest would fail with "ModuleNotFoundError: No module
    named 'models'" because Python wouldn't know where to look.

WHEN YOU RUN TESTS AS A PLAIN SCRIPT (no pytest):
    Each test file also has the same sys.path insertion at the top of the
    file, so `python tests/test_base_adapter.py` works without conftest.py.
    conftest.py only matters when you use `pytest`.

HOW TO USE:
    # Install pytest first (once):
    ..\\venv\\Scripts\\pip.exe install pytest

    # Then run all tests from the package root:
    ..\\venv\\Scripts\\pytest.exe tests/ -v
"""
import sys
from pathlib import Path

# Resolve the absolute path of CausalRepair/CausalRepair/
PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # tests/ -> CausalRepair/

# Insert at position 0 so our package takes priority over any installed version
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
