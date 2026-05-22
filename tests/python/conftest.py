# Shared pytest configuration for TardigradeDB Python tests.

# Re-export the GPU cleanup helper at conftest level so the existing AT
# (`test_conftest_gpu_cleanup`) can find it. The canonical home is the
# importable sibling module `_gpu_test_utils` — conftest.py is not
# directly importable as `from conftest import …` because pytest treats
# it specially.
import sys
from pathlib import Path

# Make the sibling `_gpu_test_utils` module importable from any test
# file via `from _gpu_test_utils import do_gpu_cleanup`. conftest.py is
# loaded by pytest but not on sys.path itself; we add the directory.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _gpu_test_utils import do_gpu_cleanup as _do_gpu_cleanup  # noqa: E402,F401


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA GPU and vLLM runtime",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require large model downloads and long runtime",
    )
    config.addinivalue_line(
        "markers",
        "live_api: marks tests that hit a real LLM API (skipped unless keys are set)",
    )
