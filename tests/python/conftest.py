# Shared pytest configuration for TardigradeDB Python tests.

import pytest


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
