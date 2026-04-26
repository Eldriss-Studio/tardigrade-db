# Shared pytest configuration for TardigradeDB Python tests.

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA GPU and vLLM runtime",
    )
