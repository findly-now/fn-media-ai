"""
Pytest configuration and shared fixtures for FN Media AI tests.

This module sets up test configuration and provides shared fixtures
that can be used across all test modules.
"""

import asyncio
import os
from typing import Generator
import pytest


# Set test environment variables
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

# Mock external APIs by default in tests
os.environ.setdefault("TEST_MOCK_EXTERNAL_APIS", "true")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for each test."""
    # This fixture runs before every test
    # Add any test setup logic here
    yield
    # Add any test cleanup logic here


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external services"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "cloud: mark test as requiring cloud services"
    )