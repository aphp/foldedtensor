import multiprocessing

import pytest


# This is necessary for CUDA to work properly in test_multiprocessing.py
@pytest.fixture(scope="session", autouse=True)
def mp_spawn():
    multiprocessing.set_start_method("spawn")
