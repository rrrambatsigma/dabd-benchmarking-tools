"""Shared helpers for the benchmark project."""
from pathlib import Path
import time

def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)