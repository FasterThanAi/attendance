"""Placeholder so `pytest` has something to collect in services/worker/tests
before Phase 3 adds real pipeline-stage tests. Deletes itself in spirit the
moment test_extract.py etc. exist -- feel free to remove this file then.
"""


def test_worker_package_imports():
    import pipeline  # noqa: F401
