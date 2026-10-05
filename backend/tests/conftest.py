"""Isolate every test run: fresh SQLite file and model dir under a temp folder."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="athlete_lens_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["MODEL_DIR"] = f"{_tmp}/models"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
