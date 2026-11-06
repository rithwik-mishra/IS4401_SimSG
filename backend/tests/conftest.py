import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def no_real_ai(monkeypatch):
    """
    Every test starts with no API keys and SIMSG_PROVIDER=openai, whatever backend/.env says,
    so the suite never calls a real AI. Tests that need the AI path set a fake key and fake clients.
    """
    for var in ("OPEN_API_KEY", "OPENAI_API_KEY", "SOCLAAS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SIMSG_PROVIDER", "openai")
