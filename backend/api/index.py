# Entrypoint for Vercel Serverless Functions
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(backend_dir)

for p in (backend_dir, parent_dir):
    if p and p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.main import app
except ImportError:
    from backend.app.main import app
