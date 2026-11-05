# Entrypoint for Vercel Serverless Functions
import sys
import os

# Add backend directory and its parent to sys.path so app.* and backend.app.* work
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(backend_dir)

for p in (backend_dir, parent_dir):
    if p and p not in sys.path:
        sys.path.insert(0, p)

from app.main import app as _fastapi_app

# Explicit top-level entrypoints recognized by @vercel/python
app = _fastapi_app
application = app
handler = app
