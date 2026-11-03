# Entrypoint for Vercel Serverless Functions
import sys
import os
# Add backend directory to sys.path so app.* imports work smoothly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
