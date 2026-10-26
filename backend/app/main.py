"""
backend/app/main.py
-------------------
Main entry point for the SimSG FastAPI application.
Integrates modular routers and exposes API documentation at /docs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Resilient imports (support both uvicorn backend.app.main:app and app.main:app)
try:
    from app.routers import test_simulation
    from app.services.simulation_context import get_simulation_context_service
except ImportError:
    from backend.app.routers import test_simulation
    from backend.app.services.simulation_context import get_simulation_context_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("simsg")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Pre-warms the SimulationContextService
    on application startup to cache Layer 1 demographics, HES data,
    and the ChromaDB persistent client connection.
    """
    logger.info("Initializing SimSG services on startup...")
    try:
        svc = get_simulation_context_service()
        logger.info(
            f"Pre-warm complete: ChromaDB collection has {svc._collection.count()} docs."
        )
    except Exception as exc:
        logger.error(f"Startup warning - could not pre-warm simulation service: {exc}")
    yield
    logger.info("Shutting down SimSG application...")


app = FastAPI(
    title="SimSG Simulation Engine API",
    description="Singapore micro-SME AI focus group simulation platform backend.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for React 19 frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------------------------

# Testing / Diagnostics module (kept isolated from future production routes)
app.include_router(test_simulation.router)

# Future production routers will be mounted here:
# app.include_router(simulation.router, prefix="/api/v1")
# app.include_router(personas.router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def root():
    return {
        "app": "SimSG Simulation Engine API",
        "status": "operational",
        "version": "0.1.0",
        "docs_url": "/docs",
        "testing_routes": {
            "diagnostics": "/api/test/diagnostics",
            "simulation_context_get": "/api/test/simulation-context?planning_area=Ang Mo Kio&user_pitch=S$7 specialty coffee",
            "simulation_context_post": "/api/test/simulation-context",
        },
    }
