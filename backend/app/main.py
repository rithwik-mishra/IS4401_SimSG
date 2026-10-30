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
    from app.routers import simulate, simulation, test_simulation
    from app.services.simulation_context import get_simulation_context_service
except ImportError:
    from backend.app.routers import simulate, simulation, test_simulation
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
    version="1.0.0",
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

# Full End-to-End Simulation Pipeline (POST /api/simulate)
app.include_router(simulate.router)

# Granular Simulation Engine Endpoints (/api/v1/simulation/*)
app.include_router(simulation.router, prefix="/api/v1")
app.include_router(simulation.router, prefix="/api")

# Testing & Diagnostics module (Layer 1 Seed, HES benchmarks, ChromaDB checks)
app.include_router(test_simulation.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "app": "SimSG Simulation Engine API",
        "status": "operational",
        "version": "1.0.0",
        "docs_url": "/docs",
        "routes": {
            "simulate": "/api/simulate",
            "simulation_preview": "/api/v1/simulation/preview",
            "simulation_run": "/api/v1/simulation/run",
            "simulation_areas": "/api/v1/simulation/areas",
            "diagnostics": "/api/test/diagnostics",
            "simulation_context_get": "/api/test/simulation-context?planning_area=Ang Mo Kio&user_pitch=S$7 specialty coffee",
            "simulation_context_post": "/api/test/simulation-context",
        },
    }
