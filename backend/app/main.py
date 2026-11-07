"""
backend/app/main.py
-------------------
Main entry point for the SimSG FastAPI application.
Integrates modular routers and exposes API documentation at /docs.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler

# Resilient imports (support both uvicorn backend.app.main:app and app.main:app)
try:
    from app.routers import simulate, simulation, test_simulation
    from app.services.simulation_context import get_simulation_context_service
    from app.services.rate_limiter import rate_limit_dependency
except ImportError:
    from backend.app.routers import simulate, simulation, test_simulation
    from backend.app.services.simulation_context import get_simulation_context_service
    from backend.app.services.rate_limiter import rate_limit_dependency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("simsg")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: Pre-warms the SimulationContextService
    on application startup to open the ChromaDB persistent client connection.
    """
    logger.info("Initializing SimSG services on startup...")
    try:
        svc = get_simulation_context_service()
        logger.info(
            f"Pre-warm complete: ChromaDB collection '{svc._collection.name}' has {svc._collection.count()} docs."
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

# CORS middleware for React 19 frontend integration and Vercel preview/production deployments
default_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://simsg.vercel.app",
]

env_cors = os.getenv("CORS_ORIGINS")
if env_cors:
    extra_origins = [orig.strip() for orig in env_cors.split(",") if orig.strip()]
    allowed_origins = list(dict.fromkeys(default_origins + extra_origins))
else:
    allowed_origins = default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https://simsg(-[a-zA-Z0-9_-]+)?\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Enforces standard HTTP security response headers."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global unhandled exception handler returning sanitized JSON error response."""
    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        return await http_exception_handler(request, exc)
    if isinstance(exc, RequestValidationError):
        return await request_validation_exception_handler(request, exc)
    logger.exception(f"Unhandled error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred while processing your request. Please try again shortly.",
            "code": "INTERNAL_SERVER_ERROR",
        },
    )

# ---------------------------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------------------------

# Full End-to-End Simulation Pipeline (POST /api/simulate and /simulate)
app.include_router(simulate.router)

# Planning-area lookup (GET /api/simulation/areas). The older /simulation/preview and /run
# routes (respondent.py pipeline) are kept unregistered in app/legacy/simulation_routes.py.
# Aliases for reverse proxies that might strip /api prefix

app.add_api_route(
    "/simulate",
    simulate.simulate,
    methods=["POST"],
    response_model=simulate.SimulateResponse,
    dependencies=[Depends(rate_limit_dependency)],
    include_in_schema=False,
)
app.add_api_route(
    "/simulate",
    simulate.simulate_info,
    methods=["GET"],
    include_in_schema=False,
)

app.include_router(simulation.router, prefix="/api")

# Testing & Diagnostics module (census sampler areas, ChromaDB checks)
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
            "simulate_preview": "/api/simulate/preview",
            "ai_status": "/api/ai-status",
            "simulation_areas": "/api/simulation/areas",
            "diagnostics": "/api/test/diagnostics",
            "simulation_context_get": "/api/test/simulation-context?planning_area=Ang Mo Kio&user_pitch=S$7 specialty coffee",
            "simulation_context_post": "/api/test/simulation-context",
        },
    }
