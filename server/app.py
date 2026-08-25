"""
FastAPI application factory for LLM Shield.

Provides:
- Auto-generated OpenAPI spec (importable into Postman at /openapi.json)
- Interactive docs at /docs (Swagger UI)
- CORS support for local development
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import execute, escalations, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("llm_shield")
    logger.info("LLM Shield server starting...")
    logger.info("Docs: http://localhost:8000/docs")
    logger.info("OpenAPI spec: http://localhost:8000/openapi.json")
    yield
    logger.info("LLM Shield server shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="LLM Shield API",
        description=(
            "Production-ready API that wraps LLM calls in a reliable control loop: "
            "Validate → Retry → Repair → Fallback → Human Escalation.\n\n"
            "**Import this API into Postman:**\n"
            "1. Open Postman → Import → Link\n"
            "2. Enter: `http://localhost:8000/openapi.json`\n"
            "3. All endpoints will be auto-generated with request/response schemas."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow all origins for local development / Postman
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router)
    app.include_router(execute.router)
    app.include_router(escalations.router)

    return app


# Application instance — used by uvicorn
app = create_app()

