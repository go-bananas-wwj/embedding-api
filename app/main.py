"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import regions, patches, embeddings, tasks, sam3, annotations, models, system_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager (startup/shutdown)."""
    from app.config import get_config

    get_config()
    logging.info("[Startup] Embedding API started successfully")
    yield
    get_config().stop_watching()
    logging.info("[Shutdown] Embedding API stopped")


# CORS origins from environment variable. Default to empty list for security;
# explicit origins must be set via CORS_ORIGINS env var.
_cors_origins = os.environ.get("CORS_ORIGINS", "")
allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
if not allow_origins:
    logger.warning("CORS_ORIGINS not set; cross-origin requests will be rejected")

# Docs URLs: default to disabled (None). Enable explicitly via env vars.
_docs_url = os.environ.get("DOCS_URL", "none")
_redoc_url = os.environ.get("REDOC_URL", "none")

app = FastAPI(
    title="Embedding API",
    description="Unified API for remote sensing embeddings and downstream tasks (Harbin & Haidian)",
    version="0.1.0",
    docs_url=_docs_url if _docs_url.lower() != "none" else None,
    redoc_url=_redoc_url if _redoc_url.lower() != "none" else None,
    lifespan=lifespan,
)

# CORS - restrict to specific origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(regions.router)
app.include_router(patches.router)
app.include_router(embeddings.router)
app.include_router(tasks.router)
app.include_router(annotations.router)
app.include_router(models.router)
app.include_router(system_models.router)
app.include_router(sam3.router)
