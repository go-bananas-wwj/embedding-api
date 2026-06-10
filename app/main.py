"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import regions, patches, embeddings, tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager (startup/shutdown)."""
    from app.config import get_config

    get_config()
    logging.info("[Startup] Embedding API started successfully")
    yield
    get_config().stop_watching()
    logging.info("[Shutdown] Embedding API stopped")


app = FastAPI(
    title="Embedding API",
    description="Unified API for remote sensing embeddings and downstream tasks (Harbin & Haidian)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS - restrict to specific origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(regions.router)
app.include_router(patches.router)
app.include_router(embeddings.router)
app.include_router(tasks.router)
