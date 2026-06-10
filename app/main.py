"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import regions, patches, embeddings, tasks

app = FastAPI(
    title="Embedding API",
    description="Unified API for remote sensing embeddings and downstream tasks (Harbin & Haidian)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(regions.router)
app.include_router(patches.router)
app.include_router(embeddings.router)
app.include_router(tasks.router)


@app.on_event("startup")
async def startup_event():
    """Initialize config on startup."""
    from app.config import get_config

    get_config()
    print("[Startup] Embedding API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    from app.config import get_config

    config = get_config()
    config.stop_watching()
    print("[Shutdown] Embedding API stopped")
