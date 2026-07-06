"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.openapi_docs import enhance_openapi_schema
from app.request_audit import request_audit_middleware
from app.routers import embeddings, logs, models, patches, regions, sam3, system_models, tasks

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

# Docs URLs: default to enabled for convenience; set to "none" to disable.
_docs_url = os.environ.get("DOCS_URL", "/docs")
_redoc_url = os.environ.get("REDOC_URL", "/redoc")

app = FastAPI(
    title="Embedding API",
    description="Unified API for remote sensing embeddings and downstream tasks (Harbin & Haidian)",
    version="0.1.0",
    docs_url=_docs_url if _docs_url.lower() != "none" else None,
    redoc_url=_redoc_url if _redoc_url.lower() != "none" else None,
    swagger_ui_parameters={
        "docExpansion": "none",
        "defaultModelsExpandDepth": 1,
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
        "syntaxHighlight.theme": "agate",
    },
    lifespan=lifespan,
)

app.middleware("http")(request_audit_middleware)


def _safe_validation_errors(exc: RequestValidationError) -> List[Dict[str, Any]]:
    """Convert validation errors into JSON-serializable dicts.

    FastAPI's default error detail may embed exception objects in `ctx.error`,
    which cannot be serialized by json.dumps. We stringify those values while
    preserving the rest of the error structure.
    """
    safe_errors: List[Dict[str, Any]] = []
    for err in exc.errors():
        safe_err = dict(err)
        ctx = safe_err.get("ctx")
        if isinstance(ctx, dict):
            safe_ctx: Dict[str, Any] = {}
            for k, v in ctx.items():
                if isinstance(v, BaseException):
                    safe_ctx[k] = str(v)
                else:
                    safe_ctx[k] = v
            safe_err["ctx"] = safe_ctx
        safe_errors.append(safe_err)
    return safe_errors


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a clearer message when the request body is completely empty.

    Other validation errors keep FastAPI's default field-level format so that
    existing clients and tests continue to work.
    """
    for err in exc.errors():
        if err.get("type") == "missing" and list(err.get("loc", [])) == ["body"]:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": (
                        "请求体不能为空。请确认 POST 请求携带了 JSON body，"
                        "并包含 name、model_type、task_type、region_id、annotations、classes 等必填字段。"
                    )
                },
            )
    return JSONResponse(status_code=422, content={"detail": _safe_validation_errors(exc)})


# CORS - restrict to specific origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with explicit Swagger groups.
app.include_router(regions.router, tags=["Regions"])
app.include_router(patches.router, tags=["Patches"])
app.include_router(embeddings.router, tags=["Embeddings"])
app.include_router(tasks.router, tags=["Task Results"])
app.include_router(models.router, tags=["Custom Models"])
app.include_router(system_models.router, tags=["System Models"])
app.include_router(sam3.router, tags=["SAM3"])
app.include_router(logs.router, tags=["Logs"])


def custom_openapi() -> Dict[str, Any]:
    """Build enriched OpenAPI docs for Swagger UI."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["tags"] = [
        {"name": "Regions", "description": "区域列表、区域详情与区域级元数据。"},
        {"name": "Patches", "description": "Patch 列表、详情、bbox 过滤和分页。"},
        {"name": "Embeddings", "description": "Embedding PNG/NPY/JSON 查询与区域 mosaic。"},
        {
            "name": "Task Results",
            "description": (
                "下游任务结果、预测、标签和瓦片。海淀最新数据推荐使用 "
                "`region_id=haidian`, `task_type=building_extraction`, "
                "`version=v1`, `month=202512` 联调。"
            ),
        },
        {"name": "Custom Models", "description": "前端标注驱动的自定义训练、单 patch 推理和批量推理。"},
        {"name": "System Models", "description": "系统预训练模型列表、类别和推理结果。"},
        {"name": "SAM3", "description": "SAM3 embedding 预加载、WGS84 点提示分割和缓存状态。"},
    ]
    app.openapi_schema = enhance_openapi_schema(openapi_schema)
    return app.openapi_schema


app.openapi = custom_openapi
