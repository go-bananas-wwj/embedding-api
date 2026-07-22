"""FastAPI application entry point."""

import logging
import os
import asyncio
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

    from app.services.summary_image_service import (
        cleanup_expired_summary_images,
        summary_image_cleanup_loop,
    )
    from app.services.custom_model_cleanup import (
        DEFAULT_TTL_HOURS,
        _positive_float_env,
        cleanup_custom_models,
        cleanup_enabled,
        custom_model_cleanup_loop,
    )

    get_config()
    cleanup_expired_summary_images()
    cleanup_task = asyncio.create_task(summary_image_cleanup_loop())
    model_cleanup_task = None
    if cleanup_enabled():
        ttl_hours = _positive_float_env("CUSTOM_MODEL_TTL_HOURS", DEFAULT_TTL_HOURS)
        await asyncio.to_thread(cleanup_custom_models, ttl_hours, False)
        model_cleanup_task = asyncio.create_task(custom_model_cleanup_loop())
    logging.info("[Startup] Embedding API started successfully")
    try:
        yield
    finally:
        cleanup_task.cancel()
        if model_cleanup_task:
            model_cleanup_task.cancel()
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
    description=(
        "遥感 Embedding、系统任务、自定义训练与 SAM3 分割服务。"
        "接口字段说明、默认值和范围请直接展开对应的 Request body 或 Parameters。"
    ),
    version="0.1.0",
    docs_url=_docs_url if _docs_url.lower() != "none" else None,
    redoc_url=_redoc_url if _redoc_url.lower() != "none" else None,
    swagger_ui_parameters={
        "docExpansion": "none",
        "defaultModelsExpandDepth": 0,
        "defaultModelExpandDepth": 1,
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
        "showExtensions": False,
        "showCommonExtensions": False,
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
                        "并且代理/网关转发时没有丢失 body。"
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
app.include_router(regions.router, tags=["区域"])
app.include_router(patches.router, tags=["Patch"])
app.include_router(embeddings.router, tags=["Embedding"])
app.include_router(tasks.router, tags=["任务结果"])
app.include_router(models.router, tags=["自定义模型"])
app.include_router(system_models.router, tags=["系统模型"])
app.include_router(sam3.router, tags=["SAM3"])
app.include_router(logs.router)


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
        {"name": "区域", "description": "查看可用区域及区域级元数据。"},
        {"name": "Patch", "description": "查询遥感切片、空间范围和分页列表。"},
        {"name": "Embedding", "description": "获取 Embedding 图像、数组、统计信息和区域拼图。"},
        {
            "name": "任务结果",
            "description": "查询道路、建筑、水体等系统任务的预测、标签与结果文件。",
        },
        {"name": "自定义模型", "description": "创建模型，并使用同一模型 ID 完成单 Patch 或批量推理。"},
        {"name": "系统模型", "description": "查看和调用服务内置的下游任务模型。"},
        {"name": "SAM3", "description": "预加载影像，并通过 WGS84 点提示完成实例分割。"},
    ]
    app.openapi_schema = enhance_openapi_schema(openapi_schema)
    return app.openapi_schema


app.openapi = custom_openapi
