"""Custom model training and inference routes.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import FileResponse

from app.config import get_config
from app.schemas.models import (
    BatchInferRequest,
    BatchInferResponse,
    BatchInferResult,
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONProperties,
    InferRequest,
    InferResult,
    JobStatusOut,
    ModelClass,
    ModelCreate,
    ModelOut,
    ModelRenameRequest,
)
from app.services.auth_service import get_current_user
from app.services.data_service import DataValidationError
from app.services.inference_engine import InferenceEngine
from app.services.model_registry import get_model_registry
from app.services.system_model_service import (
    get_system_model_info,
    infer_system_model,
    is_system_task,
)
from app.services.training_engine import (
    ChangeDetectionTrainingEngine,
    ClassificationTrainingEngine,
)

router = APIRouter(prefix="/models", tags=["models"])

# In-memory job tracking. Training jobs are ephemeral; restart clears them.
_training_jobs: dict[str, dict] = {}


def _validate_result_filename(filename: str) -> None:
    if (
        "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or not filename.endswith(".png")
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")


def _batch_response(results: List[dict]) -> dict:
    success_count = sum(1 for item in results if item.get("status") == "success")
    return {
        "total": len(results),
        "success_count": success_count,
        "error_count": len(results) - success_count,
        "results": results,
    }


def _resolve_embedding_version(region_id: str, requested: str) -> str:
    """Use the requested embedding version when available, otherwise region default."""
    region_cfg = get_config().get_region(region_id) or {}
    embeddings = region_cfg.get("embeddings") or {}
    if requested in embeddings:
        return requested
    if "v1" in embeddings:
        return "v1"
    if embeddings:
        return sorted(embeddings.keys())[0]
    raise HTTPException(
        status_code=404,
        detail=f"No embeddings configured for region '{region_id}'",
    )


@router.get("", response_model=List[ModelOut])
async def list_models(
    region_id: Optional[str] = Query(
        None,
        description="可选，传入区域 ID 后会在列表中同时返回该区域可用的系统预训练模型。",
        examples=["harbin"],
    ),
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """获取当前用户训练好的自定义模型列表，可一并返回系统预训练模型。

    用于模型管理页展示模型名称、类型、状态、精度等信息。
    传入 `region_id` 后，系统预训练模型会附加 `source: "system"` 与 `versions` 字段。
    返回 JSON 列表。
    """
    models = get_model_registry(user["user_id"]).list_models()
    for m in models:
        m.setdefault("source", "custom")

    if region_id:
        from app.services.system_model_service import list_system_models

        for sys_model in list_system_models(region_id):
            task_id = sys_model["id"]
            try:
                info = get_system_model_info(region_id, task_id)
                models.append(info)
            except Exception:
                # Skip system models that cannot be loaded for this region/version.
                pass

    return models


_CLASSIFICATION_EXAMPLE = {
    "summary": "单时间检测模型（以建筑物提取为例）",
    "description": "model_type='single_time_detection' 时传入 month；训练任务类型由 GeoJSON features[].properties.task_type 自动推导，顶层请求体不再需要 task_type。",
    "value": {
        "name": "我的建筑提取模型",
        "model_type": "single_time_detection",
        "region_id": "harbin",
        "embedding_version": "v2",
        "epochs": 100,
        "class_ids": ["cls_001"],
        "description": "基于用户标注的建筑提取模型",
        "annotations": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "patch_id": "patch_000000",
                        "region_id": "harbin",
                        "class_id": "cls_001",
                        "class_name": "建筑用地",
                        "color": "#FF0000",
                        "task_type": "building_extraction",
                        "month": "2025-04",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [126.51631, 45.743707],
                                [126.532242, 45.743707],
                                [126.532242, 45.755574],
                                [126.51631, 45.755574],
                                [126.51631, 45.743707],
                            ]
                        ],
                    },
                }
            ],
        },
        "classes": [{"id": "cls_001", "name": "建筑用地", "color": "#FF0000"}],
    },
}

_CHANGE_DETECTION_EXAMPLE = {
    "summary": "变化检测模型",
    "description": "model_type='change_detection' 时后端自动使用 change_detection 任务类型，并传入 before_month 和 after_month。",
    "value": {
        "name": "我的变化检测模型",
        "model_type": "change_detection",
        "region_id": "harbin",
        "embedding_version": "v2",
        "epochs": 100,
        "class_ids": ["cls_001"],
        "description": "基于用户标注的变化检测模型",
        "annotations": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "patch_id": "patch_000000",
                        "region_id": "harbin",
                        "class_id": "cls_001",
                        "class_name": "变化区域",
                        "color": "#FF0000",
                        "task_type": "change_detection",
                        "before_month": "2025-04",
                        "after_month": "2025-06",
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [126.51631, 45.743707],
                                [126.532242, 45.743707],
                                [126.532242, 45.755574],
                                [126.51631, 45.755574],
                                [126.51631, 45.743707],
                            ]
                        ],
                    },
                }
            ],
        },
        "classes": [{"id": "cls_001", "name": "变化区域", "color": "#FF0000"}],
    },
}


@router.post("", response_model=ModelOut)
async def create_model(
    background_tasks: BackgroundTasks = BackgroundTasks(),
    req: ModelCreate = Body(
        ...,
        openapi_examples={
            "classification": _CLASSIFICATION_EXAMPLE,
            "change_detection": _CHANGE_DETECTION_EXAMPLE,
        },
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """创建模型并启动异步训练任务。

    前端提交用户定义的模型名称、分类列表和 GeoJSON 标注数据包；
    后端解析标注包，提取训练样本并训练下游任务头。
    返回模型信息及训练任务 ID，可在 `/models/jobs/{job_id}` 轮询进度。

    """
    registry = get_model_registry(user["user_id"])

    if req.model_type not in ("single_time_detection", "change_detection"):
        raise HTTPException(
            status_code=422,
            detail="model_type must be 'single_time_detection' or 'change_detection'",
        )
    task_type = req.resolved_task_type()
    embedding_version = _resolve_embedding_version(
        req.region_id, req.embedding_version
    )

    active_class_ids = req.class_ids or list(
        {f.properties.class_id for f in req.annotations.features}
    )

    model_id = registry.create_model(
        name=req.name,
        model_type=req.model_type,
        classes=[c.model_dump() for c in req.classes],
        task_type=task_type,
        region_id=req.region_id,
        description=req.description,
    )

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _training_jobs[job_id] = {
        "job_id": job_id,
        "model_id": model_id,
        "status": "running",
        "user_id": user["user_id"],
        "started_at": datetime.now().isoformat(),
        "message": "Training started",
    }

    background_tasks.add_task(
        _do_training,
        job_id=job_id,
        model_id=model_id,
        user_id=user["user_id"],
        region_id=req.region_id,
        task_type=task_type,
        model_type=req.model_type,
        embedding_version=embedding_version,
        epochs=req.epochs,
        annotations=req.annotations,
        classes=req.classes,
        class_ids=active_class_ids,
    )
    model = registry.get_model(model_id)
    model["job_id"] = job_id
    return model


@router.get("/{model_id}", response_model=ModelOut)
async def get_model(
    model_id: str = PathParam(
        ...,
        description="模型 ID。可以是 POST /models 返回的自定义模型 ID，也可以是系统预训练任务 ID（如 'building_extraction'、'land_cover_classification'、'water_extraction'）。",
        examples=["model_ghi789"],
        openapi_examples={
            "custom": {"summary": "自定义模型", "value": "model_ghi789"},
            "system_building": {"summary": "系统预设-建筑物提取", "value": "building_extraction"},
            "system_land_cover": {"summary": "系统预设-土地覆盖分类", "value": "land_cover_classification"},
            "system_water": {"summary": "系统预设-水体提取", "value": "water_extraction"},
        },
    ),
    region_id: Optional[str] = Query(
        None,
        description="当 model_id 是系统预训练任务 ID 时，需要传入区域 ID 以定位 checkpoint。",
        examples=["harbin"],
    ),
    version: Optional[str] = Query(
        "v2",
        description="当 model_id 是系统预训练任务 ID 时，选择 checkpoint 版本。",
        examples=["v2"],
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """获取单个模型（自定义或系统预设）的详情。

    用于在模型详情页查看训练进度、精度、类别定义等信息。
    如果是系统预训练模型，需要同时传入 `region_id`；返回状态固定为 `ready`。
    """
    registry = get_model_registry(user["user_id"])
    model = registry.get_model(model_id)
    if model:
        model.setdefault("source", "custom")
        return model

    if is_system_task(model_id):
        if not region_id:
            raise HTTPException(
                status_code=422,
                detail=f"System model '{model_id}' requires region_id query parameter",
            )
        try:
            return get_system_model_info(region_id, model_id, version=version)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(status_code=404, detail="Model not found")


def _rename_model_for_user(user_id: str, model_id: str, name: str) -> dict:
    if not get_model_registry(user_id).rename_model(model_id, name):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "ok"}


@router.put("/{model_id}", response_model=Dict[str, str])
async def rename_model_put(
    model_id: str = PathParam(
        ...,
        description="Model ID returned by POST /models. Replace with the real ID from the create response.",
        examples=["model_ghi789"],
    ),
    req: ModelRenameRequest = Body(...),
    user: dict = Depends(get_current_user),
) -> dict:
    """重命名指定模型（推荐使用 PUT）。

    用于模型管理页修改显示名称。
    返回操作是否成功的状态 JSON。
    """
    return _rename_model_for_user(user["user_id"], model_id, req.name)


@router.patch("/{model_id}", response_model=Dict[str, str])
async def rename_model_patch(
    model_id: str = PathParam(
        ...,
        description="Model ID returned by POST /models. Replace with the real ID from the create response.",
        examples=["model_ghi789"],
    ),
    req: ModelRenameRequest = Body(...),
    user: dict = Depends(get_current_user),
) -> dict:
    """重命名指定模型（兼容旧 PATCH 调用）。

    前端新接入建议使用 `PUT /models/{model_id}`。
    """
    return _rename_model_for_user(user["user_id"], model_id, req.name)


@router.delete("/{model_id}", response_model=Dict[str, str])
async def delete_model(
    model_id: str = PathParam(
        ...,
        description="Model ID returned by POST /models. Replace with the real ID from the create response.",
        examples=["model_ghi789"],
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """删除指定模型及其产物。

    用于清理失败或不再使用的模型以释放空间。
    返回操作是否成功的状态 JSON。
    """
    if not get_model_registry(user["user_id"]).delete_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "ok"}


_INFER_EXAMPLES: Dict[str, Any] = {
    "single_time_detection": {
        "summary": "自定义单时间检测模型推理",
        "description": "单时间检测模型只需传入 month。",
        "value": {
            "region_id": "harbin",
            "patch_id": "patch_000000",
            "month": "2025-04",
        },
    },
    "change_detection": {
        "summary": "自定义变化检测模型推理",
        "description": "变化检测模型需同时传入 before_month 和 after_month，不要传 month。",
        "value": {
            "region_id": "harbin",
            "patch_id": "patch_000000",
            "before_month": "2025-04",
            "after_month": "2025-06",
        },
    },
    "system_model": {
        "summary": "系统预训练模型推理",
        "description": "把系统任务 ID（如 building_extraction）作为 model_id，传入 region_id、patch_id、month，可选 version。",
        "value": {
            "region_id": "harbin",
            "patch_id": "patch_000000",
            "month": "2025-04",
            "version": "v2",
        },
    },
}


@router.post("/{model_id}/infer", response_model=InferResult)
async def infer(
    model_id: str = PathParam(
        ...,
        description="模型 ID。可以是 POST /models 返回的自定义模型 ID，也可以是系统预训练任务 ID（如 'building_extraction'）。",
        examples=["model_ghi789"],
        openapi_examples={
            "custom": {"summary": "自定义模型", "value": "model_ghi789"},
            "system_building": {"summary": "系统预设-建筑物提取", "value": "building_extraction"},
            "system_land_cover": {"summary": "系统预设-土地覆盖分类", "value": "land_cover_classification"},
            "system_water": {"summary": "系统预设-水体提取", "value": "water_extraction"},
        },
    ),
    req: InferRequest = Body(..., openapi_examples=_INFER_EXAMPLES),
    user: dict = Depends(get_current_user),
) -> dict:
    """对单个 Patch 运行模型推理（支持自定义模型和系统预训练模型）。

    用于获取某个 Patch 的预测结果图片。
    自定义模型需要训练完成；系统预训练模型状态为 ready，直接可用。
    返回结果图片的访问 URL。

    """
    registry = get_model_registry(user["user_id"])
    model = registry.get_model(model_id)

    if is_system_task(model_id):
        if not req.month:
            raise HTTPException(
                status_code=422,
                detail="System models require 'month' in the request body",
            )
        results_dir = Path(f"users/{user['user_id']}/system_model_results")
        try:
            result_path = infer_system_model(
                req.region_id,
                model_id,
                req.patch_id,
                req.month,
                version=req.version or "v2",
                results_dir=results_dir,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except DataValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        filename = Path(result_path).name
        return {"result_url": f"/system-models/results/{filename}"}

    if not model or model.get("status") != "completed":
        raise HTTPException(
            status_code=400, detail="Model not trained or not found"
        )

    engine = InferenceEngine(user["user_id"])
    try:
        result_path = engine.infer(
            model_id,
            req.region_id,
            req.patch_id,
            month=req.month,
            before_month=req.before_month,
            after_month=req.after_month,
        )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = Path(result_path).name
    return {"result_url": f"/models/results/{filename}"}


_INFER_BATCH_EXAMPLES: Dict[str, Any] = {
    "single_time_detection": {
        "summary": "自定义单时间检测模型批量推理",
        "description": "单时间检测模型批量推理只需传入 month。",
        "value": {
            "region_id": "harbin",
            "patch_ids": ["patch_000000", "patch_000001"],
            "month": "2025-04",
        },
    },
    "change_detection": {
        "summary": "自定义变化检测模型批量推理",
        "description": "变化检测模型批量推理需同时传入 before_month 和 after_month，不要传 month。",
        "value": {
            "region_id": "harbin",
            "patch_ids": ["patch_000000", "patch_000001"],
            "before_month": "2025-04",
            "after_month": "2025-06",
        },
    },
    "system_model": {
        "summary": "系统预训练模型批量推理",
        "description": "把系统任务 ID 作为 model_id，传入 region_id、patch_ids、month，可选 version。",
        "value": {
            "region_id": "harbin",
            "patch_ids": ["patch_000000", "patch_000001"],
            "month": "2025-04",
            "version": "v2",
        },
    },
}


@router.post("/{model_id}/infer_batch", response_model=BatchInferResponse)
async def infer_batch(
    model_id: str = PathParam(
        ...,
        description="模型 ID。可以是 POST /models 返回的自定义模型 ID，也可以是系统预训练任务 ID（如 'building_extraction'）。",
        examples=["model_ghi789"],
        openapi_examples={
            "custom": {"summary": "自定义模型", "value": "model_ghi789"},
            "system_building": {"summary": "系统预设-建筑物提取", "value": "building_extraction"},
            "system_land_cover": {"summary": "系统预设-土地覆盖分类", "value": "land_cover_classification"},
            "system_water": {"summary": "系统预设-水体提取", "value": "water_extraction"},
        },
    ),
    req: BatchInferRequest = Body(..., openapi_examples=_INFER_BATCH_EXAMPLES),
    user: dict = Depends(get_current_user),
) -> dict:
    """对最多 100 个 Patch 批量运行模型推理（支持自定义模型和系统预训练模型）。

    用于一次性对多个 Patch 生成预测结果，提高处理效率。
    返回每个 Patch 的推理状态及结果图片 URL。

    """
    registry = get_model_registry(user["user_id"])
    model = registry.get_model(model_id)

    if is_system_task(model_id):
        if not req.month:
            raise HTTPException(
                status_code=422,
                detail="System models require 'month' in the request body",
            )
        results_dir = Path(f"users/{user['user_id']}/system_model_results")
        results = []
        for patch_id in req.patch_ids:
            try:
                result_path = infer_system_model(
                    req.region_id,
                    model_id,
                    patch_id,
                    req.month,
                    version=req.version or "v2",
                    results_dir=results_dir,
                )
                results.append(
                    {
                        "patch_id": patch_id,
                        "status": "success",
                        "result_path": str(result_path),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "patch_id": patch_id,
                        "status": "error",
                        "error": str(e),
                        "result_path": None,
                    }
                )
        items = [
            {
                "patch_id": r["patch_id"],
                "status": r["status"],
                "result_url": f"/system-models/results/{Path(r['result_path']).name}"
                if r.get("result_path")
                else None,
                "error": r.get("error"),
            }
            for r in results
        ]
        return _batch_response(items)

    if not model or model.get("status") != "completed":
        raise HTTPException(
            status_code=400, detail="Model not trained or not found"
        )

    engine = InferenceEngine(user["user_id"])
    results = engine.infer_batch(
        model_id,
        req.region_id,
        req.patch_ids,
        month=req.month,
        before_month=req.before_month,
        after_month=req.after_month,
    )
    items = [
        {
            "patch_id": r["patch_id"],
            "status": r["status"],
            "result_url": f"/models/results/{Path(r['result_path']).name}"
            if r.get("result_path")
            else None,
            "error": r.get("error"),
        }
        for r in results
    ]
    return _batch_response(items)


@router.get("/jobs/{job_id}", response_model=JobStatusOut)
async def get_job_status(
    job_id: str = PathParam(
        ...,
        description="Training job ID returned by POST /models. Replace with the real job ID from the create response.",
        examples=["job_jkl012"],
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """查询训练任务状态。

    用于前端轮询训练进度，显示当前训练阶段、准确率、样本数等信息。
    返回任务状态、模型路径及统计指标的 JSON。
    """
    job = _training_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if user.get("role") != "admin" and job.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "model_id": job["model_id"],
        "accuracy": job.get("accuracy"),
        "n_samples": job.get("n_samples"),
        "model_path": job.get("model_path"),
        "message": job.get("message"),
    }


@router.get("/results/{filename}")
async def get_result(
    filename: str = PathParam(
        ...,
        description="Inference result filename returned by /models/{model_id}/infer or /models/{model_id}/infer_batch.",
        examples=["infer_model_ghi789_harbin_patch_000000_2025-04.png"],
    ),
    user: dict = Depends(get_current_user),
) -> FileResponse:
    """下载自定义模型推理结果图片（PNG）。

    用于在页面中展示自定义模型对 Patch 的预测结果。
    返回 PNG 图片文件；Swagger UI 可能无法直接预览，建议使用 `<img>` 标签或浏览器访问。
    """
    results_dir = InferenceEngine(user["user_id"]).results_dir
    _validate_result_filename(filename)
    file_path = results_dir / filename

    try:
        resolved = file_path.resolve()
        base_resolved = results_dir.resolve()
        resolved.relative_to(base_resolved)
    except (ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(file_path, media_type="image/png")


def _do_training(
    *,
    job_id: str,
    model_id: str,
    user_id: str,
    region_id: str,
    task_type: str,
    model_type: str,
    embedding_version: str,
    epochs: int,
    annotations,
    classes,
    class_ids,
) -> None:
    try:
        if model_type == "change_detection":
            engine = ChangeDetectionTrainingEngine(user_id)
            result = engine.train(
                model_id=model_id,
                region_id=region_id,
                task_type=task_type,
                embedding_version=embedding_version,
                annotations=annotations,
                classes=classes,
                class_ids=class_ids,
                epochs=epochs,
            )
        else:
            engine = ClassificationTrainingEngine(user_id)
            result = engine.train(
                model_id=model_id,
                region_id=region_id,
                task_type=task_type,
                embedding_version=embedding_version,
                annotations=annotations,
                classes=classes,
                class_ids=class_ids,
                epochs=epochs,
            )

        registry = get_model_registry(user_id)
        registry.update_model(
            model_id,
            status="completed",
            completed_at=datetime.now().isoformat(),
            accuracy=result["accuracy"],
            n_samples=result["n_samples"],
        )
        _training_jobs[job_id].update(
            {
                "status": "completed",
                "accuracy": result["accuracy"],
                "n_samples": result["n_samples"],
                "model_path": result["model_path"],
            }
        )
    except Exception as e:
        registry = get_model_registry(user_id)
        registry.update_model(model_id, status="failed", message=str(e))
        _training_jobs[job_id].update(
            {"status": "failed", "message": str(e)}
        )
