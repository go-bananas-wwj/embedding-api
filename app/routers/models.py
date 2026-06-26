"""Custom model training and inference routes.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path as PathParam
from fastapi.responses import FileResponse

from app.schemas.models import (
    BatchInferRequest,
    BatchInferResult,
    InferRequest,
    JobStatusOut,
    ModelCreate,
    ModelOut,
    ModelRenameRequest,
)
from app.services.annotation_service import get_class_manager
from app.services.auth_service import get_current_user
from app.services.data_service import DataValidationError
from app.services.inference_engine import InferenceEngine
from app.services.model_registry import get_model_registry
from app.services.training_engine import (
    ChangeDetectionTrainingEngine,
    ClassificationTrainingEngine,
)

router = APIRouter(prefix="/models", tags=["models"])

# In-memory job tracking. Training jobs are ephemeral; restart clears them.
_training_jobs: dict[str, dict] = {}


@router.get("", response_model=List[ModelOut])
async def list_models(user: dict = Depends(get_current_user)) -> List[dict]:
    """获取当前用户训练好的自定义模型列表。

    用于模型管理页展示模型名称、类型、状态、精度等信息。
    返回 JSON 列表。
    """
    return get_model_registry(user["user_id"]).list_models()


@router.post("", response_model=ModelOut)
async def create_model(
    req: ModelCreate,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
) -> dict:
    """创建模型并启动异步训练任务。

    用于提交基于用户标注的分类或变化检测训练任务。
    返回模型信息及训练任务 ID，可在 `/models/jobs/{job_id}` 轮询进度。
    """
    registry = get_model_registry(user["user_id"])
    mgr = get_class_manager(user["user_id"])
    classes = mgr.list_classes()

    if req.model_type not in ("classification", "change_detection"):
        raise HTTPException(
            status_code=422,
            detail="model_type must be 'classification' or 'change_detection'",
        )

    model_id = registry.create_model(
        name=req.name,
        model_type=req.model_type,
        classes=classes,
        task_type=req.task_type,
    )
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _training_jobs[job_id] = {
        "job_id": job_id,
        "model_id": model_id,
        "status": "running",
        "user_id": user["user_id"],
        "started_at": datetime.now().isoformat(),
    }
    background_tasks.add_task(
        _do_training,
        job_id=job_id,
        model_id=model_id,
        user_id=user["user_id"],
        region_id=req.region_id,
        task_type=req.task_type,
        model_type=req.model_type,
        embedding_version=req.embedding_version,
    )
    model = registry.get_model(model_id)
    model["job_id"] = job_id
    return model


@router.get("/{model_id}", response_model=ModelOut)
async def get_model(
    model_id: str = PathParam(
        ...,
        description="Model ID returned by POST /models. Replace with the real ID from the create response.",
        examples=["model_ghi789"],
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """获取单个自定义模型的状态。

    用于在模型详情页查看训练进度、精度、完成时间等信息。
    返回模型状态的 JSON 详情。
    """
    model = get_model_registry(user["user_id"]).get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.patch("/{model_id}")
async def rename_model(
    model_id: str = PathParam(
        ...,
        description="Model ID returned by POST /models. Replace with the real ID from the create response.",
        examples=["model_ghi789"],
    ),
    req: ModelRenameRequest = Body(...),
    user: dict = Depends(get_current_user),
) -> dict:
    """重命名指定模型。

    用于模型管理页修改显示名称。
    返回操作是否成功的状态 JSON。
    """
    if not get_model_registry(user["user_id"]).rename_model(model_id, req.name):
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "ok"}


@router.delete("/{model_id}")
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


@router.post("/{model_id}/infer")
async def infer(
    model_id: str = PathParam(
        ...,
        description="Model ID returned by POST /models. Replace with the real ID from the create response.",
        examples=["model_ghi789"],
    ),
    req: InferRequest = Body(...),
    user: dict = Depends(get_current_user),
) -> dict:
    """对单个 Patch 运行自定义模型推理。

    用于获取某个 Patch 的预测结果图片，通常在模型训练完成后调用。
    返回结果图片的访问 URL。
    """
    registry = get_model_registry(user["user_id"])
    model = registry.get_model(model_id)
    if not model or model.get("status") != "completed":
        raise HTTPException(
            status_code=400, detail="Model not trained or not found"
        )

    engine = InferenceEngine(user["user_id"])
    try:
        result_path = engine.infer(
            model_id, req.region_id, req.patch_id, req.month
        )
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = Path(result_path).name
    return {"result_url": f"/models/results/{filename}"}


@router.post("/{model_id}/infer_batch", response_model=List[BatchInferResult])
async def infer_batch(
    model_id: str = PathParam(
        ...,
        description="Model ID returned by POST /models. Replace with the real ID from the create response.",
        examples=["model_ghi789"],
    ),
    req: BatchInferRequest = Body(...),
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """对最多 100 个 Patch 批量运行自定义模型推理。

    用于一次性对多个 Patch 生成预测结果，提高处理效率。
    返回每个 Patch 的推理状态及结果图片 URL。
    """
    if len(req.patch_ids) > 100:
        raise HTTPException(
            status_code=422, detail="Batch size exceeds 100 patches"
        )

    registry = get_model_registry(user["user_id"])
    model = registry.get_model(model_id)
    if not model or model.get("status") != "completed":
        raise HTTPException(
            status_code=400, detail="Model not trained or not found"
        )

    engine = InferenceEngine(user["user_id"])
    results = engine.infer_batch(
        model_id, req.region_id, req.patch_ids, req.month
    )
    return [
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
) -> None:
    try:
        if model_type == "change_detection":
            engine = ChangeDetectionTrainingEngine(user_id)
            result = engine.train(model_id, region_id, embedding_version)
        else:
            engine = ClassificationTrainingEngine(user_id)
            result = engine.train(
                model_id, region_id, task_type, embedding_version
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
