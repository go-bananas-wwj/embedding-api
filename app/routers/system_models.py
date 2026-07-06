"""System pre-trained model inference routes.
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Path as PathParam
from fastapi.responses import FileResponse

from app.schemas.models import ErrorResponse
from app.services.auth_service import get_current_user
from app.services.data_service import DataValidationError
from app.services.system_model_service import (
    get_system_model_classes,
    infer_system_model,
    list_system_models,
)
from app.services.user_paths import get_user_dir

router = APIRouter(prefix="/system-models", tags=["system-models"])

_SYSTEM_TASK_OPENAPI_EXAMPLES = {
    "building_extraction": {"summary": "Building extraction", "value": "building_extraction"},
    "land_cover_classification": {"summary": "Land cover classification", "value": "land_cover_classification"},
    "water_extraction": {"summary": "Water extraction", "value": "water_extraction"},
}


def _validate_result_filename(filename: str) -> None:
    if (
        "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or not filename.endswith(".png")
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")


@router.get("")
async def get_system_models(
    region_id: str = Query(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """列出某区域可用的系统预训练模型。

    用于前端展示可选的官方模型列表，如建筑物提取、土地覆盖分类等。
    返回模型任务及版本信息的 JSON 列表。
    """
    return list_system_models(region_id)


@router.get("/{task_id}/classes")
async def get_classes(
    task_id: str = PathParam(
        ...,
        description="System model task identifier.",
        examples=["building_extraction"],
        openapi_examples=_SYSTEM_TASK_OPENAPI_EXAMPLES,
    ),
    region_id: str = Query(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    version: str = Query(
        "v2",
        description="Model checkpoint version. Allowed values: v1, v2.",
        examples=["v2"],
        openapi_examples={
            "v1": {"summary": "V4-based checkpoint", "value": "v1"},
            "v2": {"summary": "V5-based checkpoint", "value": "v2"},
        },
    ),
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """获取系统预训练模型的类别定义。

    用于在结果可视化时生成图例，或在前端显示类别名称与颜色对应关系。
    返回类别 ID、名称和调色板等 JSON 列表。
    """
    try:
        return get_system_model_classes(region_id, task_id, version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{task_id}/infer")
async def infer(
    task_id: str = PathParam(
        ...,
        description="System model task identifier.",
        examples=["building_extraction"],
        openapi_examples=_SYSTEM_TASK_OPENAPI_EXAMPLES,
    ),
    region_id: str = Query(
        ...,
        description="Region identifier. Use 'harbin' or 'haidian'.",
        examples=["harbin"],
    ),
    patch_id: str = Query(
        ...,
        description="Patch identifier in the form patch_000000.",
        examples=["patch_000000"],
    ),
    month: str = Query(
        ...,
        description="Month for the source embedding, e.g. 2025-04.",
        examples=["2025-04"],
    ),
    version: str = Query(
        "v2",
        description="Model checkpoint version. Allowed values: v1, v2.",
        examples=["v2"],
        openapi_examples={
            "v1": {"summary": "V4-based checkpoint", "value": "v1"},
            "v2": {"summary": "V5-based checkpoint", "value": "v2"},
        },
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """调用系统预训练模型对单个 Patch 进行推理。

    用于快速获取建筑物提取、水体提取、土地覆盖分类等官方结果。
    返回结果图片的访问 URL。
    """
    results_dir = get_user_dir(user["user_id"]) / "system_model_results"
    try:
        result_path = infer_system_model(
            region_id, task_id, patch_id, month, version, results_dir
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = result_path.name
    return {"result_url": f"/system-models/results/{filename}"}


@router.get("/results/{filename}")
async def get_result(
    filename: str = PathParam(
        ...,
        description="System model inference result filename returned by POST /system-models/{task_id}/infer.",
        examples=["building_extraction_harbin_patch_000000_2025-04.png"],
    ),
    user: dict = Depends(get_current_user),
) -> FileResponse:
    """下载系统模型推理结果图片（PNG）。

    用于在页面中展示官方预训练模型对 Patch 的预测结果。
    返回 PNG 图片文件；Swagger UI 可能无法直接预览，建议使用 `<img>` 标签或浏览器访问。
    """
    results_dir = get_user_dir(user["user_id"]) / "system_model_results"
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
