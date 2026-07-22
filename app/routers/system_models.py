"""System pre-trained model inference routes.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path as PathParam
from fastapi.responses import FileResponse

from app.schemas.models import ErrorResponse
from app.services.auth_service import get_current_user
from app.services.data_service import DataValidationError
from app.services.system_model_service import (
    get_system_model_classes,
    infer_system_model,
    list_system_models,
    resolve_system_model_version,
)
from app.services.user_paths import get_user_dir

router = APIRouter(prefix="/system-models")

_SYSTEM_TASK_OPENAPI_EXAMPLES = {
    "building_extraction": {"summary": "建筑物提取", "value": "building_extraction"},
    "road_extraction": {"summary": "道路提取", "value": "road_extraction"},
    "water_extraction": {"summary": "水体提取", "value": "water_extraction"},
    "land_cover_classification": {"summary": "土地覆盖分类", "value": "land_cover_classification"},
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
        description="区域 ID。可填 `harbin`（哈尔滨）或 `haidian`（海淀）；海淀当前提供建筑物、道路和水体提取。",
        examples=["haidian"],
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
        description="系统模型任务 ID。海淀可填 `building_extraction`、`road_extraction` 或 `water_extraction`。",
        examples=["building_extraction"],
        openapi_examples=_SYSTEM_TASK_OPENAPI_EXAMPLES,
    ),
    region_id: str = Query(
        ...,
        description="区域 ID。可填 `harbin` 或 `haidian`。",
        examples=["haidian"],
    ),
    version: Optional[str] = Query(
        None,
        description="模型版本。可省略；后端自动选择该区域可用的最新版本。海淀当前使用 `v1`。",
        examples=["v1"],
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

    海淀土地利用/土地覆盖 V1 使用预生成月度结果，没有在线推理 checkpoint；
    本接口仍会返回对应静态图例。该例外只表示类别定义可用，不表示对应
    `/system-models/{task_id}/infer` 支持海淀实时推理。
    """
    try:
        if region_id == "haidian" and task_id in {
            "land_cover_classification",
            "land_use_classification",
        }:
            resolved_version = version or "v1"
        else:
            resolved_version = resolve_system_model_version(region_id, task_id, version)
        return get_system_model_classes(region_id, task_id, resolved_version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{task_id}/infer")
async def infer(
    task_id: str = PathParam(
        ...,
        description="系统模型任务 ID。海淀可填 `building_extraction`、`road_extraction` 或 `water_extraction`。",
        examples=["building_extraction"],
        openapi_examples=_SYSTEM_TASK_OPENAPI_EXAMPLES,
    ),
    region_id: str = Query(
        ...,
        description="区域 ID。可填 `harbin` 或 `haidian`；填写值必须与 Patch 所属区域一致。",
        examples=["haidian"],
    ),
    patch_id: str = Query(
        ...,
        description="待推理的 Patch ID，格式为 `patch_` 加 6 位数字，例如 `patch_000000`。",
        examples=["patch_000000"],
    ),
    month: str = Query(
        ...,
        description="P10C embedding 月份，格式为 `YYYYMM`。海淀可用范围为 `202512` 至 `202605`。",
        examples=["202604"],
    ),
    version: Optional[str] = Query(
        None,
        description="模型版本。可省略；后端自动选择该区域可用的最新版本。海淀当前使用 `v1`。",
        examples=["v1"],
        openapi_examples={
            "v1": {"summary": "V4-based checkpoint", "value": "v1"},
            "v2": {"summary": "V5-based checkpoint", "value": "v2"},
        },
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """调用系统预训练模型对单个 Patch 进行推理。

    海淀的建筑物、道路和水体任务使用最新 P10C 64 维 embedding，交由
    Binary Conv 3×3 二分类头推理；不包含变化检测。返回值中的 `result_url`
    是预测 PNG 的下载地址：黑色为背景，彩色像素为目标类别。
    """
    results_dir = get_user_dir(user["user_id"]) / "system_model_results"
    try:
        resolved_version = resolve_system_model_version(region_id, task_id, version)
        result_path = infer_system_model(
            region_id, task_id, patch_id, month, resolved_version, results_dir
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DataValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = result_path.name
    return {"result_url": f"/system-models/results/{filename}"}


@router.get(
    "/results/{filename}",
    response_class=FileResponse,
    responses={
        200: {"description": "PNG 推理结果", "content": {"image/png": {}}},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
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
