"""Annotation and class management routes.

Provides CRUD for user-defined classes and annotations, which feed the
custom training pipeline.
"""

from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Path

from app.schemas.models import (
    AnnotationCreate,
    AnnotationOut,
    ClassCreate,
    ClassOut,
    ClassRenameRequest,
    ErrorResponse,
    StatusOut,
)
from app.services.annotation_service import (
    get_annotation_store,
    get_class_manager,
)
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/annotations", tags=["annotations"])


# ── Classes ──


@router.get("/classes", response_model=List[ClassOut])
async def list_classes(user: dict = Depends(get_current_user)) -> List[dict]:
    """获取当前用户自定义的所有分类类别。

    用于标注面板填充类别下拉框或图例配置。
    返回类别 ID、名称和颜色等信息的 JSON 列表。
    """
    return get_class_manager(user["user_id"]).list_classes()


@router.post("/classes", response_model=ClassOut)
async def create_class(req: ClassCreate, user: dict = Depends(get_current_user)) -> dict:
    """创建一个新的标注类别。

    用于在标注前定义土地覆盖、变化类型等自定义类别。
    返回新创建类别的 ID、名称和颜色。
    """
    return get_class_manager(user["user_id"]).create_class(req.name, req.color)


@router.patch("/classes/{class_id}", response_model=StatusOut)
async def rename_class(
    class_id: str = Path(
        ...,
        description="Class ID returned by POST /annotations/classes. Replace with the real ID from the create response.",
        examples=["cls_abc123"],
    ),
    req: ClassRenameRequest = Body(...),
    user: dict = Depends(get_current_user),
) -> dict:
    """重命名指定类别。

    用于前端修改类别名称，级联保留该类别下所有标注。
    返回操作是否成功的状态 JSON。
    """
    if not get_class_manager(user["user_id"]).rename_class(class_id, req.name):
        raise HTTPException(status_code=404, detail="Class not found")
    return {"status": "ok"}


@router.delete("/classes/{class_id}", response_model=StatusOut)
async def delete_class(
    class_id: str = Path(
        ...,
        description="Class ID returned by POST /annotations/classes. Replace with the real ID from the create response.",
        examples=["cls_abc123"],
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """删除指定类别。

    用于清理不再使用的类别，会同时级联删除该类别下的所有标注。
    返回操作是否成功的状态 JSON。
    """
    mgr = get_class_manager(user["user_id"])
    store = get_annotation_store(user["user_id"])

    # Validate existence before cascading; class may belong to another user.
    if mgr.get_class(class_id) is None:
        raise HTTPException(status_code=404, detail="Class not found")

    for ann in store.list_annotations():
        if ann.get("class_id") == class_id:
            store.delete_annotation(ann["id"])
    mgr.delete_class(class_id)
    return {"status": "ok"}


# ── Annotations ──


@router.get("", response_model=List[AnnotationOut])
async def list_annotations(
    region_id: Optional[str] = Query(
        None,
        description="Filter by region identifier.",
        examples=["harbin"],
    ),
    patch_id: Optional[str] = Query(
        None,
        description="Filter by patch identifier, e.g. patch_000000.",
        examples=["patch_000000"],
    ),
    task_type: Optional[str] = Query(
        None,
        description="Filter by downstream task type, e.g. building_extraction.",
        examples=["building_extraction"],
    ),
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """列出当前用户的所有标注。

    用于在个人标注管理页展示历史标注数据。
    支持按区域、Patch、任务类型过滤，返回 JSON 列表。
    """
    return get_annotation_store(user["user_id"]).list_annotations(
        region_id=region_id, patch_id=patch_id, task_type=task_type
    )


@router.post("", response_model=AnnotationOut)
async def create_annotation(
    req: AnnotationCreate, user: dict = Depends(get_current_user)
) -> dict:
    """创建一条新的标注。

    用于在 Patch 上绘制掩膜、多边形或折线并保存为训练样本。
    返回包含几何、类别和元数据的标注详情 JSON。
    """
    mgr = get_class_manager(user["user_id"])
    classes = mgr.list_classes()
    if req.class_id not in {c["id"] for c in classes}:
        raise HTTPException(
            status_code=400, detail="Class not found. Please create a class first."
        )

    store = get_annotation_store(user["user_id"])
    try:
        return store.create_annotation(
            region_id=req.region_id,
            patch_id=req.patch_id,
            month=req.month,
            class_id=req.class_id,
            geometry=req.geometry,
            task_type=req.task_type,
            score=req.score,
            before_month=req.before_month,
            after_month=req.after_month,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{ann_id}", response_model=AnnotationOut)
async def get_annotation(
    ann_id: str = Path(
        ...,
        description="Annotation ID returned by POST /annotations. Replace with the real ID from the create response.",
        examples=["ann_def456"],
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """获取单条标注详情。

    用于点击标注列表某一行时展示完整信息。
    返回该标注的 JSON 详情。
    """
    ann = get_annotation_store(user["user_id"]).get_annotation(ann_id)
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return ann


@router.delete("/{ann_id}")
async def delete_annotation(
    ann_id: str = Path(
        ...,
        description="Annotation ID returned by POST /annotations. Replace with the real ID from the create response.",
        examples=["ann_def456"],
    ),
    user: dict = Depends(get_current_user),
) -> dict:
    """删除单条标注及其掩膜文件。

    用于撤销误标或清理训练数据。
    返回操作是否成功的状态 JSON。
    """
    if not get_annotation_store(user["user_id"]).delete_annotation(ann_id):
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"status": "ok"}
