"""Annotation and class management routes.

Provides CRUD for user-defined classes and annotations, which feed the
custom training pipeline.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.models import (
    AnnotationCreate,
    AnnotationOut,
    ClassCreate,
    ClassOut,
    ClassRenameRequest,
    ErrorResponse,
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
    """List user-defined classification classes."""
    return get_class_manager(user["user_id"]).list_classes()


@router.post("/classes", response_model=ClassOut)
async def create_class(req: ClassCreate, user: dict = Depends(get_current_user)) -> dict:
    """Create a new class for annotations."""
    return get_class_manager(user["user_id"]).create_class(req.name, req.color)


@router.patch("/classes/{class_id}")
async def rename_class(
    class_id: str, req: ClassRenameRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Rename an existing class."""
    if not get_class_manager(user["user_id"]).rename_class(class_id, req.name):
        raise HTTPException(status_code=404, detail="Class not found")
    return {"status": "ok"}


@router.delete("/classes/{class_id}")
async def delete_class(class_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Delete a class and cascade-delete its annotations."""
    mgr = get_class_manager(user["user_id"])
    store = get_annotation_store(user["user_id"])
    for ann in store.list_annotations():
        if ann.get("class_id") == class_id:
            store.delete_annotation(ann["id"])
    mgr.delete_class(class_id)
    return {"status": "ok"}


# ── Annotations ──


@router.get("", response_model=List[AnnotationOut])
async def list_annotations(
    region_id: Optional[str] = None,
    patch_id: Optional[str] = None,
    task_type: Optional[str] = None,
    user: dict = Depends(get_current_user),
) -> List[dict]:
    """List annotations, optionally filtered by region/patch/task."""
    return get_annotation_store(user["user_id"]).list_annotations(
        region_id=region_id, patch_id=patch_id, task_type=task_type
    )


@router.post("", response_model=AnnotationOut)
async def create_annotation(
    req: AnnotationCreate, user: dict = Depends(get_current_user)
) -> dict:
    """Create a new annotation from a mask, polygon, or polyline geometry."""
    mgr = get_class_manager(user["user_id"])
    classes = mgr.list_classes()
    if req.class_id not in {c["id"] for c in classes}:
        raise HTTPException(
            status_code=400, detail="Class not found. Please create a class first."
        )

    store = get_annotation_store(user["user_id"])
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


@router.get("/{ann_id}", response_model=AnnotationOut)
async def get_annotation(ann_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Get a single annotation by ID."""
    ann = get_annotation_store(user["user_id"]).get_annotation(ann_id)
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return ann


@router.delete("/{ann_id}")
async def delete_annotation(
    ann_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Delete an annotation and its mask file."""
    if not get_annotation_store(user["user_id"]).delete_annotation(ann_id):
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"status": "ok"}
