"""SAM3 interactive segmentation router."""

from fastapi import APIRouter, HTTPException

from app.config import get_config
from app.schemas.sam3 import (
    EmbedRequest,
    EmbedResponse,
    SegmentRequest,
    SegmentResponse,
    StatusResponse,
)
from app.services.sam3_service import SAM3Service

router = APIRouter()


@router.post("/regions/{region_id}/sam3/embed", response_model=EmbedResponse)
async def sam3_embed(region_id: str, req: EmbedRequest):
    """Preload patch image and compute SAM3 embedding."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    try:
        result = await svc.embed(region_id, req.patch_id, req.month)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {e}")


@router.post("/regions/{region_id}/sam3/segment", response_model=SegmentResponse)
async def sam3_segment(region_id: str, req: SegmentRequest):
    """Segment instance using cached embedding and point prompts."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    try:
        masks = await svc.segment(
            req.embedding_id,
            req.point_coords,
            req.point_labels,
            req.multimask_output,
        )
        return {"masks": masks}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Segmentation failed: {e}")


@router.get("/regions/{region_id}/sam3/status", response_model=StatusResponse)
async def sam3_status(region_id: str):
    """Get SAM3 model loading status and cache info."""
    config = get_config()
    if not config.region_exists(region_id):
        raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

    svc = SAM3Service()
    return svc.get_status()
