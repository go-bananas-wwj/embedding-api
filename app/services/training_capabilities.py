"""Machine-readable custom-training capabilities exposed to frontend clients."""

from typing import Any, Dict, Optional
from pathlib import Path
from app.config import get_config
from app.services.external_embeddings import aef_assets_available_for_region, dino_assets_available


def get_training_capabilities(region_id: Optional[str] = None) -> Dict[str, Any]:
    regions = [region_id] if region_id else ["harbin", "haidian"]
    region_cfg = get_config().get_region(region_id) if region_id else None
    s2_available = bool(region_cfg and Path(region_cfg.get("s2_dir", "")).is_dir()) if region_id else True
    aef_available = (
        aef_assets_available_for_region(region_id)
        if region_id
        else any(aef_assets_available_for_region(candidate) for candidate in regions)
    )
    dino_available = dino_assets_available() and s2_available
    return {
        "schema_version": 1,
        "default_training_method": "xuannv_earth",
        "regions": regions,
        "methods": [
            {
                "id": "xuannv_earth",
                "name": "玄女地球训练（默认）",
                "available": s2_available,
                "feature_source": "xuannv_embedding",
                "supported_model_types": ["single_time_detection", "change_detection"],
                "trainer": "auto",
                "selection_rule": "按类别分别统计：某类有效 Polygon 少于 10 个使用 PU + Query，否则使用 Binary Conv 3x3",
                "required_sensor": None,
            },
            {
                "id": "traditional_ml",
                "name": "传统方法训练",
                "available": s2_available,
                "feature_source": "sentinel2_l2a",
                "supported_model_types": ["single_time_detection"],
                "trainer": "random_forest",
                "selection_rule": "仅使用 Sentinel-2 光学波段和光谱指数进行像素级二分类",
                "required_sensor": "s2",
            },
            {
                "id": "aef",
                "name": "AEF 训练",
                "available": aef_available,
                "feature_source": "aef",
                "supported_model_types": ["single_time_detection", "change_detection"],
                "trainer": "pixel_mlp",
                "selection_rule": "AEF 为年度特征；当前任意前端月份均读取 2025 年年度 embedding",
                "unavailable_reason": None if aef_available else "该区域未找到真实 AEF embedding；请配置 AEF_EMBEDDING_DIR/<region_id>",
            },
            {
                "id": "dinov3_sat493m",
                "name": "DINOv3-SAT493M 训练",
                "available": dino_available,
                "feature_source": "dinov3_sat493m",
                "supported_model_types": ["single_time_detection", "change_detection"],
                "trainer": "pixel_mlp",
                "unavailable_reason": None if dino_available else "当前部署缺少 DINOv3-SAT493M ViT-L/16 权重或该区域 S2 影像",
            },
        ],
        "task_contracts": {
            "land_use_classification": {
                "temporal_mode": "single",
                "required_fields": ["month"],
                "description": "单时相土地利用/覆盖类别图；不得传 before_month/after_month。",
            },
            "change_detection": {
                "temporal_mode": "pair",
                "required_fields": ["before_month", "after_month"],
                "description": "双时相变化检测。",
            },
            "land_use_transition": {
                "temporal_mode": "pair",
                "available": False,
                "description": "土地类别转移任务，保留给后续独立接口；不再冒充单期土地利用分类。",
            },
        },
    }
