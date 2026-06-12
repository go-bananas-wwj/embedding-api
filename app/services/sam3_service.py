"""SAM3 interactive segmentation service."""

import asyncio
import base64
import io
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from app.config import get_config


class SAM3Service:
    """Singleton SAM3 inference service with lazy model loading and LRU cache."""

    _instance: Optional["SAM3Service"] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._init_lock = threading.Lock()
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        with self._init_lock:
            if getattr(self, "_initialized", False):
                return
            self._model = None
            self._processor = None
            self._device: Optional[str] = None
            self._cache: OrderedDict[str, dict] = OrderedDict()
            self._inference_lock: Optional[asyncio.Lock] = None
            self._model_lock = threading.Lock()
            self._cache_lock = threading.Lock()
            self._initialized = True

    def _get_inference_lock(self) -> asyncio.Lock:
        """Lazy-init asyncio lock for Python 3.9 compatibility."""
        if self._inference_lock is None:
            self._inference_lock = asyncio.Lock()
        return self._inference_lock

    def _ensure_model(self):
        """Lazy-load SAM3 model. Thread-safe double-checked locking."""
        if self._model is not None and self._processor is not None:
            return
        with self._model_lock:
            if self._model is not None and self._processor is not None:
                return
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            config = get_config().get_sam3_config()
            checkpoint_path = config.get("model_path", "models/sam3/sam3.pt")
            bpe_path = config.get("bpe_path", "models/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
            device = config.get("device", "cuda")

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            checkpoint_path = str(Path(checkpoint_path).resolve())
            bpe_path = str(Path(bpe_path).resolve())

            try:
                model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    enable_inst_interactivity=True,
                )
                model.to(device)

                # Convert float32 to bfloat16 for autocast compatibility
                for p in model.parameters():
                    if p.dtype == torch.float32:
                        p.data = p.data.to(torch.bfloat16)
                for b in model.buffers():
                    if b.dtype == torch.float32:
                        b.data = b.data.to(torch.bfloat16)

                processor = Sam3Processor(model, device=device)

                # Only assign after full success
                self._device = device
                self._model = model
                self._processor = processor
            except Exception:
                self._device = None
                self._model = None
                self._processor = None
                raise

    def _load_s2_image(self, region_id: str, patch_id: str, month: str) -> Image.Image:
        """Load S2 RGB image for a patch from configured s2_dir.

        Directory structure: {s2_dir}/{patch_id}/{YYYYMMDD}.tif
        Uses rasterio to read multi-band GeoTIFF and normalizes reflectance.
        """
        import rasterio

        config = get_config()
        region = config.get_region(region_id)
        if not region:
            raise ValueError(f"Region '{region_id}' not found")

        s2_dir = region.get("s2_dir")
        if not s2_dir:
            raise ValueError(f"s2_dir not configured for region '{region_id}'")

        s2_dir_path = Path(s2_dir).resolve()
        patch_dir = (s2_dir_path / patch_id).resolve()
        if not str(patch_dir).startswith(str(s2_dir_path)):
            raise ValueError(f"Invalid patch_id: '{patch_id}'")
        if not patch_dir.exists():
            raise FileNotFoundError(f"No S2 image directory found for {patch_id}")

        # Convert month "2025-10" to prefix "202510"
        month_prefix = month.replace("-", "")
        candidates = sorted(patch_dir.glob(f"{month_prefix}*.tif"))
        if not candidates:
            candidates = sorted(patch_dir.glob(f"{month_prefix}*.png"))
        if not candidates:
            raise FileNotFoundError(f"No S2 image found for {patch_id} {month}")

        s2_path = candidates[0]

        with rasterio.open(str(s2_path)) as ds:
            # Windowed read for large images
            if ds.height > 1024 or ds.width > 1024:
                data = ds.read(
                    out_shape=(ds.count, 256, 256),
                    resampling=rasterio.enums.Resampling.lanczos,
                )
            else:
                data = ds.read()  # [C, H, W]

        if data.shape[0] >= 3:
            # Sentinel-2 bands: B02(0), B03(1), B04(2), B08(3), ...
            # Always select B04(R), B03(G), B02(B) = indices [2, 1, 0]
            rgb = data[[2, 1, 0]].astype(np.float32)
        else:
            raise ValueError(f"Not enough bands in {s2_path}")

        rgb = np.clip(rgb / 3500.0, 0, 1)
        rgb = rgb.transpose(1, 2, 0)
        rgb = (rgb * 255).astype(np.uint8)

        img = Image.fromarray(rgb)
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return img

    def _evict_cache_entry(self, embedding_id: str):
        """Remove a cache entry and explicitly free GPU tensors."""
        import torch

        entry = self._cache.pop(embedding_id, None)
        if entry is None:
            return
        state = entry.get("state")
        if state and isinstance(state, dict):
            for k, v in list(state.items()):
                if hasattr(v, "cpu"):
                    try:
                        v.cpu()
                    except Exception:
                        pass
                del state[k]
        del entry
        if self._device and self._device != "cpu":
            torch.cuda.empty_cache()

    async def embed(self, region_id: str, patch_id: str, month: str) -> dict:
        """Load image, compute SAM3 embedding, cache it, return image + embedding_id."""
        import torch

        async with self._get_inference_lock():
            self._ensure_model()
            image = self._load_s2_image(region_id, patch_id, month)

            device = self._device or "cpu"
            try:
                with torch.autocast(device, dtype=torch.bfloat16):
                    state = self._processor.set_image(image)
            except torch.cuda.OutOfMemoryError:
                # Clear cache and retry once
                with self._cache_lock:
                    for key in list(self._cache.keys()):
                        self._evict_cache_entry(key)
                torch.cuda.empty_cache()
                with torch.autocast(device, dtype=torch.bfloat16):
                    state = self._processor.set_image(image)

            embedding_id = f"{region_id}_{patch_id}_{month}"
            with self._cache_lock:
                self._cache[embedding_id] = {
                    "state": state,
                    "shape": (state["original_height"], state["original_width"]),
                }

                # LRU eviction
                max_size = get_config().get_sam3_config().get("max_cache_size", 20)
                max_size = max(1, int(max_size))
                while len(self._cache) > max_size:
                    oldest_id = next(iter(self._cache))
                    self._evict_cache_entry(oldest_id)

            # Encode image to base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return {
                "embedding_id": embedding_id,
                "status": "ready",
                "image": {
                    "width": 256,
                    "height": 256,
                    "format": "png",
                    "data": img_b64,
                },
            }

    async def segment(
        self,
        embedding_id: str,
        point_coords: List[List[float]],
        point_labels: List[int],
        multimask_output: bool = True,
    ) -> List[dict]:
        """Run instance segmentation on cached embedding."""
        import torch

        with self._cache_lock:
            if embedding_id not in self._cache:
                raise ValueError(f"Embedding '{embedding_id}' not found. Call embed first.")

        async with self._get_inference_lock():
            self._ensure_model()
            with self._cache_lock:
                state = self._cache[embedding_id]["state"]
                img_h, img_w = self._cache[embedding_id]["shape"]

            coords = np.array(point_coords) * np.array([[img_w, img_h]])
            labels = np.array(point_labels)

            device = self._device or "cpu"
            try:
                with torch.autocast(device, dtype=torch.bfloat16):
                    masks, scores, _ = self._model.predict_inst(
                        state,
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=multimask_output,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise

            results = []
            for mask, score in zip(masks, scores.tolist()):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                ys, xs = np.where(mask)
                if len(xs) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

                results.append({
                    "data": mask_b64,
                    "score": score,
                    "bbox": bbox,
                })

            return results

    def get_status(self) -> dict:
        """Return model loading status, GPU memory, and cache info."""
        import torch

        model_loaded = self._model is not None and self._processor is not None
        gpu_mem = {"allocated_mb": 0, "reserved_mb": 0}

        if model_loaded and self._device and self._device != "cpu":
            try:
                gpu_mem["allocated_mb"] = torch.cuda.memory_allocated(self._device) // (1024 * 1024)
                gpu_mem["reserved_mb"] = torch.cuda.memory_reserved(self._device) // (1024 * 1024)
            except RuntimeError:
                pass

        with self._cache_lock:
            cache_entries = list(self._cache.keys())
            cache_size = len(self._cache)

        return {
            "model_loaded": model_loaded,
            "device": self._device or "not_loaded",
            "gpu_memory": gpu_mem,
            "cache": {
                "size": cache_size,
                "max_size": get_config().get_sam3_config().get("max_cache_size", 20),
                "entries": cache_entries,
            },
        }
