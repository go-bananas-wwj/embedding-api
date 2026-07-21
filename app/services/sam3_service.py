"""SAM3 interactive segmentation service."""

import asyncio
import base64
import io
import logging
import os
import re
import sys
import threading
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from app.config import get_config
from app.services.data_service import DataNotFoundError, DataValidationError
from app.services.mosaic_service import RAW_ROOT, _SENSOR_RGB, _get_raw_tiff_path, _to_rgb
from app.services.time_utils import normalize_quarter_date

logger = logging.getLogger(__name__)


def _configure_rasterio_proj_data() -> None:
    """Use the PROJ database bundled with the installed Rasterio build."""
    import rasterio
    from rasterio._env import set_proj_data_search_path

    bundled = Path(rasterio.__file__).resolve().parent / "proj_data"
    if (bundled / "proj.db").is_file():
        os.environ["PROJ_DATA"] = str(bundled)
        set_proj_data_search_path(str(bundled))


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

    @staticmethod
    def _cache_date_key(date: str) -> str:
        """Canonicalize equivalent month strings for stable cache keys."""
        if re.fullmatch(r"\d{8}", date):
            return date
        if re.fullmatch(r"\d{6}", date):
            return date
        match = re.fullmatch(r"(\d{4})-(\d{2})", date)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return date

    @staticmethod
    def _transform_coords(
        src_crs: Any,
        dst_crs: Any,
        xs: List[float],
        ys: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Transform coordinates, falling back to pyproj if rasterio's PROJ DB is stale."""
        from rasterio.warp import transform as warp_transform

        _configure_rasterio_proj_data()
        try:
            return warp_transform(src_crs, dst_crs, xs, ys)
        except Exception as exc:
            logger.warning(
                "rasterio coordinate transform failed; falling back to pyproj: %s",
                exc,
            )
            try:
                from pyproj import Transformer

                src = src_crs.to_wkt() if hasattr(src_crs, "to_wkt") else src_crs
                dst = dst_crs.to_wkt() if hasattr(dst_crs, "to_wkt") else dst_crs
                transformer = Transformer.from_crs(src, dst, always_xy=True)
                out_xs, out_ys = transformer.transform(xs, ys)
                return list(out_xs), list(out_ys)
            except Exception as fallback_exc:
                raise DataValidationError(
                    "Unable to transform between image CRS and WGS84"
                ) from fallback_exc

    def _ensure_model(self):
        """Lazy-load SAM3 model. Thread-safe double-checked locking."""
        if self._model is not None and self._processor is not None:
            return
        with self._model_lock:
            if self._model is not None and self._processor is not None:
                return
            import torch
            project_root = Path(__file__).resolve().parents[2]
            sam3_pkg = project_root / "sam3_pkg"
            if sam3_pkg.exists() and str(sam3_pkg) not in sys.path:
                sys.path.insert(0, str(sam3_pkg))
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
                    enable_inst_interactivity=bool(
                        config.get("enable_inst_interactivity", True)
                    ),
                )
                model.to(device)

                if torch.device(device).type == "cuda":
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

    def _autocast_context(self):
        """Use CUDA bfloat16 without passing an indexed device to autocast."""
        import torch

        device = torch.device(self._device or "cpu")
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _find_patch_for_points(
        self,
        region_id: str,
        point_coords: List[List[float]],
    ) -> Dict[str, Any]:
        """Find the patch containing all WGS84 point prompts."""
        config = get_config()
        patches = config.get_patches(region_id)
        if not patches:
            raise DataNotFoundError(f"No patches found for region '{region_id}'")

        matching_patch: Optional[Dict[str, Any]] = None
        for lon, lat in point_coords:
            current = None
            for patch in patches:
                bounds = patch.get("bounds_wgs84")
                if not bounds or len(bounds) != 4:
                    continue
                minx, miny, maxx, maxy = bounds
                if minx <= lon <= maxx and miny <= lat <= maxy:
                    current = patch
                    break
            if current is None:
                raise DataNotFoundError(
                    f"Point [{lon}, {lat}] is outside {region_id} patch coverage"
                )
            if matching_patch is None:
                matching_patch = current
            elif current.get("patch_id") != matching_patch.get("patch_id"):
                raise DataValidationError(
                    "All SAM3 prompt points must fall inside the same patch"
                )

        if matching_patch is None:
            raise DataValidationError("At least one prompt point is required")
        return matching_patch

    def _resolve_image_path(
        self,
        region_id: str,
        patch_id: str,
        date: str,
        sensor_type: str,
    ) -> str:
        """Resolve raw satellite TIFF path for a patch/date/sensor."""
        config = get_config()
        region = config.get_region(region_id)
        if not region:
            raise DataValidationError(f"Region '{region_id}' not found")

        sensor_type = sensor_type.lower()
        if sensor_type not in _SENSOR_RGB:
            raise DataValidationError(
                f"sensor_type '{sensor_type}' is not supported; "
                f"use one of {list(_SENSOR_RGB.keys())}"
            )

        periods = normalize_quarter_date(date)
        if not periods:
            raise DataValidationError(f"Invalid date format: '{date}'")

        if sensor_type == "highres" and region.get("highres_dir"):
            flat_path = self._resolve_flat_highres_path(
                Path(region["highres_dir"]), patch_id, date
            )
            if flat_path:
                return flat_path

        roots = [RAW_ROOT]
        s2_dir = region.get("s2_dir")
        if s2_dir:
            roots.append(s2_dir)

        path = _get_raw_tiff_path(region_id, patch_id, sensor_type, periods, roots=roots)
        if not path:
            raise DataNotFoundError(
                f"No raw {sensor_type} image found for {region_id}/{patch_id}/{date}"
            )
        return path

    @staticmethod
    def _resolve_flat_highres_path(
        highres_dir: Path, patch_id: str, date: str
    ) -> Optional[str]:
        """Resolve flat ``highres_optical_DATE_PATCH.tif`` archives."""
        if not highres_dir.exists() or not highres_dir.is_dir():
            return None
        requested = date.replace("-", "")
        if not re.fullmatch(r"\d{6}(?:\d{2})?", requested):
            return None

        matches = []
        try:
            candidates = highres_dir.glob(f"*_{patch_id}.tif")
            for path in candidates:
                date_match = re.search(r"(?<!\d)(\d{8})(?!\d)", path.stem)
                if not date_match:
                    continue
                image_date = date_match.group(1)
                if image_date == requested or (
                    len(requested) == 6 and image_date.startswith(requested)
                ):
                    matches.append((image_date, path))
        except OSError:
            return None
        if not matches:
            return None
        return str(max(matches, key=lambda item: (item[0], item[1].name))[1])

    def _load_geo_image(
        self,
        region_id: str,
        patch_id: str,
        date: str,
        sensor_type: str = "s2",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Load a georeferenced image as RGB for SAM3 and retain transform metadata."""
        import rasterio
        from rasterio.enums import Resampling

        _configure_rasterio_proj_data()
        image_path = self._resolve_image_path(region_id, patch_id, date, sensor_type)
        sam3_config = get_config().get_sam3_config()

        with rasterio.open(str(image_path)) as ds:
            if ds.crs is None:
                raise DataValidationError(
                    f"Source image '{image_path}' has no CRS; SAM3 WGS84 prompts require a georeferenced image"
                )
            max_side = int(
                sam3_config.get("highres_image_size", 1024)
                if sensor_type == "highres"
                else sam3_config.get("image_size", 256)
            )
            if max_side < 64:
                raise DataValidationError("SAM3 image size must be at least 64 pixels")
            scale = min(1.0, float(max_side) / float(max(ds.width, ds.height)))
            target_width = max(1, int(round(ds.width * scale)))
            target_height = max(1, int(round(ds.height * scale)))
            source_scene = Path(image_path).stem
            date_match = re.search(r"(?<!\d)(\d{8})(?!\d)", source_scene)
            source_image_date = date_match.group(1) if date_match else source_scene
            data = ds.read(
                out_shape=(ds.count, target_height, target_width),
                resampling=Resampling.lanczos,
            )
            rgba = _to_rgb(data.astype(np.float32), sensor_type)
            img = Image.fromarray(rgba[:, :, :3])
            meta = {
                "image_path": image_path,
                "source_width": ds.width,
                "source_height": ds.height,
                "source_scene": source_scene,
                "source_image_date": source_image_date,
                "sam_width": target_width,
                "sam_height": target_height,
                "transform": ds.transform,
                "crs": ds.crs,
                "patch_id": patch_id,
                "date": date,
                "sensor_type": sensor_type,
            }
        return img, meta

    def _load_s2_image(self, region_id: str, patch_id: str, month: str) -> Image.Image:
        """Compatibility wrapper for legacy embed tests and clients."""
        image, _ = self._load_geo_image(region_id, patch_id, month, sensor_type="s2")
        return image

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

    async def embed(
        self,
        region_id: str,
        patch_id: str,
        month: str,
        sensor_type: str = "s2",
    ) -> dict:
        """Load image, compute SAM3 embedding, cache it, return image + embedding_id."""
        import torch

        async with self._get_inference_lock():
            self._ensure_model()
            image, meta = self._load_geo_image(region_id, patch_id, month, sensor_type)
            cache_date = self._cache_date_key(month)
            meta["date"] = cache_date

            try:
                with self._autocast_context():
                    state = self._processor.set_image(image)
            except torch.cuda.OutOfMemoryError:
                # Clear cache and retry once
                with self._cache_lock:
                    for key in list(self._cache.keys()):
                        self._evict_cache_entry(key)
                torch.cuda.empty_cache()
                with self._autocast_context():
                    state = self._processor.set_image(image)

            embedding_id = f"{region_id}_{patch_id}_{sensor_type}_{cache_date}"
            with self._cache_lock:
                if embedding_id in self._cache:
                    self._evict_cache_entry(embedding_id)
                self._cache[embedding_id] = {
                    "state": state,
                    "shape": (state["original_height"], state["original_width"]),
                    "meta": meta,
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
                "source_scene": meta.get("source_scene"),
                "selected_image_date": meta.get("source_image_date"),
                "image": {
                    "width": meta["sam_width"],
                    "height": meta["sam_height"],
                    "format": "png",
                    "data": img_b64,
                },
            }

    async def _predict_from_cache(
        self,
        embedding_id: str,
        point_coords: List[List[float]],
        point_labels: List[int],
        multimask_output: bool = True,
    ) -> Tuple[List[Tuple[np.ndarray, float, List[int]]], Dict[str, Any]]:
        """Run model prediction against a cached SAM3 state."""
        async with self._get_inference_lock():
            self._ensure_model()
            with self._cache_lock:
                entry = self._cache.get(embedding_id)
                if entry is None:
                    raise ValueError(
                        f"Embedding '{embedding_id}' not found. Call embed first."
                    )
                self._cache.move_to_end(embedding_id)
                state = entry["state"]
                img_h, img_w = entry["shape"]
                meta = entry.get("meta", {})

            coords = np.array(point_coords) * np.array([[img_w, img_h]])
            results = self._predict_state(
                state, coords, point_labels, multimask_output
            )
            return results, meta

    async def _predict_wgs84_from_cache(
        self,
        embedding_id: str,
        point_coords: List[List[float]],
        point_labels: List[int],
        multimask_output: bool,
    ) -> Tuple[List[Tuple[np.ndarray, float, List[int]]], Dict[str, Any]]:
        """Atomically read cache metadata, convert coordinates, and predict."""
        async with self._get_inference_lock():
            self._ensure_model()
            with self._cache_lock:
                entry = self._cache.get(embedding_id)
                if entry is None:
                    raise ValueError(
                        f"Embedding '{embedding_id}' not found. Call embed first."
                    )
                self._cache.move_to_end(embedding_id)
                state = entry["state"]
                img_h, img_w = entry["shape"]
                meta = entry.get("meta", {})
            normalized_coords = self._wgs84_to_sam_pixels(meta, point_coords)
            pixel_coords = np.array(normalized_coords) * np.array([[img_w, img_h]])
            results = self._predict_state(
                state, pixel_coords, point_labels, multimask_output
            )
            return results, meta

    def _predict_state(
        self,
        state: Dict[str, Any],
        pixel_coords: np.ndarray,
        point_labels: List[int],
        multimask_output: bool,
    ) -> List[Tuple[np.ndarray, float, List[int]]]:
        """Run SAM3 prediction while the caller holds the inference lock."""
        import torch

        try:
            with self._autocast_context():
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=pixel_coords,
                    point_labels=np.array(point_labels),
                    multimask_output=multimask_output,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

        results = []
        for mask, score in zip(masks, scores.tolist()):
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask).astype(bool)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                bbox = [0, 0, 0, 0]
            else:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                bbox = [
                    int(x_min),
                    int(y_min),
                    int(x_max - x_min + 1),
                    int(y_max - y_min + 1),
                ]
            results.append((mask, float(score), bbox))
        return results

    def _wgs84_to_sam_pixels(
        self,
        meta: Dict[str, Any],
        point_coords: List[List[float]],
    ) -> List[List[float]]:
        """Convert WGS84 point prompts to SAM image pixel coordinates."""
        from rasterio.transform import rowcol

        crs = meta.get("crs")
        if crs is None:
            raise DataValidationError("Source image has no CRS; cannot use WGS84 prompts")

        lons = [p[0] for p in point_coords]
        lats = [p[1] for p in point_coords]
        xs, ys = self._transform_coords("EPSG:4326", crs, lons, lats)
        rows, cols = rowcol(meta["transform"], xs, ys)

        coords = []
        for row, col in zip(rows, cols):
            if row < 0 or col < 0 or row >= meta["source_height"] or col >= meta["source_width"]:
                raise DataValidationError("Prompt point is outside the source image")
            x = float(col) * float(meta["sam_width"]) / float(meta["source_width"])
            y = float(row) * float(meta["sam_height"]) / float(meta["source_height"])
            coords.append([x / float(meta["sam_width"]), y / float(meta["sam_height"])])
        return coords

    def _bbox_to_feature(
        self,
        bbox: List[int],
        score: float,
        meta: Dict[str, Any],
        index: int,
    ) -> Tuple[Dict[str, Any], List[float]]:
        """Convert SAM pixel bbox to a WGS84 GeoJSON polygon feature."""
        x, y, w, h = bbox
        scale_x = float(meta["source_width"]) / float(meta["sam_width"])
        scale_y = float(meta["source_height"]) / float(meta["sam_height"])
        col0 = x * scale_x
        row0 = y * scale_y
        col1 = (x + w) * scale_x
        row1 = (y + h) * scale_y

        corners_rc = [
            (row0, col0),
            (row0, col1),
            (row1, col1),
            (row1, col0),
            (row0, col0),
        ]
        native_xs = []
        native_ys = []
        for row, col in corners_rc:
            px, py = meta["transform"] * (col, row)
            native_xs.append(px)
            native_ys.append(py)

        crs = meta.get("crs")
        if crs is None:
            raise DataValidationError("Source image has no CRS; cannot return WGS84 boxes")
        lons, lats = self._transform_coords(crs, "EPSG:4326", native_xs, native_ys)
        ring = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]
        bbox_wgs84 = [
            min(p[0] for p in ring),
            min(p[1] for p in ring),
            max(p[0] for p in ring),
            max(p[1] for p in ring),
        ]
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "score": score,
                "bbox": bbox,
                "bbox_wgs84": bbox_wgs84,
                "patch_id": meta.get("patch_id"),
                "sensor_type": meta.get("sensor_type"),
                "date": meta.get("date"),
                "source_scene": meta.get("source_scene"),
                "selected_image_date": meta.get("source_image_date"),
                "candidate_index": index,
                "geometry_kind": "bbox",
            },
        }
        return feature, bbox_wgs84

    def _mask_to_feature(
        self,
        mask: np.ndarray,
        bbox: List[int],
        score: float,
        meta: Dict[str, Any],
        index: int,
    ) -> Tuple[Dict[str, Any], List[float]]:
        """Convert a SAM mask to WGS84 GeoJSON polygon geometry.

        The previous implementation returned the mask's bounding box as a
        rectangular Polygon. Frontend tools need the actual segmentation
        outline, so this vectorizes mask pixels and transforms every ring to
        WGS84.
        """
        from affine import Affine
        from rasterio.features import shapes

        crs = meta.get("crs")
        if crs is None:
            raise DataValidationError("Source image has no CRS; cannot return WGS84 polygon")

        mask_u8 = mask.astype(np.uint8)
        scale_x = float(meta["source_width"]) / float(meta["sam_width"])
        scale_y = float(meta["source_height"]) / float(meta["sam_height"])
        sam_to_native = meta["transform"] * Affine.scale(scale_x, scale_y)

        polygons = []
        native_bounds = []
        for geom, value in shapes(mask_u8, mask=mask_u8.astype(bool), transform=sam_to_native):
            if int(value) != 1:
                continue
            coords = geom.get("coordinates") or []
            if geom.get("type") != "Polygon" or not coords:
                continue
            wgs84_rings = []
            for ring in coords:
                if len(ring) < 4:
                    continue
                xs = [float(pt[0]) for pt in ring]
                ys = [float(pt[1]) for pt in ring]
                lons, lats = self._transform_coords(crs, "EPSG:4326", xs, ys)
                wgs84_ring = [[float(lon), float(lat)] for lon, lat in zip(lons, lats)]
                if len(wgs84_ring) >= 4:
                    wgs84_rings.append(wgs84_ring)
                    native_bounds.extend(wgs84_ring)
            if wgs84_rings:
                polygons.append(wgs84_rings)

        if not polygons:
            # Defensive fallback for an empty or unvectorizable mask.
            return self._bbox_to_feature(bbox, score, meta, index)

        bbox_wgs84 = [
            min(p[0] for p in native_bounds),
            min(p[1] for p in native_bounds),
            max(p[0] for p in native_bounds),
            max(p[1] for p in native_bounds),
        ]
        if len(polygons) == 1:
            geometry = {"type": "Polygon", "coordinates": polygons[0]}
        else:
            geometry = {"type": "MultiPolygon", "coordinates": polygons}

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "score": score,
                "bbox": bbox,
                "bbox_wgs84": bbox_wgs84,
                "patch_id": meta.get("patch_id"),
                "sensor_type": meta.get("sensor_type"),
                "date": meta.get("date"),
                "source_scene": meta.get("source_scene"),
                "selected_image_date": meta.get("source_image_date"),
                "candidate_index": index,
                "geometry_kind": "mask_polygon",
            },
        }
        return feature, bbox_wgs84

    async def segment_geojson(
        self,
        region_id: str,
        date: str,
        sensor_type: str,
        point_coords: List[List[float]],
        point_labels: List[int],
        multimask_output: bool = True,
        include_masks: bool = False,
    ) -> dict:
        """Segment from WGS84 point prompts and return WGS84 GeoJSON polygons."""
        patch = self._find_patch_for_points(region_id, point_coords)
        patch_id = patch["patch_id"]
        cache_date = self._cache_date_key(date)
        embedding_id = f"{region_id}_{patch_id}_{sensor_type}_{cache_date}"

        with self._cache_lock:
            cached = embedding_id in self._cache
        if not cached:
            await self.embed(region_id, patch_id, cache_date, sensor_type=sensor_type)

        predictions, meta = await self._predict_wgs84_from_cache(
            embedding_id,
            point_coords,
            point_labels,
            multimask_output,
        )

        features = []
        masks_payload = []
        for idx, (mask, score, bbox) in enumerate(predictions):
            feature, bbox_wgs84 = self._mask_to_feature(mask, bbox, score, meta, idx)
            features.append(feature)
            if include_masks:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                masks_payload.append(
                    {
                        "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                        "score": score,
                        "bbox": bbox,
                        "bbox_wgs84": bbox_wgs84,
                    }
                )

        response = {"type": "FeatureCollection", "features": features}
        if include_masks:
            response["masks"] = masks_payload
        return response

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
