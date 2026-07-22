"""Install real AlphaEarth Foundations embeddings from Source Cooperative COGs."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/external_embeddings/aef/_source/aef_index.parquet"
OUTPUT_ROOT = ROOT / "data/external_embeddings/aef"
SOURCE_PREFIX = "https://data.source.coop/tge-labs/aef/"


def dequantize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return np.square(values / 127.5) * np.sign(values)


def source_url(path: str) -> str:
    return SOURCE_PREFIX + path.split("/tge-labs/aef/", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=["harbin", "haidian"], required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--patch-id", action="append", default=[])
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    metadata_path = ROOT / config["regions"][args.region]["patches_meta"]
    patches = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.patch_id:
        wanted = set(args.patch_id)
        patches = [patch for patch in patches if patch["patch_id"] in wanted]
    if not patches:
        raise SystemExit("No matching patches")

    table = pq.read_table(
        INDEX,
        columns=["path", "year", "crs", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"],
        filters=[("year", "=", args.year)],
    ).to_pandas()
    west = min(p["bounds_wgs84"][0] for p in patches)
    south = min(p["bounds_wgs84"][1] for p in patches)
    east = max(p["bounds_wgs84"][2] for p in patches)
    north = max(p["bounds_wgs84"][3] for p in patches)
    table = table[
        (table.wgs84_east >= west) & (table.wgs84_west <= east)
        & (table.wgs84_north >= south) & (table.wgs84_south <= north)
    ]
    if table.empty:
        raise SystemExit(f"No Source Cooperative AEF tiles cover {args.region}/{args.year}")

    output_dir = OUTPUT_ROOT / args.region / str(args.year)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": "AlphaEarth Foundations Satellite Embedding Dataset",
        "source": "https://source.coop/tge-labs/aef",
        "source_version": "v1/annual",
        "year": args.year,
        "channels": 64,
        "quantization": "dequantized with ((value / 127.5) ** 2) * sign(value)",
        "output_dtype": "float16",
        "patch_grid": "API patch bounds, 128x128 at 10 m",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tiff")
    os.environ.setdefault("GDAL_CACHEMAX", "1024")
    with ExitStack() as stack:
        datasets: dict[str, rasterio.DatasetReader] = {}
        for index, patch in enumerate(patches, start=1):
            destination = output_dir / f"{patch['patch_id']}.npy"
            if destination.is_file():
                continue
            w, s, e, n = patch["bounds_wgs84"]
            candidates = table[
                (table.wgs84_east >= w) & (table.wgs84_west <= e)
                & (table.wgs84_north >= s) & (table.wgs84_south <= n)
                & (table.crs == patch["crs"])
            ]
            if candidates.empty:
                raise RuntimeError(f"No AEF tile intersects {patch['patch_id']}")
            merged = np.full((64, 128, 128), -128, dtype=np.int8)
            for row in candidates.itertuples():
                url = source_url(row.path)
                if url not in datasets:
                    datasets[url] = stack.enter_context(rasterio.open(url))
                with WarpedVRT(
                    datasets[url], crs=patch["crs"],
                    transform=from_bounds(*patch["bounds"], 128, 128),
                    width=128, height=128, resampling=Resampling.nearest, nodata=-128,
                ) as vrt:
                    current = vrt.read()
                valid = current != -128
                merged[valid] = current[valid]
            if np.any(merged == -128):
                raise RuntimeError(
                    f"AEF patch {patch['patch_id']} has {np.mean(merged == -128):.2%} nodata"
                )
            value = dequantize(merged).astype(np.float16)
            temporary = destination.with_suffix(".tmp.npy")
            np.save(temporary, value)
            temporary.replace(destination)
            print(f"[{index}/{len(patches)}] {args.region}/{args.year}/{patch['patch_id']} {value.shape}", flush=True)


if __name__ == "__main__":
    main()
