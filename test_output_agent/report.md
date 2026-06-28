# Embedding API 接口测试报告

**测试时间**: 2026-06-28 16:38:14
**基地址**: http://localhost:9061

## 总结

| 指标 | 数量 |
|------|------|
| 总测试数 | 101 |
| ✅ 通过 | 97 |
| ⚠️ 失败（可能预期） | 4 |
| ❌ 异常（非预期） | 0 |

**整体状态**: 🟢 健康（所有接口行为符合预期）

## ✅ 通过列表

| # | 接口 | URL | 状态码 | 详情 |
|---|------|-----|--------|------|
| 1 | GET /health | `/health` | 200 | status=ok, version=0.1.0, regions=['harbin', 'haidian'] |
| 2 | GET /regions | `/regions` | 200 | regions=['harbin', 'haidian'] |
| 3 | GET /regions/harbin | `/regions/harbin` | 200 | patch_count=424, tasks=['change_detection', 'building_extraction', 'land_use_classification', 'land_cover_classification', 'water_extraction'], embeddings=['v1', 'v2'] |
| 4 | GET /regions/haidian | `/regions/haidian` | 200 | patch_count=320, tasks=['change_detection', 'building_extraction', 'land_use_classification', 'land_cover_classification', 'water_extraction'], embeddings=['v1'] |
| 5 | GET /regions/harbin/patches?page=1&page_size=5 | `/regions/harbin/patches?page=1&page_size=5` | 200 | total=424, patches=['patch_000000', 'patch_000001', 'patch_000002', 'patch_000003', 'patch_000004'] |
| 6 | GET /regions/harbin/patches?bbox=... | `/regions/harbin/patches?page=1&page_size=5&bbox=126.5,45.74,126.55,45.76` | 200 | total=7, patches=['patch_000000', 'patch_000001', 'patch_000002', 'patch_000007', 'patch_000008'] |
| 7 | GET /regions/harbin/patches/patch_000000 | `/regions/harbin/patches/patch_000000` | 200 | has_embedding=True, available_tasks=['land_use_classification', 'building_extraction'] |
| 8 | GET /regions/harbin/patches/patch_000010 | `/regions/harbin/patches/patch_000010` | 200 | has_embedding=True, available_tasks=['land_use_classification', 'building_extraction'] |
| 9 | GET /regions/harbin/patches/patch_000000/embedding?format=png | `/regions/harbin/patches/patch_000000/embedding?format=png` | 200 | 64x64 mode=RGB |
| 10 | GET /regions/harbin/patches/patch_000000/embedding?format=json | `/regions/harbin/patches/patch_000000/embedding?format=json` | 200 | shape=[64, 64, 3], dtype=uint8 |
| 11 | GET /regions/harbin/patches/patch_000000/embedding?format=npy | `/regions/harbin/patches/patch_000000/embedding?format=npy` | 200 | shape=(128, 64, 64), dtype=float32, size=524288 |
| 12 | GET /regions/harbin/patches/patch_000000/embedding?format=cache | `/regions/harbin/patches/patch_000000/embedding?format=cache` | 200 | 64x64 mode=RGB |
| 13 | GET /regions/harbin/patches/patch_000000/embedding?format=invalid | `/regions/harbin/patches/patch_000000/embedding?format=invalid` | 422 | {"detail":"Invalid format 'invalid'. Use: png, npy, json, cache"} |
| 14 | GET /regions/haidian/patches/patch_000000/embedding?format=png | `/regions/haidian/patches/patch_000000/embedding?format=png` | 404 | {"detail":"PNG format not pre-generated for this patch"} |
| 15 | GET /regions/haidian/patches/patch_000000/embedding?format=json | `/regions/haidian/patches/patch_000000/embedding?format=json` | 200 | shape=[64, 128, 128], dtype=float32 |
| 16 | GET /regions/haidian/patches/patch_000000/embedding?format=npy | `/regions/haidian/patches/patch_000000/embedding?format=npy` | 200 | shape=(64, 128, 128), dtype=float32, size=1048576 |
| 17 | GET /regions/haidian/patches/patch_000000/embedding?format=cache | `/regions/haidian/patches/patch_000000/embedding?format=cache` | 404 | {"detail":"PNG format not pre-generated for this patch"} |
| 18 | GET /regions/haidian/patches/patch_000000/embedding?format=invalid | `/regions/haidian/patches/patch_000000/embedding?format=invalid` | 422 | {"detail":"Invalid format 'invalid'. Use: png, npy, json, cache"} |
| 19 | GET /regions/harbin/patches/patch_000000/tasks/change_detection/result?format=png&version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/change_detection/result?format=png&version=v1&period=2025-09_vs_2025-10` | 200 | 128x128 mode=RGB |
| 20 | GET /regions/harbin/patches/patch_000000/tasks/change_detection/label?version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/change_detection/label?version=v1&period=2025-09_vs_2025-10` | 404 | {"detail":"Label not found for patch 'patch_000000', task 'change_detection'"} |
| 21 | GET /regions/harbin/patches/patch_000000/tasks/change_detection/label_vis?version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/change_detection/label_vis?version=v1&period=2025-09_vs_2025-10` | 404 | {"detail":"Not Found"} |
| 22 | GET /regions/harbin/patches/patch_000010/tasks/change_detection/result?format=png&version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/change_detection/result?format=png&version=v1&period=2025-09_vs_2025-10` | 200 | 128x128 mode=RGB |
| 23 | GET /regions/harbin/patches/patch_000010/tasks/change_detection/label?version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/change_detection/label?version=v1&period=2025-09_vs_2025-10` | 404 | {"detail":"Label not found for patch 'patch_000010', task 'change_detection'"} |
| 24 | GET /regions/harbin/patches/patch_000010/tasks/change_detection/label_vis?version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/change_detection/label_vis?version=v1&period=2025-09_vs_2025-10` | 404 | {"detail":"Not Found"} |
| 25 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&version=v1&period=2025-10` | 200 | 128x128 mode=RGB |
| 26 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=npy&version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 27 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/prediction?version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 28 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/label?version=v1&period=2025-10` | 200 | JSON response |
| 29 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 30 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=png&version=v1&period=2025-10` | 200 | 128x128 mode=RGB |
| 31 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=npy&version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 32 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/prediction?version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 33 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/label?version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=uint8, size=4096 |
| 34 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 35 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=png&version=v2&period=2025-09_vs_2025-10` | 200 | 128x128 mode=RGB |
| 36 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=npy&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/result?format=npy&version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 37 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/prediction?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/prediction?version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 38 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/label?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/label?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Label not found for patch 'patch_000000', task 'building_extraction'"} |
| 39 | GET /regions/harbin/patches/patch_000000/tasks/building_extraction/label_vis?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/building_extraction/label_vis?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Not Found"} |
| 40 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=png&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=png&version=v2&period=2025-09_vs_2025-10` | 200 | 128x128 mode=RGB |
| 41 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=npy&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/result?format=npy&version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 42 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/prediction?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/prediction?version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 43 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/label?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/label?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Label not found for patch 'patch_000010', task 'building_extraction'"} |
| 44 | GET /regions/harbin/patches/patch_000010/tasks/building_extraction/label_vis?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/building_extraction/label_vis?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Not Found"} |
| 45 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=png&version=v1&period=2025-10` | 200 | 128x128 mode=RGB |
| 46 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=npy&version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 47 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/prediction?version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 48 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/label?version=v1&period=2025-10` | 200 | JSON response |
| 49 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 50 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=png&version=v1&period=2025-10` | 200 | 128x128 mode=RGB |
| 51 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=npy&version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 52 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/prediction?version=v1&period=2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 53 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/label?version=v1&period=2025-10` | 200 | JSON response |
| 54 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 55 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=png&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=png&version=v2&period=2025-09_vs_2025-10` | 200 | 128x128 mode=RGB |
| 56 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=npy&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/result?format=npy&version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 57 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/prediction?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/prediction?version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 58 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/label?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/label?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Label not found for patch 'patch_000000', task 'land_use_classification'"} |
| 59 | GET /regions/harbin/patches/patch_000000/tasks/land_use_classification/label_vis?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_use_classification/label_vis?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Not Found"} |
| 60 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=png&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=png&version=v2&period=2025-09_vs_2025-10` | 200 | 128x128 mode=RGB |
| 61 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=npy&version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/result?format=npy&version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 62 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/prediction?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/prediction?version=v2&period=2025-09_vs_2025-10` | 200 | shape=(64, 64), dtype=float32, size=4096 |
| 63 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/label?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/label?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Label not found for patch 'patch_000010', task 'land_use_classification'"} |
| 64 | GET /regions/harbin/patches/patch_000010/tasks/land_use_classification/label_vis?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_use_classification/label_vis?version=v2&period=2025-09_vs_2025-10` | 404 | {"detail":"Not Found"} |
| 65 | GET /regions/harbin/patches/patch_000000/tasks/land_cover_classification/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_cover_classification/result?format=png&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000000', task 'land_cover_classification'"} |
| 66 | GET /regions/harbin/patches/patch_000000/tasks/land_cover_classification/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_cover_classification/result?format=npy&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000000', task 'land_cover_classification'"} |
| 67 | GET /regions/harbin/patches/patch_000000/tasks/land_cover_classification/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_cover_classification/prediction?version=v1&period=2025-10` | 404 | {"detail":"Prediction not found for patch 'patch_000000', task 'land_cover_classification'"} |
| 68 | GET /regions/harbin/patches/patch_000000/tasks/land_cover_classification/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_cover_classification/label?version=v1&period=2025-10` | 404 | {"detail":"Label not found for patch 'patch_000000', task 'land_cover_classification'"} |
| 69 | GET /regions/harbin/patches/patch_000000/tasks/land_cover_classification/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/land_cover_classification/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 70 | GET /regions/harbin/patches/patch_000010/tasks/land_cover_classification/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_cover_classification/result?format=png&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000010', task 'land_cover_classification'"} |
| 71 | GET /regions/harbin/patches/patch_000010/tasks/land_cover_classification/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_cover_classification/result?format=npy&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000010', task 'land_cover_classification'"} |
| 72 | GET /regions/harbin/patches/patch_000010/tasks/land_cover_classification/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_cover_classification/prediction?version=v1&period=2025-10` | 404 | {"detail":"Prediction not found for patch 'patch_000010', task 'land_cover_classification'"} |
| 73 | GET /regions/harbin/patches/patch_000010/tasks/land_cover_classification/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_cover_classification/label?version=v1&period=2025-10` | 404 | {"detail":"Label not found for patch 'patch_000010', task 'land_cover_classification'"} |
| 74 | GET /regions/harbin/patches/patch_000010/tasks/land_cover_classification/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/land_cover_classification/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 75 | GET /regions/harbin/patches/patch_000000/tasks/water_extraction/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/water_extraction/result?format=png&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000000', task 'water_extraction'"} |
| 76 | GET /regions/harbin/patches/patch_000000/tasks/water_extraction/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/water_extraction/result?format=npy&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000000', task 'water_extraction'"} |
| 77 | GET /regions/harbin/patches/patch_000000/tasks/water_extraction/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/water_extraction/prediction?version=v1&period=2025-10` | 404 | {"detail":"Prediction not found for patch 'patch_000000', task 'water_extraction'"} |
| 78 | GET /regions/harbin/patches/patch_000000/tasks/water_extraction/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/water_extraction/label?version=v1&period=2025-10` | 404 | {"detail":"Label not found for patch 'patch_000000', task 'water_extraction'"} |
| 79 | GET /regions/harbin/patches/patch_000000/tasks/water_extraction/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000000/tasks/water_extraction/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 80 | GET /regions/harbin/patches/patch_000010/tasks/water_extraction/result?format=png&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/water_extraction/result?format=png&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000010', task 'water_extraction'"} |
| 81 | GET /regions/harbin/patches/patch_000010/tasks/water_extraction/result?format=npy&version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/water_extraction/result?format=npy&version=v1&period=2025-10` | 404 | {"detail":"Result not found for patch 'patch_000010', task 'water_extraction'"} |
| 82 | GET /regions/harbin/patches/patch_000010/tasks/water_extraction/prediction?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/water_extraction/prediction?version=v1&period=2025-10` | 404 | {"detail":"Prediction not found for patch 'patch_000010', task 'water_extraction'"} |
| 83 | GET /regions/harbin/patches/patch_000010/tasks/water_extraction/label?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/water_extraction/label?version=v1&period=2025-10` | 404 | {"detail":"Label not found for patch 'patch_000010', task 'water_extraction'"} |
| 84 | GET /regions/harbin/patches/patch_000010/tasks/water_extraction/label_vis?version=v1&period=2025-10 | `/regions/harbin/patches/patch_000010/tasks/water_extraction/label_vis?version=v1&period=2025-10` | 404 | {"detail":"Not Found"} |
| 85 | GET /regions/harbin/tasks/change_detection/tiles?version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/tasks/change_detection/tiles?version=v1&period=2025-09_vs_2025-10` | 200 | total=2968 |
| 86 | GET /regions/harbin/tasks/building_extraction/tiles?version=v1&period=2025-10 | `/regions/harbin/tasks/building_extraction/tiles?version=v1&period=2025-10` | 200 | total=424 |
| 87 | GET /regions/harbin/tasks/building_extraction/tiles?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/tasks/building_extraction/tiles?version=v2&period=2025-09_vs_2025-10` | 200 | total=424 |
| 88 | GET /regions/harbin/tasks/land_use_classification/tiles?version=v1&period=2025-10 | `/regions/harbin/tasks/land_use_classification/tiles?version=v1&period=2025-10` | 200 | total=424 |
| 89 | GET /regions/harbin/tasks/land_use_classification/tiles?version=v2&period=2025-09_vs_2025-10 | `/regions/harbin/tasks/land_use_classification/tiles?version=v2&period=2025-09_vs_2025-10` | 200 | total=424 |
| 90 | GET /regions/harbin/tasks/land_cover_classification/tiles?version=v1&period=2025-10 | `/regions/harbin/tasks/land_cover_classification/tiles?version=v1&period=2025-10` | 200 | total=0 |
| 91 | GET /regions/harbin/tasks/water_extraction/tiles?version=v1&period=2025-10 | `/regions/harbin/tasks/water_extraction/tiles?version=v1&period=2025-10` | 200 | total=0 |
| 92 | GET /regions/harbin/mosaic?date=2025-04&sensor_type=s2&format=png&patch_ids=patch_000000&patch_ids=patch_000001 | `/regions/harbin/mosaic?date=2025-04&sensor_type=s2&format=png&patch_ids=patch_000000&patch_ids=patch_000001` | 200 | 256x128 mode=RGBA |
| 93 | GET /regions/harbin/mosaic?date=2025-04&sensor_type=s1&format=png&patch_ids=patch_000000&patch_ids=patch_000001 | `/regions/harbin/mosaic?date=2025-04&sensor_type=s1&format=png&patch_ids=patch_000000&patch_ids=patch_000001` | 200 | 256x128 mode=RGBA |
| 94 | GET /regions/harbin/mosaic?date=2025-04&sensor_type=landsat&format=png&patch_ids=patch_000000&patch_ids=patch_000001 | `/regions/harbin/mosaic?date=2025-04&sensor_type=landsat&format=png&patch_ids=patch_000000&patch_ids=patch_000001` | 200 | 86x43 mode=RGBA |
| 95 | GET /models | `/models` | 200 | count=125 |
| 96 | GET /openapi.json | `/openapi.json` | 200 | openapi=3.1.0, title=Embedding API |
| 97 | GET /docs | `/docs` | 200 | body length=1012, contains swagger=True |

## ⚠️ 失败列表（数据缺失导致，可能预期）

| # | 接口 | URL | 状态码 | 预期 | 详情 | 备注 |
|---|------|-----|--------|------|------|------|
| 1 | GET /regions/harbin/patches/patch_000000/tasks/change_detection/result?format=npy&version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/change_detection/result?format=npy&version=v1&period=2025-09_vs_2025-10` | 404 | 200 | {"detail":"Result not found for patch 'patch_000000', task 'change_detection'"} | Data exists |
| 2 | GET /regions/harbin/patches/patch_000000/tasks/change_detection/prediction?version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000000/tasks/change_detection/prediction?version=v1&period=2025-09_vs_2025-10` | 404 | 200 | {"detail":"Prediction not found for patch 'patch_000000', task 'change_detection'"} | Data exists |
| 3 | GET /regions/harbin/patches/patch_000010/tasks/change_detection/result?format=npy&version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/change_detection/result?format=npy&version=v1&period=2025-09_vs_2025-10` | 404 | 200 | {"detail":"Result not found for patch 'patch_000010', task 'change_detection'"} | Data exists |
| 4 | GET /regions/harbin/patches/patch_000010/tasks/change_detection/prediction?version=v1&period=2025-09_vs_2025-10 | `/regions/harbin/patches/patch_000010/tasks/change_detection/prediction?version=v1&period=2025-09_vs_2025-10` | 404 | 200 | {"detail":"Prediction not found for patch 'patch_000010', task 'change_detection'"} | Data exists |

## ❌ 异常列表（非预期行为）

无异常项。

## 关键发现

- **数据缺失导致的 404**: 共 4 个接口。 land_cover_classification、water_extraction 任务在配置中已暴露但磁盘上暂无对应数据目录；部分 label/label_vis 文件对 patch_000000/patch_000010 不存在，属于已知数据缺失。
- **Embedding 接口**: harbin 和 haidian 的 patch_000000 均支持 png/json/npy/cache 格式，返回正常。invalid format 正确返回 422。
- **基础接口**: /health、/regions、/regions/harbin、/regions/haidian 均正常返回 JSON 数据。
- **专题任务接口**: change_detection/building_extraction/land_use_classification 的 result/prediction 正常返回；land_cover_classification/water_extraction 因无数据返回 404。
- **Mosaic 大图接口**: /regions/harbin/mosaic 支持 s2/s1/landsat，返回 PNG 正常。
- **自定义模型接口**: /models 列表接口返回正常。

## Mosaic 接口调用示例

```bash
# 哈尔滨全区域 Sentinel-2 真彩色 PNG（首次生成较慢，结果会缓存）
curl -s "http://localhost:9061/regions/harbin/mosaic?date=2025-04&sensor_type=s2&format=png" -o /tmp/harbin_s2_2025-04.png

# 只拼前两个 patch 的 Sentinel-1 SAR 伪彩色预览（快）
curl -s "http://localhost:9061/regions/harbin/mosaic?date=2025-04&sensor_type=s1&format=png&patch_ids=patch_000000&patch_ids=patch_000001" -o /tmp/harbin_s1_preview.png

# Landsat 全区域 GeoTIFF 原始数据（保留多波段与坐标）
curl -s "http://localhost:9061/regions/harbin/mosaic?date=2025-04&sensor_type=landsat&format=tif" -o /tmp/harbin_landsat_2025-04.tif
```

**参数取值说明**：

| 参数 | 可取值 | 默认值 | 说明 |
|------|--------|--------|------|
| `region_id` | `harbin` / `haidian` | - | 区域 ID |
| `date` | `YYYY-MM`，如 `2025-04` | - | 哈尔滨会自动映射到 `2025Q1/Q2/Q3/Q4` |
| `sensor_type` | `s2` / `s1` / `landsat` | `s2` | 传感器类型 |
| `format` | `png` / `tif` | `png` | `png` 可视化；`tif` GeoTIFF 原始数据 |
| `patch_ids` | 如 `patch_000000` | 全区域 | 可多次传入，只拼指定 patch |
