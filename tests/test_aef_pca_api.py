from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def test_haidian_aef_2025_pca_returns_png():
    response = client.get(
        "/regions/haidian/patches/patch_000106/embeddings/aef/pca"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-embedding-source"] == "AEF"
    assert response.headers["x-embedding-year"] == "2025"
    assert response.headers["x-patch-id"] == "patch_000106"
    assert response.headers["x-pca-version"] == "aef-haidian-2025-global-v1"
    image = Image.open(BytesIO(response.content))
    assert image.mode == "RGB"
    assert image.size == (128, 128)


def test_haidian_aef_pca_rejects_invalid_patch_id():
    response = client.get(
        "/regions/haidian/patches/patch_106/embeddings/aef/pca"
    )

    assert response.status_code == 422
    assert "patch_000000" in response.json()["detail"]


def test_haidian_aef_pca_returns_404_for_missing_embedding():
    response = client.get(
        "/regions/haidian/patches/patch_999999/embeddings/aef/pca"
    )

    assert response.status_code == 404
    assert "AEF 2025 embedding not found" in response.json()["detail"]


def test_aef_pca_openapi_is_png_only_and_documented_in_chinese():
    operation = client.get("/openapi.json").json()["paths"][
        "/regions/haidian/patches/{patch_id}/embeddings/aef/pca"
    ]["get"]

    assert "image/png" in operation["responses"]["200"]["content"]
    assert "仅支持海淀区" in operation["description"]
    assert "2025" in operation["description"]
