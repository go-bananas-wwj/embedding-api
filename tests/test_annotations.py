"""Annotation and class management tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestAnnotations:
    def test_create_class(self):
        response = client.post("/annotations/classes", json={"name": "Building", "color": "#ff0000"})
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Building"
        assert data["color"] == "#ff0000"
        assert data["id"].startswith("cls_")

    def test_list_classes(self):
        client.post("/annotations/classes", json={"name": "Water", "color": "#0000ff"})
        response = client.get("/annotations/classes")
        assert response.status_code == 200
        data = response.json()
        assert any(c["name"] == "Water" for c in data)

    def test_create_annotation_requires_class(self):
        response = client.post(
            "/annotations",
            json={
                "region_id": "harbin",
                "patch_id": "patch_000000",
                "month": "2025-04",
                "class_id": "cls_nonexistent",
                "geometry": {"type": "mask", "mask_b64": "iVBORw0KGgoAAAA"},
            },
        )
        assert response.status_code == 400

    def test_create_annotation_with_class(self):
        r = client.post("/annotations/classes", json={"name": "Forest", "color": "#00ff00"})
        class_id = r.json()["id"]

        # Minimal valid base64 1x1 PNG
        import base64
        from io import BytesIO
        from PIL import Image
        buf = BytesIO()
        Image.new("L", (1, 1), 255).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        response = client.post(
            "/annotations",
            json={
                "region_id": "harbin",
                "patch_id": "patch_000000",
                "month": "2025-04",
                "class_id": class_id,
                "geometry": {"type": "mask", "mask_b64": b64},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["patch_id"] == "patch_000000"
        assert data["class_id"] == class_id

    def test_list_annotations_filter(self):
        r = client.get("/annotations?region_id=harbin&patch_id=patch_000000")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_delete_annotation(self):
        r = client.post("/annotations/classes", json={"name": "Temp", "color": "#fff"})
        class_id = r.json()["id"]

        import base64
        from io import BytesIO
        from PIL import Image
        buf = BytesIO()
        Image.new("L", (1, 1), 255).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        r = client.post(
            "/annotations",
            json={
                "region_id": "harbin",
                "patch_id": "patch_000000",
                "month": "2025-04",
                "class_id": class_id,
                "geometry": {"type": "mask", "mask_b64": b64},
            },
        )
        ann_id = r.json()["id"]

        r = client.delete(f"/annotations/{ann_id}")
        assert r.status_code == 200

        r = client.get(f"/annotations/{ann_id}")
        assert r.status_code == 404
