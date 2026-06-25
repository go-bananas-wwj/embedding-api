"""Authentication dependency tests."""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.services.auth_service import get_current_user


app = FastAPI()


@app.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return user


client = TestClient(app)


class TestAuth:
    def test_no_auth_config_returns_default(self):
        response = client.get("/me")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "default"

    def test_api_key_header_resolves_user(self, monkeypatch):
        auth_cfg = {
            "type": "api_key",
            "users": {
                "key_alice": {"user_id": "alice", "name": "Alice"},
            },
        }
        monkeypatch.setattr(
            "app.services.auth_service._get_auth_config", lambda: auth_cfg
        )
        response = client.get("/me", headers={"X-API-Key": "key_alice"})
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "alice"
        assert data["name"] == "Alice"

    def test_bearer_token_resolves_user(self, monkeypatch):
        auth_cfg = {
            "type": "api_key",
            "users": {
                "key_bob": {"user_id": "bob"},
            },
        }
        monkeypatch.setattr(
            "app.services.auth_service._get_auth_config", lambda: auth_cfg
        )
        response = client.get("/me", headers={"Authorization": "Bearer key_bob"})
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "bob"

    def test_missing_key_returns_401(self, monkeypatch):
        auth_cfg = {
            "type": "api_key",
            "users": {"key_alice": {"user_id": "alice"}},
        }
        monkeypatch.setattr(
            "app.services.auth_service._get_auth_config", lambda: auth_cfg
        )
        response = client.get("/me")
        assert response.status_code == 401

    def test_invalid_key_returns_401(self, monkeypatch):
        auth_cfg = {
            "type": "api_key",
            "users": {"key_alice": {"user_id": "alice"}},
        }
        monkeypatch.setattr(
            "app.services.auth_service._get_auth_config", lambda: auth_cfg
        )
        response = client.get("/me", headers={"X-API-Key": "bad_key"})
        assert response.status_code == 401
