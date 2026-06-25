"""Authentication and user isolation utilities.

The project currently supports API-Key based user isolation. The dependency
`get_current_user` is designed to be swappable: the router code only depends
on the returned dict shape ``{"user_id": str, "name": str}``. Future OAuth2
or JWT implementations can replace this module without touching routers.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from app.config import get_config


def _get_auth_config() -> Dict[str, Any]:
    """Return the auth section from config, or an empty dict."""
    return get_config().get("auth", default={})


def _resolve_user(api_key: Optional[str]) -> Dict[str, str]:
    """Map an API key to a user record.

    If no auth is configured, all requests are treated as the default user.
    If auth is configured but the key is missing/invalid, a 401 is raised.
    """
    auth_cfg = _get_auth_config()
    auth_type = auth_cfg.get("type", "none")

    if auth_type == "none" or not auth_cfg.get("users"):
        return {"user_id": "default", "name": "Default User"}

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    users = auth_cfg.get("users", {})
    record = users.get(api_key)
    if not record:
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_id = record.get("user_id")
    if not user_id:
        raise HTTPException(status_code=500, detail="User record missing user_id")

    return {
        "user_id": str(user_id),
        "name": str(record.get("name", user_id)),
    }


async def get_current_user(request: Request) -> Dict[str, str]:
    """FastAPI dependency that resolves the current user from the request.

    Supports:
      - ``X-API-Key: <key>`` header
      - ``Authorization: Bearer <key>`` header

    Returns a dict with at least ``user_id`` and ``name``.
    """
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()

    return _resolve_user(api_key)
