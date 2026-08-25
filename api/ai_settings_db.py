"""Local, encrypted-at-rest storage for bring-your-own-key AI provider
credentials. This is a self-hosted, single-deployer tool with no login
system, so "your key" means the key of whoever runs `docker compose up` —
one shared SQLite row per provider, not per-visitor. Encryption (Fernet)
guards against the key leaking via a DB file backup or volume snapshot;
it is not a substitute for access control on the instance itself.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(os.environ.get("AI_SETTINGS_DB_PATH", "/data/ai_settings.db"))


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provider_keys (
            provider TEXT PRIMARY KEY,
            encrypted_key BLOB NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def _fernet():
    secret = os.environ.get("AI_SETTINGS_SECRET")
    if not secret:
        raise RuntimeError(
            "AI_SETTINGS_SECRET is not set — generate one with "
            '`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` '
            "and set it before storing or reading provider keys."
        )
    from cryptography.fernet import Fernet

    return Fernet(secret.encode())


def set_key(provider: str, api_key: str) -> None:
    encrypted = _fernet().encrypt(api_key.encode())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO provider_keys (provider, encrypted_key, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(provider) DO UPDATE SET encrypted_key = excluded.encrypted_key, updated_at = excluded.updated_at",
            (provider, encrypted, datetime.now(timezone.utc).isoformat()),
        )


def get_key(provider: str) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT encrypted_key FROM provider_keys WHERE provider = ?", (provider,)).fetchone()
    if row is None:
        return None
    return _fernet().decrypt(row[0]).decode()


def delete_key(provider: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM provider_keys WHERE provider = ?", (provider,))


def list_configured() -> set[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT provider FROM provider_keys").fetchall()
    return {r[0] for r in rows}
