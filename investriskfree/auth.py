"""Local, password-based user accounts for the Streamlit application.

Passwords are never stored directly.  Each password is hashed with PBKDF2-HMAC
SHA-256 and an individual random salt.  The auth database only contains account
metadata; every user's paper-trading ledger is stored in a separate SQLite file.

This is intentionally a small self-hosted authentication layer.  Production
installations that need SSO, e-mail verification or password recovery should put
an identity provider (for example Auth0/Clerk) in front of the app.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import get

PBKDF2_ITERATIONS = 390_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def password_validation_error(password: str) -> str | None:
    """Return a user-facing password validation error, or ``None``."""
    if len(password or "") < 10:
        return "Password must contain at least 10 characters."
    if not re.search(r"[A-Za-z]", password):
        return "Password must contain at least one letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    return None


class AuthStore:
    """SQLite-backed user registry with basic brute-force protection."""

    def __init__(self, db_path: str | None = None):
        root = get("data.repo_root")
        self.db_path = db_path or os.path.join(root, "data", "auth.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT,
                    salt TEXT,
                    iterations INTEGER,
                    auth_provider TEXT NOT NULL DEFAULT 'local',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")

    @staticmethod
    def _hash(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)

    def register(
        self, username: str, email: str, password: str, display_name: str = ""
    ) -> dict:
        username = _normalize_username(username)
        email = (email or "").strip().lower()
        display_name = (display_name or username).strip()[:80]
        if not USERNAME_RE.fullmatch(username):
            return {
                "ok": False,
                "error": "Username must be 3–32 characters using letters, numbers, dot, dash or underscore.",
            }
        if not EMAIL_RE.fullmatch(email):
            return {"ok": False, "error": "Enter a valid e-mail address."}
        password_error = password_validation_error(password)
        if password_error:
            return {"ok": False, "error": password_error}
        if not display_name:
            return {"ok": False, "error": "Display name is required."}

        salt = secrets.token_bytes(24)
        password_hash = self._hash(password, salt)
        user_id = uuid.uuid4().hex
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO users(
                        id, username, email, display_name, password_hash, salt,
                        iterations, auth_provider, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        user_id,
                        username,
                        email,
                        display_name,
                        password_hash.hex(),
                        salt.hex(),
                        PBKDF2_ITERATIONS,
                        "local",
                        _now(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            message = str(exc).lower()
            field = "e-mail" if "email" in message else "username"
            return {"ok": False, "error": f"That {field} is already registered."}
        return {
            "ok": True,
            "user": {
                "id": user_id,
                "username": username,
                "email": email,
                "display_name": display_name,
                "auth_provider": "local",
            },
        }

    def authenticate(self, username: str, password: str) -> dict:
        """Authenticate a local user without revealing whether the user exists."""
        username = _normalize_username(username)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)
            ).fetchone()
            if row and row["locked_until"]:
                try:
                    locked_until = datetime.fromisoformat(row["locked_until"])
                    if locked_until > datetime.now(timezone.utc):
                        return {
                            "ok": False,
                            "error": "Too many failed attempts. Try again in a few minutes.",
                        }
                except ValueError:
                    pass
            valid = False
            if row and row["is_active"] and row["auth_provider"] == "local":
                try:
                    salt = bytes.fromhex(row["salt"])
                    expected = bytes.fromhex(row["password_hash"])
                    actual = self._hash(password or "", salt, int(row["iterations"]))
                    valid = hmac.compare_digest(actual, expected)
                except (TypeError, ValueError):
                    valid = False
            if not valid:
                if row and row["auth_provider"] == "local":
                    failures = int(row["failed_attempts"] or 0) + 1
                    # Lock for five minutes after five consecutive failures.
                    locked = (
                        (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(timespec="seconds")
                        if failures >= 5
                        else None
                    )
                    conn.execute(
                        "UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
                        (failures, locked, row["id"]),
                    )
                # Do a hash even for an unknown username to reduce timing leakage.
                if row is None:
                    self._hash(password or "", b"\0" * 24)
                return {"ok": False, "error": "Invalid username or password."}

            conn.execute(
                """UPDATE users SET failed_attempts=0, locked_until=NULL,
                   last_login=? WHERE id=?""",
                (_now(), row["id"]),
            )
            return {"ok": True, "user": self._public_user(row)}

    def ensure_external_user(
        self, username: str, display_name: str = "Administrator", email: str | None = None
    ) -> dict:
        """Create/get an account authenticated by deployment secrets.

        No password is stored for an external account; callers must verify the
        deployment secret before invoking this method.
        """
        username = _normalize_username(username)
        if not USERNAME_RE.fullmatch(username):
            raise ValueError("invalid external username")
        email = (email or f"{username}@local.invalid").lower()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)
            ).fetchone()
            if row:
                if row["auth_provider"] != "deployment_secret":
                    raise ValueError(
                        "configured administrator username belongs to a local account; "
                        "choose a different ADMIN_USERNAME"
                    )
                conn.execute("UPDATE users SET last_login=? WHERE id=?", (_now(), row["id"]))
                return self._public_user(row)
            user_id = uuid.uuid4().hex
            conn.execute(
                """INSERT INTO users(id, username, email, display_name,
                   auth_provider, created_at, last_login)
                   VALUES (?,?,?,?,?,?,?)""",
                (user_id, username, email, display_name, "deployment_secret", _now(), _now()),
            )
            return {
                "id": user_id,
                "username": username,
                "email": email,
                "display_name": display_name,
                "auth_provider": "deployment_secret",
            }

    def get_user(self, user_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            return self._public_user(row) if row else None

    def active_users(self) -> list[dict]:
        """Return active users for the optional auto-trade worker."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM users WHERE is_active=1 ORDER BY created_at").fetchall()
            return [self._public_user(row) for row in rows]

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "display_name": row["display_name"],
            "auth_provider": row["auth_provider"],
            "created_at": row["created_at"],
            "last_login": row["last_login"],
        }


def user_paper_db_path(user_id: str) -> str:
    """Return an isolated paper-ledger path for a validated user id."""
    if not re.fullmatch(r"[a-f0-9]{32}", user_id or ""):
        raise ValueError("invalid user id")
    root = Path(get("data.repo_root")) / "data" / "users" / user_id
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "paper.db")
