
"""SQLite database layer for the Face Recognition Management System."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
DB_PATH = DATA_DIR / "face_system.db"


def connect() -> sqlite3.Connection:
    """Open a SQLite connection and return dictionary-like rows."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def hash_password(password: str) -> str:
    """Hash passwords for local demo authentication."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db() -> None:
    """Create all database tables and seed default accounts/settings."""
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                display_name TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                user_code TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                department TEXT NOT NULL,
                role TEXT DEFAULT 'User',
                embedding TEXT NOT NULL,
                samples_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'Active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_code TEXT NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT,
                marked_at TEXT NOT NULL,
                date TEXT NOT NULL,
                month TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unknown_visitors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                confidence REAL,
                source TEXT,
                note TEXT,
                detected_at TEXT NOT NULL,
                date TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                camera_type TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT DEFAULT 'Inactive',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

        defaults = [
            ("admin", "admin123", "Administrator", "System Administrator", "admin@example.com"),
            ("teacher", "teacher123", "Teacher/Manager", "Teacher Manager", "teacher@example.com"),
            ("user", "user123", "User", "Demo User", "user@example.com"),
        ]
        for username, password, role, display_name, email in defaults:
            conn.execute(
                """
                INSERT OR IGNORE INTO accounts
                (username, password_hash, role, display_name, email, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, hash_password(password), role, display_name, email, now_text()),
            )

        default_settings = {
            "duplicate_interval_minutes": "30",
            "recognition_threshold": "68",
            "unknown_alert_threshold": "68",
            "theme_mode": "Dark",
            "institution_name": "AI Face Recognition Management System",
        }
        for key, value in default_settings.items():
            conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))


def now_text() -> str:
    """Return a consistent timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    """Validate a login and return the account if successful."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE username = ? AND password_hash = ?",
            (username.strip(), hash_password(password)),
        ).fetchone()
    return dict(row) if row else None


def log_activity(actor: str, action: str, details: str = "") -> None:
    """Save a user-visible activity log entry."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO activity_logs (actor, action, details, created_at) VALUES (?, ?, ?, ?)",
            (actor, action, details, now_text()),
        )


def encode_embedding(embedding: np.ndarray) -> str:
    """Serialize a face embedding to JSON."""
    return json.dumps([float(value) for value in embedding.reshape(-1)])


def decode_embedding(raw: str) -> np.ndarray:
    """Deserialize a face embedding from JSON."""
    return np.array(json.loads(raw), dtype="float32")


def add_registered_user(data: dict[str, str], embedding: np.ndarray, samples_count: int) -> tuple[bool, str]:
    """Create a registered user with a stored face embedding."""
    required = ["name", "user_code", "email", "phone", "department"]
    missing = [field for field in required if not data.get(field, "").strip()]
    if missing:
        return False, "Please fill all required fields."

    timestamp = now_text()
    try:
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO users
                (name, user_code, email, phone, department, role, embedding, samples_count, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Active', ?, ?)
                """,
                (
                    data["name"].strip(),
                    data["user_code"].strip(),
                    data["email"].strip(),
                    data["phone"].strip(),
                    data["department"].strip(),
                    data.get("role", "User"),
                    encode_embedding(embedding),
                    samples_count,
                    timestamp,
                    timestamp,
                ),
            )
        return True, "Registration completed successfully."
    except sqlite3.IntegrityError:
        return False, "A user with this ID already exists."


def update_registered_user(user_id: int, data: dict[str, str]) -> None:
    """Update profile fields for an existing registered user."""
    with connect() as conn:
        conn.execute(
            """
            UPDATE users
            SET name = ?, user_code = ?, email = ?, phone = ?, department = ?, role = ?, status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                data["name"], data["user_code"], data["email"], data["phone"],
                data["department"], data.get("role", "User"), data.get("status", "Active"), now_text(), user_id,
            ),
        )


def delete_registered_user(user_id: int) -> None:
    """Delete a registered user."""
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def get_registered_users(active_only: bool = False) -> list[dict[str, Any]]:
    """Return registered users."""
    query = "SELECT * FROM users"
    params: tuple[Any, ...] = ()
    if active_only:
        query += " WHERE status = ?"
        params = ("Active",)
    query += " ORDER BY created_at DESC"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def mark_attendance(user: dict[str, Any], confidence: float, source: str, duplicate_minutes: int) -> tuple[bool, str]:
    """Mark attendance and prevent duplicates inside the configured interval."""
    current = datetime.now()
    cutoff = current - timedelta(minutes=duplicate_minutes)
    with connect() as conn:
        duplicate = conn.execute(
            """
            SELECT * FROM attendance
            WHERE user_code = ? AND marked_at >= ?
            ORDER BY marked_at DESC LIMIT 1
            """,
            (user["user_code"], cutoff.strftime("%Y-%m-%d %H:%M:%S")),
        ).fetchone()
        if duplicate:
            return False, f"Attendance already marked recently for {user['name']}."

        conn.execute(
            """
            INSERT INTO attendance
            (user_code, name, department, status, confidence, source, marked_at, date, month)
            VALUES (?, ?, ?, 'Present', ?, ?, ?, ?, ?)
            """,
            (
                user["user_code"], user["name"], user["department"], confidence, source,
                current.strftime("%Y-%m-%d %H:%M:%S"), current.strftime("%Y-%m-%d"), current.strftime("%Y-%m"),
            ),
        )
    return True, f"Attendance marked for {user['name']}."


def get_attendance(filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Return attendance records using optional filters."""
    filters = filters or {}
    clauses: list[str] = []
    params: list[Any] = []
    if filters.get("date"):
        clauses.append("date = ?")
        params.append(filters["date"])
    if filters.get("month"):
        clauses.append("month = ?")
        params.append(filters["month"])
    if filters.get("department") and filters["department"] != "All":
        clauses.append("department = ?")
        params.append(filters["department"])
    if filters.get("search"):
        clauses.append("(name LIKE ? OR user_code LIKE ?)")
        term = f"%{filters['search']}%"
        params.extend([term, term])
    query = "SELECT * FROM attendance"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY marked_at DESC"
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def add_unknown_visitor(image_path: str, confidence: float, source: str, note: str) -> None:
    """Store an unknown visitor alert."""
    current = datetime.now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO unknown_visitors (image_path, confidence, source, note, detected_at, date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_path, confidence, source, note, current.strftime("%Y-%m-%d %H:%M:%S"), current.strftime("%Y-%m-%d")),
        )


def get_unknown_visitors() -> list[dict[str, Any]]:
    """Return unknown visitor alerts."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM unknown_visitors ORDER BY detected_at DESC").fetchall()
    return [dict(row) for row in rows]


def save_camera(name: str, camera_type: str, source: str, status: str = "Inactive") -> None:
    """Save a camera source."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO cameras (name, camera_type, source, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, camera_type, source, status, now_text()),
        )


def get_cameras() -> list[dict[str, Any]]:
    """Return all configured cameras."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM cameras ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def delete_camera(camera_id: int) -> None:
    """Delete a camera source."""
    with connect() as conn:
        conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))


def get_setting(key: str, default: str = "") -> str:
    """Read one setting."""
    with connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    """Write one setting."""
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))


def get_activity_logs(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent activity logs."""
    with connect() as conn:
        rows = conn.execute("SELECT * FROM activity_logs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def dashboard_metrics() -> dict[str, Any]:
    """Return summary metrics for the dashboard."""
    today = datetime.now().strftime("%Y-%m-%d")
    month = datetime.now().strftime("%Y-%m")
    week_start = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
    with connect() as conn:
        total_users = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        today_attendance = conn.execute("SELECT COUNT(*) AS c FROM attendance WHERE date = ?", (today,)).fetchone()["c"]
        weekly_attendance = conn.execute("SELECT COUNT(*) AS c FROM attendance WHERE date >= ?", (week_start,)).fetchone()["c"]
        monthly_attendance = conn.execute("SELECT COUNT(*) AS c FROM attendance WHERE month = ?", (month,)).fetchone()["c"]
        unknown_count = conn.execute("SELECT COUNT(*) AS c FROM unknown_visitors").fetchone()["c"]
        active_cameras = conn.execute("SELECT COUNT(*) AS c FROM cameras WHERE status = 'Active'").fetchone()["c"]
    return {
        "total_users": total_users,
        "today_attendance": today_attendance,
        "weekly_attendance": weekly_attendance,
        "monthly_attendance": monthly_attendance,
        "unknown_visitors": unknown_count,
        "active_cameras": active_cameras,
    }
