from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.human_face_validation import (
    ANIMAL_FACE_MESSAGE,
    MODEL_FILENAME as ANIMAL_CLASSIFIER_MODEL_FILENAME,
    NO_FACE_MESSAGE,
    FaceInputKind,
    FaceInputRejected,
    require_validated_detection,
    validate_human_candidates,
)

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # SQLite remains available for local development.
    psycopg = None
    dict_row = None


def normalize_database_url(value: str) -> str:
    """Normalize URI text copied from Supabase or a TOML secrets line."""
    normalized = value.strip()
    if normalized.startswith("DATABASE_URL") and "=" in normalized:
        normalized = normalized.split("=", 1)[1].strip()
    if normalized.startswith("psql "):
        normalized = normalized[5:].strip()
    normalized = normalized.strip("'\"`").strip()
    if "://" not in normalized or "@" not in normalized:
        return normalized

    scheme, remainder = normalized.split("://", 1)
    user_info, server_info = remainder.rsplit("@", 1)
    if ":" not in user_info:
        return normalized
    username, password = user_info.split(":", 1)
    # Supabase passwords often contain %, @, #, or /; encode them as URI data.
    encoded_password = quote(unquote(password), safe="")
    return f"{scheme}://{username}:{encoded_password}@{server_info}"


ROOT = Path(__file__).parent
DATA = ROOT / "data"
REGISTERED_DIR = ROOT / "storage" / "registered"
UNKNOWN_DIR = ROOT / "storage" / "unknown"
REPORT_DIR = ROOT / "storage" / "reports"
MODEL_DIR = ROOT / "models"
DB_PATH = DATA / "face_system.sqlite3"
DATABASE_URL = normalize_database_url(os.getenv("DATABASE_URL", ""))
DATABASE_CONNECTION_ERROR: str | None = None

for folder in (DATA, REGISTERED_DIR, UNKNOWN_DIR, REPORT_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
ANIMAL_CLASSIFIER_MODEL_PATH = MODEL_DIR / ANIMAL_CLASSIFIER_MODEL_FILENAME
REGISTRATION_DETECTION_THRESHOLD = 0.65
LIVENESS_RECOVERY_DETECTION_THRESHOLD = 0.50
FACE_DETECTION_THRESHOLD = 0.70
RECOGNITION_COSINE_THRESHOLD = 0.50
VERIFICATION_COSINE_THRESHOLD = 0.36
LIVENESS_IDENTITY_THRESHOLD = 0.30
LIVENESS_HEAD_MOVEMENT_THRESHOLD = 0.08
RECOGNITION_MODEL_VERSION = "sface_v1"
YUNET_INFERENCE_LOCK = threading.RLock()
SFACE_INFERENCE_LOCK = threading.RLock()
DATABASE_INTEGRITY_ERRORS = (
    (sqlite3.IntegrityError, psycopg.IntegrityError)
    if psycopg is not None
    else (sqlite3.IntegrityError,)
)


@dataclass
class MatchResult:
    person: dict[str, Any] | None
    confidence: float
    distance: float


def persistent_database_enabled() -> bool:
    return bool(DATABASE_URL)


def connect() -> Any:
    global DATABASE_CONNECTION_ERROR, DATABASE_URL
    if persistent_database_enabled():
        if psycopg is None or dict_row is None:
            raise RuntimeError("PostgreSQL support is not installed. Install psycopg[binary].")
        try:
            return psycopg.connect(
                DATABASE_URL,
                row_factory=dict_row,
                connect_timeout=10,
                prepare_threshold=None,
            )
        except Exception as error:
            if not isinstance(error, psycopg.Error):
                raise
            # Never take down the presentation app because of a malformed or
            # temporarily unreachable cloud secret. Make the fallback visible.
            DATABASE_CONNECTION_ERROR = "Persistent database connection failed; local safe mode is active."
            DATABASE_URL = ""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def database_query(query: str) -> str:
    """Translate the app's portable placeholders for PostgreSQL."""
    return query.replace("?", "%s") if persistent_database_enabled() else query


def database_execute(con: Any, query: str, params: tuple[Any, ...] = ()) -> Any:
    return con.execute(database_query(query), params)


def database_script(con: Any, script: str) -> None:
    if persistent_database_enabled():
        for statement in script.split(";"):
            if statement.strip():
                con.execute(statement)
    else:
        con.executescript(script)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def normalize_username(username: str) -> str:
    """Keep login names predictable while display names remain free-form."""
    return username.strip().lower()


def valid_username(username: str) -> bool:
    """Accept simple, URL-safe login names from 3 to 50 characters."""
    return 3 <= len(username) <= 50 and all(
        character.isalnum() or character in "._-" for character in username
    )


def authenticate_user(identifier: str, password: str) -> dict[str, Any] | None:
    """Authenticate using an unambiguous username, display name, or email.

    Display names are allowed for backwards-compatible recovery from the old
    Profile screen, but duplicate matches are rejected instead of selecting an
    arbitrary account.
    """
    identifier = identifier.strip()
    if not identifier or not password:
        return None
    matches = rows(
        "auth_users",
        "WHERE password_hash = ? AND ("
        "LOWER(username) = LOWER(?) OR "
        "LOWER(display_name) = LOWER(?) OR "
        "LOWER(COALESCE(email, '')) = LOWER(?)"
        ")",
        (hash_password(password), identifier, identifier, identifier),
    )
    return matches[0] if len(matches) == 1 else None


def default_admin_credentials_active() -> bool:
    """Only advertise the starter password while it is actually valid."""
    return bool(
        rows(
            "auth_users",
            "WHERE LOWER(username) = 'admin' AND password_hash = ?",
            (hash_password("admin123"),),
        )
    )


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def init_db() -> None:
    con = connect()
    schema = """
        CREATE TABLE IF NOT EXISTS auth_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            display_name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            department TEXT NOT NULL,
            embeddings TEXT NOT NULL,
            image_paths TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            marked_at TEXT NOT NULL,
            camera TEXT,
            UNIQUE(person_id, marked_at)
        );
        CREATE TABLE IF NOT EXISTS unknown_visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            confidence REAL NOT NULL,
            detected_at TEXT NOT NULL,
            camera TEXT,
            note TEXT
        );
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Configured',
            last_seen TEXT
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL
        );
        """
    if persistent_database_enabled():
        schema = schema.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")
    database_script(con, schema)
    defaults = {
        "recognition_threshold": str(RECOGNITION_COSINE_THRESHOLD),
        "duplicate_minutes": "30",
        "dark_mode": "false",
        "blink_required": "true",
        "head_movement_required": "true",
        "camera_fps_target": "12",
    }
    for key, value in defaults.items():
        database_execute(
            con,
            "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO NOTHING",
            (key, value),
        )
    model_version = database_execute(
        con,
        "SELECT value FROM settings WHERE key = 'recognition_model_version'"
    ).fetchone()
    if model_version is None or model_version["value"] != RECOGNITION_MODEL_VERSION:
        database_execute(
            con,
            "INSERT INTO settings(key, value) VALUES('recognition_threshold', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(RECOGNITION_COSINE_THRESHOLD),),
        )
        database_execute(
            con,
            "INSERT INTO settings(key, value) VALUES('recognition_model_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (RECOGNITION_MODEL_VERSION,),
        )
    database_execute(
        con,
        """
        INSERT INTO auth_users(username, password_hash, role, display_name, email, phone, created_at)
        VALUES('admin', ?, 'Administrator', 'System Administrator', 'admin@example.com', '', ?)
        ON CONFLICT(username) DO NOTHING
        """,
        (hash_password("admin123"), now_iso()),
    )
    existing_cameras = database_execute(con, "SELECT COUNT(*) AS count FROM cameras").fetchone()["count"]
    if existing_cameras == 0:
        database_execute(
            con,
            "INSERT INTO cameras(name, kind, source, status) VALUES('Default Webcam', 'Webcam', '0', 'Configured')"
        )
    con.commit()
    con.close()


def reset_storage_files() -> None:
    for folder in (REGISTERED_DIR, UNKNOWN_DIR, REPORT_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def clear_unknown_alerts_data() -> int:
    """Delete unknown-alert records and their stored images only."""
    if UNKNOWN_DIR.exists():
        for path in UNKNOWN_DIR.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    con = connect()
    count = int(database_execute(con, "SELECT COUNT(*) AS count FROM unknown_visitors").fetchone()["count"])
    database_execute(con, "DELETE FROM unknown_visitors")
    con.commit()
    con.close()
    return count


def reset_operational_data() -> None:
    reset_storage_files()
    con = connect()
    database_script(
        con,
        """
        DELETE FROM people;
        DELETE FROM attendance;
        DELETE FROM unknown_visitors;
        DELETE FROM activity_logs;
        DELETE FROM cameras;
        """
    )
    database_execute(
        con,
        "INSERT INTO cameras(name, kind, source, status) VALUES('Default Webcam', 'Webcam', '0', 'Configured')",
    )
    database_execute(
        con,
        "INSERT INTO activity_logs(actor, action, detail, created_at) VALUES(?, ?, ?, ?)",
        (st.session_state.get("user", {}).get("username", "system"), "Fresh start reset", "Cleared operational data", now_iso()),
    )
    con.commit()
    con.close()


def rows(table: str, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    con = connect()
    query = f"SELECT * FROM {table} {where}"
    items = [dict(r) for r in database_execute(con, query, params).fetchall()]
    con.close()
    return items


def execute(query: str, params: tuple[Any, ...] = ()) -> None:
    con = connect()
    database_execute(con, query, params)
    con.commit()
    con.close()


def setting(key: str, cast: type = str) -> Any:
    value = rows("settings", "WHERE key = ?", (key,))[0]["value"]
    if cast is bool:
        return value.lower() == "true"
    return cast(value)


def set_setting(key: str, value: Any) -> None:
    execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value).lower() if isinstance(value, bool) else str(value)),
    )


def log(action: str, detail: str = "") -> None:
    actor = st.session_state.get("user", {}).get("username", "system")
    execute(
        "INSERT INTO activity_logs(actor, action, detail, created_at) VALUES(?, ?, ?, ?)",
        (actor, action, detail, now_iso()),
    )


@st.cache_resource
def cascades() -> tuple[Any, Any]:
    cascade_dir = cascade_directory()
    if not hasattr(cv2, "CascadeClassifier") or cascade_dir is None:
        return None, None
    face = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
    eye = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))
    return (None if face.empty() else face, None if eye.empty() else eye)


def cascade_directory() -> Path | None:
    """Find OpenCV's XML models across local and hosted installations."""
    candidates = []
    data_module = getattr(cv2, "data", None)
    configured_path = getattr(data_module, "haarcascades", "")
    if configured_path:
        candidates.append(Path(configured_path))
    cv2_file = getattr(cv2, "__file__", "")
    if cv2_file:
        package_dir = Path(cv2_file).resolve().parent
        candidates.extend([package_dir / "data", package_dir / "haarcascades"])
    for candidate in candidates:
        if (candidate / "haarcascade_frontalface_default.xml").is_file():
            return candidate
    return None


@st.cache_resource
def face_detectors() -> list[tuple[Any, bool, float, float]]:
    """Load complementary frontal and profile detectors.

    The boolean marks profile cascades, which are also run on a mirrored image
    so faces turned in either direction can be found.
    """
    cascade_dir = cascade_directory()
    if not hasattr(cv2, "CascadeClassifier") or cascade_dir is None:
        return []
    detector_specs = [
        # Cascade feature weights use different numeric scales, so each model
        # has its own offset and scale for the displayed confidence.
        ("haarcascade_frontalface_default.xml", False, 2.0, 2.0),
        ("haarcascade_frontalface_alt.xml", False, 100.0, 5.0),
        ("haarcascade_frontalface_alt2.xml", False, 50.0, 4.0),
        ("haarcascade_profileface.xml", True, 1.0, 1.0),
    ]
    loaded = []
    for filename, is_profile, offset, scale in detector_specs:
        detector = cv2.CascadeClassifier(str(cascade_dir / filename))
        if not detector.empty():
            loaded.append((detector, is_profile, offset, scale))
    return loaded


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def uploaded_to_bgr(file: Any) -> np.ndarray:
    return pil_to_bgr(Image.open(file))


@st.cache_resource
def yunet_detector() -> Any:
    """Load OpenCV YuNet, downloading the official MIT-licensed model once."""
    if not hasattr(cv2, "FaceDetectorYN"):
        return None
    if not YUNET_MODEL_PATH.is_file() or YUNET_MODEL_PATH.stat().st_size < 100_000:
        temporary_path = YUNET_MODEL_PATH.with_suffix(".download")
        try:
            with urllib.request.urlopen(YUNET_MODEL_URL, timeout=20) as response:
                model_bytes = response.read()
            if len(model_bytes) < 100_000:
                return None
            temporary_path.write_bytes(model_bytes)
            temporary_path.replace(YUNET_MODEL_PATH)
        except (OSError, ValueError):
            return None
    try:
        return create_yunet_detector()
    except cv2.error:
        return None


def create_yunet_detector() -> Any:
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        REGISTRATION_DETECTION_THRESHOLD,
        0.30,
        5000,
    )


def run_yunet_inference(detector: Any, frame_bgr: np.ndarray) -> Any:
    height, width = frame_bgr.shape[:2]
    detector.setInputSize((width, height))
    return detector.detect(frame_bgr)[1]


def detect_faces_yunet(
    frame_bgr: np.ndarray,
    confidence_threshold: float = FACE_DETECTION_THRESHOLD,
) -> list[dict[str, Any]] | None:
    """Detect faces with YuNet; return None only when the model is unavailable."""
    detector = yunet_detector()
    if detector is None:
        return None
    height, width = frame_bgr.shape[:2]
    if height == 0 or width == 0:
        return []
    scale = min(1.0, 1280.0 / max(height, width))
    if scale < 1.0:
        inference_frame = cv2.resize(frame_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        inference_frame = frame_bgr
    # FaceDetectorYN.setInputSize mutates the cached detector. Serialize that
    # mutation and inference so Streamlit sessions cannot race each other.
    with YUNET_INFERENCE_LOCK:
        try:
            faces = run_yunet_inference(detector, inference_frame)
        except cv2.error:
            # Recover from a damaged/mismatched cached native object once.
            try:
                faces = run_yunet_inference(create_yunet_detector(), inference_frame)
            except cv2.error:
                return None
    if faces is None:
        return []
    detections = []
    for face in faces:
        if float(face[-1]) < confidence_threshold:
            continue
        geometry = [float(value) / scale for value in face[:14]]
        x, y, w, h = geometry[:4]
        detections.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "confidence": float(face[-1]),
                "face_geometry": geometry,
            }
        )
    return detections


def has_eye_pair(gray: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Confirm that a face candidate contains a plausible pair of eyes."""
    _, eye_detector = cascades()
    if eye_detector is None:
        raise RuntimeError("The facial-feature detector could not be loaded.")
    x, y, w, h = box
    upper_face = gray[y : y + int(h * 0.68), x : x + w]
    min_eye = max(10, min(w, h) // 10)
    eyes = eye_detector.detectMultiScale(
        upper_face,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(min_eye, min_eye),
    )
    centers = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes]
    for index, first in enumerate(centers):
        for second in centers[index + 1 :]:
            horizontal_gap = abs(first[0] - second[0])
            vertical_gap = abs(first[1] - second[1])
            if horizontal_gap >= w * 0.18 and vertical_gap <= h * 0.25:
                return True
    return False


def plausible_eye_count(
    eyes: Any,
    face_width: int,
    face_height: int,
) -> int:
    """Convert noisy Haar eye boxes into zero, one, or one plausible pair."""
    centers = [
        (float(ex) + float(ew) / 2, float(ey) + float(eh) / 2)
        for ex, ey, ew, eh in eyes
    ]
    for index, first in enumerate(centers):
        for second in centers[index + 1 :]:
            horizontal_gap = abs(first[0] - second[0])
            vertical_gap = abs(first[1] - second[1])
            if horizontal_gap >= face_width * 0.18 and vertical_gap <= face_height * 0.25:
                return 2
    return 1 if centers else 0


def count_visible_eyes(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
) -> int:
    """Count plausible visible eyes only inside the detected upper-face ROI."""
    _, eye_detector = cascades()
    if eye_detector is None:
        raise RuntimeError("The eye detector could not be loaded.")
    x, y, w, h = box
    x1, y1 = max(0, x), max(0, y)
    x2 = min(gray.shape[1], x + w)
    y2 = min(gray.shape[0], y + int(h * 0.68))
    upper_face = cv2.equalizeHist(gray[y1:y2, x1:x2])
    if upper_face.size == 0:
        return 0
    min_eye = max(10, min(w, h) // 10)
    eyes = eye_detector.detectMultiScale(
        upper_face,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(min_eye, min_eye),
    )
    return plausible_eye_count(eyes, w, h)


def _detect_face_candidates(
    frame_bgr: np.ndarray,
    confidence_threshold: float = FACE_DETECTION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return raw face-shaped candidates; callers must use ``detect_faces``."""
    yunet_detections = detect_faces_yunet(frame_bgr, confidence_threshold)
    if yunet_detections is not None:
        return yunet_detections

    # Offline fallback for environments where the neural model cannot be
    # downloaded. The eye-pair gate prevents object false positives.
    detectors = face_detectors()
    if not detectors:
        raise RuntimeError("The human-face detector could not be loaded.")
    gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    detections = []
    for detector, is_profile, offset, weight_scale in detectors:
        # Profile cascades are a fallback. Running them after a frontal match
        # can add small false duplicate boxes around facial features.
        if is_profile and detections:
            continue
        views = [(gray, False), (cv2.flip(gray, 1), True)] if is_profile else [(gray, False)]
        for view, mirrored in views:
            faces, _, feature_weights = detector.detectMultiScale3(
                view,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(32, 32),
                outputRejectLevels=True,
            )
            for (x, y, fw, fh), feature_weight in zip(faces, feature_weights):
                if mirrored:
                    x = gray.shape[1] - x - fw
                confidence = float(1.0 / (1.0 + np.exp(-(float(feature_weight) - offset) / weight_scale)))
                box = (int(x), int(y), int(fw), int(fh))
                if confidence >= confidence_threshold and has_eye_pair(gray, box):
                    detections.append({"box": box, "confidence": confidence})

    # Several cascades often find the same face. Keep only the strongest
    # overlapping result so the UI reports one person once.
    kept = []
    for candidate in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        x, y, w, h = candidate["box"]
        duplicate = False
        for existing in kept:
            ex, ey, ew, eh = existing["box"]
            overlap_w = max(0, min(x + w, ex + ew) - max(x, ex))
            overlap_h = max(0, min(y + h, ey + eh) - max(y, ey))
            intersection = overlap_w * overlap_h
            union = w * h + ew * eh - intersection
            if intersection / max(1, union) >= 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect_faces(
    frame_bgr: np.ndarray,
    confidence_threshold: float = FACE_DETECTION_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return only human-validated faces or reject an animal image.

    This is the single gate used by every active workflow.  YuNet/Haar first
    finds face-shaped regions; the independent animal classifier then filters
    them.  A full-image check distinguishes animal-only images from invalid
    images even when the human-specific detector correctly returns no boxes.
    """
    candidates = _detect_face_candidates(frame_bgr, confidence_threshold)
    result = validate_human_candidates(
        frame_bgr,
        candidates,
        ANIMAL_CLASSIFIER_MODEL_PATH,
    )
    if result.kind is FaceInputKind.ANIMAL:
        raise FaceInputRejected(FaceInputKind.ANIMAL)
    return list(result.detections)


def crop_face(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    pad = int(max(w, h) * 0.14)
    y1, y2 = max(0, y - pad), min(frame_bgr.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(frame_bgr.shape[1], x + w + pad)
    return frame_bgr[y1:y2, x1:x2]


@st.cache_resource
def sface_recognizer() -> Any:
    """Load the official OpenCV SFace recognition model."""
    if not hasattr(cv2, "FaceRecognizerSF"):
        return None
    if not SFACE_MODEL_PATH.is_file() or SFACE_MODEL_PATH.stat().st_size < 1_000_000:
        temporary_path = SFACE_MODEL_PATH.with_suffix(".download")
        try:
            with urllib.request.urlopen(SFACE_MODEL_URL, timeout=120) as response:
                model_bytes = response.read()
            if len(model_bytes) < 1_000_000:
                return None
            temporary_path.write_bytes(model_bytes)
            temporary_path.replace(SFACE_MODEL_PATH)
        except (OSError, ValueError):
            return None
    try:
        return cv2.FaceRecognizerSF.create(str(SFACE_MODEL_PATH), "")
    except cv2.error:
        return None


def embedding(face_bgr: np.ndarray) -> list[float]:
    """Safely embed a standalone image only after central human validation."""
    detections = detect_faces(face_bgr)
    if not detections:
        raise FaceInputRejected(FaceInputKind.NO_FACE)
    if len(detections) != 1:
        raise RuntimeError(f"Expected one human face; found {len(detections)}.")
    return embedding_from_detection(face_bgr, detections[0])


def sface_feature(recognizer: Any, aligned_face: np.ndarray) -> list[float]:
    with SFACE_INFERENCE_LOCK:
        feature = recognizer.feature(aligned_face).reshape(-1).astype("float32")
    feature /= np.linalg.norm(feature) + 1e-9
    return feature.tolist()


def embedding_from_detection(
    frame_bgr: np.ndarray,
    detection: dict[str, Any],
) -> list[float]:
    """Align a YuNet face by its landmarks before extracting SFace features."""
    # Defense in depth: even future code cannot send a raw detector box to the
    # embedding model without first passing the shared human/animal gate.
    require_validated_detection(detection)
    recognizer = sface_recognizer()
    if recognizer is None:
        raise RuntimeError("The face-recognition model could not be loaded.")
    return sface_feature(recognizer, aligned_face_from_detection(frame_bgr, detection))


def aligned_face_from_detection(
    frame_bgr: np.ndarray,
    detection: dict[str, Any],
) -> np.ndarray:
    """Return a landmark-aligned 112x112 face for SFace inference."""
    require_validated_detection(detection)
    recognizer = sface_recognizer()
    if recognizer is None:
        raise RuntimeError("The face-recognition model could not be loaded.")
    geometry = detection.get("face_geometry")
    if geometry:
        with SFACE_INFERENCE_LOCK:
            return recognizer.alignCrop(
                frame_bgr,
                np.asarray(geometry, dtype=np.float32),
            )
    face = crop_face(frame_bgr, detection["box"])
    if face.size == 0:
        raise RuntimeError("The detected face crop was empty.")
    return cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)


def registration_frame_variants(frame_bgr: np.ndarray) -> list[np.ndarray]:
    """Retry face detection with safe lighting/contrast corrections."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    enhanced_lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    contrast_frame = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    brighter_frame = cv2.convertScaleAbs(frame_bgr, alpha=1.12, beta=12)
    return [frame_bgr, contrast_frame, brighter_frame]


def registration_face_quality(
    frame_bgr: np.ndarray,
    detection: dict[str, Any],
) -> tuple[bool, str]:
    """Reject frames whose detected face is too dark, bright, small, or blurred."""
    face = crop_face(frame_bgr, detection["box"])
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, _, width, height = detection["box"]
    min_frame_side = min(frame_bgr.shape[:2])
    if min(width, height) < min_frame_side * 0.10:
        return False, "face is too small; move closer"
    if brightness < 30:
        return False, "frame is too dark"
    if brightness > 235:
        return False, "frame is overexposed"
    if sharpness < 20:
        return False, "frame is blurred; hold still and retry"
    return True, ""


def detect_registration_face(
    frame_bgr: np.ndarray,
) -> tuple[dict[str, Any] | None, str]:
    """Retry a registration frame and return one quality-approved face."""
    for attempt, variant in enumerate(registration_frame_variants(frame_bgr)):
        try:
            detections = detect_faces(variant, REGISTRATION_DETECTION_THRESHOLD)
        except cv2.error:
            continue
        if len(detections) > 1:
            return None, f"expected one face, found {len(detections)}"
        if len(detections) == 1:
            quality_ok, reason = registration_face_quality(frame_bgr, detections[0])
            if quality_ok:
                return detections[0], "" if attempt == 0 else "recovered by detection retry"
            return None, reason
    return None, NO_FACE_MESSAGE


def detect_liveness_face(
    frame_bgr: np.ndarray,
) -> tuple[dict[str, Any] | None, str]:
    """Find one human face in a blink/turn frame with bounded retries.

    Closed eyes, mild motion, and a side turn reduce YuNet confidence. Liveness
    therefore retries the original and safely enhanced frames at one lower
    threshold. Every attempt still passes through the shared human/animal gate;
    the lower detector threshold cannot create an embedding bypass.
    """
    for variant in registration_frame_variants(frame_bgr):
        for threshold in (
            REGISTRATION_DETECTION_THRESHOLD,
            LIVENESS_RECOVERY_DETECTION_THRESHOLD,
        ):
            try:
                detections = detect_faces(variant, threshold)
            except cv2.error:
                continue
            if len(detections) > 1:
                return None, f"Exactly one human face is required; found {len(detections)}."
            if len(detections) == 1:
                return detections[0], ""
    return None, NO_FACE_MESSAGE


def cosine(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    if av.size == 0 or bv.size == 0:
        return 0.0
    return float(np.dot(av, bv) / ((np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-9))


def verification_similarity(
    first_frame: np.ndarray,
    first_detection: dict[str, Any],
    second_frame: np.ndarray,
    second_detection: dict[str, Any],
) -> float:
    """Compare pose-tolerant, landmark-aligned SFace verification features."""
    first_embedding = verification_embedding(first_frame, first_detection)
    second_embedding = verification_embedding(second_frame, second_detection)
    return max(0.0, min(1.0, cosine(first_embedding, second_embedding)))


def verification_embedding(
    frame_bgr: np.ndarray,
    detection: dict[str, Any],
) -> list[float]:
    """Average original and mirrored aligned features to reduce pose sensitivity."""
    require_validated_detection(detection)
    recognizer = sface_recognizer()
    if recognizer is None:
        raise RuntimeError("The face-recognition model could not be loaded.")
    aligned_face = aligned_face_from_detection(frame_bgr, detection)
    original = np.asarray(sface_feature(recognizer, aligned_face), dtype=np.float32)
    mirrored = np.asarray(
        sface_feature(recognizer, cv2.flip(aligned_face, 1)),
        dtype=np.float32,
    )
    combined = original + mirrored
    combined /= np.linalg.norm(combined) + 1e-9
    return combined.tolist()


def verification_match_score(similarity: float) -> float:
    """Map raw cosine similarity to an intuitive, monotonic decision score."""
    return float(1.0 / (1.0 + math.exp(-12.0 * (similarity - VERIFICATION_COSINE_THRESHOLD))))


def liveness_face_state(detection: dict[str, Any]) -> dict[str, float]:
    """Extract translation and landmark-based turn evidence from one face."""
    x, _, width, _ = detection["box"]
    geometry = detection.get("face_geometry")
    yaw = 0.0
    if geometry and len(geometry) >= 10:
        eye_midpoint_x = (float(geometry[4]) + float(geometry[6])) / 2
        nose_x = float(geometry[8])
        yaw = (nose_x - eye_midpoint_x) / max(1.0, float(width))
    return {
        "center_x": float(x) + float(width) / 2,
        "width": max(1.0, float(width)),
        "yaw": yaw,
    }


def evaluate_liveness_evidence(
    eye_counts: list[int],
    first_state: dict[str, float],
    last_state: dict[str, float],
    identity_similarities: list[float],
    blink_required: bool = True,
    movement_required: bool = True,
) -> dict[str, Any]:
    """Evaluate the ordered neutral, blink, turn-head challenge."""
    # Haar may see only one neutral eye under glasses, low light, or mild pose.
    # Require an actual drop in the deliberately blinked second frame instead
    # of insisting that the neutral frame always contains a perfect pair.
    blink_ok = (
        len(eye_counts) == 3
        and eye_counts[0] >= 1
        and eye_counts[1] < eye_counts[0]
    )
    center_shift = abs(first_state["center_x"] - last_state["center_x"]) / first_state["width"]
    yaw_shift = abs(first_state["yaw"] - last_state["yaw"])
    movement = max(center_shift, yaw_shift)
    move_ok = movement >= LIVENESS_HEAD_MOVEMENT_THRESHOLD
    identity_score = min(identity_similarities) if identity_similarities else 0.0
    identity_ok = identity_score >= LIVENESS_IDENTITY_THRESHOLD
    live = identity_ok and (blink_ok or not blink_required) and (move_ok or not movement_required)

    if not identity_ok:
        reason = "The same face was not present in all three frames."
    elif blink_required and not blink_ok:
        reason = "Blink not confirmed: capture neutral first, blink second, then turn."
    elif movement_required and not move_ok:
        reason = "Head turn was too small; turn slightly farther for the third frame."
    else:
        reason = "Ordered blink, head turn, and same-face checks passed."
    return {
        "live": live,
        "reason": reason,
        "blink_ok": blink_ok,
        "move_ok": move_ok,
        "movement": movement,
        "identity_ok": identity_ok,
        "identity_score": identity_score,
        "eye_counts": eye_counts,
    }


def analyze_liveness_frames(
    frames: list[np.ndarray],
    blink_required: bool = True,
    movement_required: bool = True,
) -> dict[str, Any]:
    """Run face, eye, movement, and identity checks over three ordered frames."""
    if len(frames) != 3:
        return {"live": False, "reason": "Exactly three liveness frames are required."}

    detections = []
    eye_counts = []
    embeddings = []
    for index, frame in enumerate(frames):
        try:
            detection, detection_reason = detect_liveness_face(frame)
        except FaceInputRejected as error:
            return {"live": False, "reason": str(error), "frame_index": index}
        except (RuntimeError, cv2.error) as error:
            return {
                "live": False,
                "reason": str(error),
                "frame_index": index,
            }
        if detection is None:
            return {
                "live": False,
                "reason": detection_reason,
                "frame_index": index,
            }
        try:
            eye_counts.append(
                count_visible_eyes(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detection["box"])
            )
            embeddings.append(embedding_from_detection(frame, detection))
        except (RuntimeError, cv2.error) as error:
            return {"live": False, "reason": str(error), "frame_index": index}
        detections.append(detection)

    identity_similarities = [
        cosine(embeddings[0], embeddings[1]),
        cosine(embeddings[0], embeddings[2]),
    ]
    return evaluate_liveness_evidence(
        eye_counts,
        liveness_face_state(detections[0]),
        liveness_face_state(detections[2]),
        identity_similarities,
        blink_required,
        movement_required,
    )


def load_people() -> list[dict[str, Any]]:
    people = rows("people", "ORDER BY name")
    for person in people:
        person["embeddings"] = json.loads(person["embeddings"])
        person["image_paths"] = json.loads(person["image_paths"])
    return people


def recognize_embedding(
    probe: list[float],
    people: list[dict[str, Any]],
    threshold: float,
) -> MatchResult:
    """Accept only identities supported by several compatible embeddings."""
    best_person = None
    best_score = -1.0
    for person in people:
        if person["status"] != "Active":
            continue
        compatible_embeddings = [item for item in person["embeddings"] if len(item) == len(probe)]
        scores = sorted(
            (cosine(probe, item) for item in compatible_embeddings),
            reverse=True,
        )
        # Averaging the best three samples prevents one accidental nearest
        # neighbor from forcing a match when only one identity is enrolled.
        support_count = min(3, len(scores))
        person_score = float(np.mean(scores[:support_count])) if support_count else -1.0
        if person_score > best_score:
            best_score = person_score
            best_person = person
    similarity = max(0.0, min(1.0, best_score))
    if best_person and best_score >= threshold:
        return MatchResult(best_person, similarity, 1 - similarity)
    return MatchResult(None, similarity, 1 - similarity)


def recognize(frame_bgr: np.ndarray, detection: dict[str, Any]) -> MatchResult:
    probe = embedding_from_detection(frame_bgr, detection)
    return recognize_embedding(
        probe,
        load_people(),
        setting("recognition_threshold", float),
    )


def draw_boxes(frame_bgr: np.ndarray, detections: list[dict[str, Any]], labels: list[str] | None = None) -> Image.Image:
    output = frame_bgr.copy()
    for idx, detection in enumerate(detections):
        x, y, w, h = detection["box"]
        label = labels[idx] if labels and idx < len(labels) else f"Face {detection['confidence']:.0%}"
        rejected = any(term in label for term in ("Unknown", "Not registered", "Spoof"))
        color = (32, 201, 151) if labels and not rejected else (0, 86, 255)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(output, (x, max(0, y - 30)), (x + min(w + 90, 420), y), color, -1)
        cv2.putText(output, label[:46], (x + 8, max(20, y - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


def save_bgr(frame_bgr: np.ndarray, folder: Path, prefix: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(str(path), frame_bgr)
    return str(path)


def mark_attendance(person: dict[str, Any], confidence: float, camera: str = "Manual Scan") -> tuple[bool, str]:
    duplicate_minutes = setting("duplicate_minutes", int)
    since = (datetime.now() - timedelta(minutes=duplicate_minutes)).isoformat(sep=" ")
    existing = rows(
        "attendance",
        "WHERE person_id = ? AND marked_at >= ? ORDER BY marked_at DESC LIMIT 1",
        (person["person_id"], since),
    )
    if existing:
        return False, f"Already marked within {duplicate_minutes} minutes."
    execute(
        "INSERT INTO attendance(person_id, name, department, status, confidence, marked_at, camera) VALUES(?, ?, ?, ?, ?, ?, ?)",
        (person["person_id"], person["name"], person["department"], "Present", confidence, now_iso(), camera),
    )
    log("Attendance marked", f"{person['name']} ({person['person_id']})")
    return True, "Attendance marked successfully."


def register_unknown(face_bgr: np.ndarray, confidence: float, camera: str) -> None:
    path = save_bgr(face_bgr, UNKNOWN_DIR, "unknown")
    execute(
        "INSERT INTO unknown_visitors(image_path, confidence, detected_at, camera, note) VALUES(?, ?, ?, ?, ?)",
        (path, confidence, now_iso(), camera, "Unknown face detected"),
    )
    log("Unknown visitor", f"Saved alert image {Path(path).name}")


def simple_pdf_bytes(df: pd.DataFrame, title: str) -> bytes:
    lines = [title, f"Generated: {now_iso()}", ""]
    if df.empty:
        lines.append("No records found.")
    else:
        lines.extend([", ".join(map(str, df.columns))])
        for _, row in df.head(120).iterrows():
            lines.append(", ".join(str(row[col])[:32] for col in df.columns))
    content = "\n".join(lines)
    stream = io.BytesIO()
    stream.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n")
    stream.write(b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n")
    safe = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    text = "BT /F1 9 Tf 40 780 Td 12 TL " + " T* ".join(f"({line})" for line in safe.splitlines()[:58]) + " ET"
    stream.write(f"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n".encode())
    stream.write(b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n")
    stream.write(f"5 0 obj<</Length {len(text)}>>stream\n{text}\nendstream endobj\n".encode())
    stream.write(b"xref\n0 6\n0000000000 65535 f \n")
    stream.write(b"trailer<</Root 1 0 R/Size 6>>\nstartxref\n0\n%%EOF")
    return stream.getvalue()


def to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Report")
    return output.getvalue()


def image_download_link(path: str) -> str:
    try:
        data = Path(path).read_bytes()
        encoded = base64.b64encode(data).decode()
        return f"data:image/jpeg;base64,{encoded}"
    except OSError:
        return ""


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --void:#030712;
            --panel:#07111f;
            --panel-2:#0b1728;
            --line:rgba(56,189,248,.42);
            --blue:#38bdf8;
            --blue-hot:#0ea5e9;
            --cyan:#67e8f9;
            --text:#e5f7ff;
            --muted:#8ddcff;
            --danger:#fb7185;
            --ok:#5eead4;
            --warn:#facc15;
            --glow:0 0 18px rgba(56,189,248,.30), 0 0 40px rgba(14,165,233,.18);
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 8%, rgba(56,189,248,.18), transparent 30%),
                linear-gradient(135deg, #020617 0%, #07111f 48%, #030712 100%);
            color:var(--text);
        }
        .stApp::before {
            content:"";
            position:fixed;
            inset:0;
            pointer-events:none;
            background-image:
                linear-gradient(rgba(56,189,248,.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(56,189,248,.06) 1px, transparent 1px);
            background-size:44px 44px;
            mask-image:linear-gradient(to bottom, rgba(0,0,0,.72), transparent 78%);
        }
        [data-testid="stSidebar"] {
            background:rgba(3,7,18,.92);
            border-right:1px solid var(--line);
            box-shadow:18px 0 44px rgba(0,0,0,.35);
        }
        [data-testid="stSidebar"] * { color:var(--text); }
        h1, h2, h3, h4, h5, h6 {
            color:var(--text);
            letter-spacing:0;
            text-shadow:0 0 18px rgba(56,189,248,.34);
        }
        h1 {
            font-weight:900;
            background:linear-gradient(90deg, #ffffff, var(--cyan), var(--blue));
            -webkit-background-clip:text;
            background-clip:text;
            color:transparent;
        }
        p, label, span, li, div, small { color:var(--text); }
        a { color:var(--cyan) !important; }
        code {
            color:var(--cyan);
            background:rgba(56,189,248,.12);
            border:1px solid rgba(56,189,248,.28);
            border-radius:6px;
        }
        .metric-card {
            background:linear-gradient(145deg, rgba(7,17,31,.96), rgba(14,35,58,.90));
            border:1px solid var(--line);
            border-radius:8px;
            padding:18px;
            box-shadow:var(--glow), inset 0 1px 0 rgba(255,255,255,.08);
            min-height:112px;
            color:var(--text);
            transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
        }
        .metric-card:hover {
            transform:translateY(-2px);
            border-color:rgba(103,232,249,.82);
            box-shadow:0 0 28px rgba(56,189,248,.42), 0 0 60px rgba(14,165,233,.24);
        }
        .metric-label { color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; }
        .metric-value { color:var(--cyan); font-size:2rem; font-weight:900; line-height:1.1; text-shadow:0 0 16px rgba(103,232,249,.55); }
        .metric-sub { color:var(--muted); font-size:.88rem; margin-top:8px; }
        .status-ok { color:var(--ok); font-weight:800; }
        .status-warn { color:var(--warn); font-weight:800; }
        .status-danger { color:var(--danger); font-weight:800; }
        .block-container { padding-top:1.8rem; max-width:1420px; }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius:8px;
            font-weight:900;
            color:#02111f;
            background:linear-gradient(90deg, var(--cyan), var(--blue));
            border:1px solid rgba(103,232,249,.70);
            box-shadow:0 0 18px rgba(56,189,248,.32);
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            color:#020617;
            border-color:#ffffff;
            box-shadow:0 0 26px rgba(103,232,249,.58);
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div,
        [data-testid="stMultiSelect"] div {
            background:rgba(5,15,29,.96) !important;
            color:var(--cyan) !important;
            border-color:rgba(56,189,248,.55) !important;
            box-shadow:inset 0 0 0 1px rgba(56,189,248,.12), 0 0 14px rgba(56,189,248,.16);
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color:rgba(141,220,255,.70) !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color:var(--cyan) !important;
            box-shadow:0 0 0 1px var(--cyan), 0 0 22px rgba(103,232,249,.30) !important;
        }
        div[data-testid="stAlert"] {
            background:rgba(7,17,31,.96);
            color:var(--cyan);
            border:1px solid var(--line);
            border-radius:8px;
            box-shadow:var(--glow);
        }
        div[data-testid="stAlert"] * { color:var(--cyan) !important; }
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"],
        [data-testid="stForm"] {
            background:rgba(7,17,31,.74);
            border:1px solid rgba(56,189,248,.28);
            border-radius:8px;
            box-shadow:0 0 22px rgba(14,165,233,.14);
            padding:10px;
        }
        [data-testid="stTabs"] button { color:var(--muted); }
        [data-testid="stTabs"] button[aria-selected="true"] { color:var(--cyan); }
        [data-baseweb="radio"] label,
        [data-baseweb="checkbox"] label {
            background:rgba(7,17,31,.70);
            border-radius:8px;
            color:var(--text);
        }
        .alert-box {
            border-left:5px solid var(--danger);
            background:rgba(251,113,133,.12);
            color:var(--text);
            padding:14px 16px;
            border-radius:8px;
            margin:8px 0 16px;
            box-shadow:0 0 18px rgba(251,113,133,.16);
        }
        .soft-box {
            background:linear-gradient(145deg, rgba(7,17,31,.96), rgba(8,31,54,.86));
            border:1px solid var(--line);
            border-radius:8px;
            padding:16px; margin-bottom:14px;
            color:var(--cyan);
            box-shadow:var(--glow);
        }
        .soft-box,
        .soft-box * {
            color:var(--cyan) !important;
            text-shadow:0 0 10px rgba(103,232,249,.28);
        }
        .soft-box b,
        .soft-box strong {
            color:var(--cyan);
            text-shadow:0 0 12px rgba(103,232,249,.45);
        }
        .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:rgba(56,189,248,.16); color:var(--cyan); border:1px solid var(--line); font-weight:800; }
        .identix-hero {
            position:relative;
            overflow:hidden;
            min-height:520px;
            padding:42px 44px;
            border:1px solid rgba(103,232,249,.30);
            border-radius:24px;
            background:
                radial-gradient(circle at 82% 18%, rgba(34,211,238,.18), transparent 30%),
                linear-gradient(145deg, rgba(7,17,31,.97), rgba(6,29,49,.88));
            box-shadow:0 32px 80px rgba(0,0,0,.34), 0 0 42px rgba(14,165,233,.15), inset 0 1px 0 rgba(255,255,255,.08);
        }
        .identix-hero::after {
            content:"";
            position:absolute;
            width:220px;
            height:220px;
            right:-72px;
            bottom:-78px;
            border:1px solid rgba(103,232,249,.22);
            border-radius:50%;
            box-shadow:0 0 0 34px rgba(56,189,248,.035), 0 0 0 68px rgba(56,189,248,.025);
        }
        .identix-brand {
            display:flex;
            align-items:center;
            gap:18px;
            margin-bottom:48px;
        }
        .identix-logo {
            width:76px;
            height:76px;
            flex:0 0 76px;
            border-radius:22px;
            background:linear-gradient(145deg, rgba(103,232,249,.18), rgba(14,165,233,.08));
            border:1px solid rgba(103,232,249,.48);
            box-shadow:0 0 32px rgba(56,189,248,.25), inset 0 0 24px rgba(56,189,248,.10);
            display:grid;
            place-items:center;
        }
        .identix-logo svg { width:54px; height:54px; filter:drop-shadow(0 0 9px rgba(103,232,249,.60)); }
        .identix-wordmark {
            color:#f2fbff !important;
            font-size:2.05rem;
            line-height:1;
            font-weight:950;
            letter-spacing:.16em;
            text-transform:uppercase;
        }
        .identix-tagline {
            color:var(--muted) !important;
            font-size:.72rem;
            font-weight:800;
            letter-spacing:.18em;
            text-transform:uppercase;
            margin-top:8px;
        }
        .identix-kicker {
            display:inline-flex;
            align-items:center;
            gap:8px;
            color:var(--cyan) !important;
            font-size:.78rem;
            font-weight:900;
            letter-spacing:.14em;
            text-transform:uppercase;
            margin-bottom:16px;
        }
        .identix-kicker::before { content:""; width:28px; height:2px; background:var(--cyan); box-shadow:0 0 12px var(--cyan); }
        .identix-headline {
            max-width:720px;
            color:#f5fcff !important;
            font-size:clamp(2.25rem, 4.2vw, 4.25rem);
            line-height:1.02;
            font-weight:950;
            letter-spacing:-.045em;
            margin:0 0 20px;
            text-wrap:balance;
        }
        .identix-headline span {
            background:linear-gradient(90deg, var(--cyan), var(--blue));
            -webkit-background-clip:text;
            background-clip:text;
            color:transparent !important;
        }
        .identix-copy {
            max-width:660px;
            color:#b9dded !important;
            font-size:1.02rem;
            line-height:1.7;
            margin-bottom:28px;
        }
        .identix-features { display:flex; flex-wrap:wrap; gap:10px; }
        .identix-feature {
            color:#d9f7ff !important;
            font-size:.78rem;
            font-weight:800;
            padding:8px 12px;
            border-radius:999px;
            border:1px solid rgba(103,232,249,.24);
            background:rgba(3,14,27,.55);
        }
        .identix-feature b { color:var(--ok) !important; margin-right:6px; }
        .demo-access {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:16px;
            padding:13px 16px;
            margin-top:14px;
            border:1px solid rgba(103,232,249,.20);
            border-radius:12px;
            background:rgba(3,14,27,.62);
        }
        .demo-access-label { color:var(--muted) !important; font-size:.75rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
        .demo-access-values { color:#dff9ff !important; font-size:.86rem; }
        .login-panel-title { margin:10px 0 2px; color:#f5fcff !important; font-size:2rem; font-weight:950; }
        .login-panel-copy { color:#9dc9dc !important; margin-bottom:20px; }
        @media (max-width: 900px) {
            .identix-hero { min-height:auto; padding:28px 24px; }
            .identix-brand { margin-bottom:32px; }
            .identix-headline { font-size:2.5rem; }
            .demo-access { align-items:flex-start; flex-direction:column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: Any, sub: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def require_role(allowed: list[str]) -> bool:
    role = st.session_state["user"]["role"]
    if role not in allowed:
        st.error("You do not have permission to access this area.")
        return False
    return True


def login_screen() -> None:
    apply_theme()
    left, right = st.columns([1.28, 0.72], gap="large", vertical_alignment="center")
    with left:
        st.markdown(
            """
            <section class="identix-hero">
                <div class="identix-brand">
                    <div class="identix-logo" aria-label="Identix logo">
                        <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                            <path d="M18 8H11a3 3 0 0 0-3 3v7M46 8h7a3 3 0 0 1 3 3v7M18 56H11a3 3 0 0 1-3-3v-7M46 56h7a3 3 0 0 0 3-3v-7" stroke="#67E8F9" stroke-width="3" stroke-linecap="round"/>
                            <path d="M21 28c1-8 5-13 11-13s10 5 11 13" stroke="#38BDF8" stroke-width="3" stroke-linecap="round"/>
                            <path d="M21 29v5c0 8 5 14 11 14s11-6 11-14v-5" stroke="#E5F7FF" stroke-width="3" stroke-linecap="round"/>
                            <path d="M25 31h2M37 31h2M28 40c2.5 2 5.5 2 8 0" stroke="#67E8F9" stroke-width="2.5" stroke-linecap="round"/>
                            <path d="M14 32h36" stroke="#0EA5E9" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="3 4" opacity=".8"/>
                        </svg>
                    </div>
                    <div>
                        <div class="identix-wordmark">Identix</div>
                        <div class="identix-tagline">Identity intelligence</div>
                    </div>
                </div>
                <div class="identix-kicker">AI-powered recognition</div>
                <h1 class="identix-headline">See the person.<br><span>Know the identity.</span></h1>
                <p class="identix-copy">A secure command center for face recognition, attendance, access control, and visitor intelligence.</p>
                <div class="identix-features">
                    <span class="identix-feature"><b>●</b>Face intelligence</span>
                    <span class="identix-feature"><b>●</b>Access controls</span>
                    <span class="identix-feature"><b>●</b>Smart attendance</span>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="identix-kicker">Secure access</div>
            <div class="login-panel-title">Welcome back</div>
            <div class="login-panel-copy">Sign in to enter the Identix command center.</div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login"):
            identifier = st.text_input("Username, display name, or email", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Enter Identix", width="stretch")
        if default_admin_credentials_active():
            st.markdown(
                """
                <div class="demo-access">
                    <span class="demo-access-label">Demo access</span>
                    <span class="demo-access-values"><code>admin</code> &nbsp;·&nbsp; <code>admin123</code></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="demo-access">
                    <span class="demo-access-label">Sign-in options</span>
                    <span class="demo-access-values">Use your username, display name, or email.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if submitted:
            user = authenticate_user(identifier, password)
            if user:
                st.session_state["user"] = user
                log("Login", "Successful login")
                st.toast("Welcome back.", icon="✅")
                st.rerun()
            else:
                st.error("Invalid username or password.")


def sidebar() -> str:
    user = st.session_state["user"]
    st.sidebar.title("AI Face System")
    st.sidebar.caption(f"{user['display_name']} · {user['role']}")
    base_pages = ["Dashboard", "Face Detection", "Face Registration", "Face Recognition", "Face Verification", "Attendance", "Unknown Alerts", "Reports", "Cameras", "Profile", "Help", "About"]
    if user["role"] == "Administrator":
        pages = base_pages + ["User Management", "Settings", "Activity Logs"]
    elif user["role"] == "Teacher/Manager":
        pages = [p for p in base_pages if p not in ("Face Registration",)] + ["Activity Logs"]
    else:
        pages = ["Dashboard", "Face Recognition", "Face Verification", "Attendance", "Profile", "Help", "About"]
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")
    st.sidebar.divider()
    dark = st.sidebar.toggle("Dark mode", value=setting("dark_mode", bool))
    if dark != setting("dark_mode", bool):
        set_setting("dark_mode", dark)
        st.rerun()
    if st.sidebar.button("Log out", width="stretch"):
        log("Logout")
        st.session_state.pop("user", None)
        st.rerun()
    return page


def dashboard() -> None:
    st.title("Command Dashboard")
    people = rows("people")
    attendance = pd.DataFrame(rows("attendance", "ORDER BY marked_at DESC"))
    unknowns = pd.DataFrame(rows("unknown_visitors", "ORDER BY detected_at DESC"))
    today = date.today().isoformat()
    today_count = 0 if attendance.empty else attendance[attendance["marked_at"].str.startswith(today)].shape[0]
    week_start = (date.today() - timedelta(days=6)).isoformat()
    week_count = 0 if attendance.empty else attendance[attendance["marked_at"] >= week_start].shape[0]
    month_prefix = date.today().strftime("%Y-%m")
    month_count = 0 if attendance.empty else attendance[attendance["marked_at"].str.startswith(month_prefix)].shape[0]
    unknown_count = len(unknowns)
    avg_acc = 0 if attendance.empty else int(attendance["confidence"].mean() * 100)
    cams = rows("cameras")
    active_cams = sum(1 for c in cams if c["status"] in ("Online", "Configured"))
    cols = st.columns(4)
    with cols[0]: metric_card("Registered Users", len(people), "Active identities")
    with cols[1]: metric_card("Today's Attendance", today_count, "Automatic and manual scans")
    with cols[2]: metric_card("Unknown Visitors", unknown_count, "Saved alert events")
    with cols[3]: metric_card("Recognition Accuracy", f"{avg_acc}%", "Average accepted confidence")
    cols = st.columns(3)
    with cols[0]: metric_card("Weekly Attendance", week_count, "Last 7 days")
    with cols[1]: metric_card("Monthly Attendance", month_count, month_prefix)
    with cols[2]: metric_card("Camera Status", f"{active_cams}/{len(cams)}", "Configured or online")

    st.subheader("Attendance Trends")
    if attendance.empty:
        st.info("Attendance charts will appear after the first recognition event.")
    else:
        attendance["day"] = pd.to_datetime(attendance["marked_at"]).dt.date.astype(str)
        daily = attendance.groupby("day", as_index=False).size().rename(columns={"size": "records"})
        st.altair_chart(alt.Chart(daily).mark_line(point=True).encode(x="day", y="records").properties(height=260), width="stretch")
        c1, c2 = st.columns(2)
        dept = attendance.groupby("department", as_index=False).size().rename(columns={"size": "records"})
        c1.altair_chart(alt.Chart(dept).mark_bar().encode(x="department", y="records", color="department").properties(height=260), width="stretch")
        status = attendance.groupby("status", as_index=False).size().rename(columns={"size": "records"})
        c2.altair_chart(alt.Chart(status).mark_arc(innerRadius=45).encode(theta="records", color="status").properties(height=260), width="stretch")

    st.subheader("Recent Activity")
    st.dataframe(pd.DataFrame(rows("activity_logs", "ORDER BY created_at DESC LIMIT 12")), width="stretch", hide_index=True)


def face_detection() -> None:
    st.title("Face Detection")
    mode = st.segmented_control("Input source", ["Upload Image", "Webcam Capture"], default="Upload Image")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) if mode == "Upload Image" else st.camera_input("Capture from webcam")
    if image_file:
        frame = uploaded_to_bgr(image_file)
        try:
            detections = detect_faces(frame)
        except RuntimeError as error:
            st.error(str(error))
            return
        st.image(draw_boxes(frame, detections), caption=f"{len(detections)} face(s) detected", width="stretch")
        if detections:
            st.dataframe(pd.DataFrame([{"Face": i + 1, "Confidence": f"{d['confidence']:.1%}", "Box": d["box"]} for i, d in enumerate(detections)]), hide_index=True, width="stretch")
        else:
            st.warning(NO_FACE_MESSAGE)


def face_registration() -> None:
    if not require_role(["Administrator"]):
        return
    st.title("Face Registration")
    st.write("Register a person with identity details and multiple face samples.")
    with st.form("registration_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Full Name")
        person_id = c2.text_input("ID")
        department = c3.text_input("Department")
        email = c1.text_input("Email")
        phone = c2.text_input("Phone Number")
        status = c3.selectbox("Status", ["Active", "Inactive"])
        uploads = st.file_uploader("Upload multiple clear face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        camera_sample = st.camera_input("Optional webcam sample")
        replace_existing = st.checkbox("Replace an existing registration with this ID")
        submitted = st.form_submit_button("Register Face", width="stretch")
    if submitted:
        errors = []
        if not all([name.strip(), person_id.strip(), department.strip(), email.strip(), phone.strip()]):
            errors.append("All identity fields are required.")
        if "@" not in email:
            errors.append("Enter a valid email address.")
        if len(phone) < 6:
            errors.append("Enter a valid phone number.")
        files = list(uploads or [])
        if camera_sample:
            files.append(camera_sample)
        if len(files) < 2:
            errors.append("Provide at least two face images for a stronger registration.")
        existing_person = rows("people", "WHERE person_id = ?", (person_id,))
        if existing_person and not replace_existing:
            errors.append("This ID is already registered. Select replacement to enroll it again.")
        if errors:
            for error in errors:
                st.error(error)
            return
        accepted_samples = []
        failed_samples = []
        for index, file in enumerate(files):
            frame = uploaded_to_bgr(file)
            try:
                detection, reason = detect_registration_face(frame)
            except FaceInputRejected as error:
                # An animal is never a retryable registration sample: stop the
                # entire enrollment before any image or embedding is stored.
                st.error(str(error))
                return
            except RuntimeError as error:
                st.error(str(error))
                return
            if detection is None:
                failed_samples.append(index + 1)
                if reason == NO_FACE_MESSAGE:
                    st.warning(NO_FACE_MESSAGE)
                    st.caption(f"Registration sample {index + 1} needs to be retaken.")
                else:
                    st.warning(f"Retry sample {index + 1}: {reason}.")
                continue
            face = crop_face(frame, detection["box"])
            try:
                sample_embedding = embedding_from_detection(frame, detection)
            except RuntimeError as error:
                st.error(str(error))
                return
            accepted_samples.append((index + 1, face, sample_embedding))

        required_samples = max(2, math.ceil(len(files) * 0.90))
        if len(accepted_samples) < required_samples:
            retry_list = ", ".join(str(number) for number in failed_samples)
            st.error(
                f"Registration paused: {len(accepted_samples)}/{len(files)} usable samples. "
                f"Retake samples {retry_list}; at least {required_samples} are required."
            )
            return

        person_dir = REGISTERED_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        embs, image_paths = [], []
        for sample_number, face, sample_embedding in accepted_samples:
            embs.append(sample_embedding)
            image_paths.append(save_bgr(face, person_dir, f"sample_{sample_number}"))

        if existing_person:
            execute(
                """
                UPDATE people SET name=?, email=?, phone=?, department=?, embeddings=?, image_paths=?, status=?, updated_at=?
                WHERE person_id=?
                """,
                (name, email, phone, department, json.dumps(embs), json.dumps(image_paths), status, now_iso(), person_id),
            )
        else:
            execute(
                """
                INSERT INTO people(person_id, name, email, phone, department, embeddings, image_paths, status, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (person_id, name, email, phone, department, json.dumps(embs), json.dumps(image_paths), status, now_iso(), now_iso()),
            )
        log("Face registration", f"{name} ({person_id})")
        st.success(f"Registration successful for {name}. Stored {len(embs)} face embeddings.")


def process_recognition_frame(frame: np.ndarray, camera_name: str, mark: bool) -> None:
    try:
        detections = detect_faces(frame)
    except RuntimeError as error:
        st.error(str(error))
        return
    labels = []
    records = []
    if not detections:
        st.warning(NO_FACE_MESSAGE)
        return
    for detection in detections:
        face = crop_face(frame, detection["box"])
        try:
            match = recognize(frame, detection)
        except RuntimeError as error:
            st.error(str(error))
            return
        if match.person:
            person = match.person
            labels.append(f"{person['name']} · {match.confidence:.0%}")
            marked_message = ""
            if mark:
                _, marked_message = mark_attendance(person, match.confidence, camera_name)
            records.append(
                {
                    "Name": person["name"],
                    "ID": person["person_id"],
                    "Department": person["department"],
                    "Recognition Confidence": f"{match.confidence:.1%}",
                    "Attendance": marked_message or "Recognition only",
                }
            )
        else:
            labels.append("Not registered")
            register_unknown(face, detection["confidence"], camera_name)
            records.append({"Name": "Not registered", "ID": "-", "Department": "-", "Recognition Confidence": f"{match.confidence:.1%}", "Attendance": "Unknown alert saved"})
    st.image(draw_boxes(frame, detections, labels), width="stretch")
    if any(r["Name"] == "Not registered" for r in records):
        st.warning("This person has never registered before. Kindly register first.")
    st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)


def face_recognition() -> None:
    st.title("Face Recognition")
    if not load_people():
        st.info("Register at least one person before recognition.")
    mark = st.toggle("Automatically mark attendance after recognition", value=True)
    camera_name = st.selectbox("Camera label", [c["name"] for c in rows("cameras", "ORDER BY name")] or ["Manual Scan"])
    mode = st.segmented_control("Input source", ["Webcam Capture", "Upload Image"], default="Webcam Capture")
    image_file = st.camera_input("Capture recognition frame") if mode == "Webcam Capture" else st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if image_file:
        process_recognition_frame(uploaded_to_bgr(image_file), camera_name, mark)


def verification() -> None:
    st.title("Face Verification")
    c1, c2 = st.columns(2)
    first = c1.file_uploader("First face image", type=["jpg", "jpeg", "png"], key="verify_a")
    second = c2.file_uploader("Second face image", type=["jpg", "jpeg", "png"], key="verify_b")
    if first and second:
        frames = [uploaded_to_bgr(first), uploaded_to_bgr(second)]
        faces = []
        detections_by_frame = []
        for idx, frame in enumerate(frames):
            try:
                detections = detect_faces(frame)
            except RuntimeError as error:
                st.error(str(error))
                return
            if not detections:
                st.error(NO_FACE_MESSAGE)
                return
            if len(detections) != 1:
                st.error(f"Image {idx + 1} must contain exactly one detectable face. Found {len(detections)}.")
                return
            detections_by_frame.append(detections[0])
            faces.append(crop_face(frame, detections[0]["box"]))
        try:
            similarity = verification_similarity(
                frames[0], detections_by_frame[0], frames[1], detections_by_frame[1]
            )
        except RuntimeError as error:
            st.error(str(error))
            return
        same = similarity >= VERIFICATION_COSINE_THRESHOLD
        st.metric("Match score", f"{verification_match_score(similarity):.1%}")
        st.caption(
            f"Feature-based decision score, not model accuracy. Raw cosine similarity: {similarity:.3f} · "
            f"acceptance threshold: {VERIFICATION_COSINE_THRESHOLD:.2f}"
        )
        if same:
            st.success("Same person")
        else:
            st.error("Different people")
        st.image([Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in faces], width=260)


def liveness() -> None:
    st.title("Liveness Detection")
    st.write("Complete the three ordered steps below. Each frame is checked for exactly one human face before it is accepted.")
    if "live_frames" not in st.session_state:
        st.session_state.live_frames = []
    frame_steps = (
        "Look straight at the camera with both eyes open.",
        "Keep your face centered and close your eyes for a clear blink.",
        "Open your eyes and turn your head slightly left or right.",
    )
    completed = min(len(st.session_state.live_frames), 3)
    if completed < 3:
        st.info(f"Step {completed + 1} of 3: {frame_steps[completed]}")
    else:
        st.success("All three frames are ready. Run liveness verification.")
    sample = st.camera_input("Capture liveness frame")
    if sample and st.button(
        "Validate and add this frame",
        disabled=completed >= 3,
        width="stretch",
    ):
        frame = uploaded_to_bgr(sample)
        error_displayed = False
        try:
            detection, reason = detect_liveness_face(frame)
        except FaceInputRejected as error:
            st.error(str(error))
            detection, reason = None, str(error)
            error_displayed = True
        except (RuntimeError, cv2.error) as error:
            st.error(str(error))
            detection, reason = None, str(error)
            error_displayed = True
        if detection is None:
            if not error_displayed:
                st.error(reason)
            st.caption(f"Step {completed + 1} was not saved. Retake that frame with your full face visible and well lit.")
        else:
            st.session_state.live_frames.append(frame)
            st.toast(f"Step {completed + 1} accepted.", icon="✅")
            st.rerun()
    cols = st.columns(3)
    frame_names = ("Neutral", "Blink", "Head turn")
    for i, frame in enumerate(st.session_state.live_frames[:3]):
        cols[i].image(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            caption=f"Step {i + 1}: {frame_names[i]}",
            width="stretch",
        )
    if st.button("Run liveness verification", disabled=len(st.session_state.live_frames) < 3, width="stretch"):
        result = analyze_liveness_frames(
            st.session_state.live_frames[:3],
            setting("blink_required", bool),
            setting("head_movement_required", bool),
        )
        if result["live"]:
            st.success("Live Person")
        else:
            if result["reason"] in (ANIMAL_FACE_MESSAGE, NO_FACE_MESSAGE):
                st.error(result["reason"])
            else:
                st.error(f"Liveness not verified: {result['reason']}")
            if "frame_index" in result:
                failed_index = int(result["frame_index"])
                st.caption(f"Problem found in step {failed_index + 1}: {frame_names[failed_index]}. Retake the three-frame session.")
        if "eye_counts" in result:
            st.write(
                {
                    "Blink evidence": result["blink_ok"],
                    "Head movement": f"{result['movement']:.1%}",
                    "Eye detections": result["eye_counts"],
                    "Same-face consistency": f"{result['identity_score']:.1%}",
                }
            )
    if st.button("Reset liveness frames"):
        st.session_state.live_frames = []
        st.rerun()


def attendance_page() -> None:
    st.title("Attendance Management")
    df = pd.DataFrame(rows("attendance", "ORDER BY marked_at DESC"))
    c1, c2, c3, c4 = st.columns(4)
    query = c1.text_input("Search")
    departments = ["All"] + sorted(df["department"].dropna().unique().tolist()) if not df.empty else ["All"]
    dept = c2.selectbox("Department", departments)
    start = c3.date_input("From", value=date.today() - timedelta(days=30))
    end = c4.date_input("To", value=date.today())
    if not df.empty:
        df["date"] = pd.to_datetime(df["marked_at"]).dt.date
        mask = (df["date"] >= start) & (df["date"] <= end)
        if dept != "All":
            mask &= df["department"].eq(dept)
        if query:
            mask &= df.apply(lambda r: query.lower() in " ".join(map(str, r.values)).lower(), axis=1)
        df = df[mask].drop(columns=["date"])
    st.dataframe(df, width="stretch", hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV", df.to_csv(index=False).encode(), "attendance.csv", "text/csv", width="stretch")
    c2.download_button("Download Excel", to_excel(df), "attendance.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    c3.download_button("Download PDF", simple_pdf_bytes(df, "Attendance Report"), "attendance.pdf", "application/pdf", width="stretch")
    if require_role(["Administrator", "Teacher/Manager"]):
        st.subheader("Edit or Delete Record")
        ids = df["id"].tolist() if not df.empty else []
        selected = st.selectbox("Record ID", ids) if ids else None
        if selected:
            c1, c2 = st.columns(2)
            new_status = c1.selectbox("Status", ["Present", "Late", "Excused", "Absent"])
            if c1.button("Update status"):
                execute("UPDATE attendance SET status = ? WHERE id = ?", (new_status, selected))
                log("Attendance edited", f"Record {selected} set to {new_status}")
                st.rerun()
            if c2.button("Delete record"):
                execute("DELETE FROM attendance WHERE id = ?", (selected,))
                log("Attendance deleted", f"Record {selected}")
                st.rerun()


def unknown_alerts() -> None:
    st.title("Unknown Visitors Log")
    df = pd.DataFrame(rows("unknown_visitors", "ORDER BY detected_at DESC"))
    if df.empty:
        st.info("No unknown visitor alerts yet.")
        return
    if st.session_state["user"]["role"] == "Administrator":
        with st.expander("Privacy controls"):
            st.caption("Permanently delete every unknown-alert image and its log entry. Registered people and attendance are preserved.")
            confirmed = st.checkbox("I understand these unknown-alert images cannot be recovered.")
            if st.button(
                "Delete all unknown alerts",
                type="primary",
                disabled=not confirmed,
                width="stretch",
            ):
                deleted = clear_unknown_alerts_data()
                log("Unknown alerts cleared", f"Deleted {deleted} unknown alert record(s)")
                st.success(f"Deleted {deleted} unknown alert record(s) and all stored alert images.")
                st.rerun()
    search = st.text_input("Search alerts")
    if search:
        df = df[df.apply(lambda r: search.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    per_page = st.selectbox("Rows per page", [10, 25, 50], index=0)
    page_count = max(1, int(np.ceil(len(df) / per_page)))
    page = st.number_input("Page", min_value=1, max_value=page_count, value=1)
    shown = df.iloc[(page - 1) * per_page : page * per_page]
    st.dataframe(shown.drop(columns=["image_path"]), width="stretch", hide_index=True)
    st.subheader("Alert Images")
    cols = st.columns(4)
    for idx, row in shown.head(8).iterrows():
        src = image_download_link(row["image_path"])
        if src:
            cols[idx % 4].image(src, caption=row["detected_at"], width="stretch")


def reports() -> None:
    st.title("Reports & Analytics")
    attendance = pd.DataFrame(rows("attendance", "ORDER BY marked_at DESC"))
    unknowns = pd.DataFrame(rows("unknown_visitors", "ORDER BY detected_at DESC"))
    people = pd.DataFrame(rows("people", "ORDER BY name"))
    report_type = st.selectbox("Report type", ["Attendance", "Registered Users", "Unknown Visitors"])
    df = {"Attendance": attendance, "Registered Users": people, "Unknown Visitors": unknowns}[report_type]
    st.dataframe(df, width="stretch", hide_index=True)
    c1, c2, c3 = st.columns(3)
    base = report_type.lower().replace(" ", "_")
    c1.download_button("CSV", df.to_csv(index=False).encode(), f"{base}.csv", "text/csv", width="stretch")
    c2.download_button("Excel", to_excel(df), f"{base}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    c3.download_button("PDF", simple_pdf_bytes(df, f"{report_type} Report"), f"{base}.pdf", "application/pdf", width="stretch")


def cameras() -> None:
    if not require_role(["Administrator", "Teacher/Manager"]):
        return
    st.title("Camera Management")
    df = pd.DataFrame(rows("cameras", "ORDER BY name"))
    st.dataframe(df, width="stretch", hide_index=True)
    with st.form("camera_form"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Camera Name")
        kind = c2.selectbox("Camera Type", ["Webcam", "USB Camera", "IP Camera", "CCTV/RTSP"])
        source = c3.text_input("Source", value="0", help="Use 0 for default webcam or an RTSP/IP URL.")
        submit = st.form_submit_button("Add Camera", width="stretch")
    if submit and name and source:
        execute("INSERT INTO cameras(name, kind, source, status) VALUES(?, ?, ?, ?)", (name, kind, source, "Configured"))
        log("Camera added", name)
        st.rerun()
    st.subheader("Live Camera Test")
    camera_map = {f"{c['name']} ({c['source']})": c for c in rows("cameras")}
    selected = st.selectbox("Camera", list(camera_map.keys())) if camera_map else None
    c1, c2, c3 = st.columns(3)
    if selected and c1.button("Start / Capture Frame", width="stretch"):
        cam = camera_map[selected]
        source: Any = int(cam["source"]) if str(cam["source"]).isdigit() else cam["source"]
        start = time.time()
        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        cap.release()
        fps = 1 / max(time.time() - start, 0.001)
        if ok:
            execute("UPDATE cameras SET status = ?, last_seen = ? WHERE id = ?", ("Online", now_iso(), cam["id"]))
            st.image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), caption=f"Online · FPS {fps:.1f}", width="stretch")
        else:
            execute("UPDATE cameras SET status = ? WHERE id = ?", ("Offline", cam["id"]))
            st.error("Unable to read from this camera source.")
    if selected and c2.button("Stop Camera", width="stretch"):
        execute("UPDATE cameras SET status = ? WHERE id = ?", ("Stopped", camera_map[selected]["id"]))
        st.rerun()
    if selected and c3.button("Delete Camera", width="stretch"):
        execute("DELETE FROM cameras WHERE id = ?", (camera_map[selected]["id"],))
        log("Camera deleted", selected)
        st.rerun()


def user_management() -> None:
    if not require_role(["Administrator"]):
        return
    st.title("User Management")
    st.dataframe(pd.DataFrame(rows("auth_users", "ORDER BY username")).drop(columns=["password_hash"], errors="ignore"), width="stretch", hide_index=True)
    with st.form("add_auth_user"):
        c1, c2, c3 = st.columns(3)
        username = c1.text_input("Username")
        display = c2.text_input("Display Name")
        role = c3.selectbox("Role", ["Administrator", "Teacher/Manager", "User"])
        email = c1.text_input("Email")
        phone = c2.text_input("Phone")
        password = c3.text_input("Temporary Password", type="password")
        submit = st.form_submit_button("Create Login", width="stretch")
    if submit:
        username = normalize_username(username)
        if not valid_username(username) or not display.strip() or len(password) < 6:
            st.error("Username, display name, and a password of at least 6 characters are required.")
        else:
            try:
                execute(
                    "INSERT INTO auth_users(username, password_hash, role, display_name, email, phone, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (username, hash_password(password), role, display, email, phone, now_iso()),
                )
                log("Login user created", username)
                st.rerun()
            except DATABASE_INTEGRITY_ERRORS:
                st.error("That username already exists.")
    st.subheader("Registered People")
    people = pd.DataFrame(rows("people", "ORDER BY name"))
    st.dataframe(people.drop(columns=["embeddings", "image_paths"], errors="ignore"), width="stretch", hide_index=True)
    if not people.empty:
        selected = st.selectbox("Person ID", people["person_id"].tolist())
        c1, c2 = st.columns(2)
        new_status = c1.selectbox("Set Status", ["Active", "Inactive"])
        if c1.button("Update person"):
            execute("UPDATE people SET status = ?, updated_at = ? WHERE person_id = ?", (new_status, now_iso(), selected))
            log("Person updated", selected)
            st.rerun()
        if c2.button("Delete person"):
            execute("DELETE FROM people WHERE person_id = ?", (selected,))
            log("Person deleted", selected)
            st.rerun()


def profile() -> None:
    st.title("Profile Management")
    user = st.session_state["user"]
    with st.form("profile"):
        username = st.text_input(
            "Username",
            value=user["username"],
            help="This is your unique login name. You may also sign in with your display name or email.",
        )
        display = st.text_input("Display Name", value=user["display_name"])
        email = st.text_input("Email", value=user.get("email") or "")
        phone = st.text_input("Phone", value=user.get("phone") or "")
        password = st.text_input(
            "New Password",
            type="password",
            help="Leave this blank to keep your current password.",
        )
        submit = st.form_submit_button("Save Profile", width="stretch")
    if submit:
        username = normalize_username(username)
        display = display.strip()
        email = email.strip()
        phone = phone.strip()
        if not valid_username(username):
            st.error("Username must be 3–50 characters and may contain letters, numbers, dots, hyphens, or underscores.")
            return
        if not display:
            st.error("Display name is required.")
            return
        if password and len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return
        duplicates = rows(
            "auth_users",
            "WHERE LOWER(username) = LOWER(?) AND id <> ?",
            (username, user["id"]),
        )
        if duplicates:
            st.error("That username is already in use.")
            return
        try:
            if password:
                execute(
                    "UPDATE auth_users SET username=?, display_name=?, email=?, phone=?, password_hash=? WHERE id=?",
                    (username, display, email, phone, hash_password(password), user["id"]),
                )
            else:
                execute(
                    "UPDATE auth_users SET username=?, display_name=?, email=?, phone=? WHERE id=?",
                    (username, display, email, phone, user["id"]),
                )
        except DATABASE_INTEGRITY_ERRORS:
            st.error("That username is already in use.")
            return
        st.session_state["user"] = rows("auth_users", "WHERE id = ?", (user["id"],))[0]
        log("Profile updated")
        st.success(
            f"Profile updated. Sign in with '{username}', your display name, or your email."
        )


def settings_page() -> None:
    if not require_role(["Administrator"]):
        return
    st.title("System Settings")
    if persistent_database_enabled():
        st.success("Persistent database connected. Registrations survive app updates and restarts.")
    elif not DATABASE_CONNECTION_ERROR:
        st.warning(
            "Local SQLite mode is active. It is suitable for local development, but Streamlit Cloud "
            "registrations may disappear after a redeployment. Add DATABASE_URL in Streamlit Secrets."
        )
    with st.form("settings"):
        threshold = st.slider("Recognition cosine threshold", 0.30, 0.80, setting("recognition_threshold", float), 0.01)
        duplicate = st.number_input("Duplicate attendance prevention interval (minutes)", 1, 480, setting("duplicate_minutes", int))
        fps = st.number_input("Target FPS indicator", 1, 60, setting("camera_fps_target", int))
        submit = st.form_submit_button("Save Settings", width="stretch")
    if submit:
        for key, value in {
            "recognition_threshold": threshold,
            "duplicate_minutes": duplicate,
            "camera_fps_target": fps,
        }.items():
            set_setting(key, value)
        log("Settings updated")
        st.success("Settings saved.")

    st.subheader("Fresh Start Reset")
    st.markdown(
        """
        <div class='soft-box'>
        <b>Reset workspace data:</b> clears registered people, attendance, unknown visitor alerts,
        saved face images, generated reports, and custom cameras. Login users and your profile stay intact.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("fresh_start_reset"):
        confirmation = st.text_input("Type RESET to confirm")
        reset_submit = st.form_submit_button("Reset workspace data", width="stretch")
    if reset_submit:
        if confirmation.strip().upper() != "RESET":
            st.error("Type RESET exactly before running the fresh start reset.")
        else:
            reset_operational_data()
            st.success("Workspace reset complete. You can start fresh now.")
            st.rerun()


def activity_logs() -> None:
    if not require_role(["Administrator", "Teacher/Manager"]):
        return
    st.title("Activity Logs")
    df = pd.DataFrame(rows("activity_logs", "ORDER BY created_at DESC"))
    search = st.text_input("Search logs")
    if search and not df.empty:
        df = df[df.apply(lambda r: search.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(df, width="stretch", hide_index=True)


def help_page() -> None:
    st.title("Help")
    st.markdown(
        """
        <div class='soft-box'><b>Registration:</b> add identity details and at least two clear face images.</div>
        <div class='soft-box'><b>Recognition:</b> capture or upload a frame. Known faces can automatically mark attendance.</div>
        <div class='soft-box'><b>Cameras:</b> use <code>0</code> for the default webcam, another device index for USB cameras, or an RTSP/IP URL.</div>
        """,
        unsafe_allow_html=True,
    )


def about_page() -> None:
    st.title("About")
    st.write("This local AI web application combines face detection, registration, recognition, verification, attendance automation, camera management, visitor alerts, analytics, reporting, and role-based access control.")
    st.info("For higher-security production deployments, connect a dedicated face-recognition model, HTTPS authentication, encrypted backups, and organization-specific privacy controls.")


def main() -> None:
    st.set_page_config(page_title="Identix | Identity Intelligence", page_icon="◈", layout="wide")
    init_db()
    apply_theme()
    if "user" not in st.session_state:
        login_screen()
        return
    page = sidebar()
    routes = {
        "Dashboard": dashboard,
        "Face Detection": face_detection,
        "Face Registration": face_registration,
        "Face Recognition": face_recognition,
        "Face Verification": verification,
        "Attendance": attendance_page,
        "Unknown Alerts": unknown_alerts,
        "Reports": reports,
        "Cameras": cameras,
        "User Management": user_management,
        "Profile": profile,
        "Settings": settings_page,
        "Activity Logs": activity_logs,
        "Help": help_page,
        "About": about_page,
    }
    routes[page]()


if __name__ == "__main__":
    main()
