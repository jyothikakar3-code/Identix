from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


ROOT = Path(__file__).parent
DATA = ROOT / "data"
REGISTERED_DIR = ROOT / "storage" / "registered"
UNKNOWN_DIR = ROOT / "storage" / "unknown"
REPORT_DIR = ROOT / "storage" / "reports"
DB_PATH = DATA / "face_system.sqlite3"
FRESH_RESET_MARKER = "fresh_reset_2026_07_17"

for folder in (DATA, REGISTERED_DIR, UNKNOWN_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)


@dataclass
class MatchResult:
    person: dict[str, Any] | None
    confidence: float
    distance: float


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def init_db() -> None:
    con = connect()
    cur = con.cursor()
    cur.executescript(
        """
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
    )
    defaults = {
        "recognition_threshold": "0.72",
        "duplicate_minutes": "30",
        "dark_mode": "false",
        "blink_required": "true",
        "head_movement_required": "true",
        "camera_fps_target": "12",
    }
    for key, value in defaults.items():
        cur.execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", (key, value))
    cur.execute(
        """
        INSERT OR IGNORE INTO auth_users(username, password_hash, role, display_name, email, phone, created_at)
        VALUES('admin', ?, 'Administrator', 'System Administrator', 'admin@example.com', '', ?)
        """,
        (hash_password("admin123"), now_iso()),
    )
    existing_cameras = cur.execute("SELECT COUNT(*) AS count FROM cameras").fetchone()["count"]
    if existing_cameras == 0:
        cur.execute(
            "INSERT INTO cameras(name, kind, source, status) VALUES('Default Webcam', 'Webcam', '0', 'Configured')"
        )
    fresh_reset_done = cur.execute(
        "SELECT COUNT(*) AS count FROM settings WHERE key = ?", (FRESH_RESET_MARKER,)
    ).fetchone()["count"]
    if fresh_reset_done == 0:
        reset_storage_files()
        cur.executescript(
            """
            DELETE FROM people;
            DELETE FROM attendance;
            DELETE FROM unknown_visitors;
            DELETE FROM activity_logs;
            DELETE FROM cameras;
            DELETE FROM sqlite_sequence WHERE name IN ('people', 'attendance', 'unknown_visitors', 'activity_logs', 'cameras');
            """
        )
        cur.execute(
            "INSERT INTO cameras(name, kind, source, status) VALUES('Default Webcam', 'Webcam', '0', 'Configured')"
        )
        cur.execute("INSERT INTO settings(key, value) VALUES(?, ?)", (FRESH_RESET_MARKER, "true"))
        cur.execute(
            "INSERT INTO activity_logs(actor, action, detail, created_at) VALUES(?, ?, ?, ?)",
            ("system", "Fresh start reset", "Cleared old people, attendance, visitors, and reports", now_iso()),
        )
    con.commit()
    con.close()


def reset_storage_files() -> None:
    for folder in (REGISTERED_DIR, UNKNOWN_DIR, REPORT_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def reset_operational_data() -> None:
    reset_storage_files()
    con = connect()
    cur = con.cursor()
    cur.executescript(
        """
        DELETE FROM people;
        DELETE FROM attendance;
        DELETE FROM unknown_visitors;
        DELETE FROM activity_logs;
        DELETE FROM cameras;
        DELETE FROM sqlite_sequence WHERE name IN ('people', 'attendance', 'unknown_visitors', 'activity_logs', 'cameras');
        """
    )
    cur.execute("INSERT INTO cameras(name, kind, source, status) VALUES('Default Webcam', 'Webcam', '0', 'Configured')")
    cur.execute(
        "INSERT INTO activity_logs(actor, action, detail, created_at) VALUES(?, ?, ?, ?)",
        (st.session_state.get("user", {}).get("username", "system"), "Fresh start reset", "Cleared operational data", now_iso()),
    )
    con.commit()
    con.close()


def rows(table: str, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    con = connect()
    query = f"SELECT * FROM {table} {where}"
    items = [dict(r) for r in con.execute(query, params).fetchall()]
    con.close()
    return items


def execute(query: str, params: tuple[Any, ...] = ()) -> None:
    con = connect()
    con.execute(query, params)
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
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None, None
    cascade_dir = getattr(cv2.data, "haarcascades", "")
    face = cv2.CascadeClassifier(str(Path(cascade_dir) / "haarcascade_frontalface_default.xml"))
    eye = cv2.CascadeClassifier(str(Path(cascade_dir) / "haarcascade_eye.xml"))
    return (None if face.empty() else face, None if eye.empty() else eye)


@st.cache_resource
def face_detectors() -> list[tuple[Any, bool, float, float]]:
    """Load complementary frontal and profile detectors.

    The boolean marks profile cascades, which are also run on a mirrored image
    so faces turned in either direction can be found.
    """
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return []
    cascade_dir = Path(getattr(cv2.data, "haarcascades", ""))
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


def detect_faces(frame_bgr: np.ndarray) -> list[dict[str, Any]]:
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
                candidate = {"box": (int(x), int(y), int(fw), int(fh)), "confidence": confidence}
                detections.append(candidate)

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


def crop_face(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    pad = int(max(w, h) * 0.14)
    y1, y2 = max(0, y - pad), min(frame_bgr.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(frame_bgr.shape[1], x + w + pad)
    return frame_bgr[y1:y2, x1:x2]


def embedding(face_bgr: np.ndarray) -> list[float]:
    if face_bgr.size == 0:
        return []
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([resized], [0], None, [48], [0, 256]).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-9)
    small = cv2.resize(resized, (24, 24), interpolation=cv2.INTER_AREA).astype("float32").flatten()
    small = (small - small.mean()) / (small.std() + 1e-6)
    vec = np.concatenate([hist, small / (np.linalg.norm(small) + 1e-9)])
    return vec.astype("float32").tolist()


def cosine(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    if av.size == 0 or bv.size == 0:
        return 0.0
    return float(np.dot(av, bv) / ((np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-9))


def load_people() -> list[dict[str, Any]]:
    people = rows("people", "ORDER BY name")
    for person in people:
        person["embeddings"] = json.loads(person["embeddings"])
        person["image_paths"] = json.loads(person["image_paths"])
    return people


def recognize(face_bgr: np.ndarray) -> MatchResult:
    probe = embedding(face_bgr)
    threshold = setting("recognition_threshold", float)
    best_person = None
    best_score = -1.0
    for person in load_people():
        if person["status"] != "Active":
            continue
        scores = [cosine(probe, item) for item in person["embeddings"]]
        if scores and max(scores) > best_score:
            best_score = max(scores)
            best_person = person
    confidence = max(0.0, min(1.0, (best_score + 1) / 2))
    if best_person and confidence >= threshold:
        return MatchResult(best_person, confidence, 1 - confidence)
    return MatchResult(None, confidence, 1 - confidence)


def draw_boxes(frame_bgr: np.ndarray, detections: list[dict[str, Any]], labels: list[str] | None = None) -> Image.Image:
    output = frame_bgr.copy()
    for idx, detection in enumerate(detections):
        x, y, w, h = detection["box"]
        label = labels[idx] if labels and idx < len(labels) else f"Face {detection['confidence']:.0%}"
        color = (32, 201, 151) if labels and "Unknown" not in label and "Spoof" not in label else (0, 86, 255)
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
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.title("Face Recognition Management System")
        st.write("A complete AI-powered attendance, access, visitor alert, and reporting workspace.")
        st.markdown(
            """
            <div class='soft-box'>
            <b>Default administrator:</b> username <code>admin</code>, password <code>admin123</code>.
            Change it from Profile after login.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        with st.form("login"):
            st.subheader("Secure Login")
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Log in", width="stretch")
        if submitted:
            user = rows("auth_users", "WHERE username = ? AND password_hash = ?", (username, hash_password(password)))
            if user:
                st.session_state["user"] = user[0]
                log("Login", "Successful login")
                st.toast("Welcome back.", icon="✅")
                st.rerun()
            else:
                st.error("Invalid username or password.")


def sidebar() -> str:
    user = st.session_state["user"]
    st.sidebar.title("AI Face System")
    st.sidebar.caption(f"{user['display_name']} · {user['role']}")
    base_pages = ["Dashboard", "Face Detection", "Face Registration", "Face Recognition", "Face Verification", "Liveness Detection", "Attendance", "Unknown Alerts", "Reports", "Cameras", "Profile", "Help", "About"]
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
            st.warning("No face was detected. Try a clearer frontal image with good lighting.")


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
        if rows("people", "WHERE person_id = ?", (person_id,)):
            errors.append("This ID is already registered.")
        if errors:
            for error in errors:
                st.error(error)
            return
        person_dir = REGISTERED_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        embs, image_paths = [], []
        for index, file in enumerate(files):
            frame = uploaded_to_bgr(file)
            faces = detect_faces(frame)
            if len(faces) != 1:
                st.warning(f"Skipped sample {index + 1}: expected one face, found {len(faces)}.")
                continue
            face = crop_face(frame, faces[0]["box"])
            embs.append(embedding(face))
            image_paths.append(save_bgr(face, person_dir, f"sample_{index + 1}"))
        if len(embs) < 2:
            st.error("Registration failed. At least two usable face samples are required.")
            return
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
    detections = detect_faces(frame)
    labels = []
    records = []
    if not detections:
        st.warning("No face detected.")
        return
    for detection in detections:
        face = crop_face(frame, detection["box"])
        match = recognize(face)
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
            labels.append(f"Unknown · {match.confidence:.0%}")
            register_unknown(face, detection["confidence"], camera_name)
            records.append({"Name": "Unknown", "ID": "-", "Department": "-", "Recognition Confidence": f"{match.confidence:.1%}", "Attendance": "Unknown alert saved"})
    st.image(draw_boxes(frame, detections, labels), width="stretch")
    if any(r["Name"] == "Unknown" for r in records):
        st.markdown("<div class='alert-box'><b>Unknown person detected.</b> Alert image and timestamp were saved.</div>", unsafe_allow_html=True)
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
        for idx, frame in enumerate(frames):
            detections = detect_faces(frame)
            if len(detections) != 1:
                st.error(f"Image {idx + 1} must contain exactly one detectable face. Found {len(detections)}.")
                return
            faces.append(crop_face(frame, detections[0]["box"]))
        similarity = (cosine(embedding(faces[0]), embedding(faces[1])) + 1) / 2
        same = similarity >= setting("recognition_threshold", float)
        st.metric("Similarity", f"{similarity:.1%}")
        st.success("Same person") if same else st.error("Different people")
        st.image([Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in faces], width=260)


def liveness() -> None:
    st.title("Liveness Detection")
    st.write("Capture three frames: neutral, blink, and turn head. The system checks face presence, eye change, and head movement.")
    if "live_frames" not in st.session_state:
        st.session_state.live_frames = []
    sample = st.camera_input("Capture liveness frame")
    if sample and st.button("Add frame to liveness check", width="stretch"):
        st.session_state.live_frames.append(uploaded_to_bgr(sample))
        st.toast(f"Captured frame {len(st.session_state.live_frames)}.", icon="📷")
    cols = st.columns(3)
    for i, frame in enumerate(st.session_state.live_frames[-3:]):
        cols[i].image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), caption=f"Frame {i + 1}", width="stretch")
    if st.button("Run liveness verification", disabled=len(st.session_state.live_frames) < 3, width="stretch"):
        latest = st.session_state.live_frames[-3:]
        face_cascade, eye_cascade = cascades()
        if face_cascade is None:
            st.error("The human-face detector could not be loaded.")
            return
        centers, eye_counts = [], []
        for frame in latest:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.08, 5, minSize=(64, 64))
            if len(faces) != 1:
                st.error("Spoof Detected: every liveness frame must contain exactly one face.")
                return
            x, y, w, h = faces[0]
            if eye_cascade is not None:
                roi = gray[y : y + h // 2, x : x + w]
                eyes = eye_cascade.detectMultiScale(roi, 1.08, 4, minSize=(18, 18))
            else:
                eyes = []
            centers.append((x + w / 2, y + h / 2, w))
            eye_counts.append(len(eyes))
        movement = abs(centers[0][0] - centers[-1][0]) / max(1, centers[0][2])
        blink_ok = min(eye_counts) < max(eye_counts) if eye_cascade is not None else False
        move_ok = movement > 0.08
        if (blink_ok or not setting("blink_required", bool)) and (move_ok or not setting("head_movement_required", bool)):
            st.success("Live Person")
        else:
            st.error("Spoof Detected")
        st.write({"Blink evidence": blink_ok, "Head movement": f"{movement:.1%}", "Eye detections": eye_counts})
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
        if not username or not display or len(password) < 6:
            st.error("Username, display name, and a password of at least 6 characters are required.")
        else:
            try:
                execute(
                    "INSERT INTO auth_users(username, password_hash, role, display_name, email, phone, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (username, hash_password(password), role, display, email, phone, now_iso()),
                )
                log("Login user created", username)
                st.rerun()
            except sqlite3.IntegrityError:
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
        display = st.text_input("Display Name", value=user["display_name"])
        email = st.text_input("Email", value=user.get("email") or "")
        phone = st.text_input("Phone", value=user.get("phone") or "")
        password = st.text_input("New Password", type="password")
        submit = st.form_submit_button("Save Profile", width="stretch")
    if submit:
        if password and len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return
        if password:
            execute("UPDATE auth_users SET display_name=?, email=?, phone=?, password_hash=? WHERE id=?", (display, email, phone, hash_password(password), user["id"]))
        else:
            execute("UPDATE auth_users SET display_name=?, email=?, phone=? WHERE id=?", (display, email, phone, user["id"]))
        st.session_state["user"] = rows("auth_users", "WHERE id = ?", (user["id"],))[0]
        log("Profile updated")
        st.success("Profile updated.")


def settings_page() -> None:
    if not require_role(["Administrator"]):
        return
    st.title("System Settings")
    with st.form("settings"):
        threshold = st.slider("Recognition confidence threshold", 0.50, 0.95, setting("recognition_threshold", float), 0.01)
        duplicate = st.number_input("Duplicate attendance prevention interval (minutes)", 1, 480, setting("duplicate_minutes", int))
        fps = st.number_input("Target FPS indicator", 1, 60, setting("camera_fps_target", int))
        blink = st.toggle("Require blink evidence for liveness", value=setting("blink_required", bool))
        movement = st.toggle("Require head movement for liveness", value=setting("head_movement_required", bool))
        submit = st.form_submit_button("Save Settings", width="stretch")
    if submit:
        for key, value in {
            "recognition_threshold": threshold,
            "duplicate_minutes": duplicate,
            "camera_fps_target": fps,
            "blink_required": blink,
            "head_movement_required": movement,
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
        <div class='soft-box'><b>Liveness:</b> capture neutral, blink, and head-turn frames to reduce photo or screen spoofing.</div>
        <div class='soft-box'><b>Cameras:</b> use <code>0</code> for the default webcam, another device index for USB cameras, or an RTSP/IP URL.</div>
        """,
        unsafe_allow_html=True,
    )


def about_page() -> None:
    st.title("About")
    st.write("This local AI web application combines face detection, registration, recognition, verification, liveness checks, attendance automation, camera management, visitor alerts, analytics, reporting, and role-based access control.")
    st.info("For higher-security production deployments, connect a dedicated face-recognition model, HTTPS authentication, encrypted backups, and organization-specific privacy controls.")


def main() -> None:
    st.set_page_config(page_title="AI Face Recognition System", page_icon="◉", layout="wide")
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
        "Liveness Detection": liveness,
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
