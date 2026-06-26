
"""OpenCV and NumPy AI utilities for detection, embeddings, recognition, and liveness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.database import decode_embedding

FACE_SIZE = (96, 96)


@dataclass
class Detection:
    """One detected face box."""

    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass
class RecognitionMatch:
    """Recognition result for one detected face."""

    user: dict[str, Any] | None
    confidence: float
    label: str
    known: bool
    box: Detection


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR."""
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _cascade(filename: str) -> cv2.CascadeClassifier:
    """Load an OpenCV cascade."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + filename)


def detect_faces_bgr(frame: np.ndarray) -> list[Detection]:
    """Detect one or more human faces in a BGR image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    boxes: list[Detection] = []
    seen: list[tuple[int, int, int, int]] = []
    height, width = gray.shape
    for name in cascade_names:
        detector = _cascade(name)
        if detector.empty():
            continue
        faces = detector.detectMultiScale(gray, scaleFactor=1.06, minNeighbors=4, minSize=(28, 28))
        for x, y, w, h in faces:
            duplicate = any(abs(x - sx) < 15 and abs(y - sy) < 15 for sx, sy, sw, sh in seen)
            if duplicate:
                continue
            seen.append((int(x), int(y), int(w), int(h)))
            face_area = (w * h) / max(1, width * height)
            confidence = min(99.0, max(55.0, 58.0 + face_area * 450.0))
            boxes.append(Detection(int(x), int(y), int(w), int(h), round(confidence, 2)))
    boxes.sort(key=lambda box: box.w * box.h, reverse=True)
    return boxes


def crop_face(frame: np.ndarray, detection: Detection) -> np.ndarray:
    """Crop a face with a small margin."""
    height, width = frame.shape[:2]
    margin_x = int(detection.w * 0.18)
    margin_y = int(detection.h * 0.22)
    x1 = max(0, detection.x - margin_x)
    y1 = max(0, detection.y - margin_y)
    x2 = min(width, detection.x + detection.w + margin_x)
    y2 = min(height, detection.y + detection.h + margin_y)
    return frame[y1:y2, x1:x2]


def face_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """Create a compact normalized embedding from a cropped face."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    vector = resized.astype("float32").reshape(-1)
    vector = (vector - np.mean(vector)) / (np.std(vector) + 1e-6)
    norm = np.linalg.norm(vector) + 1e-6
    return vector / norm


def embedding_from_image(image: Image.Image) -> tuple[np.ndarray | None, list[Detection], Image.Image]:
    """Detect the largest face from an image and return its embedding."""
    frame = pil_to_bgr(image)
    detections = detect_faces_bgr(frame)
    annotated = annotate_detections(image, detections)
    if not detections:
        return None, detections, annotated
    face = crop_face(frame, detections[0])
    return face_embedding(face), detections, annotated


def similarity_percent(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity as a percentage."""
    first = first.reshape(-1)
    second = second.reshape(-1)
    cosine = float(np.dot(first, second) / ((np.linalg.norm(first) * np.linalg.norm(second)) + 1e-6))
    return round(max(0.0, min(100.0, (cosine + 1.0) * 50.0)), 2)


def recognize_faces(image: Image.Image, users: list[dict[str, Any]], threshold: float) -> tuple[list[RecognitionMatch], Image.Image]:
    """Recognize registered users in all detected faces."""
    frame = pil_to_bgr(image)
    detections = detect_faces_bgr(frame)
    matches: list[RecognitionMatch] = []
    known_embeddings = [(user, decode_embedding(user["embedding"])) for user in users]

    for detection in detections:
        embedding = face_embedding(crop_face(frame, detection))
        best_user = None
        best_score = 0.0
        for user, known_embedding in known_embeddings:
            score = similarity_percent(embedding, known_embedding)
            if score > best_score:
                best_user = user
                best_score = score
        known = bool(best_user and best_score >= threshold)
        label = best_user["name"] if known and best_user else "Unknown"
        matches.append(RecognitionMatch(best_user if known else None, best_score, label, known, detection))

    return matches, annotate_recognition(image, matches)


def annotate_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Draw detection boxes on an image."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for face in detections:
        box = [face.x, face.y, face.x + face.w, face.y + face.h]
        draw.rectangle(box, outline=(56, 189, 248), width=4)
        draw.text((face.x, max(0, face.y - 22)), f"Human face {face.confidence:.1f}%", fill=(56, 189, 248))
    return output


def annotate_recognition(image: Image.Image, matches: list[RecognitionMatch]) -> Image.Image:
    """Draw recognition labels on an image."""
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for match in matches:
        face = match.box
        color = (34, 197, 94) if match.known else (239, 68, 68)
        draw.rectangle([face.x, face.y, face.x + face.w, face.y + face.h], outline=color, width=4)
        label = f"{match.label} {match.confidence:.1f}%" if match.known else f"Unknown {match.confidence:.1f}%"
        draw.rectangle([face.x, max(0, face.y - 28), face.x + max(170, len(label) * 9), face.y], fill=color)
        draw.text((face.x + 5, max(0, face.y - 23)), label, fill=(255, 255, 255))
    return output


def compare_two_faces(first: Image.Image, second: Image.Image) -> tuple[bool, float, str, Image.Image, Image.Image]:
    """Compare two uploaded face images."""
    first_embedding, first_detections, first_annotated = embedding_from_image(first)
    second_embedding, second_detections, second_annotated = embedding_from_image(second)
    if first_embedding is None or second_embedding is None:
        return False, 0.0, "Could not detect a human face in one or both images.", first_annotated, second_annotated
    similarity = similarity_percent(first_embedding, second_embedding)
    same = similarity >= 72.0
    message = "Same person" if same else "Different people"
    return same, similarity, message, first_annotated, second_annotated


def liveness_from_frames(open_eye_image: Image.Image | None, blink_image: Image.Image | None, turn_image: Image.Image | None) -> tuple[bool, str, dict[str, Any]]:
    """Simple liveness check using eye count change and head movement between captures."""
    if open_eye_image is None or blink_image is None or turn_image is None:
        return False, "Capture all three liveness steps first.", {}

    eye_detector = _cascade("haarcascade_eye.xml")
    frames = [pil_to_bgr(open_eye_image), pil_to_bgr(blink_image), pil_to_bgr(turn_image)]
    detections = [detect_faces_bgr(frame) for frame in frames]
    if any(len(items) == 0 for items in detections):
        return False, "Spoof Detected: a human face was not detected in every liveness step.", {"faces": [len(items) for items in detections]}

    eye_counts = []
    centers = []
    for frame, faces in zip(frames, detections):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faces[0]
        roi = gray[face.y:face.y + face.h, face.x:face.x + face.w]
        eyes = eye_detector.detectMultiScale(roi, scaleFactor=1.08, minNeighbors=4, minSize=(12, 12)) if not eye_detector.empty() else []
        eye_counts.append(len(eyes))
        centers.append((face.x + face.w / 2, face.y + face.h / 2))

    blink_ok = eye_counts[0] >= 1 and eye_counts[1] < eye_counts[0]
    movement = abs(centers[2][0] - centers[0][0])
    movement_ok = movement >= 8

    details = {"eye_counts": eye_counts, "head_movement_pixels": round(float(movement), 2)}
    if blink_ok and movement_ok:
        return True, "Live Person", details
    return False, "Spoof Detected: blink or head movement was not verified.", details


def save_unknown_image(image: Image.Image, directory: Path) -> str:
    """Save an unknown visitor image and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"unknown_{np.datetime64('now').astype(str).replace(':', '-')}.jpg"
    path = directory / filename
    image.convert("RGB").save(path, quality=90)
    return str(path)


def read_camera_frame(source: str) -> tuple[Image.Image | None, str]:
    """Read one frame from a USB/IP/RTSP camera source."""
    camera_source: int | str
    camera_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        return None, "Camera could not be opened."
    ok, frame = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    capture.release()
    if not ok:
        return None, "Could not read a frame from the camera."
    return bgr_to_pil(frame), f"Frame captured. FPS: {fps:.1f}"
