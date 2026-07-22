"""Shared human-versus-animal validation for every face workflow.

The face detector answers "does this look face-shaped?"; it is not an animal
classifier.  This module adds an independent ImageNet classifier before a
candidate may be called human.  Keeping the gate here gives registration,
recognition, verification, liveness, attendance, and future features one
fail-closed policy instead of several subtly different checks.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


NO_FACE_MESSAGE = "No face detected. Please upload an image containing a clear human face."
ANIMAL_FACE_MESSAGE = (
    "Only human faces are supported. Please upload an image containing a human face."
)
CLASSIFIER_UNAVAILABLE_MESSAGE = (
    "The human/animal classifier could not be loaded. Please try again."
)
VALIDATION_MARKER = "human_animal_guard_v1"

# Official ONNX Model Zoo EfficientNet-Lite4. It is a production mobile model
# with 80.4% ImageNet top-1 accuracy and a published 30 ms Pixel 4 CPU result.
# It executes through the already-installed OpenCV DNN runtime and covers the
# broad ImageNet animal taxonomy without sending user images to a cloud API.
MODEL_FILENAME = "efficientnet-lite4-11.onnx"
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "efficientnet-lite4/model/efficientnet-lite4-11.onnx"
)
MODEL_SHA256 = "d111689907c06eea7c82e4833ddef758da6453b9d4cf60b7e99ca05c7cbd9c12"
MODEL_MIN_BYTES = 45_000_000
MODEL_MAX_BYTES = 60_000_000

# ImageNet-1K IDs 0..397 contain fish, birds, reptiles, amphibians, insects,
# mammals, and domestic/wild animals. Teddy-bear and comic-book proxies cover
# stuffed and illustrated/cartoon animals that trigger a face detector. New
# classifier versions can extend/replace this immutable taxonomy in one place.
ANIMAL_CLASS_IDS = frozenset(range(398)) | {850, 917}

# These conservative defaults prioritize not accepting an animal as human.
# They remain configurable for a labelled deployment-specific calibration set.
ANIMAL_TOP1_MIN_PROBABILITY = float(
    os.getenv("ANIMAL_GUARD_TOP1_MIN_PROBABILITY", "0.18")
)
ANIMAL_TOP5_MIN_MASS = float(os.getenv("ANIMAL_GUARD_TOP5_MIN_MASS", "0.45"))
ANIMAL_MIN_MARGIN = float(os.getenv("ANIMAL_GUARD_MIN_MARGIN", "0.02"))
NON_FACE_OBJECT_MIN_PROBABILITY = float(
    os.getenv("ANIMAL_GUARD_NON_FACE_OBJECT_MIN_PROBABILITY", "0.65")
)
HIGH_CONFIDENCE_FACE_OVERRIDE = float(
    os.getenv("ANIMAL_GUARD_HIGH_CONFIDENCE_FACE_OVERRIDE", "0.90")
)


class FaceInputKind(str, Enum):
    HUMAN = "human"
    ANIMAL = "animal"
    NO_FACE = "no_face"


class FaceInputRejected(RuntimeError):
    """Stop a face workflow before an embedding or identity decision."""

    def __init__(self, kind: FaceInputKind, message: str | None = None) -> None:
        self.kind = kind
        default_message = ANIMAL_FACE_MESSAGE if kind is FaceInputKind.ANIMAL else NO_FACE_MESSAGE
        super().__init__(message or default_message)


@dataclass(frozen=True)
class AnimalEvidence:
    """Auditable classifier evidence used by the central validation gate."""

    animal_like: bool
    top_class_id: int
    top_probability: float
    top_animal_probability: float
    top_non_animal_probability: float
    animal_probability_in_top5: float


@dataclass(frozen=True)
class FaceValidationResult:
    """Three-way image result plus only the candidates safe for embeddings."""

    kind: FaceInputKind
    detections: tuple[dict[str, Any], ...]
    animal_evidence: tuple[AnimalEvidence, ...]

    @property
    def message(self) -> str:
        if self.kind is FaceInputKind.ANIMAL:
            return ANIMAL_FACE_MESSAGE
        if self.kind is FaceInputKind.NO_FACE:
            return NO_FACE_MESSAGE
        return ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(model_path: Path) -> Path:
    """Download and checksum the exact audited model; never run unknown bytes."""
    if (
        model_path.is_file()
        and MODEL_MIN_BYTES <= model_path.stat().st_size <= MODEL_MAX_BYTES
        and _sha256(model_path) == MODEL_SHA256
    ):
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(model_path.suffix + ".download")
    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "Identix-Animal-Guard/1.0"},
        )
        total = 0
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open("wb") as output:
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > MODEL_MAX_BYTES:
                    raise ValueError("Animal classifier download exceeded the expected size.")
                digest.update(block)
                output.write(block)
        if total < MODEL_MIN_BYTES or digest.hexdigest() != MODEL_SHA256:
            raise ValueError("Animal classifier checksum validation failed.")
        temporary_path.replace(model_path)
        return model_path
    except (OSError, ValueError) as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(CLASSIFIER_UNAVAILABLE_MESSAGE) from error


def _softmax(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    # Some ImageNet exports include a leading background logit. The audited
    # model has 1,000 outputs, but accepting 1,001 keeps model upgrades safe.
    if flattened.size == 1001:
        flattened = flattened[1:]
    if flattened.size != 1000:
        raise RuntimeError(CLASSIFIER_UNAVAILABLE_MESSAGE)
    # EfficientNet-Lite4 includes a Softmax output. Keeping the logits branch
    # makes a future checksum-pinned model replacement straightforward.
    if (
        float(np.min(flattened)) >= 0.0
        and float(np.max(flattened)) <= 1.0
        and abs(float(np.sum(flattened)) - 1.0) <= 0.01
    ):
        return flattened / (float(np.sum(flattened)) + 1e-12)
    shifted = flattened - float(np.max(flattened))
    exponentials = np.exp(shifted)
    return exponentials / (float(np.sum(exponentials)) + 1e-12)


def animal_evidence_from_logits(logits: np.ndarray) -> AnimalEvidence:
    """Turn ImageNet logits into a conservative animal/non-animal decision."""
    probabilities = _softmax(logits)
    top5 = np.argsort(probabilities)[-5:][::-1]
    top_class_id = int(top5[0])
    top_probability = float(probabilities[top_class_id])
    top_animal_id = max(ANIMAL_CLASS_IDS, key=lambda class_id: float(probabilities[class_id]))
    non_animal_ids = np.asarray(
        [class_id for class_id in range(398, probabilities.size) if class_id not in ANIMAL_CLASS_IDS],
        dtype=np.int32,
    )
    top_non_animal_probability = float(np.max(probabilities[non_animal_ids]))
    top_animal_probability = float(probabilities[top_animal_id])
    animal_top5_mass = float(
        sum(float(probabilities[class_id]) for class_id in top5 if int(class_id) in ANIMAL_CLASS_IDS)
    )
    animal_like = (
        top_class_id in ANIMAL_CLASS_IDS
        and top_animal_probability >= ANIMAL_TOP1_MIN_PROBABILITY
        and (
            top_animal_probability - top_non_animal_probability >= ANIMAL_MIN_MARGIN
            or animal_top5_mass >= ANIMAL_TOP5_MIN_MASS
        )
    )
    return AnimalEvidence(
        animal_like=animal_like,
        top_class_id=top_class_id,
        top_probability=top_probability,
        top_animal_probability=top_animal_probability,
        top_non_animal_probability=top_non_animal_probability,
        animal_probability_in_top5=animal_top5_mass,
    )


def _model_input(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
        raise FaceInputRejected(FaceInputKind.NO_FACE)
    height, width = frame_bgr.shape[:2]
    scale = 256.0 / float(min(height, width))
    resized = cv2.resize(
        frame_bgr,
        (max(224, int(round(width * scale))), max(224, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    y = (resized.shape[0] - 224) // 2
    x = (resized.shape[1] - 224) // 2
    rgb = cv2.cvtColor(resized[y : y + 224, x : x + 224], cv2.COLOR_BGR2RGB)
    # The audited EfficientNet-Lite4 ONNX export expects NHWC RGB float input
    # in approximately [-1, 1], using the Model Zoo's (pixel-127)/128 rule.
    normalized = (rgb.astype(np.float32) - 127.0) / 128.0
    return normalized[np.newaxis, ...]


class AnimalClassifier:
    """Thread-safe OpenCV DNN wrapper for the local ImageNet classifier."""

    def __init__(self, model_path: Path) -> None:
        try:
            self._network = cv2.dnn.readNetFromONNX(str(ensure_model(model_path)))
        except (cv2.error, RuntimeError) as error:
            raise RuntimeError(CLASSIFIER_UNAVAILABLE_MESSAGE) from error
        self._lock = threading.RLock()

    def classify(self, frame_bgr: np.ndarray) -> AnimalEvidence:
        blob = _model_input(frame_bgr)
        try:
            with self._lock:
                self._network.setInput(blob)
                logits = self._network.forward()
        except cv2.error as error:
            raise RuntimeError(CLASSIFIER_UNAVAILABLE_MESSAGE) from error
        return animal_evidence_from_logits(logits)


@lru_cache(maxsize=4)
def get_animal_classifier(model_path: str) -> AnimalClassifier:
    return AnimalClassifier(Path(model_path))


def expanded_crop(
    frame_bgr: np.ndarray,
    box: Sequence[int],
    padding_ratio: float = 0.35,
) -> np.ndarray:
    """Include ears/muzzle/fur around a candidate; tight crops hide animal cues."""
    x, y, width, height = (int(value) for value in box)
    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)
    x1, y1 = max(0, x - padding_x), max(0, y - padding_y)
    x2 = min(frame_bgr.shape[1], x + width + padding_x)
    y2 = min(frame_bgr.shape[0], y + height + padding_y)
    return frame_bgr[y1:y2, x1:x2]


def validate_human_candidates(
    frame_bgr: np.ndarray,
    detections: Iterable[Mapping[str, Any]],
    model_path: Path,
    classifier: Any | None = None,
) -> FaceValidationResult:
    """Classify an image and return only candidates authorized as human.

    A mixed human-and-animal image continues with its human candidates only.
    An animal-only image is rejected distinctly from an image with no face.
    Every accepted detection receives an unforgeable-by-accident marker that
    downstream embedding functions are required to check.
    """
    candidates = [dict(detection) for detection in detections]
    active_classifier = classifier or get_animal_classifier(str(model_path))
    accepted: list[dict[str, Any]] = []
    evidence_items: list[AnimalEvidence] = []
    animal_candidates = 0

    for candidate in candidates:
        face_context = expanded_crop(frame_bgr, candidate["box"])
        evidence = active_classifier.classify(face_context)
        evidence_items.append(evidence)
        if evidence.animal_like:
            animal_candidates += 1
            continue
        # Preserve the app's previous object-image fix: a clear semantic object
        # (for example an apple or car) must not become "human" merely because
        # YuNet produced a weak false-positive rectangle. Strong YuNet faces
        # override this rule so masks, low light, and unusual clothing remain
        # usable even when ImageNet describes an accessory instead of a person.
        semantic_object_false_positive = (
            evidence.top_class_id not in ANIMAL_CLASS_IDS
            and evidence.top_probability >= NON_FACE_OBJECT_MIN_PROBABILITY
            and float(candidate.get("confidence", 0.0)) < HIGH_CONFIDENCE_FACE_OVERRIDE
        )
        if semantic_object_false_positive:
            continue
        candidate["human_validation"] = VALIDATION_MARKER
        accepted.append(candidate)

    if accepted:
        return FaceValidationResult(
            FaceInputKind.HUMAN,
            tuple(accepted),
            tuple(evidence_items),
        )
    if animal_candidates:
        return FaceValidationResult(
            FaceInputKind.ANIMAL,
            (),
            tuple(evidence_items),
        )
    if candidates:
        return FaceValidationResult(
            FaceInputKind.NO_FACE,
            (),
            tuple(evidence_items),
        )

    # YuNet is human-specific and may correctly return zero for an animal.
    # Full-image classification is therefore needed to choose the requested
    # animal message instead of incorrectly reporting a generic no-face result.
    full_image_evidence = active_classifier.classify(frame_bgr)
    return FaceValidationResult(
        FaceInputKind.ANIMAL if full_image_evidence.animal_like else FaceInputKind.NO_FACE,
        (),
        (full_image_evidence,),
    )


def require_validated_detection(detection: Mapping[str, Any]) -> None:
    """Hard stop that prevents every embedding bypass, including future code."""
    if detection.get("human_validation") != VALIDATION_MARKER:
        raise RuntimeError(
            "Face embedding blocked: the candidate did not pass human/animal validation."
        )
