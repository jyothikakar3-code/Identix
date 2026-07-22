"""Measure the animal guard on a labelled, deployment-specific image set.

Expected layout:

    evaluation_set/human/*.jpg
    evaluation_set/animal/*.jpg
    evaluation_set/invalid/*.jpg
    evaluation_set/mixed/*.jpg

"mixed" means at least one human plus one animal; validated humans may proceed,
but animal regions must never reach an embedding. No image data leaves the host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from utils.human_face_validation import FaceInputKind, FaceInputRejected, VALIDATION_MARKER  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def classify(path: Path) -> str:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return "invalid"
    try:
        detections = app.detect_faces(frame)
    except FaceInputRejected as error:
        return error.kind.value
    if not detections:
        return FaceInputKind.NO_FACE.value
    if not all(item.get("human_validation") == VALIDATION_MARKER for item in detections):
        return "unsafe_unvalidated_detection"
    return FaceInputKind.HUMAN.value


def expected_prediction(category: str) -> str:
    return {
        "human": FaceInputKind.HUMAN.value,
        "animal": FaceInputKind.ANIMAL.value,
        "invalid": FaceInputKind.NO_FACE.value,
        "mixed": FaceInputKind.HUMAN.value,
    }[category]


def evaluate(dataset: Path) -> dict[str, object]:
    records: list[dict[str, str | bool]] = []
    for category in ("human", "animal", "invalid", "mixed"):
        folder = dataset / category
        for path in sorted(folder.glob("*")) if folder.is_dir() else []:
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            predicted = classify(path)
            expected = expected_prediction(category)
            records.append(
                {
                    "file": str(path.relative_to(dataset)),
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                }
            )
    total = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    animal_records = [record for record in records if record["expected"] == FaceInputKind.ANIMAL.value]
    animal_false_accepts = sum(
        record["predicted"] == FaceInputKind.HUMAN.value for record in animal_records
    )
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "animal_images": len(animal_records),
        "animal_false_accepts": animal_false_accepts,
        "animal_false_accept_rate": animal_false_accepts / len(animal_records) if animal_records else 0.0,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--minimum-accuracy", type=float, default=0.95)
    arguments = parser.parse_args()
    report = evaluate(arguments.dataset)
    print(json.dumps(report, indent=2))
    if not report["total"]:
        print("No evaluation images found.", file=sys.stderr)
        return 2
    if report["animal_false_accepts"] or report["accuracy"] < arguments.minimum_accuracy:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
