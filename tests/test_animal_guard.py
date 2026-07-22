"""Regression tests for the shared human-versus-animal safety gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import app
from utils import face_ai
from utils import image_processing
from utils import database
from utils.eigenfaces import EigenfacesRecognizer
from utils.human_face_validation import (
    ANIMAL_FACE_MESSAGE,
    CLASSIFIER_UNAVAILABLE_MESSAGE,
    NO_FACE_MESSAGE,
    AnimalEvidence,
    FaceInputKind,
    FaceInputRejected,
    FaceValidationResult,
    VALIDATION_MARKER,
    animal_evidence_from_logits,
    ensure_model,
    validate_human_candidates,
)


def evidence(animal_like: bool, class_id: int | None = None) -> AnimalEvidence:
    selected_class = class_id if class_id is not None else (207 if animal_like else 409)
    return AnimalEvidence(
        animal_like=animal_like,
        top_class_id=selected_class,
        top_probability=0.92,
        top_animal_probability=0.92 if animal_like else 0.01,
        top_non_animal_probability=0.01 if animal_like else 0.92,
        animal_probability_in_top5=0.96 if animal_like else 0.01,
    )


class FakeClassifier:
    def __init__(self, *results: AnimalEvidence) -> None:
        self.results = list(results)
        self.calls = 0

    def classify(self, frame: np.ndarray) -> AnimalEvidence:
        self.calls += 1
        return self.results.pop(0)


class HumanAnimalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.first = {"box": (20, 30, 100, 120), "confidence": 0.94}
        self.second = {"box": (180, 30, 90, 110), "confidence": 0.91}

    def test_human_candidate_receives_required_downstream_marker(self) -> None:
        result = validate_human_candidates(
            self.frame,
            [self.first],
            Path("unused.onnx"),
            FakeClassifier(evidence(False)),
        )
        self.assertEqual(result.kind, FaceInputKind.HUMAN)
        self.assertEqual(result.detections[0]["human_validation"], VALIDATION_MARKER)

    def test_animal_candidate_is_rejected(self) -> None:
        result = validate_human_candidates(
            self.frame,
            [self.first],
            Path("unused.onnx"),
            FakeClassifier(evidence(True)),
        )
        self.assertEqual(result.kind, FaceInputKind.ANIMAL)
        self.assertEqual(result.message, ANIMAL_FACE_MESSAGE)
        self.assertEqual(result.detections, ())

    def test_no_candidate_is_classified_as_animal_or_invalid(self) -> None:
        animal_result = validate_human_candidates(
            self.frame,
            [],
            Path("unused.onnx"),
            FakeClassifier(evidence(True)),
        )
        invalid_result = validate_human_candidates(
            self.frame,
            [],
            Path("unused.onnx"),
            FakeClassifier(evidence(False)),
        )
        self.assertEqual(animal_result.kind, FaceInputKind.ANIMAL)
        self.assertEqual(invalid_result.kind, FaceInputKind.NO_FACE)
        self.assertEqual(invalid_result.message, NO_FACE_MESSAGE)

    def test_mixed_image_keeps_human_and_filters_animal(self) -> None:
        result = validate_human_candidates(
            self.frame,
            [self.first, self.second],
            Path("unused.onnx"),
            FakeClassifier(evidence(False), evidence(True)),
        )
        self.assertEqual(result.kind, FaceInputKind.HUMAN)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0]["box"], self.first["box"])

    def test_weak_detector_box_on_clear_object_is_invalid_not_human(self) -> None:
        apple_like = evidence(False, class_id=948)
        weak_candidate = {**self.first, "confidence": 0.79}
        result = validate_human_candidates(
            self.frame,
            [weak_candidate],
            Path("unused.onnx"),
            FakeClassifier(apple_like),
        )
        self.assertEqual(result.kind, FaceInputKind.NO_FACE)
        self.assertEqual(result.message, NO_FACE_MESSAGE)

    def test_strong_human_detector_can_override_accessory_classification(self) -> None:
        mask_like = evidence(False, class_id=643)
        strong_candidate = {**self.first, "confidence": 0.96}
        result = validate_human_candidates(
            self.frame,
            [strong_candidate],
            Path("unused.onnx"),
            FakeClassifier(mask_like),
        )
        self.assertEqual(result.kind, FaceInputKind.HUMAN)

    def test_species_taxonomy_covers_requested_animal_families(self) -> None:
        # Representative ImageNet IDs: dog, cat, horse, cow, monkey, lion,
        # tiger, bird, rabbit, fox, wolf, bear, elephant, goat, sheep, deer.
        requested_species_ids = [207, 281, 339, 345, 365, 291, 292, 13, 330, 277, 269, 294, 386, 350, 348, 353]
        for class_id in requested_species_ids:
            logits = np.full(1000, -12.0, dtype=np.float32)
            logits[class_id] = 12.0
            self.assertTrue(animal_evidence_from_logits(logits).animal_like, class_id)

    def test_teddy_bear_is_rejected_as_animal_like(self) -> None:
        logits = np.full(1000, -12.0, dtype=np.float32)
        logits[850] = 12.0
        self.assertTrue(animal_evidence_from_logits(logits).animal_like)

    def test_non_animal_logits_do_not_fabricate_an_animal(self) -> None:
        logits = np.full(1000, -12.0, dtype=np.float32)
        logits[409] = 12.0
        self.assertFalse(animal_evidence_from_logits(logits).animal_like)

    def test_model_download_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "utils.human_face_validation.urllib.request.urlopen",
            side_effect=OSError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, CLASSIFIER_UNAVAILABLE_MESSAGE):
                ensure_model(Path(temporary_directory) / "classifier.onnx")


class WorkflowGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((180, 180, 3), dtype=np.uint8)
        self.raw_detection = {"box": (20, 20, 120, 120), "confidence": 0.96}

    def test_face_detection_uses_exact_animal_rejection(self) -> None:
        animal_result = FaceValidationResult(FaceInputKind.ANIMAL, (), (evidence(True),))
        with patch.object(app, "_detect_face_candidates", return_value=[self.raw_detection]), patch.object(
            app, "validate_human_candidates", return_value=animal_result
        ):
            with self.assertRaisesRegex(FaceInputRejected, ANIMAL_FACE_MESSAGE):
                app.detect_faces(self.frame)

    def test_embeddings_cannot_bypass_validation(self) -> None:
        with patch.object(app, "sface_recognizer") as recognizer:
            with self.assertRaisesRegex(RuntimeError, "did not pass human/animal validation"):
                app.embedding_from_detection(self.frame, self.raw_detection)
        recognizer.assert_not_called()

    def test_verification_cannot_bypass_validation(self) -> None:
        with patch.object(app, "sface_recognizer") as recognizer:
            with self.assertRaisesRegex(RuntimeError, "did not pass human/animal validation"):
                app.verification_embedding(self.frame, self.raw_detection)
        recognizer.assert_not_called()

    def test_registration_stops_immediately_on_an_animal(self) -> None:
        with patch.object(
            app,
            "detect_faces",
            side_effect=FaceInputRejected(FaceInputKind.ANIMAL),
        ):
            with self.assertRaisesRegex(FaceInputRejected, ANIMAL_FACE_MESSAGE):
                app.detect_registration_face(self.frame)

    def test_recognition_never_matches_or_saves_an_animal(self) -> None:
        with patch.object(
            app,
            "detect_faces",
            side_effect=FaceInputRejected(FaceInputKind.ANIMAL),
        ), patch.object(app, "recognize") as recognize, patch.object(
            app, "register_unknown"
        ) as register_unknown, patch.object(app.st, "error") as show_error:
            app.process_recognition_frame(self.frame, "Test camera", True)
        recognize.assert_not_called()
        register_unknown.assert_not_called()
        show_error.assert_called_once_with(ANIMAL_FACE_MESSAGE)

    def test_liveness_never_embeds_an_animal(self) -> None:
        with patch.object(
            app,
            "detect_faces",
            side_effect=FaceInputRejected(FaceInputKind.ANIMAL),
        ), patch.object(app, "embedding_from_detection") as make_embedding:
            result = app.analyze_liveness_frames([self.frame, self.frame, self.frame])
        self.assertFalse(result["live"])
        self.assertEqual(result["reason"], ANIMAL_FACE_MESSAGE)
        make_embedding.assert_not_called()

    def test_legacy_embedding_api_is_also_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not pass human/animal validation"):
            face_ai.face_embedding(self.frame)

    def test_legacy_eigenface_preprocessing_cannot_bypass_validation(self) -> None:
        image = image_processing.Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        with patch.object(
            image_processing,
            "detect_human_face_from_pil",
            return_value=(False, 0),
        ):
            with self.assertRaisesRegex(FaceInputRejected, NO_FACE_MESSAGE):
                image_processing.prepare_image_from_pil(image, (64, 64))

    def test_legacy_database_registration_cannot_store_unvalidated_embedding(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not pass human/animal validation"):
            database.add_registered_user({}, np.asarray([1.0], dtype=np.float32), 1)

    def test_legacy_eigenfaces_cannot_train_on_unvalidated_vectors(self) -> None:
        recognizer = EigenfacesRecognizer(2)
        with self.assertRaisesRegex(RuntimeError, "did not pass human/animal validation"):
            recognizer.fit([np.asarray([1.0], dtype=np.float32)], ["animal"])


if __name__ == "__main__":
    unittest.main()
