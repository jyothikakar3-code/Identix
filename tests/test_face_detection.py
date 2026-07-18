"""Regression tests for the face-detection confidence bug."""

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import app


ROOT = Path(__file__).parents[1]


class FaceDetectionTests(unittest.TestCase):
    def test_cascade_models_are_discoverable(self) -> None:
        cascade_dir = app.cascade_directory()
        self.assertIsNotNone(cascade_dir)
        self.assertTrue((cascade_dir / "haarcascade_frontalface_default.xml").is_file())

    def test_registered_face_samples_have_feature_based_scores(self) -> None:
        scores = []
        for image_path in sorted((ROOT / "storage" / "registered").rglob("*.jpg")):
            detections = app.detect_faces(cv2.imread(str(image_path)))
            self.assertGreaterEqual(len(detections), 1, image_path.name)
            scores.append(round(detections[0]["confidence"], 3))

        self.assertGreaterEqual(len(scores), 3)
        self.assertGreater(len(set(scores)), 1, "face scores must depend on image features")

    def test_object_images_return_no_face(self) -> None:
        apple = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.circle(apple, (320, 240), 115, (0, 0, 210), -1)
        cv2.rectangle(apple, (312, 85), (328, 135), (35, 90, 35), -1)

        car = np.full((480, 640, 3), 230, dtype=np.uint8)
        cv2.rectangle(car, (150, 220), (490, 340), (180, 70, 20), -1)
        cv2.circle(car, (225, 350), 42, (20, 20, 20), -1)
        cv2.circle(car, (425, 350), 42, (20, 20, 20), -1)

        noise = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)

        for image in (apple, car, noise):
            self.assertEqual(app.detect_faces(image), [])

    def test_missing_detector_never_fabricates_a_face(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(app, "detect_faces_yunet", return_value=None), patch.object(app, "face_detectors", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "could not be loaded"):
                app.detect_faces(image)

    def test_face_candidate_requires_two_eye_features(self) -> None:
        image = np.zeros((200, 200), dtype=np.uint8)
        self.assertFalse(app.has_eye_pair(image, (20, 20, 160, 160)))

    def test_yunet_output_uses_model_confidence(self) -> None:
        model_output = np.array(
            [[10, 20, 80, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.91]],
            dtype=np.float32,
        )
        detector = unittest.mock.Mock()
        detector.detect.return_value = (None, model_output)
        with patch.object(app, "yunet_detector", return_value=detector):
            detections = app.detect_faces_yunet(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(detections[0]["box"], (10, 20, 80, 90))
        self.assertAlmostEqual(detections[0]["confidence"], 0.91, places=2)

    def test_yunet_native_error_retries_with_fresh_detector(self) -> None:
        model_output = np.array(
            [[10, 20, 80, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.91]],
            dtype=np.float32,
        )
        broken_detector = unittest.mock.Mock()
        broken_detector.detect.side_effect = cv2.error("native detector failure")
        fresh_detector = unittest.mock.Mock()
        fresh_detector.detect.return_value = (None, model_output)
        with patch.object(app, "yunet_detector", return_value=broken_detector), patch.object(
            app, "create_yunet_detector", return_value=fresh_detector
        ):
            detections = app.detect_faces_yunet(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 1)
        fresh_detector.detect.assert_called_once()

    def test_registration_retries_a_missed_frame(self) -> None:
        detection = {"box": (20, 20, 120, 120), "confidence": 0.72}
        with patch.object(app, "detect_faces", side_effect=[[], [detection]]), patch.object(
            app, "registration_face_quality", return_value=(True, "")
        ):
            result, note = app.detect_registration_face(np.zeros((200, 200, 3), dtype=np.uint8))
        self.assertEqual(result, detection)
        self.assertIn("retry", note)

    def test_single_enrolled_identity_does_not_force_a_match(self) -> None:
        person = {
            "name": "Registered person",
            "status": "Active",
            "embeddings": [[1.0, 0.0], [0.98, 0.02], [0.96, 0.04]],
        }
        stranger_probe = [0.0, 1.0]
        result = app.recognize_embedding(stranger_probe, [person], threshold=0.50)
        self.assertIsNone(result.person)

    def test_match_requires_multi_sample_support(self) -> None:
        person = {
            "name": "Registered person",
            "status": "Active",
            "embeddings": [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]],
        }
        result = app.recognize_embedding([1.0, 0.0], [person], threshold=0.50)
        self.assertEqual(result.person, person)
        self.assertGreater(result.confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
