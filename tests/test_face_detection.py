"""Regression tests for the face-detection confidence bug."""

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import app


ROOT = Path(__file__).parents[1]


class FaceDetectionTests(unittest.TestCase):
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
        with patch.object(app, "face_detectors", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "could not be loaded"):
                app.detect_faces(image)


if __name__ == "__main__":
    unittest.main()
