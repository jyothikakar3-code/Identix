"""Regression tests for the face-detection confidence bug."""

from pathlib import Path
import tempfile
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


class ApplicationRegressionTests(unittest.TestCase):
    def test_reinitializing_database_never_deletes_registered_people(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_db = Path(temporary_directory) / "persistent.sqlite3"
            with patch.object(app, "DB_PATH", temporary_db), patch.object(
                app, "DATABASE_URL", ""
            ), patch.object(app, "reset_storage_files") as reset_storage:
                app.init_db()
                app.execute(
                    """
                    INSERT INTO people(
                        person_id, name, email, phone, department, embeddings,
                        image_paths, status, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "persist-1",
                        "Persistent Person",
                        "person@example.com",
                        "",
                        "Test",
                        "[]",
                        "[]",
                        "Active",
                        app.now_iso(),
                        app.now_iso(),
                    ),
                )

                app.init_db()
                people = app.rows("people", "WHERE person_id = ?", ("persist-1",))

            self.assertEqual(len(people), 1)
            self.assertEqual(people[0]["name"], "Persistent Person")
            reset_storage.assert_not_called()

    def test_postgres_placeholders_are_translated(self) -> None:
        with patch.object(app, "DATABASE_URL", "postgresql://example.invalid/database"):
            self.assertEqual(
                app.database_query("SELECT * FROM people WHERE person_id = ?"),
                "SELECT * FROM people WHERE person_id = %s",
            )

    def test_postgres_initialization_uses_portable_schema(self) -> None:
        class FakeResult:
            def __init__(self, row=None):
                self.row = row

            def fetchone(self):
                return self.row

        class FakeConnection:
            def __init__(self):
                self.statements = []
                self.committed = False
                self.closed = False

            def execute(self, query, params=()):
                self.statements.append((query, params))
                if "recognition_model_version" in query and query.lstrip().startswith("SELECT"):
                    return FakeResult(None)
                if "COUNT(*) AS count FROM cameras" in query:
                    return FakeResult({"count": 0})
                return FakeResult()

            def commit(self):
                self.committed = True

            def close(self):
                self.closed = True

        connection = FakeConnection()
        with patch.object(app, "DATABASE_URL", "postgresql://example.invalid/database"), patch.object(
            app, "connect", return_value=connection
        ):
            app.init_db()

        schema_statements = "\n".join(query for query, _ in connection.statements)
        self.assertIn("BIGSERIAL PRIMARY KEY", schema_statements)
        self.assertNotIn("AUTOINCREMENT", schema_statements)
        self.assertIn("VALUES(%s, %s)", schema_statements)
        self.assertTrue(connection.committed)
        self.assertTrue(connection.closed)

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

    def test_verification_uses_pose_tolerant_aligned_embeddings(self) -> None:
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        first_detection = {"box": (10, 10, 80, 80), "face_geometry": [0.0] * 14}
        second_detection = {"box": (12, 10, 80, 80), "face_geometry": [0.0] * 14}
        with patch.object(
            app,
            "verification_embedding",
            side_effect=[[1.0, 0.0], [0.8, 0.2]],
        ) as aligned_embedding:
            similarity = app.verification_similarity(
                frame, first_detection, frame, second_detection
            )
        self.assertGreater(similarity, app.VERIFICATION_COSINE_THRESHOLD)
        self.assertEqual(aligned_embedding.call_count, 2)

    def test_verification_match_score_is_feature_based_and_monotonic(self) -> None:
        rejected_score = app.verification_match_score(0.05)
        threshold_score = app.verification_match_score(app.VERIFICATION_COSINE_THRESHOLD)
        strong_score = app.verification_match_score(0.70)
        self.assertLess(rejected_score, threshold_score)
        self.assertAlmostEqual(threshold_score, 0.50)
        self.assertLess(threshold_score, strong_score)

    def test_verification_rejects_unrelated_features(self) -> None:
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        detection = {"box": (10, 10, 80, 80), "face_geometry": [0.0] * 14}
        with patch.object(
            app,
            "verification_embedding",
            side_effect=[[1.0, 0.0], [0.0, 1.0]],
        ):
            similarity = app.verification_similarity(frame, detection, frame, detection)
        self.assertLess(similarity, app.VERIFICATION_COSINE_THRESHOLD)
        self.assertLess(app.verification_match_score(similarity), 0.02)

    def test_eye_evidence_is_capped_at_one_plausible_pair(self) -> None:
        duplicate_boxes = np.array(
            [[10, 10, 20, 20], [12, 11, 20, 20], [70, 10, 20, 20], [72, 11, 20, 20]]
        )
        self.assertEqual(app.plausible_eye_count(duplicate_boxes, 100, 100), 2)

    def test_liveness_requires_blink_in_the_second_frame(self) -> None:
        first_state = {"center_x": 100.0, "width": 100.0, "yaw": 0.0}
        last_state = {"center_x": 112.0, "width": 100.0, "yaw": 0.10}
        result = app.evaluate_liveness_evidence(
            [0, 2, 2], first_state, last_state, [0.80, 0.70]
        )
        self.assertFalse(result["live"])
        self.assertFalse(result["blink_ok"])

    def test_liveness_accepts_ordered_same_face_session(self) -> None:
        first_state = {"center_x": 100.0, "width": 100.0, "yaw": 0.0}
        last_state = {"center_x": 112.0, "width": 100.0, "yaw": 0.10}
        result = app.evaluate_liveness_evidence(
            [2, 0, 2], first_state, last_state, [0.80, 0.70]
        )
        self.assertTrue(result["live"])
        self.assertTrue(result["blink_ok"])
        self.assertTrue(result["move_ok"])

    def test_liveness_rejects_different_faces_across_frames(self) -> None:
        first_state = {"center_x": 100.0, "width": 100.0, "yaw": 0.0}
        last_state = {"center_x": 112.0, "width": 100.0, "yaw": 0.10}
        result = app.evaluate_liveness_evidence(
            [2, 0, 2], first_state, last_state, [0.80, 0.10]
        )
        self.assertFalse(result["live"])
        self.assertFalse(result["identity_ok"])

    def test_liveness_pipeline_detects_each_face_before_feature_checks(self) -> None:
        frames = [np.zeros((160, 160, 3), dtype=np.uint8) for _ in range(3)]
        detections = [
            {"box": (20, 20, 100, 100), "face_geometry": [20, 20, 100, 100, 50, 55, 90, 55, 70, 75, 0, 0, 0, 0]},
            {"box": (20, 20, 100, 100), "face_geometry": [20, 20, 100, 100, 50, 55, 90, 55, 70, 75, 0, 0, 0, 0]},
            {"box": (30, 20, 100, 100), "face_geometry": [30, 20, 100, 100, 60, 55, 100, 55, 85, 75, 0, 0, 0, 0]},
        ]
        with patch.object(
            app, "detect_faces", side_effect=[[item] for item in detections]
        ) as face_detector, patch.object(
            app, "count_visible_eyes", side_effect=[2, 0, 2]
        ), patch.object(
            app, "embedding_from_detection", return_value=[1.0, 0.0]
        ):
            result = app.analyze_liveness_frames(frames)
        self.assertTrue(result["live"])
        self.assertEqual(face_detector.call_count, 3)
        self.assertTrue(
            all(call.args[1] == app.REGISTRATION_DETECTION_THRESHOLD for call in face_detector.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
