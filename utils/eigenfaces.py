"""PCA and Eigenfaces helpers for the face recognition app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FaceRecognitionResult:
    """Store the recognition result shown in the Streamlit interface."""

    predicted_label: str
    confidence: float
    distance: float
    nearest_index: int
    nearest_image_label: str
    image_distances: list[tuple[str, float]]
    label_distances: list[tuple[str, float]]
    recognized: bool
    unknown_threshold: float
    failure_reason: str


class EigenfacesRecognizer:
    """A beginner-friendly Eigenfaces recognizer built with scikit-learn PCA."""

    def __init__(self, n_components: int):
        """Create the recognizer with the requested number of principal components."""

        # PCA learns eigenvectors of the covariance matrix of the training images.
        self.n_components = n_components
        self.pca: PCA | None = None
        self.labels: list[str] = []
        self.mean_face: np.ndarray | None = None
        self.train_vectors: np.ndarray | None = None
        self.train_projections: np.ndarray | None = None
        self.training_distances: np.ndarray | None = None
        self.label_centroids: dict[str, np.ndarray] = {}
        self.unknown_threshold: float = 0.0

    def fit(
        self,
        image_vectors: list[np.ndarray],
        labels: list[str],
        *,
        human_validated: bool = False,
    ) -> None:
        """Train only on vectors produced by validated human preprocessing."""
        if not human_validated:
            raise RuntimeError(
                "Eigenfaces training blocked: inputs did not pass human/animal validation."
            )

        # Build matrix X where each row is one face vector.
        data_matrix = np.vstack(image_vectors).astype("float32")

        # PCA cannot use more components than rank, samples, or pixels.
        # The final component in a tiny dataset is often zero-variance, so we avoid it.
        safe_components = min(self.n_components, data_matrix.shape[0] - 1, data_matrix.shape[1])
        safe_components = max(1, safe_components)

        # Store the labels so the nearest PCA vector can be converted back to a name.
        self.labels = labels
        self.train_vectors = data_matrix

        # Fit PCA: this computes the mean face, eigenvalues, and eigenvectors/eigenfaces.
        # Whitening is disabled because it can exaggerate near-zero components in small datasets.
        self.pca = PCA(n_components=safe_components, svd_solver="full", whiten=False)
        self.train_projections = self.pca.fit_transform(data_matrix)
        self.mean_face = self.pca.mean_
        self.n_components = safe_components

        # Average all images of the same person in PCA space.
        # Recognition uses these person centroids so one person's first image cannot dominate.
        self.label_centroids = {}
        for label in sorted(set(labels)):
            label_rows = [index for index, row_label in enumerate(labels) if row_label == label]
            self.label_centroids[label] = np.mean(self.train_projections[label_rows], axis=0)

        # Estimate typical distances between training examples for a confidence score.
        self.training_distances = self._nearest_training_distances()

        # Learn a rejection threshold from the training set.
        # If a test face is farther than this, the app marks it as an unknown person.
        self.unknown_threshold = self._estimate_unknown_threshold()

    def recognize(
        self,
        test_vector: np.ndarray,
        *,
        human_validated: bool = False,
    ) -> FaceRecognitionResult:
        """Compare only a vector produced by validated human preprocessing."""
        if not human_validated:
            raise RuntimeError(
                "Eigenfaces recognition blocked: input did not pass human/animal validation."
            )

        if self.pca is None or self.train_projections is None:
            raise ValueError("The recognizer must be trained before recognizing a face.")

        # Convert the test image vector into the same PCA feature space as the training images.
        test_projection = self.pca.transform(test_vector.reshape(1, -1).astype("float32"))

        # Measure Euclidean distance between the test image and every training image in PCA space.
        image_distances_array = np.linalg.norm(self.train_projections - test_projection, axis=1)

        # The smallest image distance is used only to display the nearest example image.
        nearest_index = int(np.argmin(image_distances_array))
        nearest_image_label = self.labels[nearest_index]

        # Compare against each person's average PCA vector.
        # This is more stable when several training images are uploaded for the same person.
        label_distances_array = {
            label: float(np.linalg.norm(centroid - test_projection.reshape(-1)))
            for label, centroid in self.label_centroids.items()
        }
        predicted_label = min(label_distances_array, key=label_distances_array.get)
        nearest_label_distance = label_distances_array[predicted_label]

        # Reject the test image when it is too far from every trained person.
        recognized = nearest_label_distance <= self.unknown_threshold
        failure_reason = ""
        if not recognized:
            failure_reason = "Face Detection Failed: this face does not match the trained people."

        # Convert distance into an easy-to-read educational confidence score.
        confidence = self._distance_to_confidence(nearest_label_distance) if recognized else 0.0

        # Prepare sorted distance lists for display in the app.
        image_distances = sorted(
            [
                (f"{self.labels[index]} - training image {index + 1}", float(distance))
                for index, distance in enumerate(image_distances_array)
            ],
            key=lambda item: item[1],
        )
        label_distances = sorted(label_distances_array.items(), key=lambda item: item[1])

        return FaceRecognitionResult(
            predicted_label=predicted_label,
            confidence=confidence,
            distance=nearest_label_distance,
            nearest_index=nearest_index,
            nearest_image_label=nearest_image_label,
            image_distances=image_distances,
            label_distances=label_distances,
            recognized=recognized,
            unknown_threshold=float(self.unknown_threshold),
            failure_reason=failure_reason,
        )

    def get_eigenfaces(self, image_shape: tuple[int, int], max_faces: int = 8) -> list[np.ndarray]:
        """Return the first few eigenfaces reshaped as images."""

        if self.pca is None:
            return []

        # PCA components are vectors; reshaping them turns them into visible Eigenfaces.
        eigenfaces = []
        for component in self.pca.components_[:max_faces]:
            eigenface = component.reshape(image_shape)
            eigenfaces.append(eigenface)

        return eigenfaces

    def explained_variance_percentages(self) -> np.ndarray:
        """Return how much information each selected principal component explains."""

        if self.pca is None:
            return np.array([])

        return self.pca.explained_variance_ratio_ * 100

    def reduced_size_ratio(self) -> tuple[int, int, float]:
        """Compare original pixel-vector size with PCA-reduced vector size."""

        if self.train_vectors is None:
            return 0, 0, 0.0

        original_size = int(self.train_vectors.shape[1])
        reduced_size = int(self.n_components)
        reduction_percent = (1 - reduced_size / original_size) * 100

        return original_size, reduced_size, reduction_percent

    def _estimate_unknown_threshold(self) -> float:
        """Estimate how far a test image may be before it is treated as unknown."""

        if self.train_projections is None or not self.label_centroids:
            return 0.0

        # Measure how far each training image is from its own person's average vector.
        own_label_distances = []
        for index, projection in enumerate(self.train_projections):
            label = self.labels[index]
            own_centroid = self.label_centroids[label]
            own_label_distances.append(float(np.linalg.norm(projection - own_centroid)))

        if not own_label_distances:
            return 1.0

        own_distances = np.array(own_label_distances, dtype="float32")
        own_mean = float(np.mean(own_distances))
        own_std = float(np.std(own_distances))
        own_max = float(np.max(own_distances))

        # Start with a relaxed boundary around known training variation.
        threshold = max(own_mean + 3.0 * own_std, own_max * 1.6, 1.0)

        # When multiple people are trained, keep the boundary below the distance between identities.
        centroids = list(self.label_centroids.values())
        if len(centroids) >= 2:
            inter_label_distances = []
            for first_index, first_centroid in enumerate(centroids):
                for second_centroid in centroids[first_index + 1:]:
                    inter_label_distances.append(float(np.linalg.norm(first_centroid - second_centroid)))

            if inter_label_distances:
                closest_identity_distance = min(inter_label_distances)
                identity_boundary = closest_identity_distance * 0.55
                if identity_boundary > own_max:
                    threshold = min(threshold, identity_boundary)

        return float(max(threshold, 1.0))

    def _nearest_training_distances(self) -> np.ndarray:
        """Calculate nearest-neighbor distances among training images."""

        if self.train_projections is None or len(self.train_projections) < 2:
            return np.array([1.0])

        # Create a full distance matrix between all training projections.
        differences = self.train_projections[:, None, :] - self.train_projections[None, :, :]
        distances = np.linalg.norm(differences, axis=2)

        # Ignore each image's distance to itself by setting the diagonal to infinity.
        np.fill_diagonal(distances, np.inf)

        return np.min(distances, axis=1)

    def _distance_to_confidence(self, distance: float) -> float:
        """Convert a PCA-space distance into a simple confidence percentage."""

        if self.training_distances is None or len(self.training_distances) == 0:
            return 0.0

        # A smaller distance means a better match; this scale is for classroom demonstration.
        typical_distance = float(np.mean(self.training_distances) + np.std(self.training_distances))
        if typical_distance <= 0:
            typical_distance = 1.0

        confidence = max(0.0, 100.0 * (1.0 - distance / (typical_distance * 2.0)))
        return round(min(confidence, 100.0), 2)
