"""Image loading and preprocessing helpers for the face recognition app."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.face_ai import detect_faces_bgr
from utils.human_face_validation import FaceInputKind, FaceInputRejected


# Supported image extensions for the local dataset folder.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_human_face_from_pil(image: Image.Image) -> tuple[bool, int]:
    """Detect centrally validated humans; animal images raise immediately."""
    rgb_array = np.asarray(image.convert("RGB"))
    frame_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    detections = detect_faces_bgr(frame_bgr)
    return bool(detections), len(detections)


def prepare_image_from_pil(image: Image.Image, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Validate a human, then convert it to an Eigenfaces vector."""

    found_human, _ = detect_human_face_from_pil(image)
    if not found_human:
        raise FaceInputRejected(FaceInputKind.NO_FACE)

    # Convert the uploaded image to RGB first so OpenCV receives a predictable format.
    rgb_image = image.convert("RGB")

    # OpenCV uses NumPy arrays, so the PIL image is converted into an array.
    rgb_array = np.array(rgb_image)

    # Convert the image from RGB to grayscale because Eigenfaces works on brightness values.
    grayscale_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    # Resize every image to the same size so all vectors have the same number of pixels.
    resized_image = cv2.resize(grayscale_image, image_size, interpolation=cv2.INTER_AREA)

    # Flatten the 2D image matrix into a 1D vector for PCA.
    image_vector = resized_image.flatten().astype("float32")

    return resized_image, image_vector


def load_dataset_from_folder(dataset_path: Path, image_size: tuple[int, int]) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
    """Load training images from dataset/person_name folders."""

    # These lists store display images, vectorized images, identity labels, and filenames.
    display_images: list[np.ndarray] = []
    image_vectors: list[np.ndarray] = []
    labels: list[str] = []
    filenames: list[str] = []

    # Each subfolder name is treated as the person's label.
    for person_folder in sorted(dataset_path.iterdir()):
        if not person_folder.is_dir():
            continue

        # Every image inside the person's folder becomes one training example.
        for image_path in sorted(person_folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Open the image with Pillow, then reuse the same preprocessing pipeline.
            image = Image.open(image_path)
            grayscale_image, image_vector = prepare_image_from_pil(image, image_size)

            display_images.append(grayscale_image)
            image_vectors.append(image_vector)
            labels.append(person_folder.name)
            filenames.append(image_path.name)

    return display_images, image_vectors, labels, filenames


def infer_label_from_filename(filename: str) -> str:
    """Guess a person's label from a filename such as alice_01.jpg or bob-face.png."""

    # Remove the extension and keep the part before the first common separator.
    name_without_extension = Path(filename).stem

    # This makes beginner datasets easy: person1_01.jpg becomes person1.
    for separator in ("_", "-", " "):
        if separator in name_without_extension:
            return name_without_extension.split(separator)[0].strip() or name_without_extension

    return name_without_extension
