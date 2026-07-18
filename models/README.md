# Face detector model

At first startup, the app downloads `face_detection_yunet_2023mar.onnx` from
the official OpenCV Zoo repository. YuNet and the files in its model directory
are licensed under the MIT License. The model is cached here for later runs.

The app also downloads `face_recognition_sface_2021dec.onnx` from OpenCV Zoo
for identity embeddings and caches it here. SFace replaces the previous
grayscale pixel descriptor, which could not reliably reject unknown people.

Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

SFace source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
