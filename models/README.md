# Face detector model

At first startup, the app downloads `face_detection_yunet_2023mar.onnx` from
the official OpenCV Zoo repository. YuNet and the files in its model directory
are licensed under the MIT License. The model is cached here for later runs.

The app also downloads `face_recognition_sface_2021dec.onnx` from OpenCV Zoo
for identity embeddings and caches it here. SFace replaces the previous
grayscale pixel descriptor, which could not reliably reject unknown people.

Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

SFace source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

The shared human/animal safety gate also downloads `efficientnet-lite4-11.onnx` from
the official ONNX Model Zoo. It verifies SHA-256
`d111689907c06eea7c82e4833ddef758da6453b9d4cf60b7e99ca05c7cbd9c12`
before loading it with OpenCV DNN. The model covers ImageNet's broad animal
taxonomy; candidate classification runs before every face embedding.

EfficientNet-Lite4 source: https://github.com/onnx/models/tree/main/validated/vision/classification/efficientnet-lite4
