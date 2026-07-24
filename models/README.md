# Face detector model

`face_detection_yunet_2023mar.onnx` is bundled from the official OpenCV Zoo
repository. YuNet and the files in its model directory are licensed under the
MIT License. Keeping the verified model with the app avoids an exam/demo-time
network download after Streamlit wakes from sleep.

`face_recognition_sface_2021dec.onnx` is also bundled from OpenCV Zoo for
identity embeddings. SFace replaces the previous grayscale pixel descriptor,
which could not reliably reject unknown people.

Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

SFace source: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

The shared human/animal safety gate bundles the compact
`efficientnet-lite4-11-int8.onnx` model from the official ONNX Model Zoo. It
verifies SHA-256
`2b3cbb5077262b20df565dacddecb3724c0976c35029a87e512d13aa4eff04a2`
before loading it with OpenCV DNN. The model covers ImageNet's broad animal
taxonomy; candidate classification runs before every face embedding. The
download URL remains only as a recovery fallback if a deployment is damaged.

EfficientNet-Lite4 source: https://github.com/onnx/models/tree/main/validated/vision/classification/efficientnet-lite4
