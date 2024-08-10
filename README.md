

# 🚀 YOLOv8 Person Detection in Video 🎥

Welcome to the **YOLOv8 Person Detection** project! This repository demonstrates how to use the YOLOv8 model to detect persons in a video, drawing bounding boxes around them, and saving the processed video for easy visualization.

![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-blue) ![Python](https://img.shields.io/badge/Python-3.x-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)

## 🔥 Features

- **YOLOv8 Model**: Leverage the powerful YOLOv8n model pre-trained on the COCO dataset.
- **Real-Time Detection**: Detect persons in video frames in real-time.
- **Customizable Output**: Save the processed video with bounding boxes and labels for detected persons.
- **Easy to Use**: Simple setup and customization for different use cases.

## 📋 Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8


## 📝 Code Explanation
- Model Loading: The YOLOv8 model is loaded using the Ultralytics library.
- Video Processing: The script reads the video frame by frame and processes each frame through the YOLO model.
- Labeling: Detected persons are labeled with bounding boxes and the text "Person Detected" is displayed on the frame.
- Output: The processed frames are saved into a new video file.
