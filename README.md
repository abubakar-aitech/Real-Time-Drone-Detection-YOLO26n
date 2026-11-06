# Real-Time Drone Detection System

An AI-based real-time drone detection system developed using YOLO26n and computer vision.

## Project Overview
This project detects drones from camera input using a custom-trained YOLO26n object detection model. The trained model is exported to ONNX format and converted through MaixHub for deployment and testing on MaixCAM.

## Technologies Used
- Python
- PyTorch
- Ultralytics YOLO26n
- OpenCV
- Roboflow
- ONNX
- MaixHub
- MaixCAM
- Visual Studio Code

## Training Configuration
| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Image Size | 1280 |
| Epochs | 85 |
| Batch Size | 8 |
| Workers | 2 |
| Patience | 21 |

## Deployment Workflow
1. Dataset preparation and annotation.
2. YOLO26n model training.
3. Export to ONNX format.
4. Model conversion using MaixHub.
5. Deployment and testing on MaixCAM.

## Project Documentation
See `Drone_Detection_Documentation.pdf` for technical details.

## Future Improvements
- Improve detection of small and distant drones.
- Optimize edge-device inference speed.
- Evaluate model performance in different environments.

## Author
Developed as an AI and Computer Vision project.
