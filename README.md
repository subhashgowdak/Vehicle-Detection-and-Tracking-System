## Vehicle Detection and Tracking System

A Python-based **real-time vehicle detection, tracking, and counting system** built using YOLO, OpenCV, and video analytics.
This project demonstrates how computer vision can be used for traffic monitoring, automated counting, and basic surveillance tasks.

## 🚗 Project Overview  
This system uses the **YOLO (Ultralytics)** object detection model to identify vehicles in a video stream, track their movement, and count how many of them cross a predefined line.

**Core functionalities include:**  
- Detecting vehicles from a video file  
- Tracking objects frame by frame  
- Counting vehicles when they cross a specified line  
- Displaying real-time bounding boxes and labels  
- Showing live count of each vehicle class (e.g., car, bus, truck, bike)

## 📁 Project Structure  
```
Vehicle-Detection-and-Tracking-System/
│
├── Main.py                # Main script for detection, tracking, and counting
├── Requirements.txt       # Dependencies for running the project
├── LICENSE                # Open-source license (MIT / Apache etc.)
└── .gitignore
```

## 🧠 Technologies Used  
- **Python**  
- **OpenCV** – video processing & visualization  
- **Ultralytics YOLO** (yolo11l.pt) – vehicle detection  
- **Default Python Collections** – tracking & counting
