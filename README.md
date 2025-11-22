# Vehicle Detection and Tracking System

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

## ⚙️ How It Works  
### 1. **YOLO Model Loading**  
The script loads the `yolo11l.pt` model and its class names from Ultralytics.

### 2. **Video Processing**  
The system loads a test video stored at:
```
test_videos/cars_video_for_detection(720p).mp4
```

### 3. **Vehicle Tracking & Line Crossing**  
- A horizontal line (`line_y_red = 480`) is drawn across the frame  
- Whenever a vehicle’s bounding box center crosses the line, its class count is incremented  
- Counts are stored in a Python `defaultdict(int)`

### 4. **Real-Time Display**  
The video window shows:  
- Bounding boxes  
- Class labels  
- A red detection line  
- Total vehicle counts by category  

Press **Q** to exit.

## 🚀 Getting Started  

### 1. Clone the Repository  
```bash
git clone https://github.com/subhashgowdak/Vehicle-Detection-and-Tracking-System.git
cd Vehicle-Detection-and-Tracking-System
```

### 2. Create a Virtual Environment  
```bash
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
.\.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies  
```bash
pip install -r Requirements.txt
```

### 4. Download YOLO Model  
Download YOLO model (if not present):

```bash
yolo download yolo11l.pt
```

or manually place any **YOLO model file** in the project folder and update:

```python
model = YOLO('yolo11l.pt')
```

### 5. Run the System  
```bash
python Main.py
```
