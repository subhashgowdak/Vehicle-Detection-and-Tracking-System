import cv2
from ultralytics import YOLO
from collections import defaultdict

#Load the YOLO model
model= YOLO('yolo11l.pt')