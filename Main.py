import cv2
from ultralytics import YOLO
from collections import defaultdict

#Load the YOLO model
model= YOLO('yolo11l.pt')

#Open the video file
cap=cv2.VideoCapture('test_videos/cars_video_for_detection(720p).mp4')