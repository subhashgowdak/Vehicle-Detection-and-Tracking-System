import cv2
from ultralytics import YOLO
from collections import defaultdict

#Load the YOLO model
model= YOLO('yolo11l.pt')

#Open the video file
cap=cv2.VideoCapture('test_videos/cars_video_for_detection(720p).mp4')

while cap.isOpened():
    ret, frame=cap.read()
    if not ret:
        break

    #run yolo tracking on the frame
    results = model.track(frame, persist=True, classes=[1,2,3,5,6,7])
    #print(results)

    #Ensure results are not empty
    if results[0].boxes.data is not None:
        #get detected boxes, their class indices, & track IDs
        boxes=results[0].boxes.xyxy.cpu()
        track_ids=results[0].boxes.id.int().cpu().tolist()
        class_indices=results[0].boxes.cls.int().cpu().tolist()
        confidences=results[0].boxes.conf.cpu()