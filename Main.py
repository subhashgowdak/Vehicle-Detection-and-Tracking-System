import cv2
from ultralytics import YOLO
from collections import defaultdict

#Load the YOLO model
model= YOLO('yolo11l.pt')
class_list=model.names

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

        #loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1,y1,x2,y2=map(int,box)
            class_name=class_list[class_idx]
            cv2.putText(frame,f"ID: {track_id} {class_name}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    #show the frame
    cv2.imshow("YOLO object tracking and counting", frame)

    #exit loop if q is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break