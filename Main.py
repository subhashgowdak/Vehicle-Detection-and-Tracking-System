import cv2
from ultralytics import YOLO
from collections import defaultdict

#Load the YOLO model
model= YOLO('yolo11l.pt')
class_list=model.names

#Open the video file
cap=cv2.VideoCapture('test_videos/cars_video_for_detection(720p).mp4')

line_y_red=480 #red line position

#dictionary to store object count by class
class_counts=defaultdict(int)

#dictionary to keep track of object IDs that have crossed the line
crossed_ids=set()

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

        #draw red line on each frame
        cv2.line(frame,(5,line_y_red),(640,line_y_red),(0,0,255),3)

        #loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1,y1,x2,y2=map(int,box)
            cx= (x1+x2) // 2 #calculate the center point
            cy= (y1+y2) // 2

            class_name=class_list[class_idx]
            cv2.putText(frame,f"ID: {track_id} {class_name}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw circle at the center of each detected object

            #check if the object has crossed the red line
            if cy > line_y_red and track_id not in crossed_ids:
                #mark the object as crossed
                crossed_ids.add(track_id)
                class_counts[class_name]+=1

        #Display the counts on the frame
        y_offset=30
        for class_name, count in class_counts.items():
            cv2.putText(frame,f"{class_name}: {count}",(20,y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,250),2)
            y_offset+=30

    #show the frame
    cv2.imshow("YOLO object tracking and counting", frame)

    #exit loop if q is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

#release resources
cap.release()
cv2.destroyAllWindows()