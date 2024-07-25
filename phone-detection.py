import cv2
from ultralytics import YOLO
import cvzone


classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", 
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile phone","microwave","oven","toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"
]


def phoneDetect():
    model=YOLO('yolov8n.pt')

    cap=cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)

    flag=False

    while True:
        success, img=cap.read()
        results=model(img,stream=True)

        if not success:
            break

        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]                                
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                conf=round(float(box.conf[0]),2)                        
                id=int(box.cls[0])                                     
                class_name = classNames[id]

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)         


                if class_name == "mobile phone":
                    flag=True
                    cvzone.putTextRect(img,f'{class_name}{conf}',(max(0,x1),max(40,y1)))

                
                if flag==True:
                    cv2.putText(img, "WARNING: Phone Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 3)


        cv2.imshow("Cam footage. Press 'Q' to exit.",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return flag


if phoneDetect():
    print("The student has been detected using a mobile phone inappropriately.")  
else:
    print("The student has not been detected using unauthorized methods.")

