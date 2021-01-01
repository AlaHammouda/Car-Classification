import cv2 
import numpy as np
import math

weightsPath = "./yolov3.weights"   #Weight file
configPath = "./yolov3.cfg" #Cfg file

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #dnn stands for deep neural network

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#we now load the classes from the coco.names file (contains classes names)

classes=[]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names=net.getLayerNames()
outputLayers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#read video
cap = cv2.VideoCapture('video.mp4')

if (cap.isOpened()== False):
    print("Error Opening video")
    
while(cap.isOpened()):

       
    _,frame=cap.read()
    frame=cv2.resize(frame,(1200,650))
    	
    height,width,channels=frame.shape

    blob = cv2.dnn.blobFromImage(frame, scalefactor=(1.0/255), size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob) # we feed the preproccessed img to the network (input)
    objects = net.forward(outputLayers)  # forwward -> forward propagation, outputs = predictions of the model = the objects that the model prdicted that they exist

    class_ids = []
    confidences = []
    boxes = []

    for obj in objects:
        for information in obj:
            scores = information[5:]
            class_id=np.argmax(scores)
            if class_id in [2,5,7]: #car,bus,truck
                confidence=scores[class_id]
                if confidence > 0.85:
                    #the right object is detected
                    center_x= int(information[0]*width)
                    center_y= int (information[1]*height)
                    width_object=int(information[2]*width)
                    height_object=int(information[3]*height)
    
                    #rectangle coordinates
                    x=int(center_x - width_object/2)
                    y=int(center_y - height_object/2)

                    boxes.append([x, y, width_object, height_object,center_x,center_y])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    correct_boxes = []
    for i in range(len(boxes)):
        if i in indexes:
           correct_boxes.append(boxes[i])
             
    
    for i in range(len(correct_boxes)):
        x, y, w, h,cx,cy = correct_boxes[i]
        label = str(classes[class_ids[i]])
        if (class_ids[i]==2):
            color = (0,0,255)
        elif (class_ids[i]==4):
            color = (0,255,0)
        elif (class_ids[i]==3):
            color = (255,0,0)
        else:
            color = (0,255,255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    
    cv2.imshow("AI Project", frame)
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break
            
cap.release()
cv2.destroyAllWindows()


