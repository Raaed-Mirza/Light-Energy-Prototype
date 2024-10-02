import cv2
import numpy as np

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Object List:")
print(classes)

cap = cv2.VideoCapture(0) # 0 is the camera index, so the first camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Full HD 1920 x 1000

button_person = False

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])

        is_inside = cv2.pointPolygonTest(polygon, (x, y), False)
        if is_inside > 0:
            print("We clicked the button", x, y)

            if button_person is False:
                button_person = True

            else:
                button_person = False

            print("Button Person: ", button_person)


# Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    ret, frame = cap.read() # ret is a boolean that returns True if the frame is read correctly

    # Detect objects
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        x, y, w, h = bbox
        #print(x, y, w, h)

        class_name = classes[class_id]

        if class_name == "person" and button_person is True:
            cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 50), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 50), 3)



    #print("Class IDs: ", class_ids)
    #print("Scores: ", scores)
    #print("Bounding Boxes: ", bboxes)

    # Create a button
    #cv2.rectangle(frame, (20, 20), (220, 70), (0, 200, 50), -1)
    polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
    cv2.fillPoly(frame, polygon, (0, 200, 50))
    cv2.putText(frame, 'Person', (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow("Frame", frame) # Display the frame
    key = cv2.waitKey(1) # Wait for a key press
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()