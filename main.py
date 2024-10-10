import cv2
import numpy as np
import random

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
with open("dnn_model/classes.txt", "r") as file_object:
    classes = [class_name.strip() for class_name in file_object.readlines()]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

button_person = False
sunlight_level = random.randint(0, 100)

# Define button polygons outside the click function
person_button_polygon = np.array([(20, 20), (450, 20), (450, 70), (20, 70)])  # Quadrilateral points
generate_button_polygon = np.array([(20, 80), (280, 80), (280, 130), (20, 130)])  # Quadrilateral points

def click_button(event, x, y, flags, params):
    global button_person, sunlight_level
    if event == cv2.EVENT_LBUTTONDOWN:
        if cv2.pointPolygonTest(person_button_polygon, (x, y), False) > 0:
            button_person = not button_person
        elif cv2.pointPolygonTest(generate_button_polygon, (x, y), False) > 0:
            sunlight_level = random.randint(0, 100)

# Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

def draw_buttons(frame, people_count):
    cv2.rectangle(frame, (20, 20), (600, 70), (300, 0, 0), 3)
    label = 'Person Detected: No' if not button_person else f'Person Detected: Yes'
    cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (300, 0, 0), 3)

    cv2.rectangle(frame, (20, 80), (280, 130), (300, 0, 0), 3)
    cv2.putText(frame, 'Generate', (30, 120), cv2.FONT_HERSHEY_PLAIN, 3, (300, 0, 0), 3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    sunlight_threshold = 50
    (class_ids, scores, bboxes) = model.detect(frame)
    people_count = 0  # Reset people count

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        if classes[class_id] == "person" and button_person:
            people_count += 1
            x, y, w, h = bbox
            cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 50), 3)

    # Update lights based on people and sunlight
    light_state = "OFF" if people_count == 0 else "ON" if sunlight_level < sunlight_threshold else "DIM"
    cv2.putText(frame, f'Lights: {light_state}', (900, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0) if light_state == "ON" else (0, 255, 255), 3)
    cv2.putText(frame, f'Sunlight: {sunlight_level}', (900, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    draw_buttons(frame, people_count)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:  # Esc to exit
        break

cap.release()
cv2.destroyAllWindows()








