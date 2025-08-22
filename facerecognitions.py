import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import time

# -----------------------------
# Config
# -----------------------------
YOLO_MODEL_PATH = r"D:\workinghard\FaceRecoWebview\models\yolov8n-face.pt"

 # Select mode: "frame" or "time"
ENCODE_MODE = "frame"   # "frame" | "time"

# If mode = "frame"
FRAME_INTERVAL = 2  # encode every 2 frames

# If mode = "time"
TIME_INTERVAL = 0.5  # encode every 0.5 seconds

# -----------------------------
# Load YOLOv8 face detection model
# -----------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

# -----------------------------
# Load and encode known faces
# -----------------------------
known_face_encodings = []
known_face_names = []

data = {
    "Kan": r"D:\workinghard\FaceRecoWebview\data\Kan\kan1.png",
    "Mark": r"D:\workinghard\FaceRecoWebview\data\mark\mark1.jpg",
    "Toon": r"D:\workinghard\FaceRecoWebview\data\toon\toon1.jpg",
    "Wuna": r"D:\workinghard\FaceRecoWebview\data\wuna\wuna.jpg",
    "Pound": r"D:\workinghard\FaceRecoWebview\data\pon\pon1.jpg",
    "Ball": r"D:\workinghard\FaceRecoWebview\data\ball\ball1.jpg",
}

for name, path in data.items():
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:  # Check if a face is detected
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)

# -----------------------------
# Start camera
# -----------------------------
video_capture = cv2.VideoCapture(0)
prev_time = time.time()

frame_count = 0
last_encode_time = time.time()

face_locations = []
face_names = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # -----------------------------
    # Detect faces with YOLOv8
    # -----------------------------
    results = yolo_model(frame, verbose=False)
    new_face_locations = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2) in boxes:
            top, right, bottom, left = int(y1), int(x2), int(y2), int(x1)
            new_face_locations.append((top, right, bottom, left))

    # -----------------------------
    # Decide whether to encode or not
    # -----------------------------
    do_encode = False
    frame_count += 1

    if ENCODE_MODE == "frame":
        if frame_count % FRAME_INTERVAL == 0:
            do_encode = True
    elif ENCODE_MODE == "time":
        if current_time - last_encode_time >= TIME_INTERVAL:
            do_encode = True
            last_encode_time = current_time

    # -----------------------------
    # Encode faces (if scheduled)
    # -----------------------------
    if do_encode:
        face_locations = new_face_locations
        face_names = []
        for (top, right, bottom, left) in face_locations:
            face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])
            if face_encoding:
                face_encoding = face_encoding[0]
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index] if face_distances[best_match_index] < 0.5 else "Unknown"
                face_names.append(name)
                print(f"Detected: {name} at {current_time:.2f} seconds")
            else:
                face_names.append("Unknown")
                print(f"Detected: Unknown at {current_time:.2f} seconds")

    # -----------------------------
    # Draw results
    # -----------------------------
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.putText(frame, f"face count: {len(face_locations)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("YOLOv8 + Face Recognition (Hybrid)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
