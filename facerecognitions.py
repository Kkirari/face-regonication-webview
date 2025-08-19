import cv2
import face_recognition
import numpy as np
import time

# Load and encode known faces
known_face_encodings = []
known_face_names = []

data = {
    "Kan": r"D:\workinghard\Python\FaceRecognition\data\Kan\kan1.png",
    "Mark": r"D:\workinghard\Python\FaceRecognition\data\mark\mark1.jpg",
    "Toon": r"D:\workinghard\Python\FaceRecognition\data\toon\toon1.jpg",
    "Wuna": r"D:\workinghard\Python\FaceRecognition\data\wuna\wuna.jpg",
    "Pound": r"D:\workinghard\Python\FaceRecognition\data\pon\pon1.jpg",
    "Ball": r"D:\workinghard\Python\FaceRecognition\data\ball\ball1.jpg",
}

for name, path in data.items():
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:  # Check if a face is found in the image
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)

# Start camera
video_capture = cv2.VideoCapture(0)

frame_count = 0
face_locations = []
face_names = []

# สำหรับ FPS counter
fps = 0
frame_start = time.time()

# สำหรับการประมวลผลทุกๆ N วินาที
last_process_time = 0
process_interval = 0.3  # ประมวลผลทุกๆ 3 วินาที

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize to quarter for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    frame_count += 1
    now = time.time()

    # --- Option 1: ประมวลผลทุก 3 เฟรม ---
    # process_this_frame = frame_count % 3 == 0

    # --- Option 2: ประมวลผลทุกๆ 3 วินาที ---
    process_this_frame = (now - last_process_time) >= process_interval

    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < 0.5:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
            face_names.append(name)

        last_process_time = now  # อัปเดตเวลา

    # Draw rectangles
    scale = 4
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= scale
        right *= scale
        bottom *= scale
        left *= scale
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Detected {name} at {left}, {top}, {right}, {bottom}")

    # --- FPS Counter ---
    end = time.time()
    fps = 1 / (end - frame_start)
    frame_start = end
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.putText(frame,f"detected {len(face_names)} faces", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
