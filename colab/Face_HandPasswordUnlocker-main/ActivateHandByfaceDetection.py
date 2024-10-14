import cv2
import face_recognition
import os

# ฟังก์ชันโหลดข้อมูลใบหน้า
def load_face_encodings(folder_path):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(filename.split('.')[0])
                print(f"{filename}: Encoding found.")
            else:
                print(f"{filename}: No encoding found.")

    return known_face_encodings, known_face_names

# ตั้งค่าพาธ
face_folder = "C:/Users/Lucifer/Downloads/colab/saved_faces"
known_face_encodings, known_face_names = load_face_encodings(face_folder)

# เริ่มการจับวิดีโอ
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break  # Exit if frame not captured

    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    print(f"Frame shape: {rgb_frame.shape}")  # ควรเป็น (height, width, 3)

    # ตรวจจับใบหน้า
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = []

    print(f"Face locations: {face_locations}")

    if face_locations:  # ตรวจสอบว่ามีการตรวจจับใบหน้าหรือไม่
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if len(face_encodings) != len(face_locations):  # ตรวจสอบให้แน่ใจว่ามีการเข้ารหัสเท่ากับการตรวจจับ
                print("Mismatch between face locations and face encodings.")
            if not face_encodings:  # ตรวจสอบว่า face_encodings เป็นลิสต์ที่ไม่ว่างเปล่า
                print("No face encodings found.")
        except Exception as e:
            print(f"Error while encoding faces: {e}")

    access_granted = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if face_encodings:  # ตรวจสอบว่า face_encodings มีข้อมูล
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            print(f"Matches: {matches}")  # พิมพ์ค่า matches
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                access_granted = True  # เข้าถึงได้

        # วาดกรอบรอบใบหน้า
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if access_granted:  # ตรวจสอบการเข้าถึง
        cv2.putText(frame, "Pass", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Pass", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
