import cv2
import numpy as np
import pickle

# โหลดโมเดลการตรวจจับมือ
handCascade = cv2.CascadeClassifier("C:/Users/Lucifer/Downloads/colab/Face_HandPasswordUnlocker-main/handTracking.xml")

# ฟังก์ชันสำหรับบันทึกสัญลักษณ์มือ
def save_gesture(gesture_data):
    with open('gestures.pkl', 'wb') as f:
        pickle.dump(gesture_data, f)

# ฟังก์ชันสำหรับเปรียบเทียบสัญลักษณ์มือ
def compare_gesture(current_gesture):
    try:
        with open('gestures.pkl', 'rb') as f:
            saved_gestures = pickle.load(f)
        return current_gesture in saved_gestures
    except FileNotFoundError:
        return False

# ฟังก์ชันสำหรับบันทึกภาพ
def save_image(frame, image_count):
    image_name = f"C:/Users/Lucifer/Downloads/colab/New folder/hand_gesture_{image_count}.png"
    cv2.imwrite(image_name, frame)
    print(f"Saved: {image_name}")

# เริ่มกล้อง
video_capture = cv2.VideoCapture(0)
image_count = 0

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = handCascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # บันทึกภาพเมื่อพบมือ
    if len(hands) > 0:
        save_image(frame, image_count)
        image_count += 1

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
