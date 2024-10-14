import cv2
import random

# Update the paths
handCascade = cv2.CascadeClassifier("C:/Users/Lucifer/Downloads/ai/Face_HandPasswordUnlocker/handTracking.xml")
palmCascade = cv2.CascadeClassifier("C:/Users/Lucifer/Downloads/ai/Face_HandPasswordUnlocker/handPalm.xml")
frontalFaceCloser = cv2.CascadeClassifier("C:/Users/Lucifer/Downloads/ai/Face_HandPasswordUnlocker/frontalFaceCloser.xml")

video_capture = cv2.VideoCapture(0)

while 1:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands
    hands = handCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 60))
    
    # Detect palms
    handPalm = palmCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 60))
    
    # Detect faces
    frontalFaceCloser_detect = frontalFaceCloser.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 45))

    # Unlock hand function
    def unlock_hand():
        for i in range(10):
            for (x, y, w, h) in handPalm:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 0, 20), 3)
                print("Unlocked -- " + str(random.randint(1000, 9999)))

    # Hands activation function
    def hands_activation():
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (20, 0, 180), 3)
        if len(hands) > 0:
            unlock_hand()

    # Draw rectangles for detected faces
    for (x, y, w, h) in frontalFaceCloser_detect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    if len(frontalFaceCloser_detect) > 0:
        hands_activation()

    # Display the frame
    cv2.imshow("Face Hand Unlock", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
