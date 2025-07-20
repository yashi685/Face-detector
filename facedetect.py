import cv2
import numpy as np

# Labels
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness',
            'anger', 'disgust', 'fear', 'contempt']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load models
emotion_net = cv2.dnn.readNetFromONNX("emotion-ferplus-8.onnx")
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_color = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # ---------- Emotion ----------
        try:
            face_resized = cv2.resize(face_gray, (64, 64))
            blob_emotion = cv2.dnn.blobFromImage(face_resized, scalefactor=1/255.0, size=(64, 64),
                                                 mean=(0.0,), swapRB=False, crop=False)
            emotion_net.setInput(blob_emotion)
            emotion_preds = emotion_net.forward()
            emotion = EMOTIONS[np.argmax(emotion_preds)]
        except:
            emotion = "N/A"

        # ---------- Age & Gender ----------
        try:
            face_blob = cv2.dnn.blobFromImage(face_color, 1, (227, 227),
                                              (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            gender_net.setInput(face_blob)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]

            age_net.setInput(face_blob)
            age = AGE_LIST[age_net.forward()[0].argmax()]
        except:
            gender = "N/A"
            age = "N/A"

        # ---------- Display ----------
        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Age, Gender & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
