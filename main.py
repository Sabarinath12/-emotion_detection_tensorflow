import cv2
from keras.models import load_model
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model_path = 'C:/Users/voxs/Desktop/deepfacts/emotions.hdf5' 
emotion_model = load_model(emotion_model_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces: 
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        roi_gray_resized = cv2.resize(roi_gray, (64, 64))
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)
        roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)
        roi_gray_resized = roi_gray_resized.astype('float32')
        roi_gray_resized /= 255.0
        emotion_prediction = emotion_model.predict(roi_gray_resized)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Face Tracking with Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
