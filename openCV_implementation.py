import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/model_1.keras")

class_names = [
    "Anushka_Sharma",
    "Barack_Obama",
    "Bill_Gates",
    "Dalai_Lama",
    "Indira_Nooyi",
    "Melinda_Gates",
    "Narendra_Modi",
    "Sundar_Pichai",
    "Vikas_Khanna",
    "Virat_Kohli",
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml" # type: ignore
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize to model input size
        img = cv2.resize(face, (224, 224))

        # Normalize
        img = img / 255.0

        # Expand dimensions
        img = np.expand_dims(img, axis=0)

        # Prediction
        pred = model.predict(img, verbose=0)

        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display name and confidence
        text = f"{predicted_class} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition using OpenCV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
