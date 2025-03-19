import os
import pickle
import sys

import cv2

from utils import DEBUG, EMOTIONS, get_face_landmarks, print_info

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

emotions = EMOTIONS
# Load model using absolute path with proper extension
model_path = os.path.join(script_dir, "model.pkl")
try:
    if not os.path.exists(model_path):
        print_info(f"Model file not found at: {model_path}")
        print_info("Please train the model first using train_model.py")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print_info(f"Successfully loaded model from {model_path}")
except Exception as e:
    print_info(f"Error loading model: {str(e)}")
    sys.exit(1)

# Try camera index 0 (default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print_info("Error: Could not open camera. Trying backup index...")
    # Try alternative indices if 0 doesn't work
    for i in range(1, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print_info(f"Successfully opened camera with index {i}")
            break

    if not cap.isOpened():
        print_info("Could not open any camera. Exiting.")
        sys.exit(1)

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()
    if not ret:
        break

    if DEBUG:
        print("\n--- New frame ---")
        print(f"Frame shape: {frame.shape}")

    # Add basic face detection for visual feedback
    # This is separate from the landmark detection, just for UI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    face_landmarks = get_face_landmarks(
        frame, draw=False
    )  # Don't draw here to avoid more errors

    if DEBUG:
        print(f"Got {len(face_landmarks)} landmark values")

    if len(face_landmarks) >= 1404:  # Ensure we have the expected number of landmarks
        try:
            # Make prediction
            prediction = model.predict([face_landmarks])

            if DEBUG:
                print(f"Raw prediction: {prediction}")
                print(f"Prediction type: {type(prediction)}")
                print(
                    f"Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'N/A'}"
                )

            # Get the predicted emotion index and ensure it's within valid range
            predicted_idx = int(prediction[0])
            if predicted_idx < 0 or predicted_idx >= len(emotions):
                emotion_text = f"Invalid prediction: {predicted_idx}"
                color = (0, 0, 255)  # Red
            else:
                emotion_text = emotions[predicted_idx]
                color = (0, 255, 0)  # Green

            # Show prediction confidence if available
            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba([face_landmarks])[0]
                    confidence = probas[predicted_idx]
                    emotion_text += f" ({confidence:.2f})"
                except Exception as e:
                    if DEBUG:
                        print(f"Could not get probabilities: {e}")

            cv2.putText(
                frame,
                emotion_text,
                (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
            )
        except Exception as e:
            cv2.putText(
                frame,
                f"Error: {str(e)[:20]}",
                (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            if DEBUG:
                print(f"Prediction error: {str(e)}")
    else:
        cv2.putText(
            frame,
            f"Need atleast 1404 landmarks, got: {len(face_landmarks)}",
            (10, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Facial Expression Recognition", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
