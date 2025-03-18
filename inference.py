# Facial Emotion Classification Inference Script
# This script can be used to classify facial emotions in:
# - Single images
# - Video files
# - Live webcam feed (default)

import argparse
import os
import pickle
import sys
import time

import cv2

from utils import DEBUG, EMOTIONS, get_face_landmarks, print_info, print_error


def load_model(model_path):
    """Load the trained model from the specified path."""
    try:
        if not os.path.exists(model_path):
            print_error(f"Model file not found at: {model_path}")
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print_info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print_error(f"Error loading model: {str(e)}")
        return None


def process_frame(frame, model, emotions):
    """Process a single frame to detect face and classify emotion."""
    if frame is None:
        return frame, None

    result_frame = frame.copy()

    # Detect face for visual feedback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Get facial landmarks for the emotion classification
    face_landmarks = get_face_landmarks(frame, draw=False)

    emotion_result = None

    # If faces are detected, draw bounding boxes
    for x, y, w, h in faces:
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Classify emotion if we have enough landmarks
    if len(face_landmarks) >= 1404:
        try:
            # Make prediction
            prediction = model.predict([face_landmarks])

            if DEBUG:
                print(f"Raw prediction: {prediction}")

            # Get the predicted emotion index
            predicted_idx = int(prediction[0])

            if 0 <= predicted_idx < len(emotions):
                emotion_text = emotions[predicted_idx]
                emotion_result = emotion_text

                # Get confidence if available
                confidence = None
                if hasattr(model, "predict_proba"):
                    try:
                        probas = model.predict_proba([face_landmarks])[0]
                        confidence = probas[predicted_idx]
                    except Exception:
                        pass

                # Display prediction on image
                color = (0, 255, 0)  # Green
                text = f"{emotion_text}"
                if confidence is not None:
                    text += f" ({confidence:.2f})"

                cv2.putText(
                    result_frame,
                    text,
                    (10, result_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )
            else:
                cv2.putText(
                    result_frame,
                    f"Invalid prediction: {predicted_idx}",
                    (10, result_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        except Exception as e:
            cv2.putText(
                result_frame,
                f"Error: {str(e)[:20]}",
                (10, result_frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
    else:
        cv2.putText(
            result_frame,
            f"Not enough facial landmarks ({len(face_landmarks)})",
            (10, result_frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return result_frame, emotion_result


def process_image(image_path, model, emotions, show_result=True, output_path=None):
    """Process a single image file."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print_error(f"Could not read image: {image_path}")
        return None

    # Process the image
    result_image, emotion = process_frame(image, model, emotions)

    # Display result if requested
    if show_result:
        cv2.imshow("Emotion Classification", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save result if output path is provided
    if output_path:
        cv2.imwrite(output_path, result_image)
        print_info(f"Result saved to: {output_path}")

    return emotion


def process_video(video_path, model, emotions, show_result=True, output_path=None):
    """Process a video file or webcam feed."""
    # Open video source
    if video_path.lower() == "webcam":
        cap = cv2.VideoCapture(0)
        print_info("Using webcam for input")
    else:
        cap = cv2.VideoCapture(video_path)
        print_info(f"Processing video: {video_path}")

    if not cap.isOpened():
        print_error("Error: Could not open video source")

        # Try alternative camera indices if webcam was requested
        if video_path.lower() == "webcam":
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print_info(f"Successfully opened camera with index {i}")
                    break

        if not cap.isOpened():
            return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS if not available

    # Setup video writer if output is requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    emotions_history = []
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        result_frame, emotion = process_frame(frame, model, emotions)

        if emotion is not None:
            emotions_history.append(emotion)
            # Keep only the most recent predictions
            if len(emotions_history) > 10:
                emotions_history.pop(0)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps_text = f"FPS: {frame_count / elapsed_time:.2f}"
            cv2.putText(
                result_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            frame_count = 0
            start_time = time.time()

        # Display result if requested
        if show_result:
            cv2.imshow("Facial Emotion Classification", result_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write frame to output video if requested
        if writer is not None:
            writer.write(result_frame)

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Return the most common emotion if we collected any
    if emotions_history:
        from collections import Counter

        most_common = Counter(emotions_history).most_common(1)[0][0]
        return most_common

    return None


def main():
    """Main function to handle command-line arguments and run the classifier."""
    parser = argparse.ArgumentParser(description="Facial Emotion Classification")

    # Input sources - webcam is the default
    parser.add_argument("-i", "--image", help="Path to input image")
    parser.add_argument("-v", "--video", help="Path to input video")

    # Other options
    parser.add_argument(
        "-m",
        "--model",
        default="model.pkl",
        help="Path to model file (default: model.pkl)",
    )
    parser.add_argument("-o", "--output", help="Path to output file (optional)")
    parser.add_argument(
        "--no-display", action="store_true", help="Don't display results in window"
    )

    args = parser.parse_args()

    # Load the model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = (
        args.model
        if os.path.exists(args.model)
        else os.path.join(script_dir, args.model)
    )
    model = load_model(model_path)

    if model is None:
        return 1

    # Get emotions from utils
    emotions = EMOTIONS
    print_info(f"Using emotions: {emotions}")

    # Process according to input type
    if args.image:
        emotion = process_image(
            args.image,
            model,
            emotions,
            show_result=not args.no_display,
            output_path=args.output,
        )
        if emotion:
            print_info(f"Detected emotion: {emotion}")
    elif args.video:
        emotion = process_video(
            args.video,
            model,
            emotions,
            show_result=not args.no_display,
            output_path=args.output,
        )
        if emotion:
            print_info(f"Most frequent emotion: {emotion}")
    else:
        # Default to webcam if no input specified
        emotion = process_video(
            "webcam",
            model,
            emotions,
            show_result=not args.no_display,
            output_path=args.output,
        )
        if emotion:
            print_info(f"Most frequent emotion: {emotion}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
