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
from collections import Counter  # Import Counter at the module level

import cv2

from utils import DEBUG, EMOTIONS, get_face_landmarks, print_error, print_info


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
        return frame, None, None

    result_frame = frame.copy()
    timestamp = time.time()

    # Detect face for visual feedback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Get facial landmarks for the emotion classification
    face_landmarks = get_face_landmarks(frame, draw=False)

    emotion_result = None
    debug_info = {}  # For diagnostic information

    # If faces are detected, draw bounding boxes
    for x, y, w, h in faces:
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Classify emotion if we have enough landmarks
    if len(face_landmarks) >= 1404:
        try:
            # Make prediction
            prediction = model.predict([face_landmarks])

            # Save the raw prediction for diagnostics
            debug_info["raw_prediction"] = (
                prediction.tolist()
                if hasattr(prediction, "tolist")
                else str(prediction)
            )

            # Get the predicted emotion index
            predicted_idx = int(prediction[0])
            debug_info["predicted_index"] = predicted_idx
            debug_info["available_classes"] = len(emotions)

            if 0 <= predicted_idx < len(emotions):
                emotion_text = emotions[predicted_idx]
                emotion_result = emotion_text
                debug_info["selected_emotion"] = emotion_text

                # Get confidence if available
                confidence = None
                if hasattr(model, "predict_proba"):
                    try:
                        probas = model.predict_proba([face_landmarks])[0]
                        confidence = probas[predicted_idx]
                        debug_info["confidences"] = (
                            probas.tolist()
                            if hasattr(probas, "tolist")
                            else str(probas)
                        )
                    except Exception as e:
                        debug_info["confidence_error"] = str(e)

                # Display prediction on image
                color = (0, 255, 0)  # Green
                text = f"{emotion_text}"
                if confidence is not None:
                    text += f" ({confidence:.2f})"

                # Print the timestamp and emotion dictionary to console
                emotion_dict = {
                    "timestamp": timestamp,
                    "emotion": emotion_text,
                    "confidence": confidence if confidence is not None else "N/A",
                    "debug": debug_info,
                }
                print(f"Detection: {emotion_dict}")

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
            debug_info["error"] = str(e)
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
        debug_info["landmarks_found"] = len(face_landmarks)

    return result_frame, emotion_result, timestamp


def process_image(image_path, model, emotions, show_result=True, output_path=None):
    """Process a single image file."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print_error(f"Could not read image: {image_path}")
        return None

    # Process the image
    result_image, emotion, timestamp = process_frame(image, model, emotions)

    # Display result if requested
    if show_result:
        cv2.imshow("Emotion Classification", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save result if output path is provided
    if output_path:
        cv2.imwrite(output_path, result_image)
        print_info(f"Result saved to: {output_path}")

    # Output the detection as a dictionary
    if emotion:
        detection = {"timestamp": timestamp, "emotion": emotion}
        print(f"Final detection: {detection}")

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
    timestamps = []
    frame_count = 0
    start_time = time.time()

    # Count detections per emotion to check for bias
    emotion_counts = {emotion: 0 for emotion in emotions}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        result_frame, emotion, timestamp = process_frame(frame, model, emotions)

        if emotion is not None:
            # Track emotion and timestamp
            emotions_history.append(emotion)
            timestamps.append(timestamp)

            # Update the count for this emotion
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1

            # Print detection dictionary to console
            detection = {"timestamp": timestamp, "emotion": emotion}
            print(f"Real-time detection: {detection}")

            # Keep only the most recent predictions
            if len(emotions_history) > 10:
                emotions_history.pop(0)
                timestamps.pop(0)

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

            # Every second, show the distribution of emotion detections
            print(f"Emotion detection counts: {emotion_counts}")

        # Display result if requested
        if show_result:
            cv2.imshow("Facial Emotion Classification", result_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):  # Reset counts when 'r' is pressed
                print("Resetting emotion counts")
                emotion_counts = {emotion: 0 for emotion in emotions}

        # Write frame to output video if requested
        if writer is not None:
            writer.write(result_frame)

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Print final statistics
    print("\n--- Final Statistics ---")
    print(f"Emotion detection counts: {emotion_counts}")
    print(f"Total frames with emotions detected: {sum(emotion_counts.values())}")

    # Return the most common emotion if we collected any
    if emotions_history:
        most_common = Counter(emotions_history).most_common(1)[0][0]
        final_result = {
            "most_common_emotion": most_common,
            "emotion_counts": emotion_counts,
            "total_detections": sum(emotion_counts.values()),
        }
        print(f"\nFinal result: {final_result}")
        return most_common, emotion_counts, emotions_history

    return None, emotion_counts, []


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

    # Parse arguments before entering try block
    args = parser.parse_args()

    try:
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

        emotion = None
        emotion_counts = {}
        emotions_history = []

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
            emotion, emotion_counts, emotions_history = process_video(
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
            emotion, emotion_counts, emotions_history = process_video(
                "webcam",
                model,
                emotions,
                show_result=not args.no_display,
                output_path=args.output,
            )
            if emotion:
                print_info(f"Most frequent emotion: {emotion}")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        # Optionally save the intermediate results if an output path was specified
        if (
            args.output
            and "emotion_counts" in locals()
            and "emotions_history" in locals()
        ):
            print_info(f"Saving intermediate results to: {args.output}")
            # Save the emotion counts or other relevant data to a text file
            try:
                with open(args.output + ".txt", "w") as f:
                    f.write(f"Emotion counts: {emotion_counts}\n")
                    f.write(
                        f"Total frames with emotions detected: {sum(emotion_counts.values())}\n"
                    )
                    if emotions_history:
                        most_common = Counter(emotions_history).most_common(1)[0][0]
                        f.write(f"Most frequent emotion: {most_common}\n")
                print_info(
                    f"Successfully saved intermediate emotion counts to {args.output}.txt"
                )
            except Exception as e:
                print_error(
                    f"Error saving intermediate results: {str(e)}. Please ensure the file path is valid."
                )
        sys.exit(0)  # Exit gracefully
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
