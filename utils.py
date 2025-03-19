import os
import re

import cv2
import dotenv
import mediapipe as mp
from colorama import Fore, Style
from mediapipe.python.solutions import drawing_utils, face_mesh
from mediapipe.tasks import python

# Load environment variables from .env file
dotenv.load_dotenv()

# Determine model path relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Parse environment variables
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
IMAGES_PER_CLASS = int(os.getenv("IMAGES_PER_CLASS", "500"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "7"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
MODELS_DIR = os.getenv(
    "MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)
# Parse emotions from comma-separated list
emotions_str = os.getenv("EMOTIONS", "angry,disgust,fear,happy,neutral,sad,surprise")
EMOTIONS = [e.strip() for e in emotions_str.split(",")]
# Whether to crop only the face region or use the entire frame
CROP_FACE_ONLY = os.getenv("CROP_FACE_ONLY", "False").lower() in ("true", "1", "t")

# Initialize FaceLandmarker with the relative model path
options = python.vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
landmarker = python.vision.FaceLandmarker.create_from_options(options)


def get_face_landmarks(image, draw=False):
    """Extract face landmarks from an image using MediaPipe FaceLandmarker."""
    # Check if the image is valid
    if image is None or image.size == 0:
        print("Warning: Empty or invalid image provided to get_face_landmarks")
        return []

    try:
        # Convert BGR to RGB
        image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create mp.Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_input_rgb)

        # Detect landmarks
        results = landmarker.detect(mp_image)

        image_landmarks = []

        # Only print debug messages if DEBUG is True
        if DEBUG:
            print(f"FaceLandmarker results type: {type(results)}")
            print(f"Has face_landmarks attribute: {'face_landmarks' in dir(results)}")

        if hasattr(results, "face_landmarks") and results.face_landmarks:
            if DEBUG:
                print(f"Number of faces detected: {len(results.face_landmarks)}")
                print(f"First landmarks data type: {type(results.face_landmarks[0])}")

            if len(results.face_landmarks) > 0:
                if draw:
                    try:
                        drawing_spec = drawing_utils.DrawingSpec(
                            thickness=2, circle_radius=1
                        )
                        connections = list(face_mesh.FACEMESH_CONTOURS)
                        drawing_utils.draw_landmarks(
                            image,
                            results.face_landmarks[0],
                            connections,
                            drawing_spec,
                            drawing_spec,
                        )
                    except Exception as e:
                        print(f"Error drawing landmarks: {e}")

                # Safe extraction of coordinates
                landmarks_data = results.face_landmarks[0]

                # Check first landmark to understand structure
                first_landmark = landmarks_data[0] if landmarks_data else None
                if DEBUG:
                    print(f"First landmark type: {type(first_landmark)}")
                    print(f"First landmark attributes: {dir(first_landmark)}")

                # Extract coordinates safely
                xs_ = []
                ys_ = []
                zs_ = []

                for landmark in landmarks_data:
                    if (
                        hasattr(landmark, "x")
                        and hasattr(landmark, "y")
                        and hasattr(landmark, "z")
                    ):
                        xs_.append(landmark.x)
                        ys_.append(landmark.y)
                        zs_.append(landmark.z)
                    else:
                        # Alternative approach if attributes are not directly accessible
                        try:
                            xs_.append(float(landmark[0]) if len(landmark) > 0 else 0)
                            ys_.append(float(landmark[1]) if len(landmark) > 1 else 0)
                            zs_.append(float(landmark[2]) if len(landmark) > 2 else 0)
                        except (IndexError, TypeError):
                            print(
                                f"Could not extract coordinates from landmark: {landmark}"
                            )

                if xs_ and ys_ and zs_:
                    # Only process if we have data
                    min_x, min_y, min_z = min(xs_), min(ys_), min(zs_)
                    for j in range(len(xs_)):
                        image_landmarks.append(xs_[j] - min_x)
                        image_landmarks.append(ys_[j] - min_y)
                        image_landmarks.append(zs_[j] - min_z)

                    if DEBUG:
                        print(
                            f"Successfully extracted {len(image_landmarks)} landmark values"
                        )
                else:
                    print("No valid landmark points found")
            else:
                print("No face landmarks in results.face_landmarks")
        else:
            print("Results object does not have face_landmarks or it's empty")

        return image_landmarks
    except Exception as e:
        print(f"Error processing image in get_face_landmarks: {str(e)}")
        return []


def load_face_cascade():
    # Try to load cascade from FACE_CASCADE_PATH env variable; fallback if not found.
    cascade_file = os.getenv("FACE_CASCADE_PATH")
    if cascade_file and os.path.exists(cascade_file):
        return cv2.CascadeClassifier(cascade_file)
    else:
        default_path = os.path.join(
            cv2.data.haarcascades,  # type: ignore
            "haarcascade_frontalface_default.xml",
        )
        print(
            f"FACE_CASCADE_PATH not found. Using default cascade from: {default_path}"
        )
        return cv2.CascadeClassifier(default_path)


def detect_and_crop_face(frame):
    """
    Detect and optionally crop the face from a frame.
    If CROP_FACE_ONLY is True, returns only the face region.
    Otherwise, returns the full frame with face detection info.
    """
    if frame is None or frame.size == 0:
        return None

    # If we don't need to crop, return the full frame but still detect the face
    if not CROP_FACE_ONLY:
        # We still want to check if a face is detected
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = landmarker.detect(mp_image)

        if not results.face_landmarks:
            # Try Haar cascade as fallback
            cascade = load_face_cascade()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                print("No face detected in frame, but returning full frame anyway.")
            else:
                print("Face detected using Haar cascade, returning full frame.")

        return frame  # Return the full frame regardless

    # If CROP_FACE_ONLY is True, proceed with original cropping logic
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = landmarker.detect(mp_image)
    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        h, w, _ = frame.shape
        min_x = int(min(xs) * w)
        max_x = int(max(xs) * w)
        min_y = int(min(ys) * h)
        max_y = int(max(ys) * h)
        margin = 10
        min_x = max(min_x - margin, 0)
        min_y = max(min_y - margin, 0)
        max_x = min(max_x + margin, w)
        max_y = min(max_y + margin, h)
        return frame[min_y:max_y, min_x:max_x]
    else:
        print(
            "MediaPipe FaceLandmarker did not detect a face. Falling back to Haar cascade."
        )
        # Fallback using Haar cascade detection
        cascade = load_face_cascade()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            print("Haar cascade did not detect a face either.")
            return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        print("Second case: Haar cascade detected face.")
        return frame[y : y + h, x : x + w]


def get_project_dir():
    """Get the absolute path to the project root directory."""
    return os.path.dirname((os.path.abspath(__file__)))


def print_info(msg):
    base = Fore.CYAN
    formatted = format_numbers(msg, base)
    print(base + formatted + Style.RESET_ALL)


def get_directory_paths():
    """Return paths for logs, models, data, and templates; create them if missing."""
    project_dir = get_project_dir()
    directories = {
        "logs": os.path.join(project_dir, "logs"),
        "models": os.path.join(project_dir, "models"),
        "data": os.path.join(project_dir, "data"),
        "templates": os.path.join(project_dir, "templates"),
    }
    for name, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
    return directories


def format_numbers(msg, base_color):
    """
    Wrap numbers in the message with a different color (magenta) and then resume the base color.
    """
    return re.sub(r"(\d+)", lambda m: Fore.MAGENTA + m.group(0) + base_color, msg)


def print_error(msg):
    base = Fore.RED
    formatted = format_numbers(msg, base)
    print(base + formatted + Style.RESET_ALL)
