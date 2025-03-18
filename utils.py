import os
import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils, face_mesh
from mediapipe.tasks import python

# Determine model path relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Initialize FaceLandmarker with the relative model path
options = python.vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
landmarker = python.vision.FaceLandmarker.create_from_options(options)


def get_face_landmarks(image, draw=False):
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

        if results.face_landmarks:
            if draw:
                drawing_spec = drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
                # Convert frozenset to list for the connections
                connections = list(face_mesh.FACEMESH_CONTOURS)
                drawing_utils.draw_landmarks(
                    image,
                    results.face_landmarks[0],
                    connections,
                    drawing_spec,
                    drawing_spec,
                )
            ls_single_face = results.face_landmarks[0]
            xs_ = [lm.x for lm in ls_single_face]
            ys_ = [lm.y for lm in ls_single_face]
            zs_ = [lm.z for lm in ls_single_face]
            min_x, min_y, min_z = min(xs_), min(ys_), min(zs_)
            for j in range(len(xs_)):
                image_landmarks.append(xs_[j] - min_x)
                image_landmarks.append(ys_[j] - min_y)
                image_landmarks.append(zs_[j] - min_z)
        return image_landmarks
    except Exception as e:
        print(f"Error processing image in get_face_landmarks: {str(e)}")
        return []
