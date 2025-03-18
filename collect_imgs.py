# Project: Sign Language Detector
# Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
# Owner: VKrishna04
# Organization: Life-Experimentalist
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# Add the parent directory (project root) to sys.path so that utils can be found.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2

from utils import (
    EMOTIONS,
    IMAGES_PER_CLASS,
    detect_and_crop_face,
    get_directory_paths,
    get_project_dir,
    print_info,
)


def create_img():
    # Get project directory and data directory from utils
    PROJECT_DIR = get_project_dir()  # noqa: F841
    DATA_DIR = get_directory_paths()["data"]

    # Ensure a "train" folder exists under DATA_DIR
    train_dir = os.path.join(DATA_DIR, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    # Define emotions to match prepare_data.py structure
    emotions = EMOTIONS
    target_images = IMAGES_PER_CLASS

    cap = cv2.VideoCapture(0)
    # Loop over each emotion
    for emotion in emotions:
        emotion_dir = os.path.join(train_dir, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

        print_info(f"Collecting face data for emotion: {emotion}")

        # Wait for user confirmation to start capturing face images for the current emotion
        while True:
            ret, frame = cap.read()
            cv2.putText(
                frame,
                'Ready? Press "3" to start for emotion "{}"'.format(emotion),
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow("frame", frame)
            if cv2.waitKey(25) == ord("3"):
                break

        # Capture training images for current emotion
        counter = 0
        while counter < target_images:
            ret, frame = cap.read()
            face = detect_and_crop_face(frame)
            if face is not None:
                cv2.imshow("Face", face)
                cv2.imwrite(os.path.join(emotion_dir, f"{counter}.jpg"), face)
                print_info(f"Saved {emotion} image {counter}")
                counter += 1
            else:
                print_info("No face detected in current frame.")
                cv2.putText(
                    frame,
                    "No face detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow("frame", frame)
            cv2.waitKey(25)
    cap.release()
    print_info("Face data collection complete at {}".format(train_dir))
    cv2.destroyAllWindows()


# Allow the module to be imported without running the script and run only when executed directly
if __name__ == "__main__":
    create_img()
