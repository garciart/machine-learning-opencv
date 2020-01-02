#!/usr/bin/python3
''' Summary: Script to detect and count cars, trucks, and buses '''
import os
from pathlib import Path

import cv2
import numpy as np

import mrcnn.config
from mrcnn.model import MaskRCNN


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.5

# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("Missing the mask_rcnn_coco.h5 dataset! Downloading now...")
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                 config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Directory of images or videos to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
# VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos") # Create to process videos

# Image, video or camera to process - set this to 0 to use your webcam instead of a file
# FRAME_SOURCE = [(IMAGE_DIR + "/demo_image.jpg")]
FRAME_SOURCE = ["https://raw.githubusercontent.com/garciart/Park/master/demo_images/demo_image.jpg"]


def main():
    for f in FRAME_SOURCE:
        # Load the video file we want to run detection on
        video_capture = cv2.VideoCapture(f)

        # Attempt to capture a frame
        success, frame = video_capture.read()
        if success:
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_image = frame[:, :, ::-1]

            # Run the image through the Mask R-CNN model to get results.
            results = model.detect([rgb_image], verbose=0)

            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]

            # The r variable will now have the results of detection:
            # - r['rois'] are the bounding box of each detected object
            # - r['class_ids'] are the class id (type) of each detected object
            # - r['scores'] are the confidence scores for each detection
            # - r['masks'] are the object masks for each detected object (which gives you the object outline)

            # Filter the results to only grab the car / truck bounding boxes
            car_boxes = get_car_boxes(r['rois'], r['class_ids'])

            print("Cars found in frame of video: ", len(car_boxes))

            # Draw each box on the frame. Do not use rgb_image with cv2!
            for box in car_boxes:
                # Display the box coordinates in the console
                print("Car: ", box)
                y1, x1, y2, x2 = box
                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            # Resize image if necessary
            scaling = int(
                (768 * 100) / frame.shape[0]) if frame.shape[0] > 768 else 100
            width = int(frame.shape[1] * scaling / 100)
            height = int(frame.shape[0] * scaling / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            # Show the frame of video on the screen
            cv2.imshow('Video', frame)

            # Hit any key to quit
            print("Press any key continue...")
            cv2.waitKey(0)

        else:
            print("Cannot access image or video!")

    # Clean up everything when finished
    video_capture.release()
    cv2.destroyAllWindows()

    print("Job complete. Have an excellent day.")


if __name__ == '__main__':
    main()