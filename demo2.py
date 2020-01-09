#!/usr/bin/python3
''' Summary: Script to classify objects using MaskRCNN and the COCo Dataset '''
import os
from pathlib import Path

import cv2

import mrcnn.config
import mrcnn.utils
import mrcnn.visualize
from mrcnn.model import MaskRCNN

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.5


# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(MODEL_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("Missing the mask_rcnn_coco.h5 dataset! Downloading now...")
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Directory of images or videos to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
# VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos") # Create to process videos

# Image, video or camera to process - set this to 0 to use your webcam instead of a file
# FRAME_SOURCE = [(IMAGE_DIR + "/demo_image.jpg")]
FRAME_SOURCE = ["https://raw.githubusercontent.com/garciart/Park/master/demo_images/demo_image.jpg"]


def main():
    for f in FRAME_SOURCE:
        # Load the source we want to run detection on
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

            # Show the frame of video on the screen
            mrcnn.visualize.display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

            # Hit any key to quit
            print("Close the window to continue...")
            cv2.waitKey(0)

        else:
            print("Cannot access image or video!")

        # Clean up everything when finished
        video_capture.release()
        cv2.destroyAllWindows()

    print("Job complete. Have an excellent day.")


if __name__ == '__main__':
    main()
