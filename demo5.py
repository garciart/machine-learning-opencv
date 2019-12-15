#!python
''' Summary: Script to identify and count vehicles in zones '''
import os
import cv2
from pathlib import Path
import mrcnn.config
from mrcnn.model import MaskRCNN
import numpy as np
import datetime
from numpy import array
from shapely.geometry import asPoint
from shapely.geometry import Polygon

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
VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos")

# Image, video or camera to process - set this to 0 to use your webcam instead of a video file
# FRAME_SOURCE = [(IMAGE_DIR + "\\demo_image1.jpg"),(IMAGE_DIR + "\\demo_image2.jpg"),(IMAGE_DIR + "\\demo_image3.jpg")]
# FRAME_SOURCE = [(VIDEO_DIR + "\\demo_video1.mp4"),(VIDEO_DIR + "\\demo_video2.mp4"),(VIDEO_DIR + "\\demo_video3.mp4")]
FRAME_SOURCE = [(IMAGE_DIR + "\\demo_imagex1.jpg")]

# Get UTC time before loop
local_timezone = datetime.datetime.now(
    datetime.timezone.utc).astimezone().tzinfo
timestamp = datetime.datetime.now(
    local_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")


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

            # Read clockwise from top-left corner
            poly_coords = ([[816, 1150], [3200, 1140], [3200, 1350], [816, 1400]],
                           [[240, 1140], [815, 1150], [815, 1400], [150, 1400]])

            # BGR colors: Orange, Blue, Red, Gray, Yellow, Cyan, Pink, White
            colors = [[0, 127, 255], [255, 0, 0], [0, 0, 255], [127, 127, 127], [
                0, 255, 255], [255, 255, 0], [127, 0, 255], [255, 255, 255]]

            # Make an overlay for transparent boxes
            overlay = frame.copy()

            for index, p in enumerate(poly_coords, start=0):
                # Hold count of cars in zone
                count = 0
                # Draw the filled zones
                cv2.fillPoly(overlay, np.int32(
                    [np.array(p)]), colors[index + 4])
                # Draw each box on the frame. Do not use rgb_image with cv2!
                for box in car_boxes:
                    # Get the box coordinates
                    y1, x1, y2, x2 = box
                    # Only show cars in the zones!
                    if((Polygon([(x1, y1), (x2, y1), (x1, y2), (x2, y2)])).intersects(Polygon(asPoint(array(p))))):
                        # Draw the box and add to overlay
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), colors[index], 5)
                        # Count car in zone
                        count += 1
                        # Delete the car to avoid double counting
                        np.delete(car_boxes, box)

                print("Total cars in zone {}: {}".format(
                    poly_coords.index(p), count))

            # Set transparency for boxes
            alpha = 0.4
            # Add overlay to frame
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Draw center crosshair
            height, width, channels = frame.shape
            cv2.drawMarker(frame, (int(width / 2), int(height / 2)),
                           [255, 255, 0], cv2.MARKER_TRIANGLE_UP, 16, 2, cv2.LINE_4)
            # Add timestamp
            cv2.putText(frame, timestamp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 1)

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
