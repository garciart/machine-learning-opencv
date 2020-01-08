#!python
#!/usr/bin/python3
''' Summary: Script to process images and update the database '''
import datetime
import os
import sqlite3
from pathlib import Path
from sqlite3 import Error

import cv2
import numpy as np
from numpy import array
from shapely.geometry import Polygon, asPoint

import mrcnn.config
import mrcnn.visualize
from mrcnn.model import MaskRCNN

DB_PATH = os.path.join(os.path.dirname(__file__), 'db/park.db')

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

# Directory of images or videos to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
# VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos") # Create to process videos

try:
    # Open database connection
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # prepare a cursor object using cursor() method
    cursor = conn.cursor()
    # execute SQL query using execute() method.
    cursor.execute("select sqlite_version();")
    # Fetch a single row using fetchone() method.
    db_vers = cursor.fetchone()
    print("Connected. Database version: {}".format(db_vers[0]))

    # Get source data
    sql = "SELECT * FROM Source"
    cursor.execute(sql)
    source = cursor.fetchall()
except Error as ex:
    print("Error in connection: {}".format(ex))
    exit()

if len(source) == 0:
    print("No feeds found! Exiting now...")
    exit()
else:
    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

    # Load pre-trained model
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Get UTC time before loop
    local_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    timestamp = datetime.datetime.now(local_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
    smalltimestamp = datetime.datetime.now(local_timezone).strftime("%Y%m%d")

    def main():
        for s in source:
            if s['Active'] == True:

                # Source password decryption code would go here

                # Video file or camera feed to process
                FRAME_SOURCE = s['URI']

                # Load the source we want to run detection on
                video_capture = cv2.VideoCapture(FRAME_SOURCE)

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

                    try:
                        # Get source data
                        conn = sqlite3.connect(DB_PATH)
                        conn.row_factory = sqlite3.Row
                        # prepare a cursor object using cursor() method
                        cursor = conn.cursor()
                        sql = "SELECT Zone.*, Type.Description FROM Zone JOIN Type USING(TypeID) WHERE SourceID = {}".format(s['SourceID'])
                        cursor.execute(sql)
                        zone = cursor.fetchall()
                        if len(zone) == 0:
                            print("There are no zones defined for this source!")
                            break
                    except Error as ex:
                        print("Error in connection: {}".format(ex))
                        exit()

                    print("Cars found in frame: {}".format(len(car_boxes)))

                    for z in zone:
                        # Convert string representation of list to list
                        poly_coords = eval(z['PolyCoords'])

                        # Hold count of cars in zone
                        count = 0

                        # Draw each box on the frame
                        for box in car_boxes:
                            y1, x1, y2, x2 = box

                            if(((Polygon([(x1, y1), (x2, y1), (x1, y2), (x2, y2)])).centroid).intersects(Polygon(asPoint(array(poly_coords))))):
                                # Display the box coordinates in the console
                                print("Car: ", box)
                                # Count cars in zone
                                count += 1
                                # Delete the car to avoid double counting
                                np.delete(car_boxes, box)

                        # Make sure the number counted is not more than the number of spaces
                        count = count if count <= z['TotalSpaces'] else z['TotalSpaces']
                        print("Total cars in zone {} ({}): {}.".format(z['ZoneID'], z['Description'], count))
                        # Insert count into database
                        try:
                            conn = sqlite3.connect(DB_PATH)
                            conn.row_factory = sqlite3.Row
                            # prepare a cursor object using cursor() method
                            cursor = conn.cursor()
                            sql = "INSERT INTO OccupancyLog (ZoneID, LotID, TypeID, Timestamp, OccupiedSpaces, TotalSpaces) VALUES ({}, {}, {}, {}, {}, {})".format(z['ZoneID'], z['LotID'], z['TypeID'], "'{}'".format(timestamp), count, z['TotalSpaces'])
                            cursor.execute(sql)
                            conn.commit()
                        except Error as ex:
                            print("Error in connection: {}".format(ex))
                            exit()

                    # Clean up everything when finished
                    video_capture.release()
                    cv2.destroyAllWindows()

                else:
                    print("Cannot access source {} vic {}!".format(s['SourceID'], s['Location']))

        # Disconnect from the server
        conn.close()
        print("Job complete. Have an excellent day.")


if __name__ == '__main__':
    main()
