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
try:
    # Open database connection
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # prepare a cursor object using cursor() method
    cursor = conn.cursor()
except Error as ex:
    print("Error in connection: {}".format(ex))
    exit()

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


def draw_timestamp(display_frame):
    # Draw center crosshair
    height, width, channels = display_frame.shape
    cv2.drawMarker(display_frame, (int(width / 2), int(height / 2)), [255, 255, 0], cv2.MARKER_TRIANGLE_UP, 16, 2, cv2.LINE_4)
    # Add timestamp
    cv2.putText(display_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 1)
    return display_frame


def resize_image(display_frame):
    # Resize image if necessary
    scaling = int((768 * 100) / display_frame.shape[0]) if display_frame.shape[0] > 768 else 100
    width = int(display_frame.shape[1] * scaling / 100)
    height = int(display_frame.shape[0] * scaling / 100)
    dim = (width, height)
    display_frame = cv2.resize(display_frame, dim, interpolation=cv2.INTER_AREA)
    display_frame = draw_timestamp(display_frame)
    return display_frame

def query_database(sql):
    with conn:
        # execute SQL query using execute() method.
        cursor.execute(sql)
        # Commit on CREATE, INSERT, UPDATE, and DELETE
        if sql.lower().startswith("select") == False:
            conn.commit
        return cursor

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
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
# VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos") # Create to process videos
CAPTURE_DIR = os.path.join(ROOT_DIR, "demo_captures")

# Get database version
db_vers = query_database("SELECT sqlite_version();").fetchone()
print("Connected. Database version: {}".format(db_vers[0]))
# Get source data
source = query_database("SELECT * FROM Source").fetchall()

if len(source) == 0:
    print("No feeds found! Exiting now...")
    exit()
else:
    # Create a Mask-RCNN model in inference mode
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

    # Load pre-trained model
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    ''' NOT PART OF park.py '''
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
    ''' NOT PART OF park.py '''

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
                    # Clone image instead of using original
                    frame_copy = frame.copy()
                    # Convert the image from BGR color (which OpenCV uses) to RGB color
                    rgb_image = frame_copy[:, :, ::-1]

                    ''' NOT PART OF park.py - Start Demo 1 OpenCV Image Capture and Display '''
                    print('New image dimensions : ', rgb_image.shape)

                    # Show the frame of video on the screen
                    print("Click on the image window and press enter to continue...")
                    # Resize image if necessary
                    cv2.imshow('Video', resize_image(rgb_image))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print("Starting Mask R-CNN segmentation and detection...")
                    ''' NOT PART OF park.py - End Demo 1 Display '''

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

                    ''' NOT PART OF park.py - Start Demo 2 Mask R-CNN Segmentation and Detection Display '''
                    # Show the frame of video on the screen
                    print("Close the image window to continue...")
                    mrcnn.visualize.display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print("Starting vehicle localization...")
                    ''' NOT PART OF park.py - End Demo 2 Display '''

                    # Filter the results to only grab the car / truck bounding boxes
                    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

                    # Get zone data
                    sql = "SELECT Zone.*, Type.Description FROM Zone JOIN Type USING(TypeID) WHERE SourceID = {}".format(s['SourceID'])
                    zone = query_database(sql).fetchall()
                    if len(zone) == 0:
                        print("There are no zones defined for this source!")
                        break

                    print("Cars found in frame: {}".format(len(car_boxes)))

                    ''' NOT PART OF park.py - Demo 3 Vehicle Localization Display '''
                    # Clone image instead of using original
                    frame_copy = frame.copy()
                    # Draw each box on the frame. Do not use rgb_image with cv2!
                    for box in car_boxes:
                        # Display the box coordinates in the console
                        print("Car: ", box)
                        y1, x1, y2, x2 = box
                        # Draw the box on a copy of the original frame
                        frame_copy = cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Show the frame of video on the screen
                    print("Click on the image window and press enter to continue...")
                    # Resize image if necessary
                    cv2.imshow('Video', resize_image(frame_copy))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print("Creating and displaying zones...")
                    ''' NOT PART OF park.py - End Demo 3 Display '''

                    ''' NOT PART OF park.py - For Demo 4 & 5 Display '''
                    # Clone image instead of using original
                    frame_copy = frame.copy()
                    # Make an overlay for transparent boxes
                    overlay = frame.copy()
                    ''' NOT PART OF park.py '''

                    ''' NOT PART OF park.py - Demo 4 Creating and Displaying Zones Display '''
                    for index, z in enumerate(zone, start=0):
                        # Convert string representation of list to list
                        poly_coords = eval(z['PolyCoords'])

                        # BGR colors: Orange, Blue, Red, Gray, Yellow, Cyan, Pink, White
                        colors = [[0, 127, 255], [255, 0, 0], [0, 0, 255], [127, 127, 127],
                                  [0, 255, 255], [255, 255, 0], [127, 0, 255], [255, 255, 255]]

                        # Draw the filled zones
                        cv2.fillPoly(overlay, np.int32([np.array(poly_coords)]), colors[index + 4])
                        
                        # Set transparency for boxes
                        alpha = 0.4
                        # Add overlay to frame
                        frame_copy = cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0)

                        # Optional Draw the zone boundaries
                        cv2.polylines(frame_copy, np.int32([np.array(poly_coords)]), True, colors[index], 10)

                    # Save to file
                    saved = cv2.imwrite(CAPTURE_DIR + ("/" + smalltimestamp + "_d7_capture.jpg"), frame_copy)
                    if saved is True:
                        print("Image frame captured.")
                    else:
                        print("Could not save the image!")

                    # Show the frame of video on the screen
                    print("Click on the image window and press enter to continue...")
                    # Resize image if necessary
                    cv2.imshow('Video', resize_image(frame_copy))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print("Counting vehicles in zones...")
                    ''' NOT PART OF park.py - End Demo 4 Display '''

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
                                ''' NOT PART OF park.py - Part 1 Demo 5 display (Counting Vehicles in Zones) '''
                                # Draw the box on the frame
                                frame_copy = cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                ''' NOT PART OF park.py - End Part 1 Demo 5 display '''
                                # Count cars in zone
                                count += 1
                                # Delete the car to avoid double counting
                                np.delete(car_boxes, box)

                        # Make sure the number counted is not more than the number of spaces
                        count = count if count <= z['TotalSpaces'] else z['TotalSpaces']
                        print("Total cars in zone {} ({}): {}.".format(z['ZoneID'], z['Description'], count))
                        # Insert count into database
                        sql = "INSERT INTO OccupancyLog (ZoneID, LotID, TypeID, Timestamp, OccupiedSpaces, TotalSpaces) VALUES ({}, {}, {}, {}, {}, {})".format(z['ZoneID'], z['LotID'], z['TypeID'], "'{}'".format(timestamp), count, z['TotalSpaces'])
                        query_database(sql)

                    ''' NOT PART OF park.py - Part 2 Demo 5 display (Counting Vehicles in Zones) '''
                    # Show the frame of video on the screen
                    print("Click on the image window and press enter to continue...")
                    # Resize image if necessary
                    cv2.imshow('Video', resize_image(frame_copy))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    ''' NOT PART OF park.py - End Part 2 Demo 5 display '''

                    # Clean up everything when finished
                    video_capture.release()
                    cv2.destroyAllWindows()

                else:
                    print("Cannot access source {} vic {}!".format(s['SourceID'], s['Location']))

        cursor.close()
        conn.close()
        print("Job complete. Have an excellent day.")

if __name__ == '__main__':
    main()
