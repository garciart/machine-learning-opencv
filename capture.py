#!/usr/bin/python3
''' Summary: Script to capture lot images '''
import datetime
import os
import sqlite3
from pathlib import Path
from sqlite3 import Error

import cv2
import numpy as np

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

# Directory of images or videos to run detection on
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
                    # Get zone data
                    sql = "SELECT Zone.*, Type.Description FROM Zone JOIN Type USING(TypeID) WHERE SourceID = {}".format(s['SourceID'])
                    zone = query_database(sql).fetchall()
                    if len(zone) == 0:
                        print("There are no zones defined for this source!")
                        break

                    # Make an overlay for transparent boxes
                    overlay = frame.copy()

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
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                        # Optional Draw the zone boundaries
                        cv2.polylines(frame, np.int32([np.array(poly_coords)]), True, colors[index], 10)

                    # Draw center crosshair
                    height, width, channels = frame.shape
                    cv2.drawMarker(frame, (int(width / 2), int(height / 2)), [255, 255, 0], cv2.MARKER_TRIANGLE_UP, 16, 2, cv2.LINE_4)
                    # Add timestamp
                    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 1)
                    # Save to file
                    saved = cv2.imwrite("{}/{}_{}_capture.jpg".format(CAPTURE_DIR, smalltimestamp, z['LotID']), frame)
                    if saved is True:
                        print("Image frame captured.")
                    else:
                        print("Could not save the image!")

                    # Clean up everything when finished
                    video_capture.release()
                    cv2.destroyAllWindows()

                else:
                    print("Cannot access source {} vic {}!".format(s['SourceID'], s['Location']))

        # Disconnect from the server
        cursor.close()
        conn.close()
        print("Job complete. Have an excellent day.")


if __name__ == '__main__':
    main()
