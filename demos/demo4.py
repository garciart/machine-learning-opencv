#!python
import os
import cv2
from pathlib import Path
import numpy as np
import datetime

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent

# Directory of images or videos to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos")

# Image, video or camera to process - set this to 0 to use your webcam instead of a video file
# FRAME_SOURCE = [(IMAGE_DIR + "\\demo_image1.jpg"),(IMAGE_DIR + "\\demo_image2.jpg"),(IMAGE_DIR + "\\demo_image3.jpg")]
# FRAME_SOURCE = [(VIDEO_DIR + "\\demo_video1.mp4"),(VIDEO_DIR + "\\demo_video2.mp4"),(VIDEO_DIR + "\\demo_video3.mp4")]
FRAME_SOURCE = [(IMAGE_DIR + "\\demo_image1.jpg")]

# Get UTC time before loop
local_timezone = datetime.datetime.now(
    datetime.timezone.utc).astimezone().tzinfo
timestamp = datetime.datetime.now(
    local_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

for f in FRAME_SOURCE:
    # Load the video file we want to run detection on
    video_capture = cv2.VideoCapture(f)

    # Attempt to capture a frame
    success, frame = video_capture.read()
    if success:
        poly_pts = ([[751, 1150], [3200, 1140], [3200, 1350], [816, 1400], [816, 1300]],
                    [[150, 1400], [815, 1400], [815, 1300], [750, 1150], [240, 1140]])

        # BGR colors: Blue, Orange, Red, Gray, Cyan, Yellow, Pink, White
        colors = [[255, 0, 0], [0, 127, 255], [0, 0, 255], [127, 127, 127], [
            255, 255, 0], [0, 255, 255], [127, 0, 255], [255, 255, 255]]

        # Make an overlay for transparent boxes
        overlay = frame.copy()

        # Draw the filled zones
        for index, p in enumerate(poly_pts, start=0):
            cv2.fillPoly(overlay, np.int32([np.array(p)]), colors[index + 4])

        # Set transparency for boxes
        alpha = 0.4
        # Add overlay to frame
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Optional Draw the zone boundaries
        for index, p in enumerate(poly_pts, start=0):
            cv2.polylines(frame, np.int32(
                [np.array(p)]), True, colors[index], 10)

        # Draw center crosshair
        height, width, channels = frame.shape
        cv2.drawMarker(frame, (int(width / 2), int(height / 2)),
                       [255, 255, 0], cv2.MARKER_TRIANGLE_UP, 16, 2, cv2.LINE_4)
        # Add timestamp
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 1)

        # Resize image if necessary
        scaling = int(
            (768 * 100) / frame.shape[0]) if frame.shape[0] > 768 else 1
        print('Original image dimensions : ', frame.shape)
        width = int(frame.shape[1] * scaling / 100)
        height = int(frame.shape[0] * scaling / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        print('New image dimensions : ', frame.shape)

        # Show the frame of video on the screen
        cv2.imshow('Video', frame)
        # Save to file
        saved = cv2.imwrite("captured_frame.jpg", frame)
        if saved is True:
            print("Image frame captured.")
        else:
            print("Could not save the image!")
        # Hit any key to quit
        print("Press any key continue...")
        cv2.waitKey(0)

    else:
        print("Cannot access image or video!")

    # Clean up everything when finished
    video_capture.release()
    cv2.destroyAllWindows()

print("Job complete. Have an excellent day.")
