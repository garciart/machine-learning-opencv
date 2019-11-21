#!python
import os
import cv2
from pathlib import Path

# Root directory of the project
# ROOT_DIR = Path(".")
ROOT_DIR = os.getcwd()

# Directory of images or videos to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos")

print(IMAGE_DIR)
# Image, video or camera to process - set this to 0 to use your webcam instead of a video file
# FRAME_SOURCE = [(IMAGE_DIR + "\\demo_image1.jpg"),(IMAGE_DIR + "\\demo_image2.jpg"),(IMAGE_DIR + "\\demo_image3.jpg")]
# FRAME_SOURCE = [(VIDEO_DIR + "\\demo_video1.mp4"),(VIDEO_DIR + "\\demo_video2.mp4"),(VIDEO_DIR + "\\demo_video3.mp4")]
FRAME_SOURCE = [(IMAGE_DIR + "\\demo_image1.jpg")]
print(FRAME_SOURCE)
for f in FRAME_SOURCE:
    # Load the video file we want to run detection on
    video_capture = cv2.VideoCapture(f)

    # Attempt to capture a frame
    success, frame = video_capture.read()
    if success:
        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = frame[:, :, ::-1]

        # Show the frame of video on the screen
        cv2.imshow('Video', rgb_image)
        # Hit any key to quit
        cv2.waitKey(0)

    else:
        print("Cannot access image or video!")

    # Clean up everything when finished
    video_capture.release()
    cv2.destroyAllWindows()

print("Job complete. Have an excellent day.")