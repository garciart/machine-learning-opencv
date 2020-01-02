#!/usr/bin/python3
''' Summary: Script to capture and display images using OpenCV '''
import os
from pathlib import Path

import cv2

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent

# Directory of images or videos to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "demo_images")
# VIDEO_DIR = os.path.join(ROOT_DIR, "demo_videos") # Create to process videos

# Image, video or camera to process - set this to 0 to use your webcam instead of a file
# FRAME_SOURCE = [(IMAGE_DIR + "/demo_image.jpg")]
FRAME_SOURCE = ["https://raw.githubusercontent.com/garciart/Park/master/demo_images/demo_image.jpg"]


def main():
    print(FRAME_SOURCE)
    for f in FRAME_SOURCE:
        # Load the video file we want to run detection on
        video_capture = cv2.VideoCapture(f)

        # Attempt to capture a frame
        success, frame = video_capture.read()
        if success:
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_image = frame[:, :, ::-1]

            # Resize image if necessary
            scaling = int(
                (768 * 100) / rgb_image.shape[0]) if rgb_image.shape[0] > 768 else 100
            print('Original image dimensions : ', rgb_image.shape)
            width = int(rgb_image.shape[1] * scaling / 100)
            height = int(rgb_image.shape[0] * scaling / 100)
            dim = (width, height)
            rgb_image = cv2.resize(
                rgb_image, dim, interpolation=cv2.INTER_AREA)
            print('New image dimensions : ', rgb_image.shape)

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


if __name__ == '__main__':
    main()