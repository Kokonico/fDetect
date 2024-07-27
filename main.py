"""fDetect - a wildfire detection system using already existing cameras and machine learning algorithms"""
from os import write

import cv2
import numpy as np
import os
import time
import requests
import json
import sys
import collections
import threading

# synopsis:
# 1. get live video feed from camera
# every 15 frames, run the detection algorithm

# global variables
# TODO: make this better
camera_latest_frame_url = "https://cameras.alertcalifornia.org/public-camera-data/Axis-NorthPlacerville/latest-frame.jpg"

def crop_watermark(frame):
    """Crops the watermark from the bottom of the frame"""
    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Define the crop area (adjust these values as needed)
    bottom_crop = 30  # Crop 100 pixels from the bottom
    top_crop = 160 # Crop 156 pixels from the top

    # Crop the frame
    cropped_frame = frame[:height - bottom_crop, :]
    cropped_frame = cropped_frame[top_crop:, :]

    # Return the cropped frame
    return cropped_frame

def get_frame():
    """Returns the latest frame from the camera"""
    try:
        # Get the latest frame from the camera
        response = requests.get(camera_latest_frame_url)
        response.raise_for_status()  # Check if the request was successful

        # Convert the response to an image
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        # Crop the watermark from the frame
        image = crop_watermark(image)
        return image

    except requests.RequestException as e:
        print(f"Error fetching the frame: {e}", file=sys.stderr)
        return None
    except ValueError as e:
        print(f"Error processing the frame: {e}", file=sys.stderr)
        return None


def get_frames(directory, max_frames = 15):
    # background thread to get frames
    while True:
        frame = get_frame()
        if frame is not None:
            with open(os.path.join(directory, f"{time.time()}.jpg"), "wb") as file:
                file.write(cv2.imencode(".jpg", frame)[1].tobytes())
            if len(os.listdir(directory)) > max_frames:
                os.remove(os.path.join(directory, os.listdir(directory)[0]))
        time.sleep(15)

# start the background thread to get frames
directory = "frames"
os.makedirs(directory, exist_ok=True)
thread = threading.Thread(target=get_frames, args=(directory,))
thread.start()


