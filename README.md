# People Tracking Using YOLO and OpenCV

This project implements a people tracking system using YOLO (You Only Look Once) for object detection and OpenCV for computer vision tasks. It is designed to detect people in video footage and track their movement through specified entry and exit areas.

## Features

Real-Time Object Detection: Detects people in video frames using YOLO.

Movement Tracking: Monitors individuals as they pass through defined entry and exit areas.

Count Management: Tracks the number of people entering and exiting the specified areas.

Visualization: Provides visual feedback with bounding boxes, labels, and graphical indicators.

## Requirements

Python 3.x

OpenCV

Ultralytics YOLO

NumPy

Pandas

## Setup

Clone the Repository:

Clone this repository to your local machine and navigate to the project directory.

## Prepare the Environment:

Set up a virtual environment and install the required Python packages.

## Download YOLO Model:

Obtain the YOLOv8n model from the Ultralytics YOLO model hub and place it in the project directory.

## Update File Paths:

Ensure the script points to the correct locations for your video file and label file.

## Usage

Place your video file in the project directory and update the script to use this file.

Verify that the label file contains the correct class labels for YOLO.

Run the script to start processing the video.

## Code Explanation

Model Loading: Initializes YOLO for detecting objects in video frames.

Area Definitions: Defines entry and exit areas for tracking purposes.

Detection and Tracking: Detects people in each frame and monitors their movement relative to the defined areas.

Visualization: Draws bounding boxes and labels on video frames for real-time feedback.

## Notes

Adjust the coordinates of entry and exit areas based on your specific video footage and requirements.

Ensure the label file matches the output classes of the YOLO model.
