# Ship Distance Detection with YOLOv8n

This repository contains a project designed to detect and calculate the distance of ships in the sea. Using YOLOv8n, the model detects ships in video frames by generating bounding boxes at the point where ships make contact with the water. The distance from the camera to the ships is then calculated based on these bounding boxes.

## Features

- **YOLOv8n Model Training**: Separate code for training the YOLOv8n model using a custom dataset of ships in the sea.
- **Bounding Box Creation**: Detects ships in images and videos, generating bounding boxes at the point of contact with the water.
- **Distance Calculation**: Tools to calculate the distance of ships from the camera using the bounding box and ground measurements.
- **Annotation Tools**: Scripts for annotating videos with calculated distances.
- **Video Processing**: Tools to split videos into frames for training the model.
- **Camera Calibration**: YAML files storing intrinsic parameters and specifications for different cameras.

## Prerequisites

- Python 3.10.8
- OpenCV

To install dependencies, run:

```bash
pip install -r requirements.txt

```

## Repository Structure
- `/best_models/`: Contains weight for the YOLOv8n model.
- `/Boat_Detection/train.py`: Contains script for training the YOLOv8n model.
- `/Boat_Detection/draw_boxes.py`: Code for creating bounding boxes on new images and videos.
- `/DistanceCalculation/distance_inference.py`: Tools for calculating the distance of ships from their bounding boxes.
- `/DistanceCalculation/distance_tools.py`: Tools to annotate videos with ship distances
- `/helper/split_videos.py/`: Utilities for splitting videos into frames for training purposes.
- `/camera_specs/`: YAML files storing intrinsic values and specifications for different cameras.

## Workflow

1. **Data Preparation**
    - Video data is split into frames using provided tools.
    - The frames are annotated using a custom annotation tool, which identifies the bounding boxes where the ships contact the water. 
    - Annotations are transformed into the YOLOv8 format for training.

2. **Model Training**
    - The YOLOv8n model is trained on the annotated ship images to detect ships in new videos and images.
    - Separate scripts handle the training process, leveraging the preprocessed data.

3. **Bounding Box Detection**
    - Once trained, the model is used to detect ships and generate bounding boxes on new video frames.
    - This includes marking the point where each ship contacts the water.

4. **Distance Calculation**
    - Using the bounding box data, the distance of each ship from the camera is calculated.
    - This is done using ground measurements, camera specifications, and calibration values stored in separate YAML files for each camera.

5. **Annotation of Video with Distances**
    - After calculating distances, the videos are annotated with the distances of the detected ships, making it easier to interpret the data.



## How to Use

1. **Prepare the dataset**:
   - Use the video tools to split videos into frames and annotate them.
   - Transform annotations into YOLOv8 format.

2. **Train the model**:
   - Use the scripts in the `/Boat_Detection/` directory to train the YOLOv8n model on your dataset.

3. **Run the detection**:
   - Use the `/bounding_boxes/` code to generate bounding boxes for ships in new video frames.

4. **Calculate distances**:
   - Run the distance calculation tools on detected bounding boxes to estimate the distance of ships from the camera.

5. **Annotate videos**:
   - Annotate your videos with the calculated distances for visualization.


## Camera Specifications

- Camera calibration data (intrinsic parameters) is stored in the `/camera_specs/` directory, with separate YAML files for each camera model.
- The distance calculation tools automatically reference the correct YAML file based on the camera used for the video.