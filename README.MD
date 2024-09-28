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
## Data Structure

For training and inference using the YOLOv8n model, the data needs to be organized in a specific folder structure. The data format and folder setup are crucial for the proper functioning of the model.

### Folder Structure
- The data should be placed in a main folder that contains **two subfolders**:
  - `/images/`: This folder holds all the image files (e.g., `.jpg`, `.png`) used for training, validation, and testing.
  - `/labels/`: This folder contains the corresponding label files (e.g., `.txt`), with annotations that describe the objects in the images.

### Label Format
- Each label file should have the same name as its corresponding image but with a `.txt` extension.
- The label files contain the following information about each object in the image:
  - **Class Index**: The index number for the object's class (e.g., `0` for a boat, `1` for a person).
  - **Bounding Box Coordinates**: The bounding box is represented by four values:
    1. **x_center**: The x-coordinate of the center of the bounding box, as a ratio of the image width. For example, the center of the image would have an `x_center` value of `0.5`.
    2. **y_center**: The y-coordinate of the center of the bounding box, as a ratio of the image height. Similarly, the center of the image would have a `y_center` value of `0.5`.
    3. **width**: The width of the bounding box, as a ratio of the image width.
    4. **height**: The height of the bounding box, as a ratio of the image height.

### Example of a Label File

An example label file for a boat detected in an image might look like this:

```bash
0 0.1575 0.6317705 0.126875 0.04916699999999996
0 0.3996484375 0.7306253333333332 0.15499999999999997 0.07500000000000018
0 0.7063671875 0.6042190833333333 0.20999999999999996 0.06583299999999996

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

### 1. Data Acquisition and Preparation
- We started by obtaining a publicly available boat bounding box dataset from **OpenImagesV4**, consisting of **5000 training**, **1000 validation**, and **1000 test images**.
- The dataset was formatted to fit the YOLOv8n model's requirements.

### 2. Model Fine-Tuning
- We fine-tuned a **pre-trained YOLOv8n model** that was originally trained on **100 classes** from the OpenImagesV4 dataset, specifically for boat detection.
- Initial results showed promise, though further optimization was attempted by cleaning some faulty data.
- However, the optimization did not lead to significant improvements.

### 3. Training a New Model
- We trained a new model with the same YOLOv8n architecture but observed that the pre-trained model consistently yielded better results.

### 4. Experimenting with OpenImages v8 Dataset
- To explore further, we switched to a newer version of the dataset, **OpenImagesV8**, downloading **4000 images** (3000 train, 500 val, 500 test).
- The newly trained model using OpenImagesV8 performed slightly worse in detecting boats compared to the older model trained with OpenImagesV4.

### 5. Class Adjustments and Model Performance
- We experimented with using separate classes for different types of boats and one class for people.
- This adjustment resulted in a model that was a **better general detector**, but its performance in detecting boats was slightly worse.

### 6. Distance Calculation Algorithm Development
- After deciding to proceed with the best-performing model, we shifted focus to developing the **distance calculation algorithm**.
- Initially, we completed the **vertical distance calculation**, then combined it with **lateral distance** to calculate total ship distance.
- The algorithm was first tested with **short distances in an office environment**.

### 7. Field Testing and Final Results
- We collected field data to rigorously test our distance calculation algorithm and concluded that it worked with approximately **90% accuracy**.



## How to Use

### 1. **Prepare the dataset**:
   - Use the video tools to split videos into frames and annotate them.
   - Transform annotations into YOLOv8 format.

### 2. **Train the model**:
   - Use the scripts in the `/Boat_Detection/` directory to train the YOLOv8n model on your dataset.

### 3. **Run the detection**:
   - Use the `/bounding_boxes/` code to generate bounding boxes for ships in new video frames.

### 4. **Calculate distances**:
   - Run the distance calculation tools on detected bounding boxes to estimate the distance of ships from the camera.

### 5. **Annotate videos**:
   - Annotate your videos with the calculated distances for visualization.


## Camera Specifications

- Camera calibration data (intrinsic parameters) is stored in the `/camera_specs/` directory, with separate YAML files for each camera model.
- The distance calculation tools automatically reference the correct YAML file based on the camera used for the video.