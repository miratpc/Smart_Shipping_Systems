import os
from ultralytics import YOLO
import cv2

# Define the directories and file paths
VIDEOS_DIR = os.path.join('.', r'path/to/videos')
video_path = os.path.join(VIDEOS_DIR, r'path/to/video.mp4')
video_path_out = os.path.join(VIDEOS_DIR, r'path/to/video_out.mp4')

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame from the video.")
    cap.release()
    exit()

# Get frame dimensions
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load the model
model_path = os.path.join('groundup_best.pt')
if not os.path.isfile(model_path):
    print(f"Error: Model file not found at {model_path}")
    cap.release()
    out.release()
    exit()

model = YOLO(model_path)  # Load a custom model

threshold = 0.15

while ret:
    # Run the model on the frame
    results = model(frame)[0]

    # Process the results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper() + ' ' + str(round(score, 2)), 
                        (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write the frame with detections to the output video
    out.write(frame)

    # Read the next frame
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
